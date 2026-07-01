"""Ablation metrics for the MotionEditor x flow-matching dissertation.

Scores every method's output gif against the sharp teacher and the source video:
  - LPIPS-vs-teacher  (lower = sharper / closer to the epsilon teacher)
  - LPIPS-vs-source   (lower = better source identity / reconstruction)
  - CLIP-sim          (higher = more faithful to the edit prompt)
  - NFE               (from the registry, per method)
  - wall-clock        (optional, from a JSON timing file)

Side-by-side gifs (source|edit, width ~= 2*height) are auto-cropped to the right
(edit) panel. Single-panel gifs are used whole.

Run on the GPU box (transformers is already there; LPIPS may need `pip install
lpips`). CPU is fine — it only does a handful of forward passes.

    python3 metrics.py \
        --outputs-dir ../../fm_outputs \
        --source-dir ../../motionEditor/MotionEditor/data/case-1/images \
        --prompt "a girl is dancing" \
        --out ../../fm_outputs/metrics_table.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image, ImageSequence


# ---- method registry: name -> (gif path relative to --outputs-dir, NFE) ----
# Edit paths/NFE here as runs are added. `None` NFE prints as "-".
REGISTRY = [
    ("Baseline (DDIM-50 + null-text)", "baseline_epsilon_case1.gif", 50),
    ("+ MediaPipe (epsilon)",          "mediapipe_epsilon_case1.gif", 50),
    ("+ A/B naive flow",               "mediapipe_flow_case1.gif",    50),
    ("+ A/B naive flow (50-step ODE)", "flow_ode_fm_50step_case1.gif", 50),
    ("DPM-Solver++ (20)",              "dpm_outputs/sample-all.gif",  20),
    ("DPM-Solver++ (15)",              "dpm_15_outputs/dpm_solver_15.gif", 15),
    ("C3 consistency (1)",             "cons_outputs/cons_1step.gif",  1),
    ("C3 consistency (2)",             "cons_outputs/cons_2step.gif",  2),
    ("C3 consistency (4)",             "cons_outputs/sample-consistency.gif", 4),
    ("C3 consistency (6)",             "cons_outputs/cons_6step.gif",  6),
]

# The reference the LPIPS-vs-teacher column compares against (the sharp teacher).
TEACHER_GIF = "distill_teacher_case1.gif"


def load_gif_frames(path: str, size: int = 512):
    """Return a (N,3,size,size) float tensor in [0,1]. Auto-crops the right (edit)
    panel of a side-by-side gif (width ~= 2*height)."""
    im = Image.open(path)
    frames = []
    for fr in ImageSequence.Iterator(im):
        fr = fr.convert("RGB")
        w, h = fr.size
        if w >= 1.8 * h:                       # side-by-side source|edit -> right half
            fr = fr.crop((w // 2, 0, w, h))
        fr = fr.resize((size, size), Image.LANCZOS)
        frames.append(np.asarray(fr, dtype=np.float32) / 255.0)
    arr = np.stack(frames, 0).transpose(0, 3, 1, 2)   # N,3,H,W
    return torch.from_numpy(arr)


def load_source_frames(source_dir: str, size: int = 512):
    files = sorted(glob.glob(os.path.join(source_dir, "*.png")) +
                   glob.glob(os.path.join(source_dir, "*.jpg")))
    frames = []
    for f in files:
        fr = Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS)
        frames.append(np.asarray(fr, dtype=np.float32) / 255.0)
    arr = np.stack(frames, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def align_len(a: torch.Tensor, b: torch.Tensor):
    """Truncate both to the shorter frame count so per-frame metrics line up."""
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def make_lpips(device):
    try:
        import lpips
    except ImportError:
        print("[metrics] `lpips` not installed (pip install lpips) — "
              "LPIPS columns will be blank.")
        return None
    return lpips.LPIPS(net="alex").to(device).eval()


def lpips_score(net, a01, b01, device):
    """Mean LPIPS over frames. Inputs (N,3,H,W) in [0,1]; LPIPS wants [-1,1]."""
    if net is None:
        return None
    a, b = align_len(a01, b01)
    a = (a * 2 - 1).to(device)
    b = (b * 2 - 1).to(device)
    with torch.no_grad():
        d = net(a, b)
    return float(d.mean().item())


def make_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc


def clip_sim(clip, prompt, frames01, device):
    """Mean cosine(image, text) over frames, in [-1,1] (higher = more faithful)."""
    model, proc = clip
    imgs = [Image.fromarray((f.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            for f in frames01]
    with torch.no_grad():
        inp = proc(text=[prompt], images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp)
        img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_e = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sims = (img_e @ txt_e.T).squeeze(-1)     # (num_frames,)
    return float(sims.mean().item())


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--prompt", default="a girl is dancing")
    ap.add_argument("--teacher", default=TEACHER_GIF, help="teacher gif (rel. outputs-dir)")
    ap.add_argument("--timings", default=None, help="optional JSON {method: seconds}")
    ap.add_argument("--out", default=None, help="write markdown table here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[metrics] device={device}")

    lpips_net = make_lpips(device)
    clip = make_clip(device)
    timings = json.load(open(args.timings)) if args.timings else {}

    source = load_source_frames(args.source_dir)
    teacher = load_gif_frames(os.path.join(args.outputs_dir, args.teacher))

    rows = []
    for name, rel, nfe in REGISTRY:
        path = os.path.join(args.outputs_dir, rel)
        if not os.path.exists(path):
            print(f"[metrics] SKIP (missing): {rel}")
            continue
        frames = load_gif_frames(path)
        lp_t = lpips_score(lpips_net, frames, teacher, device)
        lp_s = lpips_score(lpips_net, frames, source, device)
        cs = clip_sim(clip, args.prompt, frames, device)
        wc = timings.get(name)
        rows.append((name, lp_t, lp_s, cs, nfe, wc))
        print(f"[metrics] {name}: LPIPS-teacher={fmt(lp_t)} "
              f"LPIPS-source={fmt(lp_s)} CLIP={fmt(cs)} NFE={nfe}")

    header = ("| Method | LPIPS-vs-teacher ↓ | LPIPS-vs-source ↓ | CLIP-sim ↑ | NFE | Wall-clock (s) ↓ |\n"
              "|---|---|---|---|---|---|\n")
    lines = [header]
    for name, lp_t, lp_s, cs, nfe, wc in rows:
        lines.append(f"| {name} | {fmt(lp_t)} | {fmt(lp_s)} | {fmt(cs)} | "
                     f"{nfe if nfe is not None else '-'} | {fmt(wc, 1)} |\n")
    table = "".join(lines)
    print("\n" + table)

    if args.out:
        with open(args.out, "w") as f:
            f.write("# Ablation metrics (case-1)\n\n")
            f.write(f"Prompt: `{args.prompt}` · teacher: `{args.teacher}`\n\n")
            f.write(table)
        print(f"[metrics] wrote {args.out}")


if __name__ == "__main__":
    main()
