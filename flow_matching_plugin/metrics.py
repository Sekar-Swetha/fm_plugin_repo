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


def laplacian_sharpness(frames01):
    """No-reference sharpness proxy: mean over frames of Var(Laplacian(gray)).

    Higher = sharper (crisp edges have large 2nd derivatives; blur washes them
    out). Unlike LPIPS-vs-teacher it needs no matched reference, so it is immune
    to the pose/colour differences between methods and measures blur directly.
    """
    x = frames01.numpy()                                  # (N,3,H,W) in [0,1]
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]   # (N,H,W)
    # 4-neighbour Laplacian via rolls (reflect-ish; edges negligible at 512px)
    lap = (-4.0 * gray
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
           + np.roll(gray, 1, 2) + np.roll(gray, -1, 2))
    return float(lap.reshape(lap.shape[0], -1).var(axis=1).mean())


def laplacian_sharpness_masked(frames01, mask):
    """No-reference subject sharpness: variance of the Laplacian computed ONLY over
    the subject-mask pixels, averaged over frames. Immune to both the pose
    mismatch and the colour shift (no reference at all), so it isolates subject
    blur — the axis reference-based subject metrics cannot measure cleanly here."""
    if mask is None:
        return None
    x = frames01.numpy()
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]      # (N,H,W)
    lap = (-4.0 * gray
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
           + np.roll(gray, 1, 2) + np.roll(gray, -1, 2))
    m = mask.numpy()[:, 0] > 0.5                                     # (N,H,W) bool
    vals = []
    for i in range(len(lap)):
        if m[i].sum() > 0:
            vals.append(float(lap[i][m[i]].var()))
    return float(np.mean(vals)) if vals else None


def make_lpips(device):
    """LPIPS in spatial mode: net(a,b) returns a per-pixel (N,1,H,W) distance map.
    .mean() recovers the old global scalar; masking the map gives region metrics."""
    try:
        import lpips
    except ImportError:
        print("[metrics] `lpips` not installed (pip install lpips) — "
              "LPIPS columns will be blank.")
        return None
    return lpips.LPIPS(net="alex", spatial=True).to(device).eval()


def lpips_map(net, a01, b01, device):
    """Per-pixel LPIPS distance map (N,1,H,W). Inputs (N,3,H,W) in [0,1]."""
    if net is None:
        return None
    a, b = align_len(a01, b01)
    a = (a * 2 - 1).to(device)
    b = (b * 2 - 1).to(device)
    with torch.no_grad():
        return net(a, b).cpu()                       # (N,1,H,W)


def lpips_score(net, a01, b01, device):
    """Global mean LPIPS over frames (continuity with the old column)."""
    m = lpips_map(net, a01, b01, device)
    return None if m is None else float(m.mean().item())


# ---- region masks + region-weighted reductions (pure, CPU-testable) ----

def bg_region(src_mask, out_mask):
    """Background = pixels that are background in BOTH source and output, so an
    intended pose change (subject in one, scene in the other) is excluded.
    Masks are (N,1,H,W) in [0,1] (subject=1). Returns (N,1,H,W)."""
    src_mask, out_mask = align_len(src_mask, out_mask)
    return (1.0 - src_mask) * (1.0 - out_mask)


def subject_region(out_mask, ref_mask):
    """Subject region = union of the output-subject and reference-subject masks."""
    out_mask, ref_mask = align_len(out_mask, ref_mask)
    return torch.clamp(out_mask + ref_mask, 0.0, 1.0)


def region_weighted_mean(dist_map, region, eps=1e-6):
    """Mask-weighted mean of a per-pixel map over a region, averaged over frames
    that have any region mass. NO zero-masking of pixels — we weight the distance
    map, so no spurious edges are introduced. dist_map/region: (N,1,H,W)."""
    if dist_map is None:
        return None
    dist_map, region = align_len(dist_map, region)
    num = (dist_map * region).flatten(1).sum(1)
    den = region.flatten(1).sum(1)
    vals = [float((num[i] / den[i]).item()) for i in range(len(den)) if den[i] > eps]
    return float(np.mean(vals)) if vals else None


def ssim_map(a01, b01, window=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Per-pixel SSIM map (N,1,H,W) on luminance, uniform window (no skimage).
    Higher = more similar. Used mask-weighted as a robustness cross-check."""
    a, b = align_len(a01, b01)
    ga = (0.299 * a[:, :1] + 0.587 * a[:, 1:2] + 0.114 * a[:, 2:3])
    gb = (0.299 * b[:, :1] + 0.587 * b[:, 1:2] + 0.114 * b[:, 2:3])
    k = torch.ones(1, 1, window, window) / (window * window)
    pad = window // 2

    def flt(x):
        return torch.nn.functional.conv2d(x, k, padding=pad)

    mu_a, mu_b = flt(ga), flt(gb)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = flt(ga * ga) - mu_a2
    vb = flt(gb * gb) - mu_b2
    vab = flt(ga * gb) - mu_ab
    return ((2 * mu_ab + C1) * (2 * vab + C2)) / ((mu_a2 + mu_b2 + C1) * (va + vb + C2))


def make_seg():
    """MediaPipe SelfieSegmentation for output/teacher subject masks. None if absent."""
    try:
        import mediapipe as mp
    except ImportError:
        print("[metrics] mediapipe not installed — region masks blank.")
        return None
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


def subject_mask_seg(seg, frames01, thresh=0.5):
    """(N,1,H,W) binary subject mask from MediaPipe SelfieSegmentation."""
    if seg is None:
        return None
    masks = []
    for f in frames01:
        img = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # HWC RGB
        m = seg.process(img).segmentation_mask                       # (H,W) float
        masks.append((m > thresh).astype(np.float32))
    return torch.from_numpy(np.stack(masks)[:, None])                # (N,1,H,W)


def load_mask_pngs(mask_dir, size=512):
    """Load per-frame source-subject mask PNGs -> (N,1,H,W) in {0,1}."""
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    ms = []
    for fp in files:
        m = Image.open(fp).convert("L").resize((size, size), Image.NEAREST)
        ms.append((np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32))
    return torch.from_numpy(np.stack(ms)[:, None]) if ms else None


def make_pose():
    """MediaPipe Pose (legacy solutions). None if mediapipe absent."""
    try:
        import mediapipe as mp
    except ImportError:
        print("[metrics] mediapipe not installed — pose-distance column blank.")
        return None
    return mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)


def pose_landmarks(pose, frames01):
    """Per-frame (33,2) normalized landmarks, or None where no person detected."""
    out = []
    for f in frames01:
        img = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # HWC RGB uint8
        res = pose.process(img)
        if res.pose_landmarks:
            out.append(np.array([[p.x, p.y] for p in res.pose_landmarks.landmark],
                                dtype=np.float32))
        else:
            out.append(None)
    return out


def pose_distance(pose, frames01, ref_lms):
    """Mean per-joint L2 (normalized coords) between output pose and the reference
    (teacher) pose, averaged over frames with a detection in both. Lower = the
    output reproduces the target pose. Skeleton-only, so immune to colour/blur."""
    if pose is None or ref_lms is None:
        return None
    out_lms = pose_landmarks(pose, frames01)
    ds = []
    for a, b in zip(out_lms, ref_lms):
        if a is None or b is None:
            continue
        n = min(len(a), len(b))
        ds.append(float(np.linalg.norm(a[:n] - b[:n], axis=1).mean()))
    return float(np.mean(ds)) if ds else None


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


def load_frames(outputs_dir, rel, size=512):
    """Prefer LOSSLESS per-frame PNGs over the quantised gif. Convention: a run at
    `<rel>.gif` has clean frames at `<rel without .gif>_frames/*.png` (dumped by
    inference.py's save_frames hook). Falls back to the gif if absent."""
    frames_dir = os.path.join(outputs_dir, rel[:-4] + "_frames") if rel.endswith(".gif") \
        else os.path.join(outputs_dir, rel + "_frames")
    pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if pngs:
        fr = []
        for p in pngs:
            im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            fr.append(np.asarray(im, dtype=np.float32) / 255.0)
        return torch.from_numpy(np.stack(fr).transpose(0, 3, 1, 2)), "png"
    return load_gif_frames(os.path.join(outputs_dir, rel), size), "gif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--mask-dir", default=None,
                    help="per-frame source-subject mask PNGs (e.g. data/case-1/man.mask)")
    ap.add_argument("--target-images-dir", default=None,
                    help="target driving RGB frames (e.g. data/case-1/target_images) "
                         "for posedist_vs_target")
    ap.add_argument("--prompt", default="a girl is dancing")
    ap.add_argument("--teacher", default=TEACHER_GIF, help="teacher gif (rel. outputs-dir)")
    ap.add_argument("--timings", default=None, help="optional JSON {method: seconds}")
    ap.add_argument("--out", default=None, help="write markdown table here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[metrics] device={device}")

    lpips_net = make_lpips(device)
    clip = make_clip(device)
    pose = make_pose()
    seg = make_seg()
    timings = json.load(open(args.timings)) if args.timings else {}

    source = load_source_frames(args.source_dir)
    teacher, tsrc = load_frames(args.outputs_dir, args.teacher)
    print(f"[metrics] teacher frames from: {tsrc}")
    teacher_lms = pose_landmarks(pose, teacher) if pose is not None else None
    teacher_mask = subject_mask_seg(seg, teacher)
    src_mask = load_mask_pngs(args.mask_dir) if args.mask_dir else None

    target_lms = None
    if args.target_images_dir and pose is not None:
        target_frames = load_source_frames(args.target_images_dir)
        target_lms = pose_landmarks(pose, target_frames)
        floor = pose_distance(pose, teacher, target_lms)     # teacher-vs-target floor
        print(f"[metrics] posedist_vs_target floor (teacher) = {fmt(floor)}")

    rows = []
    for name, rel, nfe in REGISTRY:
        if not os.path.exists(os.path.join(args.outputs_dir, rel)):
            print(f"[metrics] SKIP (missing): {rel}")
            continue
        frames, fsrc = load_frames(args.outputs_dir, rel)
        sharp = laplacian_sharpness(frames)
        pd_t = pose_distance(pose, frames, teacher_lms)
        pd_tg = pose_distance(pose, frames, target_lms)
        lp_s = lpips_score(lpips_net, frames, source, device)          # global (footnote)
        lp_t = lpips_score(lpips_net, frames, teacher, device)         # global (confounded)
        cs = clip_sim(clip, args.prompt, frames, device)

        # region-split metrics (need seg + lpips + source mask)
        bg_lp = subj_lp = bg_ss = subj_ss = subj_sharp = None
        if seg is not None:
            out_mask = subject_mask_seg(seg, frames)
            subj_sharp = laplacian_sharpness_masked(frames, out_mask)   # no-ref subject quality
        if seg is not None and lpips_net is not None:
            if src_mask is not None:
                bg = bg_region(src_mask, out_mask)
                bg_lp = region_weighted_mean(lpips_map(lpips_net, frames, source, device), bg)
                bg_ss = region_weighted_mean(ssim_map(frames, source), bg)
            if teacher_mask is not None:
                subj = subject_region(out_mask, teacher_mask)
                subj_lp = region_weighted_mean(lpips_map(lpips_net, frames, teacher, device), subj)
                subj_ss = region_weighted_mean(ssim_map(frames, teacher), subj)

        wc = timings.get(name)
        rows.append(dict(name=name, nfe=nfe, sharp=sharp, subj_sharp=subj_sharp,
                         pd_t=pd_t, pd_tg=pd_tg, bg_lp=bg_lp, subj_lp=subj_lp,
                         bg_ss=bg_ss, subj_ss=subj_ss, lp_s=lp_s, lp_t=lp_t, cs=cs,
                         wc=wc, fsrc=fsrc))
        print(f"[metrics] {name} [{fsrc}]: Sharp={fmt(sharp,5)} subjSharp={fmt(subj_sharp,5)} "
              f"PoseTgt={fmt(pd_tg)} bgSSIM={fmt(bg_ss)} NFE={nfe}")

    # Primary (all no-reference or structural, so unconfounded): full-frame Sharpness,
    # SUBJECT-region Sharpness (isolates subject blur), bg SSIM (scene preserved),
    # PoseDist-vs-target (fidelity). Reference-based subject LPIPS/SSIM kept but
    # confounded (residual pose mismatch + colour shift); global LPIPS for continuity.
    cols = ("| Method | NFE | Sharpness ↑ | **subj Sharpness ↑** | bg SSIM ↑ | "
            "PoseDist-vs-target ↓ | bg LPIPS-src ↓ | subj LPIPS-teacher ↓ (conf.) | "
            "subj SSIM ↑ (conf.) | LPIPS-src global (conf.) | frames |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [cols]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['nfe'] if r['nfe'] is not None else '-'} | "
            f"{fmt(r['sharp'],5)} | {fmt(r['subj_sharp'],5)} | {fmt(r['bg_ss'])} | "
            f"{fmt(r['pd_tg'])} | {fmt(r['bg_lp'])} | {fmt(r['subj_lp'])} | "
            f"{fmt(r['subj_ss'])} | {fmt(r['lp_s'])} | {r['fsrc']} |\n")
    table = "".join(lines)
    print("\n" + table)

    if args.out:
        with open(args.out, "w") as f:
            f.write("# Ablation metrics (case-1)\n\n")
            f.write(f"Prompt: `{args.prompt}` · teacher: `{args.teacher}`\n\n")
            f.write("Primary axes: **Sharpness** (no-ref), **PoseDist-vs-target** (fidelity to "
                    "the driving pose), **bg LPIPS-src** (scene preservation, ~0 expected), "
                    "**subj LPIPS-teacher** (subject appearance vs teacher, same pose). "
                    "`LPIPS-src (global)` is confounded by the intended pose change + background "
                    "dominance + colour shift — kept for continuity only. `frames` = whether "
                    "clean PNGs or the quantised gif were used.\n\n")
            f.write(table)
        print(f"[metrics] wrote {args.out}")
        print(f"[metrics] wrote {args.out}")


if __name__ == "__main__":
    main()
