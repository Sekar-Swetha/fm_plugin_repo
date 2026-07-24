"""FIG A & C — side-by-side / tiled panels from EXISTING output gifs (no re-render).
Needs PIL. Run on the box where the gifs live.

FIG A (DPM order-2 vs order-3, 15 NFE):
    python3 make_gif_panels.py --mode A \
        --gifs ../../fm_outputs/dpm_sweep/dpm_n15_o2.gif ../../fm_outputs/dpm_sweep/dpm_n15_o3.gif \
        --labels "order-2" "order-3" \
        --out ../../motionEditor/MotionEditor/outputs/dpm_order_compare

FIG C (C3 dial 1/2/4/6/8):
    python3 make_gif_panels.py --mode C \
        --gifs ../../fm_outputs/cons_outputs/cons_1step.gif \
               ../../fm_outputs/cons_outputs/cons_2step.gif \
               ../../fm_outputs/cons_outputs/sample-consistency.gif \
               ../../fm_outputs/cons_outputs/cons_6step.gif \
               ../../fm_outputs/cons_outputs/cons_8step.gif \
        --labels 1 2 4 6 8 \
        --out ../../motionEditor/MotionEditor/outputs/c3_dial

Writes <out>.png (first-frame montage) and <out>.gif (animated montage). Skips any
missing gif and reports it (does NOT regenerate anything).
"""
import argparse
import os

from PIL import Image, ImageSequence, ImageDraw


def load_frames(path, size=384):
    im = Image.open(path)
    frames = []
    for fr in ImageSequence.Iterator(im):
        fr = fr.convert("RGB")
        w, h = fr.size
        if w >= 1.8 * h:                       # side-by-side source|edit -> right half
            fr = fr.crop((w // 2, 0, w, h))
        frames.append(fr.resize((size, size), Image.LANCZOS))
    return frames


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, len(text) * 11 + 12, 26], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "C"], required=True)
    ap.add_argument("--gifs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    cols = []
    for g, lab in zip(args.gifs, args.labels):
        if not os.path.exists(g):
            print(f"[fig] SKIP missing: {g}")
            continue
        cols.append((load_frames(g, args.size), str(lab)))
    if not cols:
        raise SystemExit("no gifs found")

    n = len(cols)
    W = args.size * n
    H = args.size
    nframes = min(len(f) for f, _ in cols)

    # animated montage
    out_frames = []
    for k in range(nframes):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        for i, (frames, lab) in enumerate(cols):
            tile = label(frames[k].copy(), lab)
            canvas.paste(tile, (i * args.size, 0))
        out_frames.append(canvas)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_frames[0].save(args.out + ".gif", save_all=True,
                       append_images=out_frames[1:], duration=200, loop=0)
    out_frames[0].save(args.out + ".png")     # first-frame montage for slides
    print(f"[fig{args.mode}] wrote {args.out}.png and {args.out}.gif  ({n} panels)")


if __name__ == "__main__":
    main()
