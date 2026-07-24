"""Score a SINGLE gif/frames dir on the §7 no-reference axes, for ad-hoc sweeps
(EXP 3 DPM knobs, EXP 4 C3-8, EXP 5 injection) without editing metrics.py's REGISTRY.

Reuses the validated metric functions from metrics.py. Prints one line:
  full-frame Sharpness, subject Sharpness (via man.mask), PoseDist-vs-target.

    python3 score_gif.py --gif ../fm_outputs/dpm_sweep/dpm_o3_karras_15.gif \
        --mask-dir ../motionEditor/MotionEditor/data/case-1/man.mask \
        --target-images-dir ../motionEditor/MotionEditor/data/case-1/target_images \
        --label "DPM o3 karras 15"
"""
import argparse
import os

from metrics import (
    fmt,
    laplacian_sharpness,
    laplacian_sharpness_masked,
    load_gif_frames,
    load_mask_pngs,
    load_source_frames,
    make_pose,
    make_seg,
    pose_distance,
    pose_landmarks,
    subject_mask_seg,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", required=True, help="gif path (auto-crops side-by-side)")
    ap.add_argument("--mask-dir", default=None, help="man.mask (subject sharpness via seg fallback)")
    ap.add_argument("--target-images-dir", default=None, help="for PoseDist-vs-target")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    frames = load_gif_frames(args.gif)
    sharp = laplacian_sharpness(frames)

    # subject sharpness: prefer MediaPipe SelfieSegmentation mask (matches §7); it
    # tracks the moving subject per output. man.mask is a static source mask, so seg
    # is the right choice for an edited output.
    seg = make_seg()
    subj_sharp = None
    if seg is not None:
        out_mask = subject_mask_seg(seg, frames)
        subj_sharp = laplacian_sharpness_masked(frames, out_mask)

    pose = make_pose()
    pd_tg = None
    if pose is not None and args.target_images_dir:
        target_lms = pose_landmarks(pose, load_source_frames(args.target_images_dir))
        pd_tg = pose_distance(pose, frames, target_lms)

    label = args.label or os.path.basename(args.gif)
    print(f"[score] {label}: Sharpness={fmt(sharp,5)} subjSharpness={fmt(subj_sharp,5)} "
          f"PoseDist-tgt={fmt(pd_tg)}")


if __name__ == "__main__":
    main()
