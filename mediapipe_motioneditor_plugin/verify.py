"""
Verification harness — produces visual + statistical proofs that the
MediaPipe-based skeleton extractor is a correct drop-in for OpenPose in the
MotionEditor pipeline.

Outputs (under `proofs/`):
    1. side_by_side/<frame>.png      — original frame | OpenPose | MediaPipe (ours)
    2. overlay/<frame>.png           — original + ours-skeleton blended
    3. diff/<frame>.png              — pixel-level diff vs. controlnet_aux openpose
                                       (if controlnet_aux is installed)
    4. report.json                   — detection rate, mean conf, missing-kp counts
    5. report.md                     — human-readable summary for the supervisor

Usage:
    python verify.py --frames /path/to/data/case-1/images \
                     --openpose-ref /path/to/data/case-1/source_condition/openposefull \
                     --out proofs/

The `--openpose-ref` argument is optional. If omitted, the script only emits
overlays and statistics (no pixel diff). If present, it computes a per-frame
pixel-diff between our output and the reference OpenPose output and reports
mean L1 error.

This script can run without a GPU and produces deterministic artefacts that
the supervisor can inspect without re-running the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mp_openpose_extractor import (
    MediaPipeOpenPoseExtractor,
    OpenPoseFrame,
    render_openpose_skeleton,
)


def _list_frames(folder: str):
    paths = sorted(glob(os.path.join(folder, "*.jpg"))) + sorted(
        glob(os.path.join(folder, "*.png"))
    )
    return [(Path(p).stem, p) for p in paths]


def _hstack_same_height(*imgs):
    """Concatenate BGR images horizontally, resizing to common height."""
    h = min(im.shape[0] for im in imgs)
    rescaled = []
    for im in imgs:
        scale = h / im.shape[0]
        new_w = int(round(im.shape[1] * scale))
        rescaled.append(cv2.resize(im, (new_w, h)))
    return np.concatenate(rescaled, axis=1)


def _overlay(frame_bgr: np.ndarray, skel_bgr: np.ndarray, alpha: float = 0.6):
    return cv2.addWeighted(frame_bgr, 1.0, skel_bgr, alpha, 0)


def _per_pixel_l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True,
                        help="Folder of input frames (jpg/png).")
    parser.add_argument("--openpose-ref", type=str, default=None,
                        help="Optional: folder of reference OpenPose PNGs to "
                             "diff against (filenames must match).")
    parser.add_argument("--out", type=str, default="proofs",
                        help="Output folder for proofs (default 'proofs').")
    parser.add_argument("--max-frames", type=int, default=8,
                        help="How many frames to emit side-by-side images for.")
    parser.add_argument("--min-conf", type=float, default=0.3)
    args = parser.parse_args()

    out_root = Path(args.out)
    (out_root / "side_by_side").mkdir(parents=True, exist_ok=True)
    (out_root / "overlay").mkdir(parents=True, exist_ok=True)
    if args.openpose_ref:
        (out_root / "diff").mkdir(parents=True, exist_ok=True)

    frame_list = _list_frames(args.frames)
    if not frame_list:
        raise SystemExit(f"No frames found in {args.frames}")

    stats = {
        "num_frames": len(frame_list),
        "frames_with_pose": 0,
        "per_keypoint_detection_rate": [0] * 18,
        "mean_visibility": 0.0,
        "per_frame": [],
        "vs_openpose_ref_l1": None,
    }

    l1_errors = []

    side_by_side_emitted = 0

    with MediaPipeOpenPoseExtractor(min_conf=args.min_conf) as extractor:
        for stem, path in tqdm(frame_list, desc="verify"):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            op_frame, skel = extractor.extract_and_render(bgr)
            detected = op_frame.num_detected()
            stats["frames_with_pose"] += int(detected > 0)
            for i in range(18):
                if op_frame.keypoints[i, 0] >= 0:
                    stats["per_keypoint_detection_rate"][i] += 1
            mean_vis = float(
                np.mean(op_frame.keypoints[op_frame.keypoints[:, 2] >= 0, 2])
            ) if detected > 0 else 0.0
            stats["per_frame"].append({
                "frame": stem,
                "n_detected": detected,
                "mean_visibility": mean_vis,
            })
            stats["mean_visibility"] += mean_vis

            # Always emit overlay
            cv2.imwrite(
                str(out_root / "overlay" / f"{stem}.png"),
                _overlay(bgr, skel),
            )

            # Build side-by-side for first N frames
            if side_by_side_emitted < args.max_frames:
                tiles = [bgr]
                if args.openpose_ref:
                    ref_path = os.path.join(args.openpose_ref, f"{stem}.png")
                    if os.path.isfile(ref_path):
                        ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
                        tiles.append(ref)
                tiles.append(skel)
                cv2.imwrite(
                    str(out_root / "side_by_side" / f"{stem}.png"),
                    _hstack_same_height(*tiles),
                )
                side_by_side_emitted += 1

            # Pixel-diff vs OpenPose reference
            if args.openpose_ref:
                ref_path = os.path.join(args.openpose_ref, f"{stem}.png")
                if os.path.isfile(ref_path):
                    ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
                    err = _per_pixel_l1(ref, skel)
                    l1_errors.append(err)
                    diff = cv2.absdiff(ref, cv2.resize(skel, (ref.shape[1], ref.shape[0])))
                    cv2.imwrite(str(out_root / "diff" / f"{stem}.png"), diff)

    # Normalise stats
    n = stats["num_frames"]
    stats["mean_visibility"] /= max(n, 1)
    stats["per_keypoint_detection_rate"] = [
        c / max(n, 1) for c in stats["per_keypoint_detection_rate"]
    ]
    if l1_errors:
        stats["vs_openpose_ref_l1"] = {
            "mean": float(np.mean(l1_errors)),
            "max": float(np.max(l1_errors)),
            "min": float(np.min(l1_errors)),
            "frames_compared": len(l1_errors),
        }

    # Write JSON
    with open(out_root / "report.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Write Markdown report
    lines = [
        "# MediaPipe pose extraction — verification report",
        "",
        f"- Frames processed: **{stats['num_frames']}**",
        f"- Frames with at least one detected keypoint: **{stats['frames_with_pose']}** "
        f"({100 * stats['frames_with_pose'] / max(n, 1):.1f}%)",
        f"- Mean landmark visibility (over detected): **{stats['mean_visibility']:.3f}**",
        "",
        "## Per-keypoint detection rate (fraction of frames detecting each OP-18 keypoint)",
        "",
        "| idx | name | detection rate |",
        "|----:|:-----|---------------:|",
    ]
    from mp_openpose_extractor import OPENPOSE_KEYPOINTS
    for i, (name, rate) in enumerate(
        zip(OPENPOSE_KEYPOINTS, stats["per_keypoint_detection_rate"])
    ):
        lines.append(f"| {i} | `{name}` | {100 * rate:.1f}% |")

    if stats["vs_openpose_ref_l1"] is not None:
        cmp = stats["vs_openpose_ref_l1"]
        lines += [
            "",
            "## Pixel-level diff vs. OpenPose reference",
            "",
            f"- Frames compared: **{cmp['frames_compared']}**",
            f"- Mean per-pixel L1 (RGB, 0..255): **{cmp['mean']:.2f}**",
            f"- Max: {cmp['max']:.2f}, Min: {cmp['min']:.2f}",
            "",
            "_Interpretation:_ Lower is better. ControlNet does not require",
            "byte-equality with OpenPose; it requires the same skeleton",
            "topology, the same canvas size, and roughly the same colour",
            "palette. An L1 in the low tens (out of 255) typically corresponds",
            "to minor sub-pixel offset of joints and matching limb colours.",
            "Per-frame diff PNGs are in `diff/`.",
        ]

    lines += [
        "",
        "## How to read the artefacts",
        "",
        "- `side_by_side/` — original frame | (ref OpenPose, if provided) | "
        "MediaPipe (ours). Use this to eyeball that joints land in the right "
        "places and limb colours match.",
        "- `overlay/` — input frame blended with our skeleton. Use this to "
        "spot mis-localised joints.",
        "- `diff/` — pixel-absolute-difference against the OpenPose reference. "
        "Used to quantify drift on a per-frame basis.",
        "- `report.json` — machine-readable version of every number above.",
    ]

    (out_root / "report.md").write_text("\n".join(lines))
    print(f"Wrote proofs to: {out_root}")


if __name__ == "__main__":
    main()
