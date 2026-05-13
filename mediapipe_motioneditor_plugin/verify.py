"""Verification harness for the MediaPipe pose extractor.

Outputs (under `proofs/`):
    side_by_side/<frame>.png   - original | OpenPose ref | ours
    overlay/<frame>.png        - original + ours-skeleton blended
    diff/<frame>.png           - pixel diff vs OpenPose ref (if provided)
    report.json / report.md    - stats + summary

Usage:
    python verify.py --frames /path/to/data/case-1/images \\
                     --openpose-ref /path/to/data/case-1/source_condition/openposefull \\
                     --out proofs/

`--openpose-ref` is optional; without it the script only emits overlays + stats.
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

from mp_openpose_extractor import MediaPipeOpenPoseExtractor


def _list_frames(folder: str):
    paths = sorted(glob(os.path.join(folder, "*.jpg"))) + sorted(
        glob(os.path.join(folder, "*.png"))
    )
    return [(Path(p).stem, p) for p in paths]


def _hstack_same_height(*imgs):
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
                        help="Optional reference OpenPose PNG folder.")
    parser.add_argument("--out", type=str, default="proofs",
                        help="Output folder.")
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

            cv2.imwrite(
                str(out_root / "overlay" / f"{stem}.png"),
                _overlay(bgr, skel),
            )

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

            if args.openpose_ref:
                ref_path = os.path.join(args.openpose_ref, f"{stem}.png")
                if os.path.isfile(ref_path):
                    ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
                    err = _per_pixel_l1(ref, skel)
                    l1_errors.append(err)
                    diff = cv2.absdiff(ref, cv2.resize(skel, (ref.shape[1], ref.shape[0])))
                    cv2.imwrite(str(out_root / "diff" / f"{stem}.png"), diff)

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

    with open(out_root / "report.json", "w") as f:
        json.dump(stats, f, indent=2)

    lines = [
        "# MediaPipe pose extraction — verification report",
        "",
        f"- Frames processed: **{stats['num_frames']}**",
        f"- Frames with at least one detected keypoint: **{stats['frames_with_pose']}** "
        f"({100 * stats['frames_with_pose'] / max(n, 1):.1f}%)",
        f"- Mean landmark visibility (over detected): **{stats['mean_visibility']:.3f}**",
        "",
        "## Per-keypoint detection rate",
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
        ]

    (out_root / "report.md").write_text("\n".join(lines))
    print(f"Wrote proofs to: {out_root}")


if __name__ == "__main__":
    main()
