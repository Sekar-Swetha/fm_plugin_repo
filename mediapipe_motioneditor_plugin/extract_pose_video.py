"""
Drop-in replacement for MotionEditor's `data_preparation/video_skeletons.py`.

Same CLI surface — same input layout, same output layout, same filename
conventions — but uses MediaPipe under the hood instead of OpenPose.

Usage (identical to the original):

    python extract_pose_video.py \
        -d /path/to/data/case-1/images \
        -c openposefull

This produces:

    /path/to/data/case-1/openposefull/0000.png
    /path/to/data/case-1/openposefull/0001.png
    ...

which is byte-format-compatible with what `controlnet_aux.OpenposeDetector`
would have produced — same canvas size, same skeleton topology, same colours.

Also accepts a video file as input directly (`-d /path/to/video.mp4`), and a
`--write-keypoints` flag to dump per-frame JSON for auditing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from mp_openpose_extractor import (
    MediaPipeOpenPoseExtractor,
    render_openpose_skeleton,
)


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _iter_images(folder: str):
    paths = sorted(glob(os.path.join(folder, "*.jpg"))) + sorted(
        glob(os.path.join(folder, "*.png"))
    )
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        yield Path(p).stem, img


def _iter_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield f"{idx:04d}", frame
        idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe-based drop-in for MotionEditor pose preprocessing."
    )
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="Folder of frames (data/case-X/images) OR a video file path.",
    )
    parser.add_argument(
        "-c", "--which_cond", type=str, default="openposefull",
        help="Subfolder name to write into. Default 'openposefull' matches "
             "MotionEditor's expectations.",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.3,
        help="Per-landmark visibility threshold (default 0.3).",
    )
    parser.add_argument(
        "--model-complexity", type=int, default=2, choices=[0, 1, 2],
        help="MediaPipe Pose model complexity (default 2 = heavy/best).",
    )
    parser.add_argument(
        "--write-keypoints", action="store_true",
        help="Also dump keypoints.json next to the output PNGs (audit aid).",
    )
    args = parser.parse_args()

    # Match the original script's output-dir convention:
    #   data/case-X/images   ->   data/case-X/<which_cond>
    if _is_video(args.data):
        outdir = str(Path(args.data).with_suffix("")) + f"_{args.which_cond}"
        iterator = _iter_video(args.data)
        total = None  # unknown until iterated
    else:
        last_name = Path(args.data.rstrip("/")).name
        outdir = args.data.replace(last_name, args.which_cond)
        all_paths = sorted(glob(os.path.join(args.data, "*.jpg"))) + sorted(
            glob(os.path.join(args.data, "*.png"))
        )
        total = len(all_paths)
        iterator = _iter_images(args.data)

    os.makedirs(outdir, exist_ok=True)
    print(f"Writing skeleton PNGs to: {outdir}")

    keypoints_log = []
    n_total = 0
    n_with_pose = 0

    with MediaPipeOpenPoseExtractor(
        min_conf=args.min_conf,
        model_complexity=args.model_complexity,
    ) as extractor:
        for stem, bgr in tqdm(iterator, total=total, desc="MediaPipe pose"):
            frame, canvas = extractor.extract_and_render(bgr)
            cv2.imwrite(os.path.join(outdir, f"{stem}.png"), canvas)
            n_total += 1
            if frame.num_detected() > 0:
                n_with_pose += 1
            if args.write_keypoints:
                keypoints_log.append({
                    "frame": stem,
                    "width": frame.width,
                    "height": frame.height,
                    "keypoints": frame.keypoints.tolist(),
                })

    if args.write_keypoints:
        with open(os.path.join(outdir, "keypoints.json"), "w") as f:
            json.dump(keypoints_log, f)
        print(f"Wrote keypoints log: {os.path.join(outdir, 'keypoints.json')}")

    detect_rate = (n_with_pose / n_total) if n_total else 0.0
    print(
        f"Done. Frames processed: {n_total}. "
        f"With pose: {n_with_pose} ({100 * detect_rate:.1f}%)."
    )
    if detect_rate < 0.95:
        print(
            "WARNING: detection rate below 95% — review the output PNGs and "
            "consider lowering --min-conf or using `--model-complexity 2`.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
