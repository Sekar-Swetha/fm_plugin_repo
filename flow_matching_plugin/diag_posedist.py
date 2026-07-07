"""READ-ONLY PoseDist diagnostic (does NOT modify metrics.py or any model code).

Answers the anomaly: teacher PoseDist-vs-target=0.04 but DPM/C3/baseline ~0.17.
Tests whether it's a reference/convention/normalization artifact (A) or genuine
pose drift (B) by:
  - printing target + detected provenance (shape/joints/units/min-max/hash),
  - anchors PoseDist(t,t)=0,
  - the decisive control: same detector+target for teacher and DPM,
  - a POSE-NORMALIZED PoseDist (centre on hip-midpoint, scale by torso) — if the
    teacher/DPM gap collapses under normalization, the raw metric was measuring
    global position/scale, not pose,
  - visual overlays (target vs detected skeleton) for one teacher + one DPM frame.

Run on the GPU box (mediapipe present):
    cd flow_matching_plugin
    python3 diag_posedist.py \
        --outputs-dir ../fm_outputs \
        --target-images-dir ../motionEditor/MotionEditor/data/case-1/target_images
"""
import argparse
import hashlib
import os

import numpy as np
from PIL import Image, ImageDraw

from metrics import load_frames, load_source_frames, make_pose, pose_landmarks

# MediaPipe BlazePose-33 indices used for normalization
L_HIP, R_HIP, L_SHO, R_SHO = 23, 24, 11, 12


def arr_of(lms):
    """Stack a list of per-frame (33,2) arrays, dropping None frames."""
    good = [l for l in lms if l is not None]
    return np.stack(good) if good else None


def describe(name, a):
    if a is None:
        print(f"  {name}: NO DETECTION")
        return
    h = hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
    print(f"  {name}: shape={a.shape} joints={a.shape[1]} "
          f"x[min={a[...,0].min():.3f},max={a[...,0].max():.3f}] "
          f"y[min={a[...,1].min():.3f},max={a[...,1].max():.3f}] hash={h}")


def raw_posedist(out_lms, ref_lms):
    ds = []
    for a, b in zip(out_lms, ref_lms):
        if a is None or b is None:
            continue
        n = min(len(a), len(b))
        ds.append(float(np.linalg.norm(a[:n] - b[:n], axis=1).mean()))
    return float(np.mean(ds)) if ds else None


def normalize_pose(lm):
    """Centre on hip midpoint, scale by torso length (hip-mid to shoulder-mid).
    Makes the comparison translation+scale invariant -> pure pose."""
    hip = (lm[L_HIP] + lm[R_HIP]) / 2.0
    sho = (lm[L_SHO] + lm[R_SHO]) / 2.0
    torso = np.linalg.norm(sho - hip) + 1e-6
    return (lm - hip) / torso


def normed_posedist(out_lms, ref_lms):
    ds = []
    for a, b in zip(out_lms, ref_lms):
        if a is None or b is None:
            continue
        n = min(len(a), len(b))
        ds.append(float(np.linalg.norm(normalize_pose(a[:n]) - normalize_pose(b[:n]), axis=1).mean()))
    return float(np.mean(ds)) if ds else None


def overlay(frame01, target_lm, detected_lm, path):
    """Draw target (red) + detected (green) joints on the output frame."""
    img = Image.fromarray((frame01.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    W, H = img.size
    for lm, col in [(target_lm, (255, 0, 0)), (detected_lm, (0, 255, 0))]:
        if lm is None:
            continue
        for (x, y) in lm:
            px, py = x * W, y * H
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)
    img.save(path)
    print(f"  overlay -> {path}  (red=target, green=detected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--target-images-dir", required=True)
    ap.add_argument("--teacher", default="distill_teacher_case1.gif")
    ap.add_argument("--dpm", default="dpm_15_outputs/dpm_solver_15.gif")
    ap.add_argument("--baseline", default="baseline_epsilon_case1.gif")
    args = ap.parse_args()

    pose = make_pose()
    if pose is None:
        raise SystemExit("mediapipe missing — run on the box")

    target_frames = load_source_frames(args.target_images_dir)
    teacher, _ = load_frames(args.outputs_dir, args.teacher)
    dpm, _ = load_frames(args.outputs_dir, args.dpm)
    base, _ = load_frames(args.outputs_dir, args.baseline)

    target_lms = pose_landmarks(pose, target_frames)
    teacher_lms = pose_landmarks(pose, teacher)
    dpm_lms = pose_landmarks(pose, dpm)
    base_lms = pose_landmarks(pose, base)

    print("\n=== (1) TARGET provenance (shared by all rows) ===")
    describe("target (MediaPipe on target_images)", arr_of(target_lms))
    print("\n=== (2) DETECTED provenance (same detector every row) ===")
    describe("teacher detected", arr_of(teacher_lms))
    describe("DPM-15 detected", arr_of(dpm_lms))
    describe("baseline detected", arr_of(base_lms))

    print("\n=== (4) ANCHORS (must be ~0) ===")
    print(f"  PoseDist(target,   target)   = {raw_posedist(target_lms, target_lms):.4f}")
    print(f"  PoseDist(teacher,  teacher)  = {raw_posedist(teacher_lms, teacher_lms):.4f}")

    print("\n=== (4) DECISIVE CONTROL — RAW normalized-coord PoseDist vs target ===")
    print(f"  teacher  vs target = {raw_posedist(teacher_lms, target_lms):.4f}  (table shows 0.040)")
    print(f"  DPM-15   vs target = {raw_posedist(dpm_lms, target_lms):.4f}  (table shows 0.173)")
    print(f"  baseline vs target = {raw_posedist(base_lms, target_lms):.4f}  (table shows 0.171)")

    print("\n=== (4b) POSE-NORMALIZED PoseDist (centre+scale invariant) ===")
    print("  If the teacher/DPM gap COLLAPSES here, the raw metric measured global")
    print("  position/scale (convention artifact), not pose.")
    print(f"  teacher  vs target = {normed_posedist(teacher_lms, target_lms):.4f}")
    print(f"  DPM-15   vs target = {normed_posedist(dpm_lms, target_lms):.4f}")
    print(f"  baseline vs target = {normed_posedist(base_lms, target_lms):.4f}")

    print("\n=== (5) VISUAL overlays ===")
    outdir = os.path.join(args.outputs_dir, "diag")
    os.makedirs(outdir, exist_ok=True)
    overlay(teacher[0], target_lms[0], teacher_lms[0], os.path.join(outdir, "overlay_teacher.png"))
    overlay(dpm[0], target_lms[0], dpm_lms[0], os.path.join(outdir, "overlay_dpm.png"))


if __name__ == "__main__":
    main()
