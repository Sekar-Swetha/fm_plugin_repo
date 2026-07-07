"""READ-ONLY conditioning-provenance diagnostic (no metrics/model/data changes).

Answers: was the `openposefull` skeleton that actually drove DPM/C3 sourced from
MediaPipe (the swap) or the original OpenPose? "openposefull" is a FORMAT/location,
not proof of source. We hash the exact files ControlNet reads and compare them,
byte-for-byte, against the backup (original OpenPose) and the MediaPipe-rendered
version. We also report each skeleton's non-black centroid to quantify the ~0.18
x-offset between pipelines.

Run on the GPU box (has openpose_bak + mp_* dirs):
    cd flow_matching_plugin
    python3 diag_conditioning.py --data-dir ../motionEditor/MotionEditor/data/case-1
"""
import argparse
import glob
import hashlib
import os

import numpy as np
from PIL import Image


def dir_hashes(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.png"))):
        out[os.path.basename(p)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return out


def dir_centroid(d):
    """Mean normalized (x,y) of non-black skeleton pixels, averaged over frames."""
    xs, ys = [], []
    for p in sorted(glob.glob(os.path.join(d, "*.png"))):
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        m = a > 10
        if m.sum() == 0:
            continue
        yy, xx = np.nonzero(m)
        xs.append(xx.mean() / a.shape[1])
        ys.append(yy.mean() / a.shape[0])
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def compare(name, a, b):
    if a is None or b is None:
        return f"{name}: MISSING"
    common = sorted(set(a) & set(b))
    if not common:
        return f"{name}: no common frames"
    n_eq = sum(a[k] == b[k] for k in common)
    return f"{name}: {n_eq}/{len(common)} frames byte-identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="e.g. .../data/case-1")
    args = ap.parse_args()
    D = args.data_dir

    cand = {
        "target/openposefull (drives DPM/C3)": "target_condition/openposefull",
        "target/openpose_bak (orig OpenPose)": "target_condition/openpose_bak",
        "mp_target (MediaPipe-rendered)":      "mp_target",
        "source/openposefull":                "source_condition/openposefull",
        "source/openpose_bak":                "source_condition/openpose_bak",
        "mp_source":                          "mp_source",
    }
    present = {k: os.path.join(D, v) for k, v in cand.items() if os.path.isdir(os.path.join(D, v))}

    print("=== dirs present ===")
    for k in cand:
        print(f"  {'FOUND ' if k in present else 'absent'} {k}")

    hashes = {k: dir_hashes(v) for k, v in present.items()}
    cents = {k: dir_centroid(v) for k, v in present.items()}

    print("\n=== (1) PROVENANCE: does target/openposefull match OpenPose-bak or MediaPipe? ===")
    tof = "target/openposefull (drives DPM/C3)"
    if tof in hashes:
        for other in ("target/openpose_bak (orig OpenPose)", "mp_target (MediaPipe-rendered)"):
            if other in hashes:
                print("  " + compare(f"openposefull  vs  {other}", hashes[tof], hashes[other]))
        print("  -> all byte-identical to openpose_bak  => OpenPose-sourced (label '+MediaPipe' is WRONG)")
        print("  -> all byte-identical to mp_target      => MediaPipe-sourced (label correct)")

    print("\n=== (2) SPATIAL OFFSET: skeleton non-black centroid (normalized x,y) ===")
    for k in present:
        c = cents[k]
        print(f"  {k:40s} centroid={('(%.3f, %.3f)' % c) if c else 'n/a'}")
    print("  (teacher output subj x~0.38, DPM output subj x~0.56 — compare which skeleton's x each followed)")


if __name__ == "__main__":
    main()
