"""C3 Stage 1: convert recorded DPM teacher trajectories into consistency pairs.

Consumes trajectory files saved by inference.py's `dump_trajectory_to` hook
(each `{ "traj": [(t_int, latent), ...], "cond": prepared_image }`) and emits a
pooled `pairs.pt` (ConsistencyPair list). Pure CPU data transform.
"""
import argparse
import glob

import torch

from consistency_core import pairs_from_trajectory, save_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True,
                    help="trajectory file globs saved by inference dump_trajectory_to")
    ap.add_argument("--out", required=True, help="output pairs.pt")
    args = ap.parse_args()

    files = []
    for pat in args.traj:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"no trajectory files matched: {args.traj}")

    all_pairs = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        traj = [(int(t), x) for t, x in d["traj"]]
        all_pairs.extend(pairs_from_trajectory(traj, d["cond"]))
    save_pairs(all_pairs, args.out)
    print(f"[c3] {len(files)} trajectories -> {len(all_pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
