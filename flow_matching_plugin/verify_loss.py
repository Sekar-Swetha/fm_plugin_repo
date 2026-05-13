"""Toy 2-D sanity check for the CFM-OT loss.

Trains a tiny MLP velocity model on two-moons and emits loss curve, sample
plots, and a markdown report. CPU-only, no MotionEditor or GPU needed.

Usage:
    python verify_loss.py --out proofs/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_matching_loss import (  # noqa: E402
    FlowMatchingConfig,
    build_fm_training_batch,
    euler_sample,
)


def make_two_moons(n: int) -> torch.Tensor:
    rng = np.random.default_rng(0)
    t = rng.uniform(0, np.pi, size=n // 2)
    x0 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    data = np.concatenate([x0, x1], axis=0)
    data += 0.05 * rng.standard_normal(data.shape)
    return torch.tensor(data, dtype=torch.float32)


class TinyV(torch.nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        t = t.view(-1, 1).float()
        return self.net(torch.cat([x, t], dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="proofs")
    parser.add_argument("--n-data", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = FlowMatchingConfig(sigma_min=0.0, num_train_timesteps=1000)
    data = make_two_moons(args.n_data)

    v = TinyV()
    opt = torch.optim.Adam(v.parameters(), lr=args.lr)

    losses = []
    for step in range(args.n_steps):
        idx = torch.randint(0, data.shape[0], (128,))
        x1 = data[idx]
        x_t, t_cont, _, target = build_fm_training_batch(x1, cfg)
        pred = v(x_t, t_cont)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    plt.figure(figsize=(6, 4))
    plt.plot(losses, lw=0.6)
    plt.xlabel("step"); plt.ylabel("CFM-OT loss")
    plt.title("CFM-OT loss on toy 2-D data")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png", dpi=120)
    plt.close()

    def velocity_fn(x, t_idx):
        t_cont = t_idx.float() / (cfg.num_train_timesteps - 1)
        return v(x, t_cont)

    torch.manual_seed(args.seed + 1)
    x0 = torch.randn(2048, 2)

    samples_1step = euler_sample(velocity_fn, x0, num_steps=1, config=cfg)
    samples_10step = euler_sample(velocity_fn, x0, num_steps=10, config=cfg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, pts, title in zip(
        axes,
        [data.numpy(), samples_1step.detach().numpy(), samples_10step.detach().numpy()],
        ["data (target)", "samples (Euler, 1 step)", "samples (Euler, 10 steps)"],
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5)
        ax.set_title(title); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig(out / "samples_before_after.png", dpi=120)
    plt.close()

    N_traj = 30; N_step = 20
    x = torch.randn(N_traj, 2)
    trajs = [x.clone()]
    dt = 1.0 / N_step
    for k in range(N_step):
        t = float(k) / N_step
        t_for = torch.full((N_traj,), int(round(t * (cfg.num_train_timesteps - 1))), dtype=torch.long)
        with torch.no_grad():
            x = x + dt * velocity_fn(x, t_for)
        trajs.append(x.clone())
    trajs = torch.stack(trajs, dim=0).numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=3, alpha=0.2, c="grey", label="data")
    for j in range(N_traj):
        plt.plot(trajs[:, j, 0], trajs[:, j, 1], lw=0.6, alpha=0.6)
    plt.scatter(trajs[0, :, 0], trajs[0, :, 1], s=20, c="red", label="x_0 (noise)")
    plt.scatter(trajs[-1, :, 0], trajs[-1, :, 1], s=20, c="green", label="x_1 (sampled)")
    plt.title("OT trajectories under trained v_theta")
    plt.legend(); plt.grid(alpha=0.3); plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out / "straight_paths.png", dpi=120)
    plt.close()

    def _stats(pts):
        return pts.mean(0).tolist(), pts.std(0).tolist()
    data_m, data_s = _stats(data.numpy())
    s1_m, s1_s = _stats(samples_1step.detach().numpy())
    s10_m, s10_s = _stats(samples_10step.detach().numpy())

    starts = trajs[0]
    ends = trajs[-1]
    interp_lines = (
        np.linspace(0, 1, N_step + 1).reshape(-1, 1, 1) * (ends - starts)[None]
        + starts[None]
    )
    path_dev = float(np.mean(np.linalg.norm(trajs - interp_lines, axis=-1)))

    final_loss = float(np.mean(losses[-200:]))
    initial_loss = float(np.mean(losses[:200]))
    lines = [
        "# CFM-OT empirical proof",
        "",
        "## Setup",
        "- Toy data: two-moons, 2-D, 2048 points.",
        f"- Model: 3-layer MLP, hidden 128, ~{sum(p.numel() for p in v.parameters())} params.",
        f"- Training: {args.n_steps} Adam steps, lr {args.lr}, batch 128.",
        f"- Loss: CFM-OT with sigma_min = {cfg.sigma_min}.",
        "",
        "## Loss",
        f"- Loss(first 200 steps avg)  = **{initial_loss:.4f}**",
        f"- Loss(last 200 steps avg)   = **{final_loss:.4f}**",
        f"- Ratio (first / last)        = **{initial_loss / max(final_loss, 1e-9):.1f}x**",
        "",
        "## Sample distribution (mean, std per dim)",
        f"- Data:                       mean={data_m}, std={data_s}",
        f"- 1-step Euler samples:       mean={s1_m}, std={s1_s}",
        f"- 10-step Euler samples:      mean={s10_m}, std={s10_s}",
        "",
        "## Path straightness",
        f"- Mean per-step deviation from straight-line interpolation: **{path_dev:.4f}**",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(f"Wrote proofs to {out}/")


if __name__ == "__main__":
    main()
