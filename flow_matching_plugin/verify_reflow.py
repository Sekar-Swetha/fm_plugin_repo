"""Empirical proof generator for reflow (Contribution C).

CPU-only, no MotionEditor / GPU. Trains a 1-rectified flow on toy 2-D data
(Contribution A objective), performs one reflow round (C1), and emits the
artefacts named in README_contrib_BC.md C.5:

  - path_straightness_reflow.png : path-length integral for {round 0, round 1}.
                                   Reflow should lower it (straighter ODE).
  - nfe_vs_quality.png           : sample quality (distance-to-data) vs NFE for
                                   round 0 vs round 1. Round 1 should be better
                                   at low NFE (1-4 steps).
  - report_reflow.md             : quantitative summary.

Usage:
    python verify_reflow.py --out proofs/
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
from flow_matching_loss import FlowMatchingConfig, compute_x_t, sample_t  # noqa: E402
from flow_inversion import flow_sample  # noqa: E402


def two_clusters(n: int) -> torch.Tensor:
    """A curved transport target: two well-separated clusters from N(0, I)."""
    c = torch.tensor([[3.0, 3.0], [-3.0, -3.0]])
    idx = torch.randint(0, 2, (n,))
    return c[idx] + 0.3 * torch.randn(n, 2)


def mlp() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(3, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 2),
    )


def _vf(model, cfg):
    def vf(x, t_idx):
        t_cont = t_idx.float() / (cfg.num_train_timesteps - 1)
        return model(torch.cat([x, t_cont.view(-1, 1)], dim=-1))
    return vf


def train_flow(model, z0, z1, cfg, steps, lr=3e-3):
    """Train the velocity field on a (z0, z1) coupling (reflow == same loss)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = z0.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (256,))
        a, b = z0[idx], z1[idx]
        t = sample_t(256, a.device, eps=cfg.t_eps)
        x_t = compute_x_t(a, b, t, sigma_min=cfg.sigma_min)
        t_idx = (t * (cfg.num_train_timesteps - 1)).round().long()
        v = _vf(model, cfg)(x_t, t_idx)
        target = b - (1.0 - cfg.sigma_min) * a
        loss = ((v - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def path_length(model, z0, cfg, n_steps=50) -> float:
    """Integral of ‖v‖ along the sampled ODE path: straighter == shorter."""
    res = flow_sample(_vf(model, cfg), z0, num_steps=n_steps, method="euler",
                      config=cfg, return_trajectory=True)
    traj = torch.stack(res.trajectory, dim=0)           # (n+1, B, 2)
    seg = (traj[1:] - traj[:-1]).norm(dim=-1)            # (n, B)
    return seg.sum(dim=0).mean().item()


def data_distance(samples, data) -> float:
    """Mean nearest-neighbour distance from samples to the data cloud."""
    d = torch.cdist(samples, data)                      # (S, N)
    return d.min(dim=1).values.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="proofs")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = FlowMatchingConfig(sigma_min=0.0, num_train_timesteps=1000)

    data = two_clusters(4096)

    # --- Round 0: 1-rectified flow (Contribution A), random N(0,I)->data pairs.
    z0_r0 = torch.randn_like(data)
    model0 = train_flow(mlp(), z0_r0, data, cfg, steps=args.steps)

    # --- Round 1 (C1): pairs are (z0, model0's own transport of z0).
    z0_r1 = torch.randn(4096, 2)
    with torch.no_grad():
        z1_r1 = flow_sample(_vf(model0, cfg), z0_r1, num_steps=50,
                            method="euler", config=cfg).latent
    model1 = train_flow(mlp(), z0_r1, z1_r1, cfg, steps=args.steps)

    # --- path_straightness_reflow.png --------------------------------------
    torch.manual_seed(args.seed + 1)
    probe = torch.randn(512, 2)
    pl0 = path_length(model0, probe.clone(), cfg)
    pl1 = path_length(model1, probe.clone(), cfg)
    straight_line = (
        flow_sample(_vf(model0, cfg), probe.clone(), 50, "euler", cfg).latent
        - probe
    ).norm(dim=-1).mean().item()
    plt.figure(figsize=(5, 4))
    plt.bar(["Round 0\n(Contrib A)", "Round 1\n(C1 reflow)"], [pl0, pl1],
            color=["#888", "#4c72b0"])
    plt.axhline(straight_line, ls="--", c="k", lw=0.8,
                label=f"straight-line lower bound ≈ {straight_line:.2f}")
    plt.ylabel(r"path length  $\int_0^1 \|v\|\,dt$")
    plt.title("Reflow shortens ODE paths (straighter transport)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "path_straightness_reflow.png", dpi=120); plt.close()

    # --- nfe_vs_quality.png ------------------------------------------------
    nfes = [1, 2, 4, 8, 16]
    q0, q1 = [], []
    for n in nfes:
        torch.manual_seed(args.seed + 2)
        s0 = flow_sample(_vf(model0, cfg), torch.randn(1024, 2), n, "euler", cfg).latent
        torch.manual_seed(args.seed + 2)
        s1 = flow_sample(_vf(model1, cfg), torch.randn(1024, 2), n, "euler", cfg).latent
        q0.append(data_distance(s0, data))
        q1.append(data_distance(s1, data))
    plt.figure(figsize=(5, 4))
    plt.plot(nfes, q0, "o-", label="Round 0 (Contrib A)")
    plt.plot(nfes, q1, "s-", label="Round 1 (C1 reflow)")
    plt.xlabel("NFE (Euler steps)"); plt.ylabel("mean dist. to data (lower=better)")
    plt.title("Few-step sample quality: reflow vs base flow")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "nfe_vs_quality.png", dpi=120); plt.close()

    report = [
        "# Reflow empirical proof (Contribution C1)",
        "",
        "Toy 2-D two-cluster transport, CPU. Round 0 = 1-rectified flow",
        "(Contribution A). Round 1 = one C1 reflow round on the round-0 model's",
        "own (z0 -> transport(z0)) pairs.",
        "",
        "## Path straightness",
        f"- Round 0 path length: **{pl0:.4f}**",
        f"- Round 1 path length: **{pl1:.4f}**  "
        f"({'shorter ✓' if pl1 < pl0 else 'NOT shorter ✗'})",
        f"- Straight-line lower bound: {straight_line:.4f}",
        "  (See path_straightness_reflow.png.)",
        "",
        "## Few-step sample quality (mean distance to data, lower better)",
        "| NFE | Round 0 | Round 1 |",
        "|----:|--------:|--------:|",
        *[f"| {n} | {a:.4f} | {b:.4f} |" for n, a, b in zip(nfes, q0, q1)],
        "",
        f"- At NFE=1, reflow improves quality by "
        f"{(q0[0] - q1[0]) / q0[0] * 100:.1f}% on this toy problem.",
        "- The real-clip CLIP/LPIPS-T-vs-NFE rows require the GPU pipeline (C.6).",
    ]
    (out / "report_reflow.md").write_text("\n".join(report))
    print(f"Wrote reflow proofs to {out}/  (path len {pl0:.3f} -> {pl1:.3f})")


if __name__ == "__main__":
    main()
