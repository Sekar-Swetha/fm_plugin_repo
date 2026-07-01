"""Consistency Distillation (C3) core math — pure, CPU-testable, no model deps.

See docs/superpowers/specs/2026-07-01-consistency-distillation-c3-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """Consistency function f(x_t, t) in predict-x0 form.

    x0 = (x_t - sigma_t * eps) / alpha_t. At t=0 (alpha=1, sigma=0) this is the
    identity, so the consistency boundary condition f(x_0, 0) = x_0 holds without
    any extra c_skip/c_out preconditioning.
    """
    return (x_t - sigma_t * eps) / alpha_t


def pseudo_huber_loss(a: torch.Tensor, b: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """LCM-style pseudo-Huber: sqrt((a-b)^2 + delta^2) - delta, averaged.

    Smoother than L2 near zero, less outlier-sensitive than L1 — the loss used in
    consistency distillation to stabilise training.
    """
    diff = a - b
    return (torch.sqrt(diff * diff + delta * delta) - delta).mean()


@torch.no_grad()
def ema_update(ema_params, params, decay: float = 0.95) -> None:
    """In-place EMA of the target network params: ema = decay*ema + (1-decay)*param."""
    for e, p in zip(ema_params, params):
        e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


@dataclass
class ConsistencyPair:
    x_hi: torch.Tensor   # higher-noise latent, at t_hi
    t_hi: int
    x_lo: torch.Tensor   # next lower-noise teacher latent, at t_lo
    t_lo: int
    cond: torch.Tensor   # target-skeleton condition tensor for this clip


def save_pairs(pairs, path: str) -> str:
    torch.save(
        {
            "x_hi": [p.x_hi.cpu() for p in pairs],
            "t_hi": [int(p.t_hi) for p in pairs],
            "x_lo": [p.x_lo.cpu() for p in pairs],
            "t_lo": [int(p.t_lo) for p in pairs],
            "cond": [p.cond.cpu() for p in pairs],
        },
        path,
    )
    return path


def load_pairs(path: str):
    d = torch.load(path, map_location="cpu")
    return [
        ConsistencyPair(x_hi=xh, t_hi=th, x_lo=xl, t_lo=tl, cond=c)
        for xh, th, xl, tl, c in zip(d["x_hi"], d["t_hi"], d["x_lo"], d["t_lo"], d["cond"])
    ]


def pairs_from_trajectory(traj, cond) -> list:
    """Consecutive states of a deterministic teacher trajectory are exactly the
    'one teacher solver step' pairs consistency distillation needs. `traj` is in
    sampling order (descending noise): traj[i] is higher-noise than traj[i+1]."""
    pairs = []
    for (t_hi, x_hi), (t_lo, x_lo) in zip(traj[:-1], traj[1:]):
        pairs.append(ConsistencyPair(
            x_hi=x_hi, t_hi=int(t_hi), x_lo=x_lo, t_lo=int(t_lo), cond=cond))
    return pairs
