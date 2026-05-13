"""Conditional Flow Matching with Optimal Transport paths.

Reference: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023),
Section 4.1, Eq. 20-21, Theorem 3.

For OT paths:
    x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1                    (1)
    u_t(x_t | x_1) = x_1 - (1 - sigma_min) * x_0                       (2)

At sigma_min = 0 this is rectified flow's target (Liu et al. 2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class FlowMatchingConfig:
    sigma_min: float = 0.0
    num_train_timesteps: int = 1000
    t_eps: float = 1e-5


def sample_t(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-5,
) -> torch.Tensor:
    t = torch.rand(batch_size, device=device, dtype=dtype)
    return t.clamp(min=eps, max=1.0 - eps)


def compute_x_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    if x_0.shape != x_1.shape:
        raise ValueError(f"x_0 and x_1 must have the same shape: {x_0.shape} vs {x_1.shape}")
    t_b = _reshape_t_like(t, x_0)
    coef_x0 = 1.0 - (1.0 - sigma_min) * t_b
    return coef_x0 * x_0 + t_b * x_1


def compute_target_velocity(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    return x_1 - (1.0 - sigma_min) * x_0


def cfm_ot_loss(
    v_pred: torch.Tensor,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    sigma_min: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    target = compute_target_velocity(x_0, x_1, sigma_min=sigma_min)
    if reduction == "mean":
        return torch.mean((v_pred.float() - target.float()) ** 2)
    if reduction == "sum":
        return torch.sum((v_pred.float() - target.float()) ** 2)
    if reduction == "none":
        return (v_pred.float() - target.float()) ** 2
    raise ValueError(f"Unknown reduction: {reduction}")


def build_fm_training_batch(
    x_1: torch.Tensor,
    config: FlowMatchingConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = x_1.shape[0]
    device = x_1.device
    dtype = x_1.dtype
    if generator is None:
        x_0 = torch.randn_like(x_1)
        t = torch.rand(bsz, device=device, dtype=torch.float32)
    else:
        x_0 = torch.randn(x_1.shape, generator=generator, device=device, dtype=dtype)
        t = torch.rand(bsz, generator=generator, device=device, dtype=torch.float32)
    t = t.clamp(min=config.t_eps, max=1.0 - config.t_eps)

    x_t = compute_x_t(x_0, x_1, t, sigma_min=config.sigma_min)
    target = compute_target_velocity(x_0, x_1, sigma_min=config.sigma_min)
    t_for_unet = (t * (config.num_train_timesteps - 1)).round().long()
    return x_t, t, t_for_unet, target


@torch.no_grad()
def euler_sample(
    velocity_fn,
    x_0: torch.Tensor,
    num_steps: int,
    config: FlowMatchingConfig,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    dt = 1.0 / num_steps
    x = x_0
    for k in range(num_steps):
        t = float(k) / num_steps
        t = max(t, config.t_eps)
        t_for_unet = torch.full(
            (x.shape[0],),
            int(round(t * (config.num_train_timesteps - 1))),
            dtype=torch.long,
            device=x.device,
        )
        v = velocity_fn(x, t_for_unet)
        x = x + dt * v
    return x


def _reshape_t_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t.dim() != 1:
        raise ValueError(f"t must be 1-D (batch,); got shape {tuple(t.shape)}")
    target_shape = [t.shape[0]] + [1] * (ref.dim() - 1)
    return t.view(*target_shape).to(dtype=ref.dtype)


__all__ = [
    "FlowMatchingConfig",
    "build_fm_training_batch",
    "cfm_ot_loss",
    "compute_target_velocity",
    "compute_x_t",
    "euler_sample",
    "sample_t",
]
