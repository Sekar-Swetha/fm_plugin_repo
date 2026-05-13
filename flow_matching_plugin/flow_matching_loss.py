"""
Conditional Flow Matching (CFM) with Optimal Transport (OT) probability paths.

This is the training-time replacement for the DDPM epsilon-prediction loss
used by MotionEditor (`train_adaptor.py`, lines ~333-376).

Reference
---------
Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023).
Specifically: Section 4.1 ("Example II: Optimal Transport conditional VFs"),
Eq. 20-21 and Theorem 3.

For Optimal Transport paths with `sigma_min` small (and exactly zero, which
is the limit and the choice we use):

    mu_t(x_1)    = t * x_1
    sigma_t(x_1) = 1 - (1 - sigma_min) * t

The flow map x_t = sigma_t * x_0 + mu_t reduces to a *linear interpolation*
between the noise `x_0 ~ N(0, I)` and the data `x_1`:

    x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1                    (1)

The closed-form target velocity (Theorem 3 of Lipman et al.) is

    u_t(x_t | x_1) = d/dt psi_t(x_0)  =  x_1 - (1 - sigma_min) * x_0   (2)

This is what we regress against. Note at sigma_min = 0 it simplifies to the
rectified-flow regression target `x_1 - x_0` (Liu et al. 2022).

What this changes in MotionEditor
---------------------------------
- The DDPM noise schedule (alpha_t, beta_t) is no longer used during training.
- The U-Net is asked to predict the *velocity* `v_theta(x_t, t, c)` instead of
  the *noise* `epsilon`.
- The UNet's timestep input still uses the standard 0..N-1 integer index for
  positional embedding, but it now means "round(t * N)" with t in [0, 1].
- Sampling at inference is no longer DDIM-step-based; it is an ODE integration
  of `v_theta`. (Inference handled in Contribution B; this module is training
  only. A sanity-check Euler sampler is provided in `inference_fm_sanity.py`.)

What this does NOT change
-------------------------
- U-Net architecture, ControlNet, motion adapter, attention blocks: untouched.
- Optimizer, lr schedule, accelerator setup, dataloader: untouched.
- The trainable-parameter set ("controlnet_adapter" etc.): untouched.

Author: Swetha (TCD dissertation, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FlowMatchingConfig:
    """Hyper-parameters for the CFM-OT loss.

    sigma_min : float
        Minimum noise scale at t = 1. With `sigma_min = 0` the path is the
        ideal Optimal Transport (straight-line) path between Gaussian noise
        and data — identical to rectified flow's training target. Lipman et
        al. use `sigma_min = 1e-4` in their image-generation experiments to
        avoid the very-small-variance regime at t = 1.
    num_train_timesteps : int
        Used only to scale continuous t in [0, 1] back to the integer
        timestep index the U-Net's sinusoidal positional embedding expects.
        The actual training loss does not depend on this number.
    t_eps : float
        Small clamp on the sampled t so we never query the exact endpoints
        (t = 0 or t = 1), which are degenerate (no noise mixing / no data
        mixing).
    """
    sigma_min: float = 0.0
    num_train_timesteps: int = 1000
    t_eps: float = 1e-5


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def sample_t(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sample t ~ Uniform[eps, 1 - eps] for each item in the batch.

    Returns a 1-D tensor of shape (batch_size,).
    """
    t = torch.rand(batch_size, device=device, dtype=dtype)
    return t.clamp(min=eps, max=1.0 - eps)


def compute_x_t(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    """Linear-interp OT path from noise x_0 to data x_1 at time t.

        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1                (eq. 1)

    Shapes
    ------
    x_0, x_1 : (B, ...) — must broadcast.
    t        : (B,)     — same batch dim; broadcast over the rest.
    """
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
    """Target velocity field for the OT-CFM loss.

        u_t(x_t | x_1) = x_1 - (1 - sigma_min) * x_0                   (eq. 2)

    Note this is *time-independent* — a property of the OT path — which is
    one of the main reasons OT-CFM trains more stably than diffusion-path
    flow matching: the regression target has no schedule curvature.
    """
    return x_1 - (1.0 - sigma_min) * x_0


def cfm_ot_loss(
    v_pred: torch.Tensor,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    sigma_min: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """The full Conditional Flow Matching loss for the OT path.

    L_CFM(theta) = E_{t, x_0, x_1} [ || v_theta(x_t, t, c) - u_t(x_t | x_1) ||^2 ]

    The expectation over t, x_0, x_1 is implicit in the call: caller is
    responsible for sampling them and constructing `x_t`. This helper just
    computes the squared error against the closed-form target.
    """
    target = compute_target_velocity(x_0, x_1, sigma_min=sigma_min)
    if reduction == "mean":
        return torch.mean((v_pred.float() - target.float()) ** 2)
    if reduction == "sum":
        return torch.sum((v_pred.float() - target.float()) ** 2)
    if reduction == "none":
        return (v_pred.float() - target.float()) ** 2
    raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------------------------------------------------------
# Convenience: one-shot sample + interp + target
# ---------------------------------------------------------------------------

def build_fm_training_batch(
    x_1: torch.Tensor,
    config: FlowMatchingConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-call helper that does sampling + interpolation + target build.

    Returns (x_t, t_continuous, t_for_unet, target_velocity), where:
      - x_t              : (B, ...) the interpolation point
      - t_continuous     : (B,) the sampled t in [eps, 1-eps]
      - t_for_unet       : (B,) integer-cast timestep index in
                           {0, ..., num_train_timesteps - 1} suitable for
                           feeding to the U-Net's existing positional embed
      - target_velocity  : (B, ...) the regression target u_t

    `generator` lets you make the sampling deterministic (used in tests).
    """
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


# ---------------------------------------------------------------------------
# Euler ODE sampler — used by `inference_fm_sanity.py` to confirm the model
# learnt something. Not the inference pipeline (that's Contribution B).
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    velocity_fn,
    x_0: torch.Tensor,
    num_steps: int,
    config: FlowMatchingConfig,
) -> torch.Tensor:
    """Forward Euler integration of x' = v(x, t) from t = 0 to t = 1.

    Parameters
    ----------
    velocity_fn
        Callable `(x_t, t_for_unet) -> v_pred`. The caller wraps whatever
        external conditioning (text embeds, controlnet residuals, etc.) into
        this closure.
    x_0
        Initial state ~ N(0, I), shape (B, ...).
    num_steps
        Number of Euler steps. 1 step works for fully-rectified flows; more
        steps reduce discretisation error if the flow isn't perfectly straight.

    Returns x_1_hat, shape same as x_0.
    """
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _reshape_t_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape (B,) to (B, 1, 1, ..., 1) to broadcast over `ref`'s feature dims."""
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
