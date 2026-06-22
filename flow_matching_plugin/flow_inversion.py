"""ODE inversion / sampling for a CFM-OT velocity field (Contribution B).

Replaces MotionEditor's DDIM inversion + null-text optimisation with a direct
ODE solve of the velocity field trained by Contribution A.

The probability-flow ODE is
    dx/dt = v_theta(x_t, t),     t in [0, 1].
Forward (sampling) integrates t: 0 -> 1 from a noise latent x_0 to a data
latent x_1. Inversion integrates t: 1 -> 0 from a data latent x_1 back to the
noise latent x_0. Because CFM-OT paths are near-straight, a handful of Euler /
Heun steps recover the latent that DDIM needs ~50 steps + null-text
optimisation to approximate.

Reference: Lipman et al. 2023 (CFM-OT); Liu et al. 2022 (rectified flow,
straight ODE paths). Companion to `flow_matching_loss.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from flow_matching_loss import FlowMatchingConfig

# A velocity closure: (x_t, t_idx_long) -> velocity tensor (same shape as x_t).
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ODESolveResult:
    """Result of a forward/backward ODE solve.

    `latent` is the final integrated state. `latent_0` / `latent_1` are
    semantic aliases: `latent_0` is the t=0 (noise) end of the path,
    `latent_1` is the t=1 (data) end, regardless of integration direction.
    """

    latent: torch.Tensor
    latent_0: torch.Tensor
    latent_1: torch.Tensor
    trajectory: List[torch.Tensor] = field(default_factory=list)


def _t_to_idx(t: float, config: FlowMatchingConfig, bsz: int, device: torch.device) -> torch.Tensor:
    t = min(max(t, config.t_eps), 1.0 - config.t_eps)
    idx = int(round(t * (config.num_train_timesteps - 1)))
    return torch.full((bsz,), idx, dtype=torch.long, device=device)


@torch.no_grad()
def flow_sample(
    velocity_fn: VelocityFn,
    x_0: torch.Tensor,
    num_steps: int,
    method: str = "heun",
    config: Optional[FlowMatchingConfig] = None,
    return_trajectory: bool = False,
) -> ODESolveResult:
    """Forward ODE solve t: 0 -> 1. Maps a noise latent x_0 to a data latent."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    cfg = config or FlowMatchingConfig()
    method = method.lower()
    dt = 1.0 / num_steps
    bsz, device = x_0.shape[0], x_0.device
    x = x_0
    traj = [x.clone()] if return_trajectory else []

    for k in range(num_steps):
        t = k / num_steps
        t_next = (k + 1) / num_steps
        v1 = velocity_fn(x, _t_to_idx(t, cfg, bsz, device))
        if method == "euler":
            x = x + dt * v1
        elif method == "heun":
            x_pred = x + dt * v1
            v2 = velocity_fn(x_pred, _t_to_idx(t_next, cfg, bsz, device))
            x = x + dt * 0.5 * (v1 + v2)
        else:
            raise ValueError(f"Unknown method: {method!r} (expected 'euler' or 'heun')")
        if return_trajectory:
            traj.append(x.clone())

    return ODESolveResult(latent=x, latent_0=x_0, latent_1=x, trajectory=traj)


@torch.no_grad()
def flow_invert(
    velocity_fn: VelocityFn,
    x_1: torch.Tensor,
    num_steps: int,
    method: str = "heun",
    config: Optional[FlowMatchingConfig] = None,
    return_trajectory: bool = False,
) -> ODESolveResult:
    """Backward ODE solve t: 1 -> 0. Recovers the noise latent from data x_1."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    cfg = config or FlowMatchingConfig()
    method = method.lower()
    dt = 1.0 / num_steps
    bsz, device = x_1.shape[0], x_1.device
    x = x_1
    traj = [x.clone()] if return_trajectory else []

    for k in range(num_steps):
        t = 1.0 - k / num_steps
        t_next = 1.0 - (k + 1) / num_steps
        v1 = velocity_fn(x, _t_to_idx(t, cfg, bsz, device))
        if method == "euler":
            x = x - dt * v1
        elif method == "heun":
            x_pred = x - dt * v1
            v2 = velocity_fn(x_pred, _t_to_idx(t_next, cfg, bsz, device))
            x = x - dt * 0.5 * (v1 + v2)
        else:
            raise ValueError(f"Unknown method: {method!r} (expected 'euler' or 'heun')")
        if return_trajectory:
            traj.append(x.clone())

    return ODESolveResult(latent=x, latent_0=x, latent_1=x_1, trajectory=traj)


@torch.no_grad()
def roundtrip_error(
    velocity_fn: VelocityFn,
    x_1: torch.Tensor,
    num_steps: int,
    method: str = "heun",
    config: Optional[FlowMatchingConfig] = None,
) -> float:
    """`‖sample(invert(x_1)) − x_1‖₂`, averaged over the batch."""
    cfg = config or FlowMatchingConfig()
    x_0 = flow_invert(velocity_fn, x_1, num_steps, method, cfg).latent
    x_1_hat = flow_sample(velocity_fn, x_0, num_steps, method, cfg).latent
    return torch.linalg.vector_norm((x_1_hat - x_1).flatten(1), dim=1).mean().item()


def make_velocity_fn(
    unet,
    cond,
    *,
    controlnet=None,
    text_encoder=None,
    prompt_ids=None,
    prepare_image_fn=None,
    extra_unet_kwargs=None,
) -> VelocityFn:
    """Build the velocity closure `flow_invert` / `flow_sample` expect.

    Mirrors a ControlNet + U-Net forward pass. Reused by the reflow pair
    generator (Contribution C) so inversion and pair generation share one
    code path. `cond` is the skeleton conditioning (image or tensor).

    Defensive about model return types so the closure works with both real
    diffusers modules and lightweight test stubs:
      - `text_encoder(prompt_ids)` may return a tensor, a tuple/list (uses [0]),
        or an object with `.last_hidden_state`.
      - `unet(...)` may return a tensor or an object with `.sample`.
    """

    encoder_hidden_states = None
    if text_encoder is not None and prompt_ids is not None:
        if prompt_ids.dim() == 1:           # CLIP needs (batch, seq_len)
            prompt_ids = prompt_ids.unsqueeze(0)
        enc = text_encoder(prompt_ids)
        encoder_hidden_states = _unwrap(enc, "last_hidden_state")

    controlnet_cond = cond
    if prepare_image_fn is not None:
        controlnet_cond = prepare_image_fn(cond)

    extra = dict(extra_unet_kwargs or {})

    def velocity_fn(x_t: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        down_res = mid_res = None
        if controlnet is not None:
            cn_out = controlnet(
                x_t, t_idx, encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
            )
            down_res, mid_res = _unwrap_controlnet(cn_out)

        kwargs = {}
        if encoder_hidden_states is not None:
            kwargs["encoder_hidden_states"] = encoder_hidden_states
        if down_res is not None:
            kwargs["down_block_additional_residuals"] = down_res
            kwargs["mid_block_additional_residual"] = mid_res
        elif controlnet is None and cond is not None and not extra:
            # No ControlNet and no explicit unet kwargs (e.g. the demo stub):
            # pass cond straight through. Real U-Nets get `extra_unet_kwargs`.
            kwargs["cond"] = controlnet_cond
        kwargs.update(extra)

        out = unet(x_t, t_idx, **kwargs)
        return _unwrap(out, "sample")

    return velocity_fn


def _unwrap(obj, attr: str):
    if isinstance(obj, (tuple, list)):
        return obj[0]
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return obj


def _unwrap_controlnet(out):
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    if hasattr(out, "down_block_res_samples"):
        return out.down_block_res_samples, out.mid_block_res_sample
    return None, None


class MotionEditorFlowInversion:
    """Drop-in inversion object for the MotionEditor inference site.

    Wraps the U-Net + (optional) ControlNet + text encoder into a velocity
    field and exposes `.invert(latents, prompt_ids)` returning the recovered
    noise latent — the replacement for `MyNullInversion(...).invert(...)` /
    `ddim_inversion(...)`.
    """

    def __init__(
        self,
        unet,
        controlnet=None,
        text_encoder=None,
        source_skeleton=None,
        num_steps: int = 8,
        method: str = "heun",
        prepare_image_fn=None,
        config: Optional[FlowMatchingConfig] = None,
        extra_unet_kwargs=None,
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if method.lower() not in ("euler", "heun"):
            raise ValueError(f"method must be 'euler' or 'heun', got {method!r}")
        self.unet = unet
        self.controlnet = controlnet
        self.text_encoder = text_encoder
        self.source_skeleton = source_skeleton
        self.num_steps = num_steps
        self.method = method.lower()
        self.prepare_image_fn = prepare_image_fn
        self.config = config or FlowMatchingConfig()
        self.extra_unet_kwargs = extra_unet_kwargs

    def velocity_fn(self, prompt_ids=None, skeleton=None) -> VelocityFn:
        return make_velocity_fn(
            self.unet,
            self.source_skeleton if skeleton is None else skeleton,
            controlnet=self.controlnet,
            text_encoder=self.text_encoder,
            prompt_ids=prompt_ids,
            prepare_image_fn=self.prepare_image_fn,
            extra_unet_kwargs=self.extra_unet_kwargs,
        )

    @torch.no_grad()
    def invert(self, latents: torch.Tensor, prompt_ids=None) -> torch.Tensor:
        vf = self.velocity_fn(prompt_ids=prompt_ids)
        return flow_invert(vf, latents, self.num_steps, self.method, self.config).latent

    @torch.no_grad()
    def sample(self, noise: torch.Tensor, prompt_ids=None, skeleton=None) -> torch.Tensor:
        vf = self.velocity_fn(prompt_ids=prompt_ids, skeleton=skeleton)
        return flow_sample(vf, noise, self.num_steps, self.method, self.config).latent


def assert_cfm_ot_checkpoint(loss_type: Optional[str]) -> None:
    """Guard the inference site: flow inversion needs a velocity-prediction
    (cfm_ot) checkpoint, not an epsilon-prediction one."""
    if loss_type is None:
        raise ValueError(
            "Could not determine `loss_type` from the training config sidecar. "
            "Flow inversion requires a CFM-OT (velocity-prediction) checkpoint."
        )
    if str(loss_type).lower() != "cfm_ot":
        raise ValueError(
            f"Flow inversion requires loss_type='cfm_ot', but the checkpoint was "
            f"trained with loss_type='{loss_type}'. Re-train with Contribution A "
            f"or use the DDIM/null-text inversion path instead."
        )


__all__ = [
    "ODESolveResult",
    "VelocityFn",
    "flow_sample",
    "flow_invert",
    "roundtrip_error",
    "make_velocity_fn",
    "MotionEditorFlowInversion",
    "assert_cfm_ot_checkpoint",
]
