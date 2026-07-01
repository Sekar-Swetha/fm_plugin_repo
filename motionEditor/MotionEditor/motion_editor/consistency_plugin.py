"""Consistency-distilled (C3) few-step sampling + shared conditioned-eps forward.

Isolated from the flow-matching and DPM-Solver++ paths. Fires only when the eval
config sets `use_consistency: True`.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "flow_matching_plugin"))
from consistency_core import predict_x0_from_eps  # noqa: E402


def conditioned_eps(unet, controlnet, x, t, cond_images, encoder_hidden_states, video_length):
    """Single-branch conditioned epsilon (no two-branch injection, no CFG).

    Mirrors train_adaptor_fm.py:355-397: ControlNet on the target skeleton, then
    the UNet non-injection branch (mid batch != 4). The two-branch injection the
    teacher used is distilled into the student weights, not recomputed here.
    """
    from einops import rearrange  # lazy: only needed for the real GPU forward

    cn_in = rearrange(x, "b c f h w -> (b f) c h w").to(dtype=controlnet.dtype)
    cn_cond = torch.cat([cond_images] * x.shape[0], dim=0)
    down, mid = controlnet(
        cn_in, t,
        encoder_hidden_states=encoder_hidden_states.repeat_interleave(video_length, dim=0),
        controlnet_cond=cn_cond, conditioning_scale=1.0, return_dict=False,
    )
    down = [rearrange(s, "(b f) c h w -> b c f h w", f=video_length) for s in down]
    mid = rearrange(mid, "(b f) c h w -> b c f h w", f=video_length)
    return unet(
        x, t, encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=down, mid_block_additional_residual=mid,
        source_masks=None, target_masks=None, rectangle_source_masks=None, skeleton=None,
    ).sample


def consistency_sample(model_fn, x_init, timesteps, alphas, sigmas, generator=None):
    """Multistep consistency sampling. model_fn(x, t_int) -> eps.

    At each step predict x0 via the consistency function; if not the last step,
    renoise to the next (lower) timestep. Returns the final x0.
    """
    x = x_init
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        eps = model_fn(x, t)
        alpha_t = alphas[t].to(x.device)
        sigma_t = sigmas[t].to(x.device)
        x0 = predict_x0_from_eps(x, eps, alpha_t, sigma_t)
        if i == n - 1:
            return x0
        t_next = timesteps[i + 1]
        a_n = alphas[t_next].to(x.device)
        s_n = sigmas[t_next].to(x.device)
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        x = a_n * x0 + s_n * noise
    return x
