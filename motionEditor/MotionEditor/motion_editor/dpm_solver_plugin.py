"""DPM-Solver++ sampling plugin for MotionEditor (self-contained).

Theory
------
Flow matching's *sampling* object is the probability-flow (PF) ODE. For a
Gaussian diffusion, the score and the epsilon prediction are interconvertible at
every noise level:

    score(x, t) = -eps_theta(x, t) / sigma_t

and the PF-ODE velocity can be written purely in terms of the frozen epsilon
network. So we do **not** need to relearn a velocity field (that step is what
discarded Stable Diffusion's pretrained sharpness prior and overfit the single
training video). Instead we take the *sharp epsilon MotionEditor as-is* and
integrate its PF-ODE with a high-order deterministic solver.

DPM-Solver++ is exactly such a solver: it exploits the semi-linear structure of
the PF-ODE (exact linear drift + high-order approximation of the epsilon term),
giving sharp samples in ~10-20 steps instead of DDIM-50 + null-text
optimization. The sharpness is inherited from the frozen epsilon backbone; no
degenerate single-video velocity training is involved.

This module is intentionally isolated from the flow-matching code paths
(`flow_sampling`, `_flow_velocity`, Contributions A/B/C). It only swaps the
pipeline's *scheduler* object; the existing sharp-epsilon denoising loop in
`pipeline_motion_editor.py` is reused unchanged (it already calls
`self.scheduler.step(...)`).

Usage
-----
    from motion_editor.dpm_solver_plugin import enable_dpm_solver
    if getattr(validation_data, "use_dpm_solver", False):
        enable_dpm_solver(
            validation_pipeline,
            algorithm_type=getattr(validation_data, "dpm_algorithm_type", "dpmsolver++"),
            solver_order=getattr(validation_data, "dpm_solver_order", 2),
            use_karras_sigmas=getattr(validation_data, "dpm_use_karras_sigmas", False),
        )

Then run with a *non-flow*, epsilon-checkpoint config:
    use_flow_inversion: False   # no flow path
    use_null_inv: False         # plain DDIM inversion (avoids null-text assert)
    use_dpm_solver: True
    guidance_scale: 7.5         # CFG on, like the sharp baseline
    num_inference_steps: 20     # DPM-Solver++ is sharp at ~15-20; sweep 10/15/20/25

The number of forward sampling steps is taken from `num_inference_steps` because
the pipeline calls `self.scheduler.set_timesteps(num_inference_steps)` inside its
denoising loop.
"""

from __future__ import annotations

import inspect
from typing import Optional

from diffusers import DPMSolverMultistepScheduler


def build_dpm_solver_scheduler(
    source_scheduler,
    algorithm_type: str = "dpmsolver++",
    solver_order: int = 2,
    solver_type: str = "midpoint",
    use_karras_sigmas: bool = False,
) -> DPMSolverMultistepScheduler:
    """Build a DPM-Solver++ scheduler that inherits the source scheduler's beta
    schedule and prediction type (epsilon), so it samples the *same* pretrained
    diffusion model, just with a higher-order deterministic solver.

    Kwargs are filtered against the installed DPMSolverMultistepScheduler
    signature so this works across diffusers versions (the lab box is pinned to
    0.15.1, where some newer kwargs like ``use_karras_sigmas`` may be absent).
    """
    candidate_kwargs = {
        "algorithm_type": algorithm_type,
        "solver_order": solver_order,
        "solver_type": solver_type,
        "use_karras_sigmas": use_karras_sigmas,
    }
    accepted = set(
        inspect.signature(DPMSolverMultistepScheduler.__init__).parameters.keys()
    )
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    dropped = [k for k in candidate_kwargs if k not in accepted]
    if dropped:
        print(f"[dpm] scheduler ignores unsupported kwargs on this diffusers "
              f"version: {dropped}")

    # from_config carries num_train_timesteps, beta_start/end, beta_schedule,
    # steps_offset, prediction_type='epsilon', etc. from the SD scheduler.
    scheduler = DPMSolverMultistepScheduler.from_config(
        source_scheduler.config, **kwargs
    )
    print(f"[dpm] DPM-Solver++ scheduler built: algorithm_type={algorithm_type}, "
          f"solver_order={solver_order}, solver_type={solver_type}, "
          f"prediction_type={scheduler.config.prediction_type}")
    return scheduler


def enable_dpm_solver(
    pipeline,
    algorithm_type: str = "dpmsolver++",
    solver_order: int = 2,
    solver_type: str = "midpoint",
    use_karras_sigmas: bool = False,
) -> DPMSolverMultistepScheduler:
    """Swap ``pipeline.scheduler`` in place for a DPM-Solver++ scheduler.

    Only touches the scheduler. The pipeline's denoising loop, ControlNet path,
    two-branch attention injection, CFG, and the flow-matching branch are all
    left exactly as they are. Returns the new scheduler.
    """
    new_scheduler = build_dpm_solver_scheduler(
        pipeline.scheduler,
        algorithm_type=algorithm_type,
        solver_order=solver_order,
        solver_type=solver_type,
        use_karras_sigmas=use_karras_sigmas,
    )
    pipeline.scheduler = new_scheduler
    print("[dpm] pipeline.scheduler replaced with DPM-Solver++ "
          "(flow-matching code paths untouched)")
    return new_scheduler
