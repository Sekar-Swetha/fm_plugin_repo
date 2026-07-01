# Consistency Distillation (C3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distill the sharp DPM-Solver++ epsilon teacher into a 1–4 step editor for case-1 via Consistency Distillation.

**Architecture:** Two stages that never co-reside in GPU memory. Stage 1 records the real DPM sampling trajectory (via the pipeline's existing `callback`) and turns consecutive states into consistency pairs on disk. Stage 2 trains a student (epsilon UNet, unfrozen `attn*.to_q`/`attn_temp` subset + EMA target) on the cached pairs with a pseudo-Huber consistency loss. Stage 3 samples the student in 1–4 steps behind a `use_consistency` config flag. Flow and DPM code paths are untouched.

**Tech Stack:** Python, PyTorch, diffusers 0.15.1 (pinned), MotionEditor vendored UNet/ControlNet, accelerate. Lab GPU box (RTX PRO 6000, 96 GB, conda `me310`).

## Global Constraints

- diffusers **must stay 0.15.1** (MotionEditor forks diffusers internals). Do not upgrade.
- **`n_sample_frames` is fixed at 8** — the controlnet adapter bakes `num_frames=8`; lowering it breaks a rearrange. Never change it.
- **Do not modify** flow-matching code paths (`flow_sampling`, `_flow_velocity`, `use_flow_inversion`) or the DPM-Solver++ path (`dpm_solver_plugin.py`, `use_dpm_solver`). C3 lives in new files behind `use_consistency`.
- Student unfrozen subset = exactly `("attn1.to_q", "attn2.to_q", "attn_temp")` (same as `train_adaptor_fm.py`). No LoRA library.
- Defaults (approved): `N=50` discretization, EMA decay `0.95`, pseudo-Huber loss, student NFE target 4 (sweep to 1).
- Latent tensor layout everywhere: `(B, 4, F=8, H=64, W=64)`. Conditions/cond tensors follow the existing `ReflowRecord` shapes.
- CPU unit tests must run on the laptop with no GPU and no diffusers model download (use tensor-level fixtures / tiny stubs).
- Commit after every task.

---

### Task 1: x0-from-epsilon conversion + boundary condition

**Files:**
- Create: `flow_matching_plugin/consistency_core.py`
- Test: `flow_matching_plugin/tests/test_consistency.py`

**Interfaces:**
- Produces: `predict_x0_from_eps(x_t: Tensor, eps: Tensor, alpha_t: Tensor, sigma_t: Tensor) -> Tensor` — returns `(x_t - sigma_t*eps) / alpha_t`, broadcasting scalar `alpha_t/sigma_t` over the batch. This is the consistency function `f`; at `t=0` (`alpha=1, sigma=0`) it returns `x_t`, giving the boundary condition for free.

- [ ] **Step 1: Write the failing test**

```python
# flow_matching_plugin/tests/test_consistency.py
import torch
from consistency_core import predict_x0_from_eps


def test_predict_x0_boundary_is_identity_at_t0():
    # At t=0: alpha=1, sigma=0 -> f(x,0) must equal x (boundary condition).
    x = torch.randn(2, 4, 8, 8, 8)
    eps = torch.randn_like(x)
    alpha0 = torch.tensor(1.0)
    sigma0 = torch.tensor(0.0)
    out = predict_x0_from_eps(x, eps, alpha0, sigma0)
    assert torch.allclose(out, x, atol=1e-6)


def test_predict_x0_matches_closed_form():
    x = torch.randn(1, 4, 8, 8, 8)
    eps = torch.randn_like(x)
    alpha = torch.tensor(0.6)
    sigma = torch.tensor(0.8)
    out = predict_x0_from_eps(x, eps, alpha, sigma)
    expected = (x - sigma * eps) / alpha
    assert torch.allclose(out, expected, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'predict_x0_from_eps'`.

- [ ] **Step 3: Write minimal implementation**

```python
# flow_matching_plugin/consistency_core.py
"""Consistency Distillation (C3) core math — pure, CPU-testable, no model deps."""
from __future__ import annotations

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add flow_matching_plugin/consistency_core.py flow_matching_plugin/tests/test_consistency.py
git commit -m "feat(c3): predict_x0_from_eps with boundary condition"
```

---

### Task 2: pseudo-Huber consistency loss + EMA update

**Files:**
- Modify: `flow_matching_plugin/consistency_core.py`
- Test: `flow_matching_plugin/tests/test_consistency.py`

**Interfaces:**
- Consumes: `predict_x0_from_eps` (Task 1).
- Produces:
  - `pseudo_huber_loss(a: Tensor, b: Tensor, delta: float = 1.0) -> Tensor` — scalar mean of `sqrt((a-b)^2 + delta^2) - delta`.
  - `ema_update(ema_params: list[Tensor], params: list[Tensor], decay: float = 0.95) -> None` — in-place `ema = decay*ema + (1-decay)*param`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consistency.py
from consistency_core import pseudo_huber_loss, ema_update


def test_pseudo_huber_zero_when_equal():
    a = torch.randn(3, 5)
    assert torch.allclose(pseudo_huber_loss(a, a.clone(), delta=1.0),
                          torch.tensor(0.0), atol=1e-6)


def test_pseudo_huber_positive_and_scalar():
    a = torch.zeros(4)
    b = torch.ones(4)
    loss = pseudo_huber_loss(a, b, delta=0.5)
    assert loss.ndim == 0 and loss.item() > 0.0


def test_ema_update_moves_toward_params():
    ema = [torch.zeros(2, 2)]
    p = [torch.ones(2, 2)]
    ema_update(ema, p, decay=0.9)
    # 0.9*0 + 0.1*1 = 0.1
    assert torch.allclose(ema[0], torch.full((2, 2), 0.1), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: FAIL with `ImportError: cannot import name 'pseudo_huber_loss'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to flow_matching_plugin/consistency_core.py


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add flow_matching_plugin/consistency_core.py flow_matching_plugin/tests/test_consistency.py
git commit -m "feat(c3): pseudo-Huber consistency loss + EMA update"
```

---

### Task 3: consistency pair dataset (save/load round-trip)

**Files:**
- Modify: `flow_matching_plugin/consistency_core.py`
- Test: `flow_matching_plugin/tests/test_consistency.py`

**Interfaces:**
- Produces:
  - `ConsistencyPair` dataclass: fields `x_hi: Tensor`, `t_hi: int`, `x_lo: Tensor`, `t_lo: int`, `cond: Tensor` (`x_hi` = higher-noise state, `x_lo` = the next, lower-noise teacher state).
  - `save_pairs(pairs: list[ConsistencyPair], path: str) -> str`
  - `load_pairs(path: str) -> list[ConsistencyPair]`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consistency.py
import os
from consistency_core import ConsistencyPair, save_pairs, load_pairs


def test_pair_save_load_roundtrip(tmp_path):
    pairs = [
        ConsistencyPair(
            x_hi=torch.randn(4, 8, 8, 8), t_hi=900,
            x_lo=torch.randn(4, 8, 8, 8), t_lo=850,
            cond=torch.randn(8, 3, 64, 64),
        ),
        ConsistencyPair(
            x_hi=torch.randn(4, 8, 8, 8), t_hi=850,
            x_lo=torch.randn(4, 8, 8, 8), t_lo=800,
            cond=torch.randn(8, 3, 64, 64),
        ),
    ]
    p = os.path.join(tmp_path, "pairs.pt")
    save_pairs(pairs, p)
    out = load_pairs(p)
    assert len(out) == 2
    assert out[0].t_hi == 900 and out[0].t_lo == 850
    assert torch.allclose(out[1].x_hi, pairs[1].x_hi)
    assert out[0].cond.shape == (8, 3, 64, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: FAIL with `ImportError: cannot import name 'ConsistencyPair'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to flow_matching_plugin/consistency_core.py
from dataclasses import dataclass


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add flow_matching_plugin/consistency_core.py flow_matching_plugin/tests/test_consistency.py
git commit -m "feat(c3): ConsistencyPair dataset save/load"
```

---

### Task 4: build pairs from a recorded trajectory (CPU)

**Files:**
- Modify: `flow_matching_plugin/consistency_core.py`
- Test: `flow_matching_plugin/tests/test_consistency.py`

**Interfaces:**
- Consumes: `ConsistencyPair` (Task 3).
- Produces: `pairs_from_trajectory(traj: list[tuple[int, Tensor]], cond: Tensor) -> list[ConsistencyPair]` — `traj` is a list of `(timestep_int, latent)` in **sampling order** (descending noise). Emits one `ConsistencyPair` per consecutive `(hi, lo)` step.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_consistency.py
from consistency_core import pairs_from_trajectory


def test_pairs_from_trajectory_consecutive():
    cond = torch.randn(8, 3, 64, 64)
    traj = [
        (900, torch.randn(4, 8, 8, 8)),
        (600, torch.randn(4, 8, 8, 8)),
        (300, torch.randn(4, 8, 8, 8)),
    ]
    pairs = pairs_from_trajectory(traj, cond)
    assert len(pairs) == 2
    assert pairs[0].t_hi == 900 and pairs[0].t_lo == 600
    assert pairs[1].t_hi == 600 and pairs[1].t_lo == 300
    assert torch.allclose(pairs[0].x_lo, traj[1][1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: FAIL with `ImportError: cannot import name 'pairs_from_trajectory'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to flow_matching_plugin/consistency_core.py


def pairs_from_trajectory(traj, cond) -> list:
    """Consecutive states of a deterministic teacher trajectory are exactly the
    'one teacher solver step' pairs consistency distillation needs. `traj` is in
    sampling order (descending noise): traj[i] is higher-noise than traj[i+1]."""
    pairs = []
    for (t_hi, x_hi), (t_lo, x_lo) in zip(traj[:-1], traj[1:]):
        pairs.append(ConsistencyPair(
            x_hi=x_hi, t_hi=int(t_hi), x_lo=x_lo, t_lo=int(t_lo), cond=cond))
    return pairs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add flow_matching_plugin/consistency_core.py flow_matching_plugin/tests/test_consistency.py
git commit -m "feat(c3): pairs_from_trajectory"
```

---

### Task 5: shared conditioned-epsilon forward + few-step consistency sampler (CPU, stubbed)

**Files:**
- Create: `motionEditor/MotionEditor/motion_editor/consistency_plugin.py`
- Test: `flow_matching_plugin/tests/test_consistency_sampler.py`

**Interfaces:**
- Consumes: `predict_x0_from_eps` (Task 1).
- Produces:
  - `conditioned_eps(unet, controlnet, x, t, cond_images, encoder_hidden_states, video_length) -> Tensor` — single-branch conditioned epsilon: ControlNet on the target skeleton → UNet non-injection path (mid batch != 4). Mirrors `train_adaptor_fm.py:355-397`. Returns epsilon of shape `x.shape`.
  - `consistency_sample(model_fn, x_init, timesteps, alphas, sigmas, generator=None) -> Tensor` — `model_fn(x, t_int) -> eps`; multistep consistency sampling; returns final x0. `timesteps` descending ints; `alphas`/`sigmas` are 1-D tensors indexed by timestep.

The sampler is model-agnostic (takes `model_fn`), so it is fully CPU-testable with a stub. `conditioned_eps` needs real models and is exercised on GPU in Task 7/8.

- [ ] **Step 1: Write the failing test**

```python
# flow_matching_plugin/tests/test_consistency_sampler.py
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "motionEditor", "MotionEditor", "motion_editor"))
from consistency_plugin import consistency_sample


def test_consistency_sample_step_count_and_shape():
    calls = {"n": 0}
    def model_fn(x, t):
        calls["n"] += 1
        return torch.zeros_like(x)          # eps=0 -> x0 = x/alpha
    x = torch.randn(1, 4, 8, 8, 8)
    T = 1000
    alphas = torch.linspace(1.0, 0.1, T + 1)
    sigmas = torch.linspace(0.0, 0.99, T + 1)
    timesteps = [900, 600, 300, 0]          # 4-step schedule
    out = consistency_sample(model_fn, x, timesteps, alphas, sigmas,
                             generator=torch.Generator().manual_seed(0))
    assert calls["n"] == 4                  # one model eval per step
    assert out.shape == x.shape


def test_consistency_sample_last_step_returns_x0_not_renoised():
    # With eps=0 and final t=0 (alpha=1, sigma=0), final x0 == input-at-that-step.
    def model_fn(x, t):
        return torch.zeros_like(x)
    x = torch.ones(1, 4, 4, 4, 4)
    T = 10
    alphas = torch.ones(T + 1)              # alpha=1 everywhere -> x0 = x
    sigmas = torch.zeros(T + 1)
    out = consistency_sample(model_fn, x, [5, 0], alphas, sigmas)
    assert torch.allclose(out, x, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency_sampler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'consistency_plugin'`.

- [ ] **Step 3: Write minimal implementation**

```python
# motionEditor/MotionEditor/motion_editor/consistency_plugin.py
"""Consistency-distilled (C3) few-step sampling + shared conditioned-eps forward.

Isolated from the flow-matching and DPM-Solver++ paths. Fires only when the eval
config sets `use_consistency: True`.
"""
from __future__ import annotations

import torch
from einops import rearrange

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "flow_matching_plugin"))
from consistency_core import predict_x0_from_eps


def conditioned_eps(unet, controlnet, x, t, cond_images, encoder_hidden_states, video_length):
    """Single-branch conditioned epsilon (no two-branch injection, no CFG).

    Mirrors train_adaptor_fm.py:355-397: ControlNet on the target skeleton, then
    the UNet non-injection branch (mid batch != 4). The two-branch injection the
    teacher used is distilled into the student weights, not recomputed here.
    """
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency_sampler.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add motionEditor/MotionEditor/motion_editor/consistency_plugin.py flow_matching_plugin/tests/test_consistency_sampler.py
git commit -m "feat(c3): conditioned-eps forward + few-step consistency sampler"
```

---

### Task 6: Stage-1 trajectory dump hook in inference.py (GPU integration)

**Files:**
- Modify: `motionEditor/MotionEditor/inference.py` (guarded block near the DPM hook + a callback around the sampling call)
- Create: `flow_matching_plugin/generate_consistency_pairs.py`

**Interfaces:**
- Consumes: existing `validation_pipeline(...)` `callback`/`callback_steps` args; `pairs_from_trajectory`, `save_pairs`, `load_pairs` (Tasks 3–4).
- Produces: on a DPM run with `dump_trajectory_to: <path>` set, writes a trajectory file `{ "traj": [(t_int, latent_cpu), ...], "cond": cond_cpu }`. `generate_consistency_pairs.py` converts one or more trajectory files into a pooled `pairs.pt`.

This task has no CPU unit test (needs the real pipeline + GPU). Its gate is the manual run below. The CPU-testable pairing logic it relies on is already covered by Task 4.

- [ ] **Step 1: Add the guarded callback in `inference.py`**

Add near the other `getattr(validation_data, ...)` flags, immediately before the `sample = validation_pipeline(...)` call:

```python
            # ---- C3 Stage 1: record the DPM teacher trajectory for consistency pairs ----
            _c3_traj = []
            _c3_dump = getattr(validation_data, "dump_trajectory_to", None)
            def _c3_callback(step, t, latents):
                # record the edit branch (index 1) at each sampling step
                _c3_traj.append((int(t.item()) if torch.is_tensor(t) else int(t),
                                 latents[1:2].detach().cpu().float()))
            _c3_cb = _c3_callback if _c3_dump else None
```

Then pass `callback=_c3_cb, callback_steps=1` into the `validation_pipeline(...)` call (add these kwargs; the pipeline already accepts them). After the call, add:

```python
            if _c3_dump:
                import torch as _t
                _cond = torch.unsqueeze(target_skeleton[-1], dim=0).detach().cpu().float() \
                    if target_skeleton.ndim == 4 else target_skeleton.detach().cpu().float()
                _t.save({"traj": _c3_traj, "cond": _cond}, _c3_dump)
                print(f"[c3] saved {len(_c3_traj)}-point trajectory to {_c3_dump}")
```

- [ ] **Step 2: Write `generate_consistency_pairs.py`**

```python
# flow_matching_plugin/generate_consistency_pairs.py
"""C3 Stage 1: convert recorded DPM teacher trajectories into consistency pairs."""
import argparse
import glob
import torch

from consistency_core import pairs_from_trajectory, save_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True,
                    help="one or more trajectory files saved by inference dump_trajectory_to")
    ap.add_argument("--out", required=True, help="output pairs.pt")
    args = ap.parse_args()

    files = []
    for pat in args.traj:
        files.extend(sorted(glob.glob(pat)))
    all_pairs = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        traj = [(int(t), x) for t, x in d["traj"]]
        all_pairs.extend(pairs_from_trajectory(traj, d["cond"]))
    save_pairs(all_pairs, args.out)
    print(f"[c3] {len(files)} trajectories -> {len(all_pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: GPU gate — dump 3 trajectories (seeds) and pool them**

Run on the box (edit `seed` in the DPM config between runs, or add `--seed`; three seeds give coverage):

```bash
# add to configs/case-1/eval-motion-dpm.yaml:  dump_trajectory_to: runs/c3/traj_s33.pt
python3 inference.py --config configs/case-1/eval-motion-dpm.yaml     # seed 33
# repeat with dump path traj_s34.pt / traj_s35.pt after changing seed
python3 ../../flow_matching_plugin/generate_consistency_pairs.py \
    --traj "runs/c3/traj_s*.pt" --out runs/c3/pairs.pt
```
Expected: prints `3 trajectories -> ~45-60 pairs -> runs/c3/pairs.pt` (≈15–20 steps × 3).

- [ ] **Step 4: Commit**

```bash
git add motionEditor/MotionEditor/inference.py flow_matching_plugin/generate_consistency_pairs.py
git commit -m "feat(c3): Stage-1 DPM trajectory dump + pair generation"
```

---

### Task 7: Stage-2 student trainer (GPU integration + CPU smoke test)

**Files:**
- Create: `flow_matching_plugin/train_consistency.py`
- Test: `flow_matching_plugin/tests/test_consistency_train_smoke.py`

**Interfaces:**
- Consumes: `load_pairs`, `predict_x0_from_eps`, `pseudo_huber_loss`, `ema_update` (Tasks 1–3); `conditioned_eps` (Task 5).
- Produces: `consistency_train_step(student_eps_fn, ema_eps_fn, pair, alphas, sigmas, delta=1.0) -> Tensor` (scalar loss) — the pure per-pair step, testable with stub eps functions. Plus a `main()` CLI that wires the real UNet/ControlNet, unfreezes the subset, runs EMA, and saves a checkpoint.

- [ ] **Step 1: Write the failing CPU smoke test**

```python
# flow_matching_plugin/tests/test_consistency_train_smoke.py
import torch
from consistency_core import ConsistencyPair
from train_consistency import consistency_train_step


def test_train_step_returns_scalar_with_grad():
    T = 1000
    alphas = torch.linspace(1.0, 0.1, T + 1)
    sigmas = torch.linspace(0.0, 0.99, T + 1)
    pair = ConsistencyPair(
        x_hi=torch.randn(1, 4, 8, 8, 8), t_hi=900,
        x_lo=torch.randn(1, 4, 8, 8, 8), t_lo=600,
        cond=torch.randn(8, 3, 64, 64),
    )
    w = torch.zeros(1, requires_grad=True)   # a trainable "student" parameter
    def student_eps_fn(x, t):
        return x * 0.0 + w                   # depends on w so grad flows
    def ema_eps_fn(x, t):
        return torch.zeros_like(x)
    loss = consistency_train_step(student_eps_fn, ema_eps_fn, pair, alphas, sigmas)
    assert loss.ndim == 0
    loss.backward()
    assert w.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency_train_smoke.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_consistency'`.

- [ ] **Step 3: Write `train_consistency.py`**

```python
# flow_matching_plugin/train_consistency.py
"""C3 Stage 2: consistency-distill the student on cached teacher-trajectory pairs.

Only the student is in memory (no teacher UNet). The teacher's one-solver-step
target is precomputed in the cached x_lo. Student unfreezes the same subset as
train_adaptor_fm.py; an EMA copy provides the consistency target.
"""
import argparse
import torch

from consistency_core import (
    predict_x0_from_eps, pseudo_huber_loss, ema_update, load_pairs,
)

TRAINABLE = ("attn1.to_q", "attn2.to_q", "attn_temp")


def consistency_train_step(student_eps_fn, ema_eps_fn, pair, alphas, sigmas, delta: float = 1.0):
    """One consistency step. student_eps_fn/ema_eps_fn: (x, t_int) -> eps.

    L = pseudo_huber( f_student(x_hi, t_hi), stopgrad f_ema(x_lo, t_lo) ).
    """
    a_hi, s_hi = alphas[pair.t_hi], sigmas[pair.t_hi]
    a_lo, s_lo = alphas[pair.t_lo], sigmas[pair.t_lo]
    student_x0 = predict_x0_from_eps(pair.x_hi, student_eps_fn(pair.x_hi, pair.t_hi), a_hi, s_hi)
    with torch.no_grad():
        target_x0 = predict_x0_from_eps(pair.x_lo, ema_eps_fn(pair.x_lo, pair.t_lo), a_lo, s_lo)
    return pseudo_huber_loss(student_x0, target_x0, delta=delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True, help="pairs.pt from generate_consistency_pairs")
    ap.add_argument("--init-checkpoint", required=True, help="epsilon UNet checkpoint dir")
    ap.add_argument("--adapter-weight-path", required=True, help="epsilon controlnet_adapter .pth")
    ap.add_argument("--pretrained-model-path", default="checkpoints/stable-diffusion-v1-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-steps", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--delta", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float16

    # --- load models (student UNet + frozen ControlNet + text encoder) ---
    from diffusers import DDPMScheduler, ControlNetModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from motion_editor.models.unet_2d_condition import UNet2DConditionModel

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(args.init_checkpoint, subfolder="unet").to(device, dtype)
    controlnet = ControlNetModel.from_pretrained("checkpoints/sd-controlnet-openpose", torch_dtype=dtype).to(device)
    controlnet.controlnet_adapter_load(torch.load(args.adapter_weight_path)) \
        if hasattr(controlnet, "controlnet_adapter_load") else None
    noise_sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    ac = noise_sched.alphas_cumprod
    alphas = torch.sqrt(ac)
    sigmas = torch.sqrt(1.0 - ac)

    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)
    trainable = []
    for name, p in unet.named_parameters():
        if any(k in name for k in TRAINABLE):
            p.requires_grad_(True)
            trainable.append(p)
    ema = [p.detach().clone() for p in trainable]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    # cache the prompt embedding once
    prompt_ids = tokenizer("a girl is dancing", return_tensors="pt").input_ids.to(device)
    ehs = text_encoder(prompt_ids)[0]
    video_length = 8

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "motionEditor", "MotionEditor", "motion_editor"))
    from consistency_plugin import conditioned_eps

    pairs = load_pairs(args.pairs_dir)

    def student_eps_fn(x, t):
        ts = torch.full((x.shape[0] * video_length,), int(t), device=device, dtype=torch.long)
        return conditioned_eps(unet, controlnet, x.to(device, dtype), ts,
                               cur_pair_cond.to(device, dtype), ehs, video_length).float()

    def ema_eps_fn(x, t):
        # swap trainable params -> EMA, forward, swap back
        backup = [p.detach().clone() for p in trainable]
        for p, e in zip(trainable, ema):
            p.data.copy_(e.data)
        ts = torch.full((x.shape[0] * video_length,), int(t), device=device, dtype=torch.long)
        out = conditioned_eps(unet, controlnet, x.to(device, dtype), ts,
                              cur_pair_cond.to(device, dtype), ehs, video_length).float()
        for p, b in zip(trainable, backup):
            p.data.copy_(b.data)
        return out

    step = 0
    while step < args.num_steps:
        for pair in pairs:
            cur_pair_cond = pair.cond
            loss = consistency_train_step(student_eps_fn, ema_eps_fn, pair,
                                          alphas, sigmas, delta=args.delta)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(ema, trainable, decay=args.ema_decay)
            if step % 50 == 0:
                print(f"[c3] step {step} loss {loss.item():.5f}")
            step += 1
            if step >= args.num_steps:
                break

    os.makedirs(args.out, exist_ok=True)
    unet.save_pretrained(os.path.join(args.out, "unet"))
    print(f"[c3] saved student to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the CPU smoke test to verify it passes**

Run: `cd flow_matching_plugin && python -m pytest tests/test_consistency_train_smoke.py -q`
Expected: PASS (1 passed). (Only `consistency_train_step` is exercised; `main()` needs GPU.)

- [ ] **Step 5: GPU gate — short training run**

```bash
python3 flow_matching_plugin/train_consistency.py \
    --pairs-dir runs/c3/pairs.pt \
    --init-checkpoint outputs/train-case-1-motion/checkpoint-300 \
    --adapter-weight-path outputs/train-case-1-motion/controlnet_adapter_checkpoint-300.pth \
    --out outputs/train-case-1-cons --num-steps 500
```
Expected: loss prints and **trends down**; `outputs/train-case-1-cons/unet` saved; no OOM.

- [ ] **Step 6: Commit**

```bash
git add flow_matching_plugin/train_consistency.py flow_matching_plugin/tests/test_consistency_train_smoke.py
git commit -m "feat(c3): Stage-2 consistency student trainer"
```

---

### Task 8: Stage-3 inference hook + config (GPU integration)

**Files:**
- Modify: `motionEditor/MotionEditor/inference.py`
- Create: `motionEditor/MotionEditor/configs/case-1/eval-motion-consistency.yaml`

**Interfaces:**
- Consumes: `consistency_sample`, `conditioned_eps` (Task 5); the student checkpoint from Task 7; the DDIM-inverted source latent already produced by the `else` inversion branch.
- Produces: a guarded block that, when `use_consistency: True`, samples the student in `consistency_steps` steps and returns the edited video, bypassing the DDIM/flow/DPM loops.

- [ ] **Step 1: Add the guarded consistency block in `inference.py`**

Immediately after `ddim_inv_latent = ...` is computed (the inversion) and before the per-prompt sampling loop, add:

```python
        if getattr(validation_data, "use_consistency", False):
            import torch as _t
            from motion_editor.consistency_plugin import consistency_sample, conditioned_eps
            from diffusers import DDPMScheduler as _DDPM
            _ns = _DDPM.from_pretrained(pretrained_model_path, subfolder="scheduler")
            _ac = _ns.alphas_cumprod
            _alphas = _t.sqrt(_ac); _sigmas = _t.sqrt(1.0 - _ac)
            _steps = getattr(validation_data, "consistency_steps", 4)
            _sched = [int(round((1 - i / _steps) * 999)) for i in range(_steps)]  # descending
            _ehs = text_encoder(input_dataset.prompt_ids.to(latents.device).unsqueeze(0))[0]
            _imgs = validation_pipeline.prepare_image(
                image=target_skeleton, width=input_data.width, height=input_data.height,
                batch_size=1, num_images_per_prompt=1, device=latents.device,
                dtype=controlnet.dtype, do_classifier_free_guidance=False)
            _imgs = rearrange(_imgs, "b f c h w -> (b f) c h w").to(controlnet.device, controlnet.dtype)
            def _model_fn(x, t):
                ts = _t.full((x.shape[0] * video_length,), int(t), device=x.device, dtype=_t.long)
                return conditioned_eps(unet, controlnet, x, ts, _imgs, _ehs, video_length).float()
            _x0 = consistency_sample(_model_fn, ddim_inv_latent, _sched, _alphas, _sigmas,
                                     generator=generator)
            _video = validation_pipeline.decode_latents(_x0.to(weight_dtype))
            save_videos_grid(_t.from_numpy(_video), f"{output_dir}/sample-consistency.gif")
            print(f"[c3] consistency sample ({_steps} steps) saved")
            return
```
(Use the module's existing `save_videos_grid` / video-save helper and `output_dir` variable — match the names already imported in inference.py.)

- [ ] **Step 2: Create the config**

```yaml
# motionEditor/MotionEditor/configs/case-1/eval-motion-consistency.yaml
pretrained_model_path: checkpoints/stable-diffusion-v1-5
output_dir: "outputs/eval-case-1-cons"
resume_from_checkpoint: "outputs/train-case-1-cons"
adapter_weight_path: "outputs/train-case-1-motion/controlnet_adapter_checkpoint-300.pth"

input_data:
  video_dir: "data/case-1"
  prompt: "a girl is dancing"
  n_sample_frames: 8
  width: 512
  height: 512
  sample_start_idx: 0
  sample_frame_rate: 1
  condition: [openposefull]
  video_suffix: .png
  condition_suffix: .png
  noise_level: 10000
  image_embed_drop: 0.1
  source_mask_dir: man.mask

validation_data:
  prompts:
    - "a girl is dancing"
  video_length: 8
  width: 512
  height: 512
  noise_level: 0
  num_inference_steps: 20
  num_inv_steps: 50
  guidance_scale: 1.0
  use_null_inv: False
  controlnet_conditioning_scale: 1.0
  # ---- C3 consistency few-step sampling ----
  use_consistency: True
  consistency_steps: 4          # sweep 4 -> 1
  # ---- flow / dpm OFF ----
  use_flow_inversion: False
  use_dpm_solver: False

input_batch_size: 1
seed: 33
mixed_precision: "no"
gradient_checkpointing: True
enable_xformers_memory_efficient_attention: True
use_sc_attn: True
use_st_attn: False
st_attn_idx: 0
```

- [ ] **Step 3: Verify inference.py still compiles**

Run: `cd motionEditor/MotionEditor && python3 -m py_compile inference.py`
Expected: no output (success).

- [ ] **Step 4: GPU gate — few-step consistency run**

```bash
python3 inference.py --config configs/case-1/eval-motion-consistency.yaml
```
Expected: `outputs/eval-case-1-cons/sample-consistency.gif` — sharp edited person at 4 steps. Then set `consistency_steps: 2` then `1`, rerun; find the lowest step count still sharp.

- [ ] **Step 5: Commit**

```bash
git add motionEditor/MotionEditor/inference.py motionEditor/MotionEditor/configs/case-1/eval-motion-consistency.yaml
git commit -m "feat(c3): Stage-3 few-step consistency inference hook + config"
```

---

### Task 9: full-suite regression + docs

**Files:**
- Modify: `flow_matching_plugin/README.md` (add a C3 section)

- [ ] **Step 1: Run the whole CPU test suite**

Run: `cd flow_matching_plugin && python -m pytest tests/ -q`
Expected: all prior tests + the new `test_consistency*.py` PASS.

- [ ] **Step 2: Add a short C3 section to `flow_matching_plugin/README.md`**

Document the 3-stage flow (dump trajectory → generate pairs → train → few-step infer) with the exact commands from Tasks 6–8.

- [ ] **Step 3: Commit**

```bash
git add flow_matching_plugin/README.md
git commit -m "docs(c3): consistency distillation usage"
```

---

## Self-Review

**Spec coverage:** §3 theory → Tasks 1,7 (predict-x0 + consistency step). §4 Stage 1 → Tasks 4,6. §4 Stage 2 → Tasks 2,3,7 (EMA, loss, dataset, trainer). §4 Stage 3 → Tasks 5,8 (sampler, hook). §6 interfaces → each task's Interfaces block. §7 testing → Tasks 1–5,7 CPU tests + GPU gates in 6,7,8. §8 memory risk → two-stage split realized in Tasks 6 (teacher only) and 7 (student only). All spec sections covered.

**Note vs spec (Stage 1 realization):** the spec described `x_t = α·z_edit + σ·ε` + one solver step. This plan realizes Stage 1 by **recording the real DPM trajectory** (consecutive states = the teacher solver-step pairs) to avoid re-implementing the guided-epsilon forward. Same intent (teacher-trajectory pairs), higher fidelity, lower risk. If coverage is thin, forward-noise augmentation can be added later without changing the trainer.

**Placeholder scan:** no TBD/TODO; every code step shows full code. The Task 8 hook references `save_videos_grid`/`output_dir` "match existing names" — flagged because those symbols already exist in inference.py and must be used verbatim.

**Type consistency:** `ConsistencyPair(x_hi, t_hi, x_lo, t_lo, cond)` used identically in Tasks 3,4,7. `consistency_sample(model_fn, x_init, timesteps, alphas, sigmas, generator)` and `conditioned_eps(unet, controlnet, x, t, cond_images, encoder_hidden_states, video_length)` consistent across Tasks 5,7,8. `predict_x0_from_eps(x_t, eps, alpha_t, sigma_t)` consistent Tasks 1,5,7.
