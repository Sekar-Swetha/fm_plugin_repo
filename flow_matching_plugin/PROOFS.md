# Correctness arguments — Contribution A (CFM-OT training)

This file pairs with `verify_loss.py` and `tests/test_flow_matching.py`.
Each claim is stated, justified mathematically, then mapped to a runnable
test or visual artefact.

The goal: convince a supervisor that the new loss is correct *before* they
spend GPU hours re-training MotionEditor on it.

---

## 0. What the change actually is

In `MotionEditor/train_adaptor.py`, lines 333–376 implement DDPM
epsilon-prediction:

```python
noise = torch.randn_like(latents)                                      # x_0
timesteps = torch.randint(0, num_train_timesteps, ...).long()           # discrete t
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)    # x_t (curved path)
target = noise                                                          # eps target
model_pred = unet(noisy_latents, timesteps, ...)                        # predicts eps
loss = F.mse_loss(model_pred.float(), target.float())                   # eps loss
```

We replace those with CFM-OT (Lipman et al. 2023, §4.1):

```python
t = uniform(eps, 1 - eps)                                              # continuous t in (0,1)
x_0 = randn_like(latents)                                              # x_0 ~ N(0, I)
x_1 = latents                                                          # data
x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1                        # straight OT interpolation
target = x_1 - (1 - sigma_min) * x_0                                   # velocity target (eq. 2)
timesteps = round(t * (num_train_timesteps - 1))                       # for unet's time embed
model_pred = unet(x_t, timesteps, ...)                                 # predicts velocity
loss = F.mse_loss(model_pred.float(), target.float())                  # velocity loss
```

Everything outside this block — model architecture, ControlNet, motion
adapter, optimizer, dataloader, attention injection — is byte-for-byte
unchanged.

---

## 1. Path endpoints match the OT specification

**Claim.** The constructed interpolation `x_t` satisfies the two boundary
conditions of an OT probability path:
- `x_t = x_0` at `t = 0` (pure noise),
- `x_t = sigma_min * x_0 + x_1 ≈ x_1` at `t = 1` (data, with tiny residual
  noise when `sigma_min > 0`; exactly `x_1` when `sigma_min = 0`).

**Justification.** Direct substitution into the equation
`x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1`:
- at `t = 0` → coef of `x_0 = 1`, coef of `x_1 = 0` → `x_t = x_0`. ✓
- at `t = 1` → coef of `x_0 = sigma_min`, coef of `x_1 = 1` →
  `x_t = sigma_min * x_0 + x_1`. ✓

**Backed by.**
`tests/test_flow_matching.py::TestPathEndpoints::test_t_zero_returns_noise`,
`test_t_one_returns_data_when_sigma_min_zero`,
`test_t_one_with_sigma_min_positive`.

---

## 2. The closed-form target velocity is the time-derivative of the path

**Claim.** The regression target `u_t = x_1 - (1 - sigma_min) * x_0` equals
`d/dt psi_t(x_0)`, i.e. the velocity of the conditional flow.

**Justification (analytic).** Differentiate `x_t` w.r.t. `t`:
```
d/dt [ (1 - (1 - sigma_min) * t) * x_0 + t * x_1 ]
  = -(1 - sigma_min) * x_0  +  x_1
  = x_1 - (1 - sigma_min) * x_0       ← target velocity
```
This matches Theorem 3, eq. 15 in Lipman et al. when specialised to the
OT-path mean/std (their Example II, eq. 20).

**Justification (numeric).** Replace the analytic derivative with the
finite difference `(x_{t+h} - x_t) / h` for tiny `h` and confirm
agreement.

**Backed by.**
`TestTargetVelocity::test_closed_form_matches_finite_difference` (passes
for sigma_min ∈ {0, 1e-3, 1e-2}).

---

## 3. The OT target is time-independent

**Claim.** `u_t` depends on `t` only through `x_t` (and not explicitly).
For the OT path, the closed-form target `x_1 - (1 - sigma_min) * x_0` has
no explicit `t` — this is what makes OT-CFM train more stably than
diffusion-path FM.

**Justification.** Inspection of eq. 2 — no `t` term. Contrast with the
VP-SDE diffusion path (Lipman et al. eq. 19) where the target contains
`alpha'_{1-t} / (1 - alpha_{1-t}^2)` factors that explode near `t = 0`.

**Implication for training.** The regression target's variance is bounded
by `Var[x_1] + (1 - sigma_min)^2 * Var[x_0]` independent of `t`, so the
gradient signal is well-conditioned at every sampled timestep. Compare
the eps-loss, whose effective SNR varies by many orders of magnitude
across the schedule.

**Backed by.** `TestTargetVelocity::test_velocity_is_time_independent`.

---

## 4. At `sigma_min = 0` this is rectified flow

**Claim.** Setting `sigma_min = 0` reduces the OT-CFM training objective
to exactly the rectified-flow training objective from Liu et al. 2022
(`x_1 - x_0` target on linear interpolation `t*x_1 + (1-t)*x_0`).

**Justification.** Substitute `sigma_min = 0`:
- `x_t = (1 - t) * x_0 + t * x_1`        (Liu et al. 2022, eq. 1, `X_t`)
- target = `x_1 - x_0`                   (Liu et al. 2022, eq. 1, `(X_1 - X_0)`)

This is *not* an approximation — it is a literal algebraic identity.

**Why this matters for Contribution C (reflow).** Because the trained
network is already a 1-rectified flow at `sigma_min = 0`, applying Liu et
al.'s reflow procedure later (Contribution C) becomes a single direct
substitution — no objective rewrite needed.

**Backed by.** `TestTargetVelocity::test_rectified_flow_equivalence_at_sigma_zero`.

---

## 5. The loss is well-formed (non-negative; zero at the optimum)

**Claim.** `cfm_ot_loss(v_pred, x_0, x_1)` returns:
- a non-negative scalar for any inputs,
- exactly zero when `v_pred == target`.

**Justification.** It is the mean of `(v_pred - target)^2`, a square →
non-negative, zero iff residual is zero.

**Backed by.** `TestLossInvariants::test_perfect_prediction_zero_loss`,
`test_loss_is_nonnegative`, `test_loss_reduction_modes`.

---

## 6. The batch builder produces correctly-shaped tensors compatible with
the MotionEditor U-Net

**Claim.** `build_fm_training_batch(x_1, cfg)` returns four tensors with
the right shapes and dtypes for plugging directly into the existing
`UNet2DConditionModel` forward call:
- `x_t` has the same shape as `x_1` (MotionEditor latent shape is
  `(B, C=4, F, H, W)`).
- `t_for_unet` is a long tensor of shape `(B,)` valid for
  `unet(..., timesteps=...)`.
- `target` has the same shape as `x_1`.

**Justification.** Direct shape arithmetic; the implementation uses
`torch.randn_like(x_1)`, `torch.rand(B, ...)`, and `_reshape_t_like` for
broadcasting. The `t_for_unet` cast preserves the integer-positional-
embedding interface that diffusers' time-embedding layers consume.

**Backed by.** `TestBatchBuilder::test_shapes`, `test_t_in_valid_range`,
`test_reproducible_with_generator`.

---

## 7. The training procedure actually learns

**Claim.** A tiny MLP trained on synthetic 2-D data with the new loss
converges to a velocity field whose Euler integration approximates the
data distribution.

**Justification.** This is the *system-level* sanity check: an end-to-end
demonstration that the loss code, batch builder, and Euler sampler all
work together. If the loss were buggy (wrong sign, missing factor, etc.),
the trained model would produce samples that look nothing like the
target.

**Backed by.**
- Deterministic test:
  `TestTrainingTrajectoryEquivalence::test_tiny_model_converges` —
  asserts post-training sample mean/std are within a tolerance of the
  data's mean/std.
- Visual:
  `proofs/samples_before_after.png` — left = data, middle =
  trained model 1-Euler-step samples, right = 10-Euler-step samples.
  The 10-step samples must visually trace the two-moons shape.

---

## 8. Sampler correctness with an oracle velocity

**Claim.** Given an *exact* velocity field, the Euler sampler recovers
`x_1` from `x_0` in a single step (when `sigma_min = 0`).

**Justification.** With `v = x_1 - x_0` and `dt = 1`, the Euler update is
`x_0 + 1 * (x_1 - x_0) = x_1`. Exactly.

This is the one-step-generation property of straight flows that rectified
flow exploits.

**Backed by.** `TestEulerSampler::test_one_step_with_oracle_velocity` and
`test_many_steps_with_oracle_velocity`.

---

## 9. Drop-in compatibility with MotionEditor's training loop

**Claim.** `train_adaptor_fm.py` differs from `train_adaptor.py` only
inside the inner training-loop loss block (and three new config knobs).
The model, dataloader, optimizer, accelerator setup, validation pipeline,
and checkpoint format are untouched.

**Justification.** Side-by-side inspection. The diff is annotated with
`# ====== Contribution A: ... ======` markers around the changed
section. The rest of the file is a verbatim copy.

**Backed by.** Manual diff:
```
diff -u MotionEditor/train_adaptor.py flow_matching_plugin/train_adaptor_fm.py
```
The only meaningful changes are in lines ~333–376 of the original (loss
block) and a small extension of `main()`'s signature for new config
fields. The U-Net `controlnet_adapter` checkpoint format is preserved
(same `.state_dict()` keys, different file suffix `_fm`) so the
inference-side code can load either kind interchangeably.

---

## How the supervisor reads this

| Time available | Do | What it proves |
|---|---|---|
| 1 min | Read this file | Theory is sound, tests exist. |
| 2 min | `python -m unittest discover -s tests -v` | Claims 1–8 deterministically. |
| 5 min | `python verify_loss.py --out proofs/` then inspect `proofs/*.png` | End-to-end training works. |
| 30 min | Run `train_adaptor_fm.py` on case-1 on a GPU box | Claim 9 + real-world convergence on the MotionEditor latent space. |

Only the last row needs a GPU. Everything above is CPU-only.
