# Per-video Consistency Distillation (Contribution C3) — Design

_Date: 2026-07-01. Project: MotionEditor × flow matching (TCD MSc dissertation)._

## 1. Problem & motivation

MotionEditor edits are sharp because sharpness comes from the **frozen pretrained
Stable Diffusion epsilon prior**. The flow-matching attempts (Contributions A/B
naive CFM-OT, C2 single-pair reflow distillation) produced **soft / ghosted**
edits on the moving subject: swapping to a learned velocity field discarded that
prior and the single-video coupling was a degenerate transport (over/underfits,
no sweet spot).

The **DPM-Solver++ PF-ODE** result (`fm_outputs/dpm_outputs/`,
`dpm_15_outputs/`) fixed sharpness by sampling the frozen epsilon editor directly
with a high-order deterministic solver — **sharp at 15–20 NFE**. That is the
working teacher.

**C3 goal:** distill that sharp teacher into a **1–4 step** editor for this video
(case-1), using **Consistency Distillation (CD)**. Novelty: a few-step,
inversion-light, sharp video *motion* editor — consistency distillation applied
to MotionEditor's two-branch motion-adapter editor. This is the headline
few-step result; DPM-Solver++ remains the guaranteed-sharp safety net; naive
CFM-OT is the analyzed negative.

## 2. Scope

- **In:** per-video (one-shot) distillation for case-1. The "distribution" the
  consistency loss learns over is timesteps + injected noise on this one video's
  sharp teacher trajectory. Method = **Consistency Distillation with a
  fixed-guidance teacher** (chosen over LCM-LoRA and progressive distillation).
- **Out:** cross-video generalization (would need many teacher trajectories +
  weeks of compute); variable guidance-scale conditioning (LCM w-embedding);
  metrics scripting (done later, separate task).
- **Non-negotiable constraint:** must not touch the flow-matching or DPM-Solver++
  code paths. All C3 code lives in new files behind a new config flag.

## 3. Method (theory)

Teacher = the epsilon editor at CFG 7.5 + two-branch injection + target-skeleton
ControlNet (the recipe that produced the sharp DPM output). Its probability-flow
ODE is integrated by DPM-Solver++.

Consistency Distillation learns a student `f_θ(x_t, t)` that maps **any** point on
a teacher PF-ODE trajectory to the trajectory's clean endpoint `x_0`. Trained so
adjacent points map to the same endpoint:

```
L = d( f_θ(x_{t_{n+1}}, t_{n+1}),  f_θ⁻(x̂_{t_n}, t_n) )
```

- `x̂_{t_n}` = one teacher DPM-Solver++ step from `x_{t_{n+1}}` (ODE integrator).
- `f_θ⁻` = EMA (target) copy of the student.
- Boundary condition: `f_θ(x_0, 0) = x_0`.
- `d` = Huber (pseudo-Huber) loss.

Because the loss spans the **whole trajectory** (many `t`, many noise draws) — not
one pair — it avoids C2's single-pair degeneracy. For one video the student
effectively memorizes the full denoising path of this edit → few-step sharp.

Data `x_0` for the forward-noising is the teacher's **sharp edited latent**
`z_edit` (dumped from the DPM run): `x_t = α_t·z_edit + σ_t·ε`.

## 4. Architecture — two stages (mirrors existing generate_reflow_pairs → train_reflow)

Splitting into offline pair-gen + student training keeps the **teacher and
student from being co-resident in GPU memory** (the top risk at 8 frames / 96 GB).

### Stage 1 — `flow_matching_plugin/generate_consistency_pairs.py`
- **Inputs:** `z_edit_sharp`, `cond_tgt`, `z_src` (all dumped from the DPM sharp
  run via the existing `distill_dump_to` / `save_records` path,
  inference.py:510-517).
- **In memory:** frozen epsilon editor (fp16, no_grad) + DPM-Solver++ scheduler
  (reuse `motion_editor/dpm_solver_plugin.build_dpm_solver_scheduler`).
- **Procedure:** discretize the schedule into `N=50` steps. For sampled indices
  `n`: `x_{t_{n+1}} = α·z_edit + σ·ε`; one teacher solver step (skip `k=1`) →
  `x̂_{t_n}`. Save `(x_{t_{n+1}}, t_{n+1}, x̂_{t_n}, t_n, cond_tgt)` shards to disk.
- **Output:** `runs/case-1-cons-pairs/` cached tensors.
- **OOM fallback:** generate pairs in timestep chunks.

### Stage 2 — `flow_matching_plugin/train_consistency.py`
- **In memory:** ONLY the student UNet (+ optimizer state for the unfrozen
  subset). No teacher.
- **Student:** epsilon UNet with `trainable_modules = (attn1.to_q, attn2.to_q,
  attn_temp)` unfrozen (same convention as `train_adaptor_fm.py`); everything else
  frozen. EMA target `f_θ⁻` over the unfrozen params (decay 0.95).
- **Predict-x0 parameterization:** convert the UNet epsilon output to `x0_hat` via
  the scheduler's `α_t, σ_t`, with the boundary skip so `f(x_0,0)=x_0`.
- **Loss:** Huber( `f_θ(x_{t_{n+1}})`, stop_grad `f_θ⁻(x̂_{t_n})` ) + boundary term.
- **Output:** `outputs/train-case-1-cons/checkpoint-cons` (+ adapter `.pth`,
  matching the epsilon checkpoint layout).

### Stage 3 — `motion_editor/consistency_plugin.py` + guarded `inference.py` hook
- Fires only when config sets `use_consistency: True`. Mutually exclusive with
  `use_flow_inversion` and `use_dpm_solver`. Flow/DPM branches untouched.
- **Sampling:** multistep consistency sampling, 1–4 steps. Start from the
  **DDIM-inverted source latent** (preserves source content, like the DPM run),
  conditioned on the target skeleton (ControlNet + injection). Each step:
  `x0_hat = f_θ(x_t, t)` → renoise to the next (smaller) `t` → repeat. Decode.
- **Config:** `configs/case-1/eval-motion-consistency.yaml` (student checkpoint,
  `use_consistency: True`, `consistency_steps: 4`).

## 5. Data flow

```
DPM sharp run ──distill_dump_to──> z_edit_sharp, cond_tgt, z_src  (.pt)
      │ (Stage 1: teacher + DPM++ in mem, no_grad)
      ▼
generate_consistency_pairs ──> runs/case-1-cons-pairs/  (x_t, x̂_{t-k} shards)
      │ (Stage 2: ONLY student in mem)
      ▼
train_consistency ──> outputs/train-case-1-cons/checkpoint-cons (+EMA, +adapter.pth)
      │ (Stage 3: guarded hook, no CFG)
      ▼
inference use_consistency ──> outputs/eval-case-1-cons/*.gif  (1–4 step sharp edit)
```

## 6. Interfaces (unit boundaries)

- `generate_consistency_pairs.py`: CLI `--dump <records.pt> --checkpoint <epsilon>
  --out runs/case-1-cons-pairs/ --num-steps 50 --skip 1`. Pure data producer.
- `train_consistency.py`: CLI `--pairs-dir <> --init-checkpoint <epsilon> --out <>
  --ema-decay 0.95 --num-steps <>`. Consumes cached pairs; emits a checkpoint.
- `consistency_plugin.py`: `enable_consistency(pipeline, steps, ...)` +
  `consistency_sample(...)`; swaps in the few-step sampler, touches nothing else.

Each unit is testable in isolation: pair-gen (shapes + cache round-trip), trainer
(loss/EMA/boundary), sampler (step count + decode shape).

## 7. Testing

**CPU unit tests** (`flow_matching_plugin/tests/test_consistency.py`, no GPU):
- boundary condition `f(x_0, 0) == x_0` (numerically);
- EMA update math;
- consistency-loss shape + gradient flows only to the unfrozen subset;
- pair-cache save/load round-trip.

**GPU gates:**
- Stage 1: pairs generated without OOM; `x̂_{t_n}` closer to `z_edit` than `x_t`
  is (solver moves toward the endpoint).
- Stage 3: 1–4 step output sharpness ≈ DPM/teacher (eyeball first; LPIPS-vs-teacher
  in the later metrics task). Sweep steps 4→1.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 8-frame / 96 GB memory (teacher+student co-resident) | two-stage split — never co-resident; Stage-1 chunking if needed |
| One-video overfit collapses to a constant | consistency loss spans many t + noise (not one pair); EMA target stabilizes; eyeball across frames |
| Student can't reproduce two-branch injection at few-step | inference reuses the same injection/ControlNet conditioning path; start from inverted source latent (not pure noise) |
| Predict-x0 ↔ epsilon parameterization bug | CPU unit test on the boundary condition + α/σ conversion before any GPU run |

## 9. Deliverables

1. `generate_consistency_pairs.py`, `train_consistency.py`,
   `motion_editor/consistency_plugin.py`, inference.py guarded hook,
   `configs/case-1/eval-motion-consistency.yaml`.
2. `tests/test_consistency.py` (CPU).
3. Result gif(s) in `fm_outputs/` + ablation row (NFE 1–4).

## 10. Out of scope / future

Cross-video generalization; LCM guidance-embedding; automated metrics; other
cases (case-2+). Each is a separate spec/plan cycle.
