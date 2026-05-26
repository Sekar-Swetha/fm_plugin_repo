# Contributions B & C — Flow Inversion + Reflow + Runnable Small-Set Demo

Date: 2026-05-26
Status: Approved, in implementation.

## Goal

1. Implement Contributions B (flow inversion) and C (reflow) per
   `flow_matching_plugin/README_contrib_BC.md`.
2. Provide a CPU-only, runnable small-set demo so the code can be executed
   end-to-end without the heavy MotionEditor GPU pipeline (no GPU, no trained
   checkpoint, no working OpenPose).

## Context / constraints

- Environment: CPU-only, `torch 2.11.0`, `matplotlib`, `numpy` installed.
- The README's Status snapshot claims `flow_inversion.py` and
  `tests/test_inversion.py` are "Implemented" — **both are absent**. All of B
  and C are built from scratch.
- The full MotionEditor pipeline (VAE encode, trained ControlNet+U-Net,
  null-text optimisation) cannot run here. Real model-dependent metrics
  (CLIP, LPIPS-T, DDIM-50 baseline) are therefore shipped as **labelled
  stubs** with the harness wired for a future GPU run.
- Real pose skeletons ARE available: `mpipe/source_video_openposefull/`
  contains 94 pre-rendered OpenPose-format skeleton PNGs; `mpipe/frames/`
  has 91 source frames; `mediapipe_motioneditor_plugin/` can re-render
  skeletons if `mediapipe` is importable.

## Design decision (user-approved)

The runnable demo uses **real mpipe skeletons as conditioning** and a **tiny
stub U-Net + synthetic latents seeded per-frame from those skeletons** as the
stand-in for VAE latents + trained model. This exercises the real
`flow_invert` / `flow_sample` / reflow code paths on real-skeleton
conditioning, runs in seconds on CPU, and emits proof artefacts.

## Files

### Contribution B (`flow_matching_plugin/`)
- `flow_inversion.py`
  - `flow_invert(velocity_fn, x_1, num_steps, method)` — backward Euler/Heun ODE, `t: 1→0`.
  - `flow_sample(velocity_fn, x_0, num_steps, method)` — forward ODE, `t: 0→1`.
  - `roundtrip_error(velocity_fn, x_1, num_steps, method)` — `‖sample(invert(x_1)) − x_1‖₂`.
  - `make_velocity_fn(unet, controlnet, text_encoder, cond, ...)` — public closure builder, reused by C's pair generator.
  - `MotionEditorFlowInversion` — wraps U-Net + ControlNet + text encoder into the `velocity_fn` signature; `.invert(latents, prompt_ids)`.
  - Integrator results expose `.latent_0` and `.latent_1` (README references `.latent_0`).
- `verify_inversion.py` — emits `proofs/roundtrip_vs_steps.png`,
  `proofs/inversion_trace.png`, `proofs/inversion_vs_ddim.md` (stub table;
  model-dependent rows labelled "requires GPU pipeline"), `report.md`.
- `tests/test_inversion.py` — identity-velocity round trip, OT-velocity round
  trip, batch shapes, Heun-vs-Euler error scaling, `MotionEditorFlowInversion`
  mock path, end-to-end smoke on `(1,4,4,8,8)`, regression pin of round-trip
  error at `num_steps=8` (fixed seed, 3 dp).

### Contribution C (`flow_matching_plugin/`)
- `generate_reflow_pairs.py` — `--mode c1|c2`; writes sharded `.pt`
  `(Z_0, Z_1, cond)` tuples (≤2 GB/shard) + sidecar JSON (seed + checkpoint
  hash). `cond` saved as the skeleton tensor, not post-ControlNet residuals.
- `train_reflow.py` — reads pairs from disk instead of sampling
  `Z_0∼N`; reuses `compute_x_t` / `compute_target_velocity` / loss from
  `flow_matching_loss.py`; inits U-Net from Contribution A checkpoint; asserts
  `loss_type == cfm_ot` on startup; writes to a new checkpoint dir.
- `configs/train-motion-reflow.yaml` — `reflow_mode`, `reflow_pairs_dir`,
  `reflow_round`, `init_checkpoint`, `num_pair_gen_steps`.
- `verify_reflow.py` — emits `proofs/path_straightness_reflow.png`,
  `proofs/nfe_vs_quality.png`, `proofs/c2_directness.md`, `proofs/report.md`.
- `tests/test_reflow.py` — `test_pair_generator_shapes`,
  `test_reflow_loss_matches_cfm_at_round0` (bit-for-bit),
  `test_round1_straighter_than_round0` (toy 2-D, fixed seed).

### MotionEditor inference patch
- `motionEditor/MotionEditor/inference.py` — gated `--use_flow_inversion`
  patch at the inversion block (B.2.1). Baseline DDIM / null-text path stays
  bit-identical. CFM-OT checkpoint loads; epsilon checkpoint errors early with
  a clear message (checks `loss_type` in the training config sidecar).

### Runnable small-set demo (`flow_matching_plugin/demo_small_set/`)
- `demo_pipeline.py` + `run_demo.sh`:
  1. Pick 8 frames from `mpipe/frames/`; load skeleton refs from
     `mpipe/source_video_openposefull/` (re-extract via the mediapipe plugin
     only if `mediapipe` importable).
  2. Encode each skeleton PNG → small deterministic cond tensor (avg-pool to ~`4×8×8`).
  3. Seed synthetic video latents `Z_1` per frame from skeleton + noise; build
     a tiny CPU stub velocity U-Net conditioned on cond.
  4. Run B: invert `Z_1`→`Z_0`, sample back, report round-trip error.
  5. Run C: generate c1/c2 pairs → reflow-train the stub a few steps → show
     path straightening.
  6. Emit `demo_small_set/proofs/*.png` + `DEMO.md` walkthrough.
- CPU-only, completes in seconds.

## Conventions to follow (from Contribution A)
- `from __future__ import annotations`, module docstring with paper reference.
- Tests via `unittest`, `sys.path.insert(0, ...)` to import the library,
  deterministic seeds.
- Verify scripts: `matplotlib.use("Agg")`, `--out` arg, write a `report.md`.
- `__all__` export list in library modules.

## Out of scope
- Real GPU training / inference.
- Computing true CLIP / LPIPS-T / DDIM-50 numbers (stubbed, harness-ready).
- Any refactor of Contribution A code or MotionEditor beyond the gated patch.

## Acceptance
- `python -m unittest discover -s flow_matching_plugin/tests` passes
  (existing + new B & C tests).
- `python flow_matching_plugin/verify_inversion.py` and `verify_reflow.py`
  run on CPU and emit their artefacts.
- `bash flow_matching_plugin/demo_small_set/run_demo.sh` runs end-to-end on
  CPU and produces `demo_small_set/proofs/`.
- MotionEditor `inference.py` baseline path unchanged when
  `--use_flow_inversion` is absent.
</content>
</invoke>
