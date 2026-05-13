# Contribution A — Flow Matching loss for MotionEditor

Drop-in replacement for the DDPM-epsilon training loss in
`MotionEditor/train_adaptor.py`. Uses Conditional Flow Matching with
Optimal Transport probability paths (CFM-OT) from Lipman et al., 2023.

At `sigma_min = 0` this reduces exactly to rectified flow (Liu et al.
2022), so the same training code supports both papers without changes.

## Why

MotionEditor inherits the entire DDPM stack (epsilon-prediction loss +
DDIM inversion + null-text optimisation). The loss is the part with the
fewest dependencies and the biggest downstream effect: switching it
- gives a velocity field instead of a noise field,
- produces near-straight ODE trajectories (no curved diffusion paths),
- enables few-step or one-step sampling later (Contributions B + C),
- removes the schedule-coupled gradient-variance problem at small `t`.

See `../MotionEditor_Gap_Analysis.md` §4 Contribution A for the full
motivation.

## Folder layout

```
flow_matching_plugin/
├── README.md                  ← this file
├── PROOFS.md                  ← formal correctness arguments
├── flow_matching_loss.py      ← core library (loss, sampler, helpers)
├── train_adaptor_fm.py        ← drop-in replacement for train_adaptor.py
├── verify_loss.py             ← empirical-proof generator (toy 2-D)
├── configs/
│   └── train-motion-fm.yaml   ← example MotionEditor config
└── tests/
    └── test_flow_matching.py  ← deterministic offline unit tests
```

## Install

The library itself only needs PyTorch (>= 1.13). For the verify script:
```
pip install torch matplotlib numpy
```

For running the full MotionEditor training (`train_adaptor_fm.py`), use
MotionEditor's existing environment — no extra dependencies on top of
what `MotionEditor/requirements.txt` installs.

## Usage

### 1. Smoke-test the loss (CPU, ~1 second)

```bash
cd flow_matching_plugin
python -m unittest discover -s tests -v
```

Expect 15/15 tests pass. Covers: path endpoints, target-velocity
finite-difference check, rectified-flow equivalence, loss invariants,
batch-builder shapes, Euler sampler correctness, end-to-end tiny-model
convergence.

### 2. Generate empirical proof artefacts (CPU, ~30 seconds)

```bash
python verify_loss.py --out proofs/
```

Produces under `proofs/`:
- `loss_curve.png` — CFM-OT loss over 4000 Adam steps on toy 2-D data.
- `samples_before_after.png` — target distribution | 1-step Euler |
  10-step Euler. The 10-step image must look like the target.
- `straight_paths.png` — trajectories of `x_0 → x_1` overlaid on data.
  Lines should be nearly straight.
- `report.md` — quantitative summary.

These are the "show your supervisor" artefacts. They do not require
MotionEditor or a GPU.

### 3. Train MotionEditor with CFM-OT (GPU, ~5–10 minutes on A100)

From the MotionEditor repo root:

```bash
# Original (baseline) — DDPM eps loss
accelerate launch train_adaptor.py --config configs/case-1/train-motion.yaml

# Contribution A — CFM-OT loss
accelerate launch ../flow_matching_plugin/train_adaptor_fm.py \
    --config ../flow_matching_plugin/configs/train-motion-fm.yaml
```

The output goes to a different `output_dir` so the baseline and CFM-OT
checkpoints can coexist. The checkpoint format is unchanged (same
state-dict keys); the filename suffix is `_fm` for traceability.

## Config keys added on top of MotionEditor

The CFM trainer accepts everything `train_adaptor.py` accepts plus:

| key | default | meaning |
|---|---|---|
| `loss_type` | `cfm_ot` | `cfm_ot` for the new loss; `epsilon` for the original DDPM behaviour (A/B comparison flag). |
| `flow_sigma_min` | `0.0` | OT-path noise minimum. `0.0` = straight line = rectified flow. `1e-4` matches Lipman et al.'s ImageNet config. |
| `flow_t_eps` | `1e-5` | Clamps the sampled `t` away from the degenerate endpoints. |

To run a baseline comparison without switching scripts, set
`loss_type: epsilon` in the YAML — the trainer falls back to
MotionEditor's original loss path.

## A/B comparison recipe

1. Train baseline:
   `accelerate launch train_adaptor.py --config configs/case-1/train-motion.yaml`
2. Train CFM-OT with same seed/steps:
   ```
   accelerate launch ../flow_matching_plugin/train_adaptor_fm.py \
       --config ../flow_matching_plugin/configs/train-motion-fm.yaml
   ```
3. Compare:
   - training loss curves (logged via accelerator's tracker),
   - checkpoint-300 size (must match — same params, same dtype),
   - downstream inference quality (covered by Contribution B).

## What this does NOT change

- The U-Net architecture (`UNet2DConditionModel` with the
  controlnet_adapter is reused identically).
- ControlNet weights or freezing pattern.
- The high-fidelity attention injection at inference (Contribution B
  will revisit inference; this file is training-only).
- DDIM inversion / null-text optimisation at inference — those still
  work for sanity-check inference even with a CFM-OT-trained U-Net,
  because the U-Net still operates in the same latent space; the only
  thing that changes is the interpretation of its output.

## Mapping to the gap-analysis document

| `MotionEditor_Gap_Analysis.md` claim | Implementation in this folder |
|---|---|
| §4 Contribution A: swap DDPM ε-loss for CFM-OT | `train_adaptor_fm.py` lines marked `# ====== Contribution A ======` |
| §2.7 Probability path locked to diffusion | Removed: see `flow_matching_loss.py:compute_x_t` (linear interpolation, no `alpha_t`/`beta_t`). |
| §3.1 Why straight paths are better | Empirically shown in `proofs/straight_paths.png`. |
| §3.2 Rectified-flow connection | Proven algebraically in `PROOFS.md §4`, tested in `test_flow_matching.py::test_rectified_flow_equivalence_at_sigma_zero`. |

## Next steps

Once CFM-OT training is validated on at least one MotionEditor case:
1. **Contribution B** — replace DDIM inversion + null-text optimisation
   with backward-Euler integration of the trained velocity field. Lives
   in a separate plugin folder.
2. **Contribution C** — reflow (Liu et al. 2022 §2.1) on either
   noise→data (C1) or source-pose→target-pose latent pairs (C2). Both
   re-use the same `flow_matching_loss.py` library.
