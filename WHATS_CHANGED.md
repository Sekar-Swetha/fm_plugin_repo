# What's in the code & what changed

_Session of 2026-06-18. Companion to `PROJECT_STATUS.md`._

This file explains, in plain terms, **what each piece of code does** and **what
was changed in this session** to close the three gaps listed in
`PROJECT_STATUS.md §3`.

---

## TL;DR of this session

Three gaps were open; all three are now closed at the code level (CPU/offline).
The real GPU run is still the next phase — see `PROJECT_STATUS.md §6`.

| Gap (was TODO) | Now | How it's proven |
|---|---|---|
| **B inference patch** not wired into MotionEditor | Wired, gated behind `use_flow_inversion` | `inference.py` parses; baseline path untouched. Needs GPU to run. |
| **C real-data mode** was a stub that refused to run | Runs on real pre-encoded latents (CPU or GPU), no blanket `SystemExit` | 3 new unit tests + CPU smoke run |
| **`verify_reflow.py` + C proofs** missing | Added; actually executed | `proofs/path_straightness_reflow.png`, `nfe_vs_quality.png`, `report_reflow.md` |

Test count: **34 tests pass** (`flow_matching_plugin/tests/`), up from 31
(added 3 for the C real-data helpers).

---

## 1. The big picture (how the code fits together)

```
 MediaPipe poses ──► A: CFM-OT loss ──► B: flow inversion ──► C: reflow (fast sampling)
 (mediapipe_…)       (flow_matching_     (flow_inversion.py)    (generate_reflow_pairs.py
                      loss.py +                                   + train_reflow.py)
                      train_adaptor_fm.py)
```

Everything in `flow_matching_plugin/` is a **library + scripts** that plug into
the real MotionEditor repo (vendored in `motionEditor/MotionEditor/`). The
library is pure-PyTorch and CPU-testable; the scripts have a `--stub` CPU dev
path and a real path that needs the MotionEditor environment + a GPU.

---

## 2. File-by-file: what each does and what changed

### `flow_matching_plugin/flow_inversion.py` — Contribution B (unchanged this session)
The ODE inversion library.
- `flow_invert(velocity_fn, x_1, ...)` — integrates the velocity field
  **backwards** (t: 1→0) to recover the noise latent. Replaces DDIM inversion.
- `flow_sample(...)` — integrates **forwards** (t: 0→1) to generate/edit.
- `MotionEditorFlowInversion` — wraps a U-Net into `.invert()` / `.sample()`.
- `make_velocity_fn(unet, cond, ...)` — builds the velocity closure both B and C
  share (defensive about real-diffusers vs stub return types).
- `assert_cfm_ot_checkpoint(loss_type)` — refuses an epsilon checkpoint (B needs
  a velocity field).

### `motionEditor/MotionEditor/inference.py` — **CHANGED (Gap 1)**
This is the real MotionEditor inference script. Two edits, both additive:

1. **New import block** (near the other imports): tries to import the flow
   plugin; sets `_FLOW_INV_AVAILABLE`. Wrapped in `try/except` so if the plugin
   isn't on the path, the baseline still runs.

2. **New gated branch** in the inversion section (~line 275): when the eval
   config has `use_flow_inversion: true`, it builds `MotionEditorFlowInversion`
   and calls `.invert(latents, prompt_ids)` instead of null-text/DDIM. It first
   calls `assert_cfm_ot_checkpoint(...)` to fail early on a wrong checkpoint.
   When the flag is off/absent (`getattr(..., False)`), the original
   `use_null_inv` / `ddim_inversion` paths run **bit-identically** — nothing else
   in the file changed.

> ⚠️ This patch **parses** but has **not been executed** — running it needs the
> MotionEditor weights + a GPU. It is faithful to the spec in
> `README_contrib_BC.md §B.2.1`; validate it on the first GPU run.

### `flow_matching_plugin/generate_reflow_pairs.py` — **CHANGED (Gap 2)**
Builds the `(Z_0, Z_1, cond)` training pairs for reflow.

What it already did: `generate_pairs(...)` (mode c1 = noise→data, c2 =
source→edited), `save_shards(...)`. The CLI used to **`raise SystemExit`** on any
non-stub run.

**New this session:**
- `save_records()` / `load_records()` — persist VAE-encoded clips + skeleton
  conditions to one `.pt`. This lets you do the heavy VAE-encode **once** inside
  the MotionEditor env, then generate pairs anywhere (CPU or GPU). *(unit-tested:
  round-trip equality)*
- `real_velocity_factory(unet, ...)` — turns a real (or stub) MotionEditor U-Net
  into the `cond → velocity_fn` factory `generate_pairs` needs, reusing
  `make_velocity_fn`. *(unit-tested with a stub U-Net + end-to-end pair gen)*
- `load_motioneditor_velocity_factory(checkpoint)` — real path; lazily imports
  the MotionEditor U-Net and loads the CFM-OT checkpoint. Clear error if the
  MotionEditor package isn't importable.
- **CLI rewritten**: now supports `--records` (real pre-encoded latents) and
  `--checkpoint` (real model). The blanket `SystemExit` is gone — real mode
  actually exists. `--stub` / `--stub-model` keep the CPU dev paths.

### `flow_matching_plugin/train_reflow.py` — **CHANGED (Gap 2)**
Trains the reflow model on the pairs above. The loss is **identical** to
Contribution A's (`reflow_training_step` reuses `cfm_ot_loss`); only the data
source differs. The generic `train(...)` loop was already there and tested.

**New this session:**
- `load_motioneditor_velocity_model(init_checkpoint)` — real path; lazily inits
  the MotionEditor U-Net from the Contribution A checkpoint and returns a
  `model_call` adapter matching its forward signature.
- **`main()` restructured**: `--stub` trains the tiny CPU model; `--init-checkpoint`
  trains the real U-Net; otherwise a clear error. No more blanket `SystemExit`.
  Always writes to a **new** checkpoint dir (Contribution A weights untouched).

### `flow_matching_plugin/verify_reflow.py` — **NEW (Gap 3)**
CPU-only empirical proof for reflow, mirroring `verify_inversion.py`. Trains a
1-rectified flow on toy 2-D data, does one C1 reflow round, and emits:
- `path_straightness_reflow.png` — path-length integral, round 0 vs round 1.
- `nfe_vs_quality.png` — sample quality vs number of Euler steps.
- `report_reflow.md` — the numbers.

**Result from the actual run:** path length 3.549 → 3.433 (now ~the straight-line
bound 3.451), and **1-step (NFE=1) sample quality improved 89.4%** after one
reflow round. This is the toy-scale evidence that reflow works as intended.

### `flow_matching_plugin/tests/test_reflow.py` — **CHANGED (Gap 2)**
Added `TestRealDataMode` (3 tests): `save_records`/`load_records` round-trip,
`real_velocity_factory` on a stub U-Net, and an end-to-end pair-gen run.

### `flow_matching_plugin/README_contrib_BC.md` — **CHANGED**
Status snapshot table updated: the rows that were `TODO` (pair generator, reflow
trainer, config, proofs) are now marked implemented, with the inference patch
flagged "needs GPU to validate".

---

## 3. How to run everything now (CPU, today)

```bash
cd flow_matching_plugin

# 1. All unit tests (34 pass)
python3 -m pytest tests/ -q

# 2. Reflow proofs (Gap 3) — produces proofs/*.png + report_reflow.md
python3 verify_reflow.py --out proofs/

# 3. Reflow pairs from real pre-encoded latents (Gap 2), CPU dev pass
#    (replace records.pt with latents you VAE-encode in the MotionEditor env)
python3 generate_reflow_pairs.py --records records.pt --stub-model \
    --mode c2 --out runs/pairs-c2/ --num-steps 4
python3 train_reflow.py --stub --reflow-pairs-dir runs/pairs-c2/ --num-steps 300
```

## 4. How to run the real thing (GPU + MotionEditor env, next phase)

```bash
# B — flow inversion at inference: just add to the eval YAML and run inference.py
#     use_flow_inversion: true
#     flow_inv_steps: 8
#     flow_inv_method: heun
#     loss_type: cfm_ot

# C — real pairs + reflow from the Contribution A checkpoint
python3 generate_reflow_pairs.py --records runs/clips.pt \
    --checkpoint runs/case-1-fm/checkpoint-300 --mode c1 --out runs/pairs-c1/
python3 train_reflow.py --reflow-pairs-dir runs/pairs-c1/ \
    --init-checkpoint runs/case-1-fm/checkpoint-300 --reflow-round 1
```

---

## 5. What is still NOT done (honest)

- **No real GPU run yet** of B or C. The inference patch is unexecuted.
- **C2 real pairs** depend on a working A+B on real data first.
- The real-clip metrics (CLIP / LPIPS-T / wall-clock) tables are still empty —
  they need the GPU pipeline. See `PROJECT_STATUS.md §6` for the full plan.
