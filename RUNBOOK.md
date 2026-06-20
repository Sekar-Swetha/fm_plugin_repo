# RUNBOOK — getting real results

_Companion to `PROJECT_STATUS.md` (what's done) and `WHATS_CHANGED.md` (what the
code does). This file is the step-by-step to take the project from "CPU-tested"
to "real edited videos + metrics on a GPU"._

Order is strict: **Phase 2 (train A) must finish before Phase 3 (B) and Phase 4
(C)** — B and C need the CFM-OT velocity checkpoint.

```
Phase 0  env + weights      ─┐
Phase 1  MediaPipe poses     │  goal 1 (replace OpenPose)
Phase 2  train CFM-OT (A)    │  REQUIRED before B/C
Phase 3  flow inversion (B)  │  validates the inference.py patch
Phase 4  reflow (C1, C2)     │  fast sampling + direct editor
Phase 5  ablation table      ─┘  dissertation core
```

---

## Phase 0 — Environment + weights (the real blocker)

You could not run MotionEditor on your laptop. You do **not** need to fix that —
you need a GPU box. Pick one:

### Option A — Google Colab (free/Pro; good for 1 case, quick start)
```bash
# In a Colab notebook cell (GPU runtime: Runtime > Change runtime type > GPU)
!nvidia-smi                                 # confirm a GPU is attached
!git clone <your repo url> fm_plugin_repo
%cd fm_plugin_repo
```
- Pros: zero setup, free T4 runs 1 case.
- Cons: session timeouts (~12 h), T4 is slower than A100 (training still minutes,
  inference a bit slower than the README's A100 numbers), disk is ephemeral —
  save checkpoints/proofs to Google Drive.
```python
from google.colab import drive; drive.mount('/content/drive')   # persist outputs
```

### Option B — Kaggle Notebooks (free P100/T4 ×2, 30 h/week)
- Similar to Colab; longer sessions, persistent `/kaggle/working`. Add the
  MotionEditor weights as a Kaggle Dataset to avoid re-downloading.

### Option C — Rented cloud GPU (Lambda / RunPod / Vast.ai — best for full runs)
- Rent an A100 (matches the paper). Persistent disk, no timeout. ~$1–2/h.
- Use this for Phase 5 (all cases) once Phases 1–4 work on Colab.

### Common setup (all options)
```bash
# 1. MotionEditor's own environment
cd motionEditor/MotionEditor
pip install -r requirements.txt          # if a requirements file exists; else
                                         # install: torch, diffusers, transformers,
                                         # accelerate, einops, opencv-python, decord
# 2. Plugin deps
pip install mediapipe opencv-python matplotlib numpy
# 3. Put the flow plugin on the import path (used by inference.py patch + scripts)
export PYTHONPATH=$PYTHONPATH:$(pwd)/../../flow_matching_plugin
```

### Weights to download (Phase 0 checklist — nothing runs without these)
- [ ] Stable Diffusion 1.5 (base diffusion model).
- [ ] ControlNet-OpenPose checkpoint.
- [ ] MotionEditor's pretrained motion-adapter / controlnet-adapter checkpoint
      (from the MotionEditor repo's README links).
- [ ] One evaluation case's data, e.g. `data/case-1/images/` frames.

> If any MotionEditor weight link is dead, that is your first blocker — resolve
> before Phase 2. Check `motionEditor/MotionEditor/README.md` for the links.

---

## Phase 1 — MediaPipe poses (Goal 1: replace OpenPose)

Prove the pose swap before touching flow matching — it's independent and is your
headline contribution.

```bash
# Extract OpenPose-format skeletons with MediaPipe (CPU, no GPU needed)
python mediapipe_motioneditor_plugin/extract_pose_video.py \
    -d data/case-1/images -c openposefull

# Prove correctness. --openpose-ref is optional but strongly wanted: it produces
# the side-by-side + pixel-diff against a real OpenPose reference.
python mediapipe_motioneditor_plugin/verify.py \
    --frames data/case-1/images \
    --openpose-ref data/case-1/source_condition/openposefull \
    --out proofs/
```
**Gate:** in `proofs/side_by_side/` joints land on the right body parts and limb
colours match OpenPose; `report.md` shows ~100% detection. Archive `proofs/` per
case for the dissertation appendix.

> If you have **no** OpenPose reference (because OpenPose won't build), run
> `verify.py` without `--openpose-ref` — you still get overlays + stats, enough
> to show the skeletons are correct. The diff rows fill in if/when you get a ref.

---

## Phase 2 — Train CFM-OT (Contribution A) — REQUIRED before B/C

```bash
accelerate launch flow_matching_plugin/train_adaptor_fm.py \
    --config flow_matching_plugin/configs/train-motion-fm.yaml
```
- ~5–10 min/case on A100 (longer on T4). Fine-tunes the adapter, not from scratch.
- Output: a `checkpoint-300` (velocity-prediction head). Note its `output_dir`.

**A/B baseline (optional but strong for the thesis):** set `loss_type: epsilon`
in a copy of the config and train the original loss with the same seed/steps, so
you can compare loss curves and downstream edit quality.

**Gate:** training loss decreases and stabilises; checkpoint saved.

---

## Phase 3 — Flow inversion (Contribution B) — validates the inference.py patch

Add these keys to the eval config (`configs/case-1/eval-motion.yaml`):
```yaml
use_flow_inversion: true
flow_inv_steps: 8
flow_inv_method: heun
loss_type: cfm_ot          # the guard refuses an epsilon checkpoint
```
Run inference, pointing at the Phase 2 checkpoint:
```bash
python motionEditor/MotionEditor/inference.py \
    --config motionEditor/MotionEditor/configs/case-1/eval-motion.yaml
```

> ⚠️ **This is the first real execution of the `inference.py` patch.** It parses
> but has never run on a GPU. MotionEditor fuses ControlNet into the U-Net, so
> the velocity closure in `flow_inversion.make_velocity_fn` may need adjusting to
> match the real `unet(...)` call (encoder_hidden_states / controlnet_cond
> arguments). **Expect to debug here.** Symptoms to watch:
> - shape mismatch in the U-Net forward → fix how `make_velocity_fn` passes cond;
> - garbage output → the U-Net may need the same conditioning path inference uses
>   (source skeleton through the adapter), not a bare `cond=` kwarg.
> Use `flow_matching_plugin/verify_inversion.py` numbers as the sanity target
> (round-trip error should be tiny).

**Gate:** edited video quality ≥ the DDIM-50 + null-text baseline on the same
clip, and total inference time drops (README estimate ~10 min → ~1–2 min/video on
A100, mostly from dropping null-text optimisation).

**Baseline to compare against** (flag off):
```bash
# same config but use_flow_inversion: false, use_null_inv: true
```

---

## Phase 4 — Reflow (Contribution C)

### Step 4.1 — pre-encode clips once (needs the MotionEditor VAE)
In the MotionEditor env, VAE-encode the clips into `ReflowRecord`s and save them,
so pair generation can run anywhere afterwards:
```python
# scratch script, run once on the GPU box
import torch
from generate_reflow_pairs import ReflowRecord, save_records
# latents: (1, 4, F, H, W) per clip from vae.encode(...).latent_dist.sample()*0.18215
# cond_src / cond_tgt: the source / target skeleton tensors
records = [ReflowRecord(z_src=lat, cond_src=cs, cond_tgt=ct) for lat, cs, ct in clips]
save_records(records, "runs/clips.pt")
```

### Step 4.2 — C1 (generation reflow → few-step sampling)
```bash
python flow_matching_plugin/generate_reflow_pairs.py \
    --records runs/clips.pt \
    --checkpoint runs/case-1-fm/checkpoint-300 \
    --mode c1 --out runs/pairs-c1/ --num-steps 8

python flow_matching_plugin/train_reflow.py \
    --reflow-pairs-dir runs/pairs-c1/ \
    --init-checkpoint runs/case-1-fm/checkpoint-300 \
    --reflow-round 1 --num-steps 1000
```
Then inference at low NFE:
```yaml
use_flow_inversion: true
flow_inv_steps: 1            # or 2-4
flow_inv_method: euler
```
**Gate:** acceptable edit quality at NFE 1–4 (vs 50 for the baseline).

### Step 4.3 — C2 (transport reflow → direct video-to-video editor, the novel bit)
Only after A + B are solid on real data (C2 pairs are A+B outputs).
```bash
python flow_matching_plugin/generate_reflow_pairs.py \
    --records runs/clips.pt --checkpoint runs/case-1-fm/checkpoint-300 \
    --mode c2 --out runs/pairs-c2/ --num-steps 8
python flow_matching_plugin/train_reflow.py \
    --reflow-pairs-dir runs/pairs-c2/ \
    --init-checkpoint runs/case-1-fm/checkpoint-300 --reflow-round 1
```
**Gate (`proofs/c2_directness.md`):** running C2 on a source latent **without**
explicit inversion produces an edit whose pose-distance to the target skeleton
matches A+B within tolerance.

---

## Phase 5 — Ablation table (dissertation core)

Run the relevant rows on MotionEditor's evaluation cases and fill:

| Config | Pose src | Inversion | Sampling | CLIP ↑ | LPIPS-T ↓ | Recon-LPIPS ↓ | NFE | Wall-clock ↓ |
|---|---|---|---|---|---|---|---|---|
| Baseline | OpenPose | DDIM-50 + nulltext | DDPM-50 | | | | 50 | |
| + MediaPipe | **MediaPipe** | DDIM-50 + nulltext | DDPM-50 | | | | 50 | |
| + A | MediaPipe | DDIM | CFM-OT | | | | 50 | |
| + A + B | MediaPipe | **FlowInv-8** | CFM-OT | | | | 8 | |
| + A + B + C1 | MediaPipe | FlowInv-1 | **reflow-1** | | | | 1–4 | |
| + A + B + C2 | MediaPipe | **none (direct)** | reflow-C2 | | | | 1–4 | |

Each row is independently valid — partial completion still gives a defensible
thesis (e.g. MediaPipe + A + B done, C2 as future work).

Add a few before/after video stills per case.

---

## Quick reference — what runs WITHOUT a GPU (today, to stay productive)

```bash
cd flow_matching_plugin
python3 -m pytest tests/ -q                       # 34 tests
python3 verify_loss.py --out proofs/              # A proofs
python3 verify_inversion.py --out proofs/         # B proofs
python3 verify_reflow.py --out proofs/            # C proofs
# C dev pass on synthetic / real-shaped latents:
python3 generate_reflow_pairs.py --stub --mode c1 --out runs/pairs --num-steps 2
python3 train_reflow.py --stub --reflow-pairs-dir runs/pairs --num-steps 100
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named cv2` | opencv not installed | `pip install opencv-python` |
| `ModuleNotFoundError: motion_editor...` (real pair gen/train) | plugin run outside MotionEditor env | run from the MotionEditor repo env, or use `--stub` / `--records --stub-model` for dev |
| `use_flow_inversion=True but flow_matching_plugin is not importable` | plugin not on PYTHONPATH | `export PYTHONPATH=$PYTHONPATH:/path/to/flow_matching_plugin` |
| `Flow inversion requires loss_type='cfm_ot'` | pointed at an epsilon checkpoint | train Phase 2 first, or set the inversion off |
| U-Net shape mismatch in Phase 3 | `make_velocity_fn` cond path ≠ MotionEditor's fused control | adjust the closure to pass conditioning the way `inference.py` does |
| Colab session died, lost outputs | ephemeral disk | mount Drive, write checkpoints/proofs there |
```
