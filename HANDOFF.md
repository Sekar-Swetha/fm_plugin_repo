# HANDOFF — run the GPU pipeline

_For the senior with GPU access. Everything CPU-side is done and verified; this
doc is the GPU part. Should take ~1–2 h of mostly-waiting on one case._

Contact / owner: Swetha (sekars@tcd.ie). MSc dissertation: improving
**MotionEditor** (CVPR2024 video motion editing) with (1) MediaPipe replacing
OpenPose and (2) Flow Matching replacing the diffusion training/inversion.

---

## What you're being asked to do

Run **one MotionEditor case end-to-end on a GPU** with the flow-matching plugin,
and send back the outputs. Concretely:
1. Train the CFM-OT adaptor (Contribution A).
2. Run inference twice — baseline, then with flow inversion (Contribution B).
3. Send back the output videos + any error logs.

You do **not** need to understand flow matching. Just run the commands and report
what happens. The one place that may need a code fix is flagged in **Step 5**.

---

## What's already done (no action needed)

- MediaPipe pose extraction replaces OpenPose — verified on all 6 cases (CPU).
- Flow-matching library (loss, inversion, reflow) — 34 unit tests pass (CPU).
- Toy correctness proofs — in `flow_matching_plugin/proofs/`.
- The only **un-run** part is the real GPU pipeline below.

Background docs if curious: `PROJECT_STATUS.md`, `WHATS_CHANGED.md`, `RUNBOOK.md`.

---

## Step 0 — get the repo

```bash
git clone -b contrib-bc-flow-inversion-reflow <REPO_URL> fm_plugin_repo
cd fm_plugin_repo
```
**Important:** the work is on branch `contrib-bc-flow-inversion-reflow`, NOT `main`.
The MotionEditor code is vendored inside the repo (`motionEditor/MotionEditor/`),
so this single clone has everything.

## Step 1 — environment

```bash
# MotionEditor deps (torch, diffusers, transformers, accelerate, einops, opencv, decord, xformers)
cd motionEditor/MotionEditor
pip install -r requirements.txt            # if present; else install the above
pip install xformers                       # recommended (speed/memory)
cd ../..
# Flow plugin needs only torch (already installed) + matplotlib for proofs
pip install matplotlib
```
GPU check: `nvidia-smi` should show a CUDA GPU. The paper used A100; a 16–24 GB
card handles one case (8 frames, 512×512). If you OOM, see Troubleshooting.

## Step 2 — download weights (~5 GB)

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
ME = "motionEditor/MotionEditor/checkpoints"
snapshot_download("lllyasviel/sd-controlnet-openpose", local_dir=f"{ME}/sd-controlnet-openpose")
# runwayml/stable-diffusion-v1-5 was removed from HF in 2024; community re-host:
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", local_dir=f"{ME}/stable-diffusion-v1-5")
PY
# sanity:
ls motionEditor/MotionEditor/checkpoints/stable-diffusion-v1-5   # expect unet/ vae/ text_encoder/ model_index.json
ls motionEditor/MotionEditor/checkpoints/sd-controlnet-openpose
```
The ControlNet path `checkpoints/sd-controlnet-openpose` is **hardcoded** in
`train_adaptor.py` and `inference.py`, so it must sit exactly there (relative to
the `motionEditor/MotionEditor/` dir). case-1 already ships its frames, masks
(`man.mask`), and poses — no GroundedSAM / data-prep needed.

## Step 3 — train (run from inside motionEditor/MotionEditor)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/flow_matching_plugin
cd motionEditor/MotionEditor

# stage 1: background reconstruction (paper step)
python3 train_bg.py --config configs/case-1/train-bg.yaml

# Contribution A: CFM-OT adaptor training (drop-in for train_adaptor.py)
accelerate launch ../../flow_matching_plugin/train_adaptor_fm.py \
    --config configs/case-1/train-motion.yaml
```
Outputs land in `outputs/train-case-1-motion/` (checkpoint-300 +
`controlnet_adapter_checkpoint-300.pth`). A few minutes each on A100.

## Step 4 — inference, BASELINE first (proves the pipeline runs)

```bash
# still inside motionEditor/MotionEditor
accelerate launch inference.py --config configs/case-1/eval-motion.yaml
```
This uses the original DDIM/null-text inversion. Output GIFs/videos under
`outputs/eval-case-1-motion/`. **If this fails, stop here and send the log** —
it means the base setup (weights/data) is off, unrelated to our contribution.

## Step 5 — inference WITH flow inversion (Contribution B) ← the one to watch

Edit `configs/case-1/eval-motion.yaml`, add these keys under `validation_data:`
(or top level, matching the file's style):
```yaml
use_flow_inversion: true
flow_inv_steps: 8
flow_inv_method: heun
loss_type: cfm_ot
```
Run again:
```bash
accelerate launch inference.py --config configs/case-1/eval-motion.yaml
```

> ⚠️ **This is the first-ever real run of our flow-inversion patch** (added to
> `inference.py`, gated by `use_flow_inversion`). It passes unit tests on stubs
> but has never touched the real U-Net. **Most likely place to need a fix:** the
> velocity closure in `flow_matching_plugin/flow_inversion.py::make_velocity_fn`
> may not pass conditioning the way MotionEditor's control-fused U-Net expects.
> Symptoms + guidance are in `RUNBOOK.md` Phase 3. If it errors, **send the full
> traceback** — Swetha can patch `make_velocity_fn` and push; you re-pull and
> re-run.

---

## What to send back to Swetha

1. `outputs/eval-case-1-motion/` videos from Step 4 (baseline) and Step 5 (flow inv).
2. Wall-clock time printed for each inference run (baseline vs flow inv).
3. Any error tracebacks (esp. Step 5).
4. `nvidia-smi` GPU model + peak memory if you noticed OOM.

That's enough for the next iteration. Thank you!

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `motionEditor/MotionEditor` empty after clone | you cloned `main`; checkout `contrib-bc-flow-inversion-reflow` |
| `checkpoints/sd-controlnet-openpose` not found | weights not in `motionEditor/MotionEditor/checkpoints/`, or not running from that dir |
| `runwayml/stable-diffusion-v1-5` 404 on download | use `stable-diffusion-v1-5/stable-diffusion-v1-5` (already in Step 2) |
| CUDA out of memory | set `n_sample_frames` lower in the config, ensure `gradient_checkpointing: True` + xformers on; or use a bigger GPU |
| `use_flow_inversion=True but flow_matching_plugin is not importable` | `export PYTHONPATH=$PYTHONPATH:/abs/path/to/flow_matching_plugin` |
| `Flow inversion requires loss_type='cfm_ot'` | Step 3 (train A) must finish first; eval must set `loss_type: cfm_ot` |
| Step 5 crashes in the U-Net forward | expected — send traceback; needs a `make_velocity_fn` tweak |
