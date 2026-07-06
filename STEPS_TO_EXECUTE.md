# Steps to execute

## Step 0 — get the repo

```bash
git clone -b contrib-bc-flow-inversion-reflow <REPO_URL> fm_plugin_repo
cd fm_plugin_repo
```
Work is on branch `contrib-bc-flow-inversion-reflow`, NOT `main`. MotionEditor is
vendored inside the repo, so this single clone has everything.

## Step 1 — environment

```bash
cd motionEditor/MotionEditor
pip install -r requirements.txt            # if present; else: torch diffusers transformers accelerate einops opencv-python decord
pip install xformers                       # recommended
cd ../..
pip install matplotlib
```
`nvidia-smi` should show a CUDA GPU. One case (8 frames, 512×512) fits on a
16–24 GB card.

## Step 2 — download weights (~5 GB)

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
ME = "motionEditor/MotionEditor/checkpoints"
snapshot_download("lllyasviel/sd-controlnet-openpose", local_dir=f"{ME}/sd-controlnet-openpose")
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", local_dir=f"{ME}/stable-diffusion-v1-5")
PY

ls motionEditor/MotionEditor/checkpoints/stable-diffusion-v1-5   # expect unet/ vae/ text_encoder/ model_index.json
ls motionEditor/MotionEditor/checkpoints/sd-controlnet-openpose
```
The path `checkpoints/sd-controlnet-openpose` is hardcoded, so weights must sit
exactly there (relative to `motionEditor/MotionEditor/`). case-1 already ships
its frames, masks, and poses — no data prep needed.

## Step 3 — train (run from inside motionEditor/MotionEditor)

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/flow_matching_plugin
cd motionEditor/MotionEditor

python3 train_bg.py --config configs/case-1/train-bg.yaml

accelerate launch ../../flow_matching_plugin/train_adaptor_fm.py \
    --config configs/case-1/train-motion.yaml
```
Outputs land in `outputs/train-case-1-motion/` (checkpoint-300 +
`controlnet_adapter_checkpoint-300.pth`).

## Step 4 — inference, baseline

```bash
# still inside motionEditor/MotionEditor
accelerate launch inference.py --config configs/case-1/eval-motion.yaml
```
Output videos under `outputs/eval-case-1-motion/`. If this fails, send the log
and stop — it means the base setup (weights/data) is off.

## Step 5 — inference with flow inversion

Add to `configs/case-1/eval-motion.yaml` under `validation_data:`:
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
This is the first real run of the flow-inversion patch. If it errors in the
U-Net forward, send the full traceback (likely needs a tweak in
`flow_matching_plugin/flow_inversion.py::make_velocity_fn`).

## Step 6 — Contribution C: reflow (experimental — attempt, send what breaks)

This part is newer and less battle-tested than Steps 1–5. Run it as far as it
goes; whatever errors, send the traceback. All commands run from inside
`motionEditor/MotionEditor` with `PYTHONPATH` set (as in Step 3).

### 6a — generate real reflow pairs (uses the trained model in-memory)

Add to `configs/case-1/eval-motion.yaml` under `validation_data:` (remove/comment
the Step 5 flow-inversion keys first, or keep `loss_type: cfm_ot`):
```yaml
gen_reflow_pairs: c1                 # c1 = noise->data pairs
reflow_pairs_out: runs/pairs-c1
flow_inv_steps: 8
flow_inv_method: heun
```
```bash
accelerate launch inference.py --config configs/case-1/eval-motion.yaml
ls runs/pairs-c1/                    # expect pairs-0000.pt + .json
```

### 6b — train the reflow model

```bash
accelerate launch ../../flow_matching_plugin/train_reflow.py \
    --reflow-pairs-dir runs/pairs-c1 \
    --init-checkpoint ../../motionEditor/MotionEditor/checkpoints/stable-diffusion-v1-5/unet \
    --reflow-round 1 --num-steps 1000 --out runs/reflow
ls runs/reflow/checkpoint-reflow-1/  # expect model.pt
```
> ⚠️ This is the most likely step to need a fix — initialising the reflow U-Net
> from the MotionEditor checkpoint format is not fully wired. If it errors, send
> the traceback and stop here; 6a's pairs are already the key C artefact.

### 6c — few-step inference with the reflow model

Add to `eval-motion.yaml` under `validation_data:`:
```yaml
reflow_unet_path: runs/reflow/checkpoint-reflow-1/model.pt
use_flow_inversion: true
flow_inv_steps: 1                    # 1-4 steps (the reflow payoff)
flow_inv_method: euler
loss_type: cfm_ot
```
(remove the `gen_reflow_pairs` / `reflow_pairs_out` keys so it edits instead of
dumping)
```bash
accelerate launch inference.py --config configs/case-1/eval-motion.yaml
```

## Send back

1. `outputs/eval-case-1-motion/` videos from Step 4 (baseline) and Step 5 (flow inversion).
2. Step 6: whether 6a produced `runs/pairs-c1/`, and any 6b/6c tracebacks.
3. Wall-clock time printed for each inference run.
4. Any error tracebacks (especially Step 5 and Step 6b).
5. GPU model + peak memory if OOM.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `motionEditor/MotionEditor` empty after clone | you cloned `main`; checkout `contrib-bc-flow-inversion-reflow` |
| `checkpoints/sd-controlnet-openpose` not found | weights not in `motionEditor/MotionEditor/checkpoints/`, or not running from that dir |
| `runwayml/stable-diffusion-v1-5` 404 | use `stable-diffusion-v1-5/stable-diffusion-v1-5` (already in Step 2) |
| CUDA out of memory | lower `n_sample_frames` in the config; keep `gradient_checkpointing: True` + xformers on; or use a bigger GPU |
| `flow_matching_plugin is not importable` | `export PYTHONPATH=$PYTHONPATH:/abs/path/to/flow_matching_plugin` |
| `Flow inversion requires loss_type='cfm_ot'` | Step 3 must finish first; eval must set `loss_type: cfm_ot` |
| Step 5 crashes in the U-Net forward | expected — send traceback; needs a `make_velocity_fn` tweak |
