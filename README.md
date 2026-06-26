# fm_plugin — Flow Matching improvements for MotionEditor

Dissertation project (TCD, 2026): improving MotionEditor (CVPR 2024) using
Conditional Flow Matching with Optimal Transport paths (Lipman et al.
2023) and rectified flow (Liu et al. 2022).

Base paper: **MotionEditor: Editing Video Motion via Content-Aware Diffusion**
(Tu et al., CVPR 2024). See [`motionEditor/NOTICE.md`](motionEditor/NOTICE.md)
for full attribution.

## Repository layout

```
fm_plugin/
├── flow_matching_plugin/          ← Contribution A: CFM-OT training loss
├── mediapipe_motioneditor_plugin/ ← MediaPipe → OpenPose drop-in
├── motionEditor/                  ← Upstream MotionEditor (CVPR 2024, Apache-2.0)
├── mpipe/                         ← Pose-extraction sandbox (test video + outputs)
├── LICENSE                        ← MIT for original contributions
└── README.md                      ← this file
```

Each plugin directory has its own `README.md`, `PROOFS.md`, tests, and
proof artefacts.

## Contributions in this repo

### 1. `mediapipe_motioneditor_plugin/`
Replaces OpenPose (broken on recent toolchains) with MediaPipe Pose for
ControlNet-OpenPose skeleton preprocessing. The output PNGs are
byte-format-compatible with what `controlnet_aux.OpenposeDetector`
produces — same canvas size, same OpenPose COCO-18 topology, same limb
palette — so the ControlNet weights used by MotionEditor work without
retraining.

- Drop-in CLI: `extract_pose_video.py` mirrors
  `MotionEditor/data_preparation/video_skeletons.py`.
- 13 deterministic offline unit tests (no MediaPipe needed).
- `verify.py` produces side-by-side and overlay proofs against an
  OpenPose reference.

### 2. `flow_matching_plugin/`
Swaps MotionEditor's DDPM ε-prediction training loss for Conditional
Flow Matching with Optimal Transport paths. At `sigma_min = 0` this
reduces exactly to rectified flow's `x_1 - x_0` regression target.

- Drop-in trainer: `train_adaptor_fm.py` mirrors
  `MotionEditor/train_adaptor.py`, with the loss block swapped under
  clearly marked `# ====== Contribution A ======` comments.
- 15 deterministic offline unit tests.
- `verify_loss.py` shows end-to-end convergence on a 2-D toy problem
  with straight OT trajectories.
- `PROOFS.md` enumerates 9 numbered correctness claims with math +
  test pointers.

### 3. `mpipe/`
Pose-extraction sandbox the supervisor used to verify MediaPipe works
where OpenPose did not. Includes `source_video.mp4`,
`mpipe.py` (earlier prototype), and `source_video_openposefull/`
containing the byte-compatible OpenPose-style PNGs produced by the
plugin in (1).

## Running the current experiment (GPU lab box)

Distilled direct flow editor (Contribution C2): the source latent is mapped
directly to the edited latent by the flow model, sampled in a single ODE step.
The evaluation config `motionEditor/MotionEditor/configs/case-1/eval-motion.yaml`
is committed with the required settings.

```bash
conda activate me310
cd ~/fm_plugin_repo
git stash; git pull; git stash drop
cd motionEditor/MotionEditor
export PYTHONPATH=$(pwd):$(pwd)/../../flow_matching_plugin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 inference.py --config configs/case-1/eval-motion.yaml

# collect all output gifs to inspect
mkdir -p ~/c2_run && cp outputs/eval-case-1-motion/*.gif ~/c2_run/
```

The edited result is the right-hand panel of the side-by-side `sample-all*.gif`
in `~/c2_run/`. Please send back the contents of `~/c2_run/`.

Optional CPU-only sanity check of the flow-matching core (no GPU needed):

```bash
cd ~/fm_plugin_repo/flow_matching_plugin
python -m pytest tests/test_flow_matching.py -q
python verify_loss.py
```

## Reproducing the proofs

Each plugin is self-contained. From the repo root:

### MediaPipe pose plugin

```bash
cd mediapipe_motioneditor_plugin
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m unittest discover -s tests -v          # 13 tests
venv/bin/python extract_pose_video.py \
    -d ../mpipe/source_video.mp4 -c openposefull --write-keypoints
venv/bin/python verify.py --frames /tmp/raw_frames --out proofs/
```

### Flow matching plugin

```bash
cd flow_matching_plugin
# uses the same venv as above plus matplotlib
venv/bin/pip install matplotlib
venv/bin/python -m unittest discover -s tests -v          # 15 tests
venv/bin/python verify_loss.py --out proofs/              # 30 seconds, CPU-only
```

### Full MotionEditor training with CFM-OT (GPU required)

```bash
cd motionEditor/MotionEditor
# Install MotionEditor deps + download checkpoints per motionEditor/MotionEditor/README.md
accelerate launch ../../flow_matching_plugin/train_adaptor_fm.py \
    --config ../../flow_matching_plugin/configs/train-motion-fm.yaml
```

## Python version note

MediaPipe pinned to `<0.10.21` because newer releases removed the
legacy `mp.solutions` API. Python **3.10–3.12** are supported; Python
3.13+ is not. Use `python3.11` on macOS.

## Author

Swetha Sekar (TCD MSc/PhD dissertation, 2026).

## Citation

If you use this work, please also cite the upstream MotionEditor paper
as well as the flow-matching / rectified-flow references — see
`motionEditor/NOTICE.md` and individual plugin `PROOFS.md` files for
BibTeX entries.
