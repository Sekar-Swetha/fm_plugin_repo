# MediaPipe → OpenPose plugin for MotionEditor

A drop-in replacement for MotionEditor's OpenPose-based pose preprocessing
(`data_preparation/video_skeletons.py`). Same input layout, same output
layout, same skeleton topology and colour palette — so the existing
ControlNet-OpenPose checkpoint used by MotionEditor accepts the output
without retraining.

This package replaces *only* the pose-extraction step. The rest of MotionEditor
(motion adapter, two-branch attention injection, skeleton alignment, U-Net
inflation) is **untouched**.

## Why this exists

MotionEditor uses `controlnet_aux.OpenposeDetector` to render
ControlNet-compatible skeleton PNGs from input frames. OpenPose is fragile to
install (the CMU Caffe build chain is no longer maintained on recent Linux /
macOS toolchains). MediaPipe Pose ships as a pip-installable wheel, runs on
CPU, and produces equivalent body keypoints. The catch is that MediaPipe's
BlazePose topology is 33-landmark and the ControlNet checkpoint expects the
OpenPose 18-keypoint COCO format with specific limb colours; this plugin
handles the remap and the colour-faithful re-rendering.

## Folder layout

```
mediapipe_motioneditor_plugin/
├── README.md                 ← this file
├── requirements.txt          ← pip dependencies
├── mp_openpose_extractor.py  ← core library: remap + render
├── extract_pose_video.py     ← CLI: drop-in for video_skeletons.py
├── verify.py                 ← visual + statistical proof generator
├── tests/test_geometry.py    ← offline unit tests (no MediaPipe needed)
└── proofs/                   ← (created by verify.py) artefacts to inspect
```

## Install

```bash
pip install -r requirements.txt
```

That installs MediaPipe, OpenCV, NumPy, Pillow, tqdm. No GPU required.

## Use (drop-in for MotionEditor)

The original MotionEditor pose preprocessing is run via

```bash
python data_preparation/video_skeletons.py -d data/case-1/images -c openposefull
```

Replace that call with:

```bash
python mediapipe_motioneditor_plugin/extract_pose_video.py \
    -d data/case-1/images -c openposefull
```

The output PNGs land in `data/case-1/openposefull/0000.png` etc. — exactly
where the original script puts them, so the downstream MotionEditor scripts
(`train_adaptor.py`, `inference.py`) need no edits.

Optional flags:
- `--min-conf 0.3` (default) — MediaPipe visibility threshold per landmark.
- `--model-complexity 2` (default, heavy/best) — set to `1` for ~5× speedup
  on CPU at minor accuracy cost.
- `--write-keypoints` — also dumps a `keypoints.json` for auditing.

The CLI also accepts a video file directly (`-d source.mp4`), which is handy
for first-pass smoke tests without unpacking frames.

## Use (with a video file directly)

```bash
python mediapipe_motioneditor_plugin/extract_pose_video.py \
    -d /path/to/source.mp4 -c openposefull --write-keypoints
```

Output goes to `<path-without-extension>_openposefull/0000.png ...`.

## Proofs — how to convince yourself (and the supervisor) it's correct

Run the verification harness:

```bash
python mediapipe_motioneditor_plugin/verify.py \
    --frames data/case-1/images \
    --openpose-ref data/case-1/source_condition/openposefull \
    --out proofs/
```

This emits, under `proofs/`:

1. **`side_by_side/<frame>.png`** — original frame | OpenPose reference (if
   provided) | MediaPipe (ours). Eyeball test: same person, same pose, same
   limb-colour scheme.
2. **`overlay/<frame>.png`** — input frame with our skeleton blended over it.
   Quick sanity check that joints land on the right body parts.
3. **`diff/<frame>.png`** — pixel-level `abs(ours - openpose_ref)`. Bright
   pixels = disagreement. A clean run should look mostly black with thin
   bright contours along limb edges (sub-pixel offsets are normal).
4. **`report.json`** — machine-readable: detection rate, per-keypoint
   coverage, mean visibility, mean/max/min L1 vs. reference.
5. **`report.md`** — human-readable summary table. This is what to paste
   into the dissertation appendix.

The `--openpose-ref` argument is optional. If you don't have an OpenPose
reference yet (e.g. because OpenPose itself doesn't build on your machine),
the harness still produces overlays and statistics — enough for the
supervisor to confirm that the skeletons look right.

## Offline unit tests

`tests/test_geometry.py` exercises the remap + render logic against synthetic
landmark data. It does **not** invoke MediaPipe and runs in under a second.
This is the test the supervisor can run to confirm the library logic is
correct independent of MediaPipe's model:

```bash
python -m unittest discover -s tests -v
```

Coverage:
- Spec sanity: 18 keypoint slots, 17 limb pairs, indices in range.
- Remap correctness: neck = shoulder midpoint, MediaPipe-left → OpenPose-left,
  pixel scaling matches canvas size, low-confidence landmarks are dropped,
  `None` input is safe.
- Render correctness: output canvas matches input dimensions, empty frame
  → fully-black canvas, all 18 joint colours appear in the rendered output,
  partial poses render without errors.

## Mapping table (MediaPipe BlazePose 33 → OpenPose COCO-18)

| OP idx | OP name      | source                                 |
|-------:|:-------------|:---------------------------------------|
|      0 | `nose`       | MP 0 (nose)                            |
|      1 | `neck`       | midpoint(MP 11, MP 12) (shoulders)     |
|      2 | `r_shoulder` | MP 12 (right_shoulder)                 |
|      3 | `r_elbow`    | MP 14 (right_elbow)                    |
|      4 | `r_wrist`    | MP 16 (right_wrist)                    |
|      5 | `l_shoulder` | MP 11 (left_shoulder)                  |
|      6 | `l_elbow`    | MP 13 (left_elbow)                     |
|      7 | `l_wrist`    | MP 15 (left_wrist)                     |
|      8 | `r_hip`      | MP 24 (right_hip)                      |
|      9 | `r_knee`     | MP 26 (right_knee)                     |
|     10 | `r_ankle`    | MP 28 (right_ankle)                    |
|     11 | `l_hip`      | MP 23 (left_hip)                       |
|     12 | `l_knee`     | MP 25 (left_knee)                      |
|     13 | `l_ankle`    | MP 27 (left_ankle)                     |
|     14 | `r_eye`      | MP 5  (right_eye)                      |
|     15 | `l_eye`      | MP 2  (left_eye)                       |
|     16 | `r_ear`      | MP 8  (right_ear)                      |
|     17 | `l_ear`      | MP 7  (left_ear)                       |

Both MediaPipe and OpenPose use the person's own perspective for left/right.
"Left shoulder" = the shoulder on the subject's left side = the shoulder on
the camera's right side. No mirror flip required.

## What this plugin does *not* do

- It does **not** modify any MotionEditor code. The plugin sits beside
  `data_preparation/` and is invoked instead of `video_skeletons.py`.
- It does **not** handle the depth modality (`-c depth`) — keep using the
  original `video_skeletons.py` for that.
- It does **not** detect hands or faces in the OpenPose-full sense (the
  ControlNet-OpenPose checkpoint used by MotionEditor is called with
  `hand_and_face=False` in `video_skeletons.py`, so the body-only 18-point
  skeleton is the correct target).
- It assumes a **single person** per frame. MediaPipe Pose tracks one
  primary subject. MotionEditor's evaluation cases are all single-person.

## Next steps (after pose plugin is verified)

1. Run `verify.py` on case-1..case-6 → archive `proofs/` per case for the
   dissertation appendix.
2. Re-run `train_adaptor.py` and `inference.py` against the MediaPipe-derived
   skeletons. The remaining MotionEditor code should not need edits.
3. Move on to the next planned change (CFM-OT training swap — see
   `MotionEditor_Gap_Analysis.md §4 Contribution A`).
