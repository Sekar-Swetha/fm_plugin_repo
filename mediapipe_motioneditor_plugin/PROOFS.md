# Correctness arguments — MediaPipe pose plugin

This document is the *paper* counterpart to `verify.py`. It records the
correctness claims that justify treating the MediaPipe-based output as a
valid input for MotionEditor's ControlNet-OpenPose conditioning, *without*
needing to run the pipeline end-to-end.

Each claim is stated, then justified, then mapped to the runnable test or
artefact that empirically backs it.

---

## Claim 1 — Output topology matches OpenPose COCO-18

**Statement.** The plugin emits 18 keypoints in the order
`(nose, neck, R-shoulder, R-elbow, R-wrist, L-shoulder, L-elbow, L-wrist,`
`R-hip, R-knee, R-ankle, L-hip, L-knee, L-ankle, R-eye, L-eye, R-ear, L-ear)`,
connected by the same 17 limbs as OpenPose.

**Justification.**
- The list `OPENPOSE_KEYPOINTS` in `mp_openpose_extractor.py` is the
  COCO-18 ordering used by both the original OpenPose and by
  `controlnet_aux.open_pose.util`.
- The constant `LIMB_SEQ_1IDX` is copied verbatim from
  `controlnet_aux.open_pose.util.draw_bodypose` (the function MotionEditor's
  preprocessor ultimately calls).
- The conversion table `MP_TO_OP` maps every OpenPose slot (except neck) to
  exactly one MediaPipe landmark. Neck is computed as the midpoint of the
  two shoulders, which is the OpenPose convention (CMU OpenPose itself
  derives the neck this way — it is not in COCO ground-truth keypoints).

**Backed by.** `tests/test_geometry.py::TestSpec` (length checks, index
ranges, MP→OP coverage) and `TestRemap::test_neck_is_shoulder_midpoint`.

---

## Claim 2 — Left/right orientation matches OpenPose convention

**Statement.** OpenPose's "left shoulder" (idx 5) is the shoulder on the
subject's own left side. MediaPipe's "left shoulder" (idx 11) is also on
the subject's own left side. We map MP 11 → OP 5 and MP 12 → OP 2; no
mirror flip is applied.

**Justification.**
- MediaPipe BlazePose documentation defines `LEFT_*` and `RIGHT_*`
  landmarks from the subject's anatomical perspective, consistent with
  OpenPose. Both libraries match the COCO keypoint convention.
- The cross-check in `tests/test_geometry.py::TestRemap::test_left_right_orientation`
  passes a synthetic person with `MP 11` at x=0.2 and `MP 12` at x=0.8
  and asserts that the resulting `OP[5]` (l_shoulder) has x=20 and `OP[2]`
  (r_shoulder) has x=80 on a 100-pixel canvas.

**Backed by.** `tests/test_geometry.py::TestRemap::test_left_right_orientation`.

---

## Claim 3 — Pixel coordinates are correct

**Statement.** The plugin produces keypoints in absolute pixel coordinates
on the same canvas as the input image. The (0, 0) origin is the top-left
corner, x grows right, y grows down — same as OpenCV and same as OpenPose.

**Justification.** MediaPipe returns landmarks in normalised
`[0, 1]` coordinates with the same origin convention; the plugin multiplies
by `width` / `height`. No further transformation is applied.

**Backed by.** `tests/test_geometry.py::TestRemap::test_pixel_scaling` plus
the visual overlay artefacts in `proofs/overlay/`.

---

## Claim 4 — Rendered skeleton uses ControlNet's exact colour palette

**Statement.** The 17 limb colours and the 18 joint-dot colours used in
the rendered PNG match `controlnet_aux.open_pose.util.colors` exactly
(in RGB).

**Justification.** The constants `LIMB_COLORS_RGB` and `JOINT_COLORS_RGB`
in `mp_openpose_extractor.py` are byte-for-byte copies of the lists in
`controlnet_aux.open_pose.util`. The render function in the plugin uses
the same drawing primitives in the same order:
1. Joint dots (`cv2.circle`, radius 4, filled, joint colour).
2. Limb ellipses (`cv2.ellipse2Poly` + `cv2.fillConvexPoly` +
   `cv2.addWeighted(0.4, 0.6)`), per the upstream code.

This matters because the ControlNet-OpenPose checkpoint *learned* to
recognise these specific colours as limb identifiers. Any other palette
would produce out-of-distribution conditioning input.

**Backed by.** `tests/test_geometry.py::TestRender::test_palette_subset`
asserts each of the 18 joint colours appears in a rendered canvas of a
full synthetic skeleton.

---

## Claim 5 — Rendered skeleton uses the input canvas size and a black background

**Statement.** Output is a uint8 BGR image of shape
`(input_height, input_width, 3)` on a fully-black background.

**Justification.** `render_openpose_skeleton(...)` allocates
`np.zeros((H, W, 3), dtype=np.uint8)` and draws on it. No resize, no
padding.

**Backed by.** `TestRender::test_canvas_dimensions` and
`TestRender::test_background_is_black`. Empirically also visible in
`proofs/side_by_side/`.

---

## Claim 6 — Missing keypoints fail closed

**Statement.** When MediaPipe reports a landmark below the configurable
visibility threshold (`--min-conf`, default 0.3), the plugin marks that
keypoint as missing (`x = y = -1`). Missing keypoints are skipped during
rendering (no spurious dot at origin). Limbs are drawn only when both
endpoints are present.

**Justification.** This matches the convention used inside
`controlnet_aux.open_pose.util.draw_bodypose`, which checks `subset[n][i]
!= -1` before drawing each joint and limb. If we instead emitted (0, 0)
for low-confidence joints, ControlNet would see phantom keypoints at the
top-left corner — a known failure mode.

**Backed by.**
- `TestRemap::test_low_visibility_marked_missing`,
- `TestRemap::test_none_landmarks_is_safe`,
- `TestRender::test_partial_pose_is_safe`.

---

## Claim 7 — Plugin is a drop-in for `video_skeletons.py`

**Statement.** Replacing the call

```bash
python data_preparation/video_skeletons.py -d data/case-X/images -c openposefull
```

with

```bash
python mediapipe_motioneditor_plugin/extract_pose_video.py -d data/case-X/images -c openposefull
```

leaves the rest of the MotionEditor pipeline untouched: the output PNG
filenames, their parent folder, and their content type all match.

**Justification.** Both scripts derive `outdir` from
`opt.data.replace(last_name, which_cond)` and write
`{frame_stem}.png`. Filenames are preserved by reading the input via
`sorted(glob(...))` and using `Path(p).stem`.

**Backed by.** Empirically, by running both scripts on `data/case-1/images`
and confirming that
`data/case-1/openposefull/0000.png` is produced by either script with
similar visual content. `verify.py --openpose-ref` quantifies the L1
diff per frame; expected mean L1 in low tens out of 255 (sub-pixel joint
offsets + identical limb colours).

---

## Claim 8 — Plugin does not silently degrade temporal continuity

**Statement.** The plugin uses one shared `mp.solutions.pose.Pose`
instance across all frames of a video, with `static_image_mode=False`
and tracking enabled. This allows MediaPipe's internal tracker to use
the previous frame's pose as a prior for the current frame, which
matches OpenPose's frame-by-frame behaviour (OpenPose has no tracker
state, but it is also fully redetect-per-frame, which is acceptable).

**Justification.** In `MediaPipeOpenPoseExtractor.__init__` we construct
the Pose object once; `extract(...)` reuses it. The constructor sets
`static_image_mode=False` and `min_tracking_confidence=0.5`, which
matches the user's existing `mpipe.py` that they validated on a dummy
video. Statelessness can be re-enabled by setting
`static_image_mode=True` if desired.

**Backed by.** Empirical detection rate in `report.json`; if MediaPipe
tracking is destabilising the output the rate drops below ~95%, which is
flagged as a warning by `extract_pose_video.py`.

---

## What the supervisor needs to read / run

| Artefact | Purpose | Needs MediaPipe? |
|---|---|---|
| `mp_openpose_extractor.py` | The library. Read once. | No (just read). |
| `PROOFS.md` (this file) | Correctness arguments + test pointers. | No. |
| `tests/test_geometry.py` | Offline deterministic tests. | No. Pytest is enough. |
| `extract_pose_video.py` | Drop-in CLI. | Yes. |
| `verify.py` + `proofs/` | Visual + statistical evidence. | Yes (for input images). |
| `README.md` | Usage. | No. |

If the supervisor only has 5 minutes: run
`python -m unittest discover -s tests -v`. That covers Claims 1, 2, 3,
4, 5, 6 deterministically and offline.

If the supervisor has 10 minutes and a GPU/CPU box: run `verify.py` on
one case and inspect `proofs/side_by_side/` and `proofs/report.md`. That
covers Claims 7 and 8.
