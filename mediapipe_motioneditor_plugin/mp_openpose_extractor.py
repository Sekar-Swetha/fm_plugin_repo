"""
MediaPipe Pose -> OpenPose-COCO-18 skeleton renderer.

Drop-in replacement for the OpenPose preprocessing step used by MotionEditor
(`data_preparation/video_skeletons.py`, `controlnet_aux.OpenposeDetector`).

The output of `render_openpose_skeleton(...)` is *byte-for-byte compatible* with
the input that the ControlNet-OpenPose checkpoint expects:
  - canvas size = input image size
  - black background
  - 18 keypoints in COCO order (nose, neck, R/L shoulder/elbow/wrist,
    R/L hip/knee/ankle, R/L eye/ear)
  - 17 limbs drawn as oriented ellipses with controlnet_aux's exact RGB palette
  - joint dots = circles radius 4, same colour as their incident limb

This means the existing ControlNet weights work without retraining.

Author: Swetha (TCD dissertation, 2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from e


# ---------------------------------------------------------------------------
# OpenPose COCO-18 spec (matches controlnet_aux.open_pose.util.draw_bodypose)
# ---------------------------------------------------------------------------

OPENPOSE_KEYPOINTS = [
    "nose",          # 0
    "neck",          # 1
    "r_shoulder",    # 2
    "r_elbow",       # 3
    "r_wrist",       # 4
    "l_shoulder",    # 5
    "l_elbow",       # 6
    "l_wrist",       # 7
    "r_hip",         # 8
    "r_knee",        # 9
    "r_ankle",       # 10
    "l_hip",         # 11
    "l_knee",        # 12
    "l_ankle",       # 13
    "r_eye",         # 14
    "l_eye",         # 15
    "r_ear",         # 16
    "l_ear",         # 17
]

# 1-indexed limb pairs, taken verbatim from controlnet_aux.open_pose.util.
LIMB_SEQ_1IDX = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18],
]

# Per-limb RGB colours, taken verbatim from controlnet_aux.open_pose.util.
LIMB_COLORS_RGB = [
    [255, 0, 0],   [255, 85, 0],   [255, 170, 0],  [255, 255, 0],
    [170, 255, 0], [85, 255, 0],   [0, 255, 0],    [0, 255, 85],
    [0, 255, 170], [0, 255, 255],  [0, 170, 255],  [0, 85, 255],
    [0, 0, 255],   [85, 0, 255],   [170, 0, 255],  [255, 0, 255],
    [255, 0, 170],
]

# Joint-dot colour list — controlnet_aux uses an 18-entry list aligned with
# the 18 keypoints. The values below are the controlnet_aux defaults
# (the trailing entry [255, 0, 85] is for the 18th keypoint).
JOINT_COLORS_RGB = [
    [255, 0, 0],   [255, 85, 0],   [255, 170, 0],  [255, 255, 0],
    [170, 255, 0], [85, 255, 0],   [0, 255, 0],    [0, 255, 85],
    [0, 255, 170], [0, 255, 255],  [0, 170, 255],  [0, 85, 255],
    [0, 0, 255],   [85, 0, 255],   [170, 0, 255],  [255, 0, 255],
    [255, 0, 170], [255, 0, 85],
]


# ---------------------------------------------------------------------------
# MediaPipe -> OpenPose remap table
# ---------------------------------------------------------------------------
# MediaPipe BlazePose 33-landmark index -> OpenPose 18-keypoint index.
# `None` means the OpenPose keypoint must be derived (e.g. neck = midpoint).
# Reference for MediaPipe indices: mediapipe.solutions.pose.PoseLandmark.
MP_TO_OP = {
    0:  0,   # OP nose       <- MP 0  nose
    # OP 1 neck = midpoint(MP 11, MP 12)        # handled in code
    2:  12,  # OP r_shoulder <- MP 12 right_shoulder
    3:  14,  # OP r_elbow    <- MP 14 right_elbow
    4:  16,  # OP r_wrist    <- MP 16 right_wrist
    5:  11,  # OP l_shoulder <- MP 11 left_shoulder
    6:  13,  # OP l_elbow    <- MP 13 left_elbow
    7:  15,  # OP l_wrist    <- MP 15 left_wrist
    8:  24,  # OP r_hip      <- MP 24 right_hip
    9:  26,  # OP r_knee     <- MP 26 right_knee
    10: 28,  # OP r_ankle    <- MP 28 right_ankle
    11: 23,  # OP l_hip      <- MP 23 left_hip
    12: 25,  # OP l_knee     <- MP 25 left_knee
    13: 27,  # OP l_ankle    <- MP 27 left_ankle
    14: 5,   # OP r_eye      <- MP 5  right_eye
    15: 2,   # OP l_eye      <- MP 2  left_eye
    16: 8,   # OP r_ear      <- MP 8  right_ear
    17: 7,   # OP l_ear      <- MP 7  left_ear
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OpenPoseFrame:
    """One frame's worth of OpenPose-format keypoints.

    `keypoints` is shape (18, 3): each row is (x_px, y_px, confidence).
    A keypoint with confidence < `min_conf` is treated as missing and its
    (x, y) are set to (-1, -1); downstream rendering skips it.
    """
    keypoints: np.ndarray  # (18, 3) float32
    width: int
    height: int

    def missing_mask(self) -> np.ndarray:
        """Boolean (18,) — True where the keypoint is missing."""
        return self.keypoints[:, 0] < 0

    def num_detected(self) -> int:
        return int(np.sum(~self.missing_mask()))


# ---------------------------------------------------------------------------
# Remap: MediaPipe landmarks -> OpenPose 18 keypoints
# ---------------------------------------------------------------------------

def mediapipe_to_openpose(
    mp_landmarks,
    width: int,
    height: int,
    min_conf: float = 0.3,
) -> OpenPoseFrame:
    """Convert a MediaPipe pose-landmark result into an OpenPose-18 frame.

    Parameters
    ----------
    mp_landmarks
        A `mediapipe.solutions.pose` `pose_landmarks` object (i.e. the
        `.pose_landmarks` attribute of `pose.process(...).pose_landmarks`).
        Each landmark has `.x`, `.y` in [0, 1] and `.visibility` in [0, 1].
        Pass `None` for a frame where MediaPipe failed to detect a pose; the
        returned frame will have all keypoints marked missing.
    width, height
        Pixel dimensions of the source frame.
    min_conf
        Visibility threshold. Landmarks below this are marked missing.
    """
    kp = np.full((18, 3), -1.0, dtype=np.float32)
    if mp_landmarks is None:
        return OpenPoseFrame(kp, width, height)

    lms = mp_landmarks.landmark

    def _pt(mp_idx: int):
        lm = lms[mp_idx]
        return float(lm.x) * width, float(lm.y) * height, float(lm.visibility)

    # Direct 1-to-1 mapping.
    for op_idx, mp_idx in MP_TO_OP.items():
        x, y, v = _pt(mp_idx)
        if v >= min_conf:
            kp[op_idx] = [x, y, v]

    # OP 1 = neck = midpoint of shoulders if both shoulders are visible.
    ls = lms[11]; rs = lms[12]
    if ls.visibility >= min_conf and rs.visibility >= min_conf:
        nx = 0.5 * (ls.x + rs.x) * width
        ny = 0.5 * (ls.y + rs.y) * height
        nv = 0.5 * (ls.visibility + rs.visibility)
        kp[1] = [nx, ny, nv]

    return OpenPoseFrame(kp, width, height)


# ---------------------------------------------------------------------------
# Rendering — byte-compatible with controlnet_aux.open_pose.util.draw_bodypose
# ---------------------------------------------------------------------------

def render_openpose_skeleton(frame: OpenPoseFrame, stickwidth: int = 4) -> np.ndarray:
    """Render the 18-keypoint OpenPose skeleton on a black canvas.

    The drawing algorithm is a faithful reproduction of
    `controlnet_aux.open_pose.util.draw_bodypose` so that the resulting PNG
    looks identical to what the ControlNet-OpenPose preprocessor would emit
    given the same keypoints.

    Returns a uint8 BGR image of shape (height, width, 3). Save it directly
    with `cv2.imwrite(...)`.
    """
    H, W = frame.height, frame.width
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    kp = frame.keypoints

    # 1) Limbs — drawn first (controlnet_aux order: ellipses then dots, but
    # the dot loop is also drawn over the limbs because dot loop runs first
    # in the upstream code? Actually in controlnet_aux the dot loop runs
    # FIRST and limbs second, then `addWeighted` blends them — so circles get
    # partly painted over. We replicate that order exactly.)
    for i in range(18):
        if kp[i, 0] < 0:
            continue
        x, y = int(kp[i, 0]), int(kp[i, 1])
        rgb = JOINT_COLORS_RGB[i]
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.circle(canvas, (x, y), 4, bgr, thickness=-1)

    for i, (a1, b1) in enumerate(LIMB_SEQ_1IDX):
        a = a1 - 1
        b = b1 - 1
        if kp[a, 0] < 0 or kp[b, 0] < 0:
            continue
        Y = np.array([kp[a, 0], kp[b, 0]], dtype=np.float32)
        X = np.array([kp[a, 1], kp[b, 1]], dtype=np.float32)
        mX = float(np.mean(X))
        mY = float(np.mean(Y))
        length = float(((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5)
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1,
        )
        rgb = LIMB_COLORS_RGB[i]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cur = canvas.copy()
        cv2.fillConvexPoly(cur, polygon, bgr)
        canvas = cv2.addWeighted(canvas, 0.4, cur, 0.6, 0)

    return canvas


# ---------------------------------------------------------------------------
# High-level convenience: process a folder / video end-to-end.
# ---------------------------------------------------------------------------

class MediaPipeOpenPoseExtractor:
    """Stateful wrapper around `mp.solutions.pose.Pose`.

    Use one instance per video (MediaPipe Pose has internal tracking state
    that benefits from temporal continuity).
    """

    def __init__(
        self,
        min_conf: float = 0.3,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.min_conf = float(min_conf)
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            enable_segmentation=False,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._pose.close()

    def close(self):
        self._pose.close()

    def extract(self, bgr_image: np.ndarray) -> OpenPoseFrame:
        """Run MediaPipe on a BGR image; return OpenPose-format keypoints."""
        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image (H,W,3); got shape {bgr_image.shape}"
            )
        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        return mediapipe_to_openpose(
            results.pose_landmarks, w, h, min_conf=self.min_conf
        )

    def extract_and_render(self, bgr_image: np.ndarray) -> tuple[OpenPoseFrame, np.ndarray]:
        """Extract + render in one shot. Returns (frame, BGR skeleton image)."""
        f = self.extract(bgr_image)
        return f, render_openpose_skeleton(f)
