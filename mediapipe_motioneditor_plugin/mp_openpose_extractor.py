"""MediaPipe Pose -> OpenPose-COCO-18 skeleton renderer.

Drop-in replacement for the OpenPose preprocessing step used by MotionEditor.
Output of `render_openpose_skeleton(...)` is byte-compatible with what
ControlNet-OpenPose expects (same canvas size, 18 keypoints in COCO order,
17 limbs as oriented ellipses with controlnet_aux's palette).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from e


OPENPOSE_KEYPOINTS = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

# Verbatim from controlnet_aux.open_pose.util (1-indexed).
LIMB_SEQ_1IDX = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18],
]

LIMB_COLORS_RGB = [
    [255, 0, 0],   [255, 85, 0],   [255, 170, 0],  [255, 255, 0],
    [170, 255, 0], [85, 255, 0],   [0, 255, 0],    [0, 255, 85],
    [0, 255, 170], [0, 255, 255],  [0, 170, 255],  [0, 85, 255],
    [0, 0, 255],   [85, 0, 255],   [170, 0, 255],  [255, 0, 255],
    [255, 0, 170],
]

JOINT_COLORS_RGB = [
    [255, 0, 0],   [255, 85, 0],   [255, 170, 0],  [255, 255, 0],
    [170, 255, 0], [85, 255, 0],   [0, 255, 0],    [0, 255, 85],
    [0, 255, 170], [0, 255, 255],  [0, 170, 255],  [0, 85, 255],
    [0, 0, 255],   [85, 0, 255],   [170, 0, 255],  [255, 0, 255],
    [255, 0, 170], [255, 0, 85],
]

# MediaPipe BlazePose 33-landmark index -> OpenPose 18-keypoint index.
# OP 1 (neck) derived as midpoint of shoulders in code.
MP_TO_OP = {
    0:  0,
    2:  12,
    3:  14,
    4:  16,
    5:  11,
    6:  13,
    7:  15,
    8:  24,
    9:  26,
    10: 28,
    11: 23,
    12: 25,
    13: 27,
    14: 5,
    15: 2,
    16: 8,
    17: 7,
}


@dataclass
class OpenPoseFrame:
    keypoints: np.ndarray  # (18, 3) float32: (x_px, y_px, confidence)
    width: int
    height: int

    def missing_mask(self) -> np.ndarray:
        return self.keypoints[:, 0] < 0

    def num_detected(self) -> int:
        return int(np.sum(~self.missing_mask()))


def mediapipe_to_openpose(
    mp_landmarks,
    width: int,
    height: int,
    min_conf: float = 0.3,
) -> OpenPoseFrame:
    kp = np.full((18, 3), -1.0, dtype=np.float32)
    if mp_landmarks is None:
        return OpenPoseFrame(kp, width, height)

    lms = mp_landmarks.landmark

    def _pt(mp_idx: int):
        lm = lms[mp_idx]
        return float(lm.x) * width, float(lm.y) * height, float(lm.visibility)

    for op_idx, mp_idx in MP_TO_OP.items():
        x, y, v = _pt(mp_idx)
        if v >= min_conf:
            kp[op_idx] = [x, y, v]

    ls = lms[11]; rs = lms[12]
    if ls.visibility >= min_conf and rs.visibility >= min_conf:
        nx = 0.5 * (ls.x + rs.x) * width
        ny = 0.5 * (ls.y + rs.y) * height
        nv = 0.5 * (ls.visibility + rs.visibility)
        kp[1] = [nx, ny, nv]

    return OpenPoseFrame(kp, width, height)


def render_openpose_skeleton(frame: OpenPoseFrame, stickwidth: int = 4) -> np.ndarray:
    """Render 18-keypoint OpenPose skeleton on black canvas.

    Faithful reproduction of `controlnet_aux.open_pose.util.draw_bodypose`:
    dots are drawn first, then limb ellipses are blended via addWeighted.
    """
    H, W = frame.height, frame.width
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    kp = frame.keypoints

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


class MediaPipeOpenPoseExtractor:
    """Stateful wrapper around `mp.solutions.pose.Pose`.

    Use one instance per video — Pose has internal tracking state.
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
        f = self.extract(bgr_image)
        return f, render_openpose_skeleton(f)
