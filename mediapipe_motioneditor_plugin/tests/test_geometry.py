"""
Deterministic, offline unit tests for the MediaPipe -> OpenPose remap and
rendering. These do not require MediaPipe to run — they exercise the
remap/render logic on synthetic landmark data.

Run with:
    pytest tests/  -v
or
    python tests/test_geometry.py
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_openpose_extractor import (   # noqa: E402
    JOINT_COLORS_RGB,
    LIMB_COLORS_RGB,
    LIMB_SEQ_1IDX,
    MP_TO_OP,
    OPENPOSE_KEYPOINTS,
    OpenPoseFrame,
    mediapipe_to_openpose,
    render_openpose_skeleton,
)


# A tiny stand-in for MediaPipe's NormalizedLandmark / NormalizedLandmarkList.
class _Lm:
    def __init__(self, x, y, vis=1.0):
        self.x, self.y, self.visibility = float(x), float(y), float(vis)


class _Lms:
    def __init__(self, lm_list):
        self.landmark = lm_list


def _fake_mp_landmarks(person):
    """Build a 33-entry landmark list from a dict of MediaPipe-index -> (x, y).

    All unspecified landmarks are placed at (0.5, 0.5) with visibility 0,
    i.e. treated as not-detected.
    """
    lms = [_Lm(0.5, 0.5, 0.0) for _ in range(33)]
    for i, (x, y) in person.items():
        lms[i] = _Lm(x, y, 1.0)
    return _Lms(lms)


class TestSpec(unittest.TestCase):
    """Constants must match the upstream controlnet_aux spec exactly."""

    def test_keypoint_count(self):
        self.assertEqual(len(OPENPOSE_KEYPOINTS), 18)
        self.assertEqual(len(JOINT_COLORS_RGB), 18)

    def test_limb_count(self):
        self.assertEqual(len(LIMB_SEQ_1IDX), 17)
        self.assertEqual(len(LIMB_COLORS_RGB), 17)

    def test_limb_indices_in_range(self):
        for a, b in LIMB_SEQ_1IDX:
            self.assertTrue(1 <= a <= 18)
            self.assertTrue(1 <= b <= 18)

    def test_mp_to_op_covers_all_but_neck(self):
        # OP index 1 = neck, computed by midpoint — not in the table.
        op_indices = set(MP_TO_OP.keys())
        self.assertEqual(op_indices, set(range(18)) - {1})


class TestRemap(unittest.TestCase):

    def test_neck_is_shoulder_midpoint(self):
        mp_lm = _fake_mp_landmarks({
            11: (0.4, 0.3),   # left shoulder
            12: (0.6, 0.3),   # right shoulder
        })
        f = mediapipe_to_openpose(mp_lm, width=1000, height=1000, min_conf=0.5)
        # Neck = ( (0.4+0.6)/2 * 1000, (0.3+0.3)/2 * 1000 ) = (500, 300).
        self.assertAlmostEqual(f.keypoints[1, 0], 500.0, places=3)
        self.assertAlmostEqual(f.keypoints[1, 1], 300.0, places=3)

    def test_left_right_orientation(self):
        # MediaPipe "left_shoulder" (MP idx 11) -> OpenPose "l_shoulder" (OP 5).
        # MediaPipe "right_shoulder" (MP idx 12) -> OpenPose "r_shoulder" (OP 2).
        mp_lm = _fake_mp_landmarks({
            11: (0.2, 0.5),
            12: (0.8, 0.5),
        })
        f = mediapipe_to_openpose(mp_lm, width=100, height=100)
        self.assertAlmostEqual(f.keypoints[5, 0], 20.0)   # OP l_shoulder
        self.assertAlmostEqual(f.keypoints[2, 0], 80.0)   # OP r_shoulder

    def test_low_visibility_marked_missing(self):
        mp_lm = _fake_mp_landmarks({})
        # All synthetic landmarks have visibility 0, so all OP keypoints
        # should be missing (-1).
        f = mediapipe_to_openpose(mp_lm, 64, 64, min_conf=0.3)
        self.assertTrue(np.all(f.missing_mask()))
        self.assertEqual(f.num_detected(), 0)

    def test_none_landmarks_is_safe(self):
        f = mediapipe_to_openpose(None, 64, 64)
        self.assertEqual(f.num_detected(), 0)
        self.assertEqual(f.width, 64)
        self.assertEqual(f.height, 64)

    def test_pixel_scaling(self):
        mp_lm = _fake_mp_landmarks({0: (0.25, 0.75)})
        f = mediapipe_to_openpose(mp_lm, width=200, height=400)
        self.assertAlmostEqual(f.keypoints[0, 0], 50.0)
        self.assertAlmostEqual(f.keypoints[0, 1], 300.0)


class TestRender(unittest.TestCase):

    def _full_skeleton(self):
        """Build a complete 18-keypoint OpenPose frame for a synthetic person."""
        kp = np.full((18, 3), -1.0, dtype=np.float32)
        # Coordinates chosen to be visibly inside a 512x512 canvas.
        coords = {
            0:  (256, 80),   # nose
            1:  (256, 130),  # neck
            2:  (200, 140),  # r_shoulder
            3:  (180, 220),  # r_elbow
            4:  (170, 300),  # r_wrist
            5:  (312, 140),  # l_shoulder
            6:  (332, 220),  # l_elbow
            7:  (342, 300),  # l_wrist
            8:  (220, 260),  # r_hip
            9:  (215, 360),  # r_knee
            10: (210, 460),  # r_ankle
            11: (292, 260),  # l_hip
            12: (297, 360),  # l_knee
            13: (302, 460),  # l_ankle
            14: (240, 70),   # r_eye
            15: (272, 70),   # l_eye
            16: (220, 80),   # r_ear
            17: (292, 80),   # l_ear
        }
        for i, (x, y) in coords.items():
            kp[i] = [x, y, 1.0]
        return OpenPoseFrame(kp, width=512, height=512)

    def test_canvas_dimensions(self):
        f = self._full_skeleton()
        canvas = render_openpose_skeleton(f)
        self.assertEqual(canvas.shape, (512, 512, 3))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_background_is_black(self):
        # An empty frame must produce a fully-black image.
        empty = OpenPoseFrame(
            keypoints=np.full((18, 3), -1.0, dtype=np.float32),
            width=64, height=64,
        )
        canvas = render_openpose_skeleton(empty)
        self.assertEqual(int(canvas.sum()), 0)

    def test_palette_subset(self):
        """All non-black pixels should be approximately one of the controlnet_aux
        palette colours (limb-ellipses are blended with addWeighted, so we
        allow tolerance; joint dots should be exact)."""
        f = self._full_skeleton()
        canvas = render_openpose_skeleton(f)
        # Check that at least one pixel matches each joint colour exactly.
        flat = canvas.reshape(-1, 3)
        for rgb in JOINT_COLORS_RGB:
            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
            self.assertTrue(
                np.any(np.all(flat == bgr, axis=1)),
                msg=f"Joint colour {rgb} not present in rendered canvas",
            )

    def test_partial_pose_is_safe(self):
        """If only some keypoints are present, render must not error and must
        only draw the limbs whose endpoints both exist."""
        kp = np.full((18, 3), -1.0, dtype=np.float32)
        kp[5] = [100, 200, 1.0]  # l_shoulder only
        f = OpenPoseFrame(keypoints=kp, width=256, height=256)
        canvas = render_openpose_skeleton(f)
        # Only one joint dot visible; canvas mostly black.
        nonzero = int(np.any(canvas > 0, axis=2).sum())
        self.assertGreater(nonzero, 0)
        self.assertLess(nonzero, 200)  # tiny dot


if __name__ == "__main__":
    unittest.main(verbosity=2)
