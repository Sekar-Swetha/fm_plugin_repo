"""Unit tests for MediaPipe -> OpenPose remap and rendering.

Synthetic landmark data; MediaPipe runtime not required.
"""

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


class _Lm:
    def __init__(self, x, y, vis=1.0):
        self.x, self.y, self.visibility = float(x), float(y), float(vis)


class _Lms:
    def __init__(self, lm_list):
        self.landmark = lm_list


def _fake_mp_landmarks(person):
    lms = [_Lm(0.5, 0.5, 0.0) for _ in range(33)]
    for i, (x, y) in person.items():
        lms[i] = _Lm(x, y, 1.0)
    return _Lms(lms)


class TestSpec(unittest.TestCase):

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
        # OP 1 (neck) derived from shoulder midpoint, not table.
        op_indices = set(MP_TO_OP.keys())
        self.assertEqual(op_indices, set(range(18)) - {1})


class TestRemap(unittest.TestCase):

    def test_neck_is_shoulder_midpoint(self):
        mp_lm = _fake_mp_landmarks({
            11: (0.4, 0.3),
            12: (0.6, 0.3),
        })
        f = mediapipe_to_openpose(mp_lm, width=1000, height=1000, min_conf=0.5)
        self.assertAlmostEqual(f.keypoints[1, 0], 500.0, places=3)
        self.assertAlmostEqual(f.keypoints[1, 1], 300.0, places=3)

    def test_left_right_orientation(self):
        mp_lm = _fake_mp_landmarks({
            11: (0.2, 0.5),
            12: (0.8, 0.5),
        })
        f = mediapipe_to_openpose(mp_lm, width=100, height=100)
        self.assertAlmostEqual(f.keypoints[5, 0], 20.0)
        self.assertAlmostEqual(f.keypoints[2, 0], 80.0)

    def test_low_visibility_marked_missing(self):
        mp_lm = _fake_mp_landmarks({})
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
        kp = np.full((18, 3), -1.0, dtype=np.float32)
        coords = {
            0:  (256, 80),
            1:  (256, 130),
            2:  (200, 140),
            3:  (180, 220),
            4:  (170, 300),
            5:  (312, 140),
            6:  (332, 220),
            7:  (342, 300),
            8:  (220, 260),
            9:  (215, 360),
            10: (210, 460),
            11: (292, 260),
            12: (297, 360),
            13: (302, 460),
            14: (240, 70),
            15: (272, 70),
            16: (220, 80),
            17: (292, 80),
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
        empty = OpenPoseFrame(
            keypoints=np.full((18, 3), -1.0, dtype=np.float32),
            width=64, height=64,
        )
        canvas = render_openpose_skeleton(empty)
        self.assertEqual(int(canvas.sum()), 0)

    def test_palette_subset(self):
        f = self._full_skeleton()
        canvas = render_openpose_skeleton(f)
        flat = canvas.reshape(-1, 3)
        for rgb in JOINT_COLORS_RGB:
            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
            self.assertTrue(
                np.any(np.all(flat == bgr, axis=1)),
                msg=f"Joint colour {rgb} not present in rendered canvas",
            )

    def test_partial_pose_is_safe(self):
        kp = np.full((18, 3), -1.0, dtype=np.float32)
        kp[5] = [100, 200, 1.0]
        f = OpenPoseFrame(keypoints=kp, width=256, height=256)
        canvas = render_openpose_skeleton(f)
        nonzero = int(np.any(canvas > 0, axis=2).sum())
        self.assertGreater(nonzero, 0)
        self.assertLess(nonzero, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
