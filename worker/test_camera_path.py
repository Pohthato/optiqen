# worker/test_camera_path.py
import math
import unittest

import cv2
import numpy as np

from geometry.camera import Camera
from simulation.camera_path import stable_handheld, tripod

BASE = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, (1920, 1080))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(cv2.Rodrigues(a @ b.T)[0])))


class CameraPathTests(unittest.TestCase):
    def test_tripod_never_moves(self):
        path = tripod(BASE, 5)
        self.assertEqual(len(path), 5)
        self.assertTrue(all(camera is BASE for camera in path))

    def test_handheld_is_deterministic_per_seed(self):
        a = stable_handheld(BASE, 30, 60.0, seed=1)
        b = stable_handheld(BASE, 30, 60.0, seed=1)
        c = stable_handheld(BASE, 30, 60.0, seed=2)
        np.testing.assert_array_equal(a[10].rvec, b[10].rvec)
        self.assertFalse(np.allclose(a[10].rvec, c[10].rvec))

    def test_shake_stays_within_the_limits_but_does_move(self):
        path = stable_handheld(BASE, 600, 60.0, seed=3, rotation_deg=0.3, translation_m=0.015)
        angles = [angle_deg(camera.rotation, BASE.rotation) for camera in path]
        offsets = [float(np.linalg.norm(camera.centre - BASE.centre)) for camera in path]
        self.assertLessEqual(max(angles), 0.3 * math.sqrt(3) + 1e-9)
        self.assertLessEqual(max(offsets), 0.015 * math.sqrt(3) + 1e-9)
        self.assertGreater(max(angles), 0.06)
        self.assertGreater(max(offsets), 0.003)

    def test_motion_is_smooth_between_frames(self):
        path = stable_handheld(BASE, 600, 60.0, seed=4)
        steps = [angle_deg(path[i + 1].rotation, path[i].rotation) for i in range(len(path) - 1)]
        self.assertLess(max(steps), 0.07)

    def test_intrinsics_are_unchanged(self):
        camera = stable_handheld(BASE, 3, 60.0)[2]
        self.assertEqual((camera.focal_px, camera.cx, camera.cy, camera.k1), (BASE.focal_px, BASE.cx, BASE.cy, BASE.k1))


if __name__ == "__main__":
    unittest.main()
