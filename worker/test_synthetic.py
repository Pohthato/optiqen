# worker/test_synthetic.py
import unittest

import numpy as np

from geometry.court_model import FLOOR_KEYPOINT_NAMES, KEYPOINTS
from geometry.synthetic import observe, standard_scenes


class SyntheticSceneTests(unittest.TestCase):
    def test_scene_names(self):
        self.assertEqual(
            set(standard_scenes()),
            {"behind_elevated", "behind_low", "side_elevated", "corner_elevated", "behind_portrait"},
        )

    def test_every_scene_sees_enough_floor_keypoints(self):
        for name, (camera, size) in standard_scenes().items():
            floor = [key for key in observe(camera, size) if key in FLOOR_KEYPOINT_NAMES]
            self.assertGreaterEqual(len(floor), 8, name)

    def test_noise_free_observations_match_projection(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size)
        for name, pixel in observed.items():
            expected = camera.project(KEYPOINTS[name][None, :])[0]
            np.testing.assert_allclose(pixel, expected, atol=1e-9)

    def test_observations_stay_inside_the_image(self):
        for camera, size in standard_scenes().values():
            for x, y in observe(camera, size, noise_px=3.0, seed=4).values():
                self.assertTrue(0 <= x < size[0] and 0 <= y < size[1])

    def test_noise_is_deterministic_and_has_the_requested_scale(self):
        camera, size = standard_scenes()["behind_elevated"]
        clean = observe(camera, size)
        first = observe(camera, size, noise_px=1.0, seed=7)
        again = observe(camera, size, noise_px=1.0, seed=7)
        self.assertEqual(first, again)
        diffs = np.array([np.subtract(first[name], clean[name]) for name in first if name in clean])
        rms = float(np.sqrt(np.mean(diffs**2)))
        self.assertGreater(rms, 0.5)
        self.assertLess(rms, 2.0)


if __name__ == "__main__":
    unittest.main()
