# worker/test_camera.py
import unittest

import numpy as np

from geometry.camera import Camera

SIZE = (1920, 1080)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, SIZE)

    def test_look_at_target_lands_on_the_principal_point(self):
        pixel = self.camera.project(np.array([[2.59, 7.0, 0.0]]))[0]
        np.testing.assert_allclose(pixel, [960.0, 540.0], atol=1e-6)

    def test_centre_is_the_camera_position(self):
        np.testing.assert_allclose(self.camera.centre, [2.59, -3.5, 3.0], atol=1e-9)

    def test_world_right_is_image_right(self):
        left = self.camera.project(np.array([[0.0, 7.0, 0.0]]))[0]
        right = self.camera.project(np.array([[5.18, 7.0, 0.0]]))[0]
        self.assertLess(left[0], right[0])

    def test_floor_round_trip(self):
        floor = np.array([[0.0, 0.0, 0.0], [5.18, 13.4, 0.0], [2.59, 6.7, 0.0], [1.0, 4.0, 0.0]])
        pixels = self.camera.project(floor)
        np.testing.assert_allclose(self.camera.pixel_to_plane(pixels, 0.0), floor[:, :2], atol=1e-6)

    def test_round_trip_with_lens_distortion(self):
        camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, SIZE, k1=-0.08)
        floor = np.array([[0.0, 13.4, 0.0], [5.18, 13.4, 0.0], [0.0, 3.0, 0.0], [5.18, 3.0, 0.0]])
        pixels = camera.project(floor)
        np.testing.assert_allclose(camera.pixel_to_plane(pixels, 0.0), floor[:, :2], atol=1e-6)

    def test_elevated_plane_round_trip(self):
        point = np.array([[1.0, 5.0, 1.5]])
        pixels = self.camera.project(point)
        np.testing.assert_allclose(self.camera.pixel_to_plane(pixels, 1.5), point[:, :2], atol=1e-6)

    def test_point_behind_camera_is_nan(self):
        self.assertTrue(np.isnan(self.camera.project(np.array([[2.59, -10.0, 0.0]]))).all())

    def test_ray_above_the_horizon_misses_the_floor(self):
        self.assertTrue(np.isnan(self.camera.pixel_to_plane(np.array([[960.0, 0.0]]), 0.0)).all())


if __name__ == "__main__":
    unittest.main()
