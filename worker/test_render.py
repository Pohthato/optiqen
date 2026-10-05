# worker/test_render.py
import unittest

import numpy as np

from geometry.camera import Camera
from simulation.render import render_frame

SIZE = (640, 360)
CAMERA = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 / 3, SIZE)


def pixel_at(image: np.ndarray, world) -> np.ndarray:
    x, y = CAMERA.project(np.array([world], dtype=float))[0]
    return image[int(round(y)), int(round(x))].astype(int)


class RenderFrameTests(unittest.TestCase):
    def test_frame_shape_and_type(self):
        image = render_frame(CAMERA, SIZE)
        self.assertEqual(image.shape, (360, 640, 3))
        self.assertEqual(image.dtype, np.uint8)

    # At 640x360 the far baseline is ~0.2 px wide (anti-aliased, not white), so these tests
    # use nearer lines that span whole pixels.
    def test_court_lines_are_white_and_the_floor_is_green(self):
        image = render_frame(CAMERA, SIZE, noise_sigma=0)
        self.assertTrue(np.all(pixel_at(image, (1.3, 0.76, 0.0)) > 170))
        blue, green, red = pixel_at(image, (1.3, 3.0, 0.0))
        self.assertGreater(green, red + 30)
        self.assertGreater(green, blue + 30)

    def test_the_net_tape_is_drawn(self):
        image = render_frame(CAMERA, SIZE, noise_sigma=0)
        self.assertTrue(np.all(pixel_at(image, (1.0, 6.7, 1.50)) > 200))

    def test_a_player_hides_the_line_behind_them(self):
        clear = render_frame(CAMERA, SIZE, noise_sigma=0)
        hidden = render_frame(CAMERA, SIZE, players=[(1.5, 3.0)], noise_sigma=0)
        self.assertTrue(np.all(pixel_at(clear, (1.3, 4.72, 0.0)) > 170))
        self.assertLess(int(pixel_at(hidden, (1.3, 4.72, 0.0)).sum()), 300)

    def test_the_shuttle_is_a_bright_blob_where_it_projects(self):
        without = render_frame(CAMERA, SIZE, noise_sigma=0)
        with_shuttle = render_frame(CAMERA, SIZE, shuttle=(2.59, 8.0, 2.0), shuttle_previous=(2.59, 7.8, 2.1), noise_sigma=0)
        self.assertTrue(np.all(pixel_at(with_shuttle, (2.59, 8.0, 2.0)) > 220))
        self.assertFalse(np.all(pixel_at(without, (2.59, 8.0, 2.0)) > 220))

    def test_objects_behind_the_camera_are_skipped(self):
        image = render_frame(CAMERA, SIZE, shuttle=(2.59, -10.0, 1.0), shuttle_previous=(2.59, -9.0, 1.0), players=[(2.59, -8.0)])
        self.assertEqual(image.shape, (360, 640, 3))

    def test_noise_is_seeded(self):
        a = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=3)
        b = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=3)
        c = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


if __name__ == "__main__":
    unittest.main()
