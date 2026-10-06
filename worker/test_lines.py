# worker/test_lines.py
import unittest

import cv2
import numpy as np

from geometry.camera import Camera
from geometry.court_model import court_lines
from geometry.lines import find_line_offsets, line_response, sample_court_lines
from simulation.render import render_frame

SIZE = (640, 360)
CAMERA = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 / 3, SIZE)


def normals_at(camera: Camera, points: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = camera.project(points)
    ahead = camera.project(points + 0.05 * directions)
    tangent = ahead - pixels
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return pixels, np.column_stack([-tangent[:, 1], tangent[:, 0]])


def ridge_image(columns: list[tuple[float, float, float]], size=(120, 80)) -> np.ndarray:
    """Vertical bright stripes as (centre x, half width, strength) on a black response image."""
    width, height = size
    x = np.arange(width, dtype=np.float32)
    row = np.zeros(width, dtype=np.float32)
    for centre, half, strength in columns:
        row = np.maximum(row, strength * np.clip(half + 0.5 - np.abs(x - centre), 0.0, 1.0))
    return np.repeat(row[None, :], height, axis=0)


class SampleCourtLinesTests(unittest.TestCase):
    def test_samples_lie_on_painted_lines_with_unit_directions(self):
        points, directions, line_ids = sample_court_lines(0.25)
        lines = court_lines()
        self.assertEqual(len(points), len(directions))
        self.assertEqual(len(points), len(line_ids))
        self.assertEqual(set(line_ids.tolist()), set(range(len(lines))))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        for point, direction, line_id in zip(points, directions, line_ids):
            _, start, end = lines[line_id]
            along = (end - start) / np.linalg.norm(end - start)
            np.testing.assert_allclose(np.cross(point - start, along), 0.0, atol=1e-9)
            np.testing.assert_allclose(direction, along)

    def test_line_ends_are_not_sampled(self):
        points, _, line_ids = sample_court_lines(0.25)
        for line_id, (_, start, end) in enumerate(court_lines()):
            mine = points[line_ids == line_id]
            self.assertGreater(np.linalg.norm(mine - start, axis=1).min(), 0.1)
            self.assertGreater(np.linalg.norm(mine - end, axis=1).min(), 0.1)


class LineResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=3)
        cls.response = line_response(cls.frame)

    def test_shape_and_range(self):
        self.assertEqual(self.response.shape, (SIZE[1], SIZE[0]))
        self.assertEqual(self.response.dtype, np.float32)
        self.assertGreaterEqual(float(self.response.min()), 0.0)
        self.assertLessEqual(float(self.response.max()), 1.0)

    def test_painted_lines_respond_and_open_floor_does_not(self):
        points, _, _ = sample_court_lines(0.25)
        pixels = CAMERA.project(points).astype(np.float32)
        inside = (pixels[:, 0] > 2) & (pixels[:, 0] < SIZE[0] - 3) & (pixels[:, 1] > 2) & (pixels[:, 1] < SIZE[1] - 3)
        on_lines = cv2.remap(self.response, pixels[inside, 0][None, :], pixels[inside, 1][None, :], cv2.INTER_LINEAR)
        self.assertGreater(float(np.median(on_lines)), 0.2)
        # Floor points at least 0.3 m from every painted line.
        open_floor = np.array([[x, y, 0.0] for x in (0.65, 1.95, 3.25, 4.55) for y in (2.5, 3.5, 9.9, 10.9)])
        floor_pixels = np.round(CAMERA.project(open_floor)).astype(int)
        values = self.response[floor_pixels[:, 1], floor_pixels[:, 0]]
        self.assertLess(float(values.max()), 0.05)


class FindLineOffsetsTests(unittest.TestCase):
    def test_recovers_a_known_shift_on_a_rendered_court(self):
        response = line_response(render_frame(CAMERA, SIZE, noise_sigma=2.0, seed=4))
        points, directions, _ = sample_court_lines(0.25)
        pixels, normals = normals_at(CAMERA, points, directions)
        inside = (pixels[:, 0] > 12) & (pixels[:, 0] < SIZE[0] - 12) & (pixels[:, 1] > 12) & (pixels[:, 1] < SIZE[1] - 12)
        offsets, strengths = find_line_offsets(response, pixels[inside] + 2.5 * normals[inside], normals[inside], 8.0)
        found = np.isfinite(offsets)
        self.assertGreater(found.mean(), 0.8)
        self.assertLess(float(np.median(np.abs(offsets[found] + 2.5))), 0.3)
        self.assertTrue((strengths[found] > 0).all())

    def test_nothing_to_find_gives_nan(self):
        response = np.zeros((80, 120), dtype=np.float32)
        offsets, strengths = find_line_offsets(response, np.array([[60.0, 40.0]]), np.array([[1.0, 0.0]]), 8.0)
        self.assertTrue(np.isnan(offsets[0]))
        self.assertEqual(strengths[0], 0.0)

    def test_points_outside_the_image_give_nan(self):
        response = ridge_image([(60.0, 1.0, 0.8)])
        offsets, _ = find_line_offsets(response, np.array([[60.0, -5.0], [200.0, 40.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]), 8.0)
        self.assertTrue(np.isnan(offsets).all())

    def test_prefers_the_nearer_of_two_strong_parallel_lines(self):
        # Singles and doubles sidelines can sit a few pixels apart far from the phone.
        response = ridge_image([(40.0, 1.0, 1.0), (52.0, 1.0, 0.7)])
        offsets, _ = find_line_offsets(response, np.array([[50.0, 40.0]]), np.array([[1.0, 0.0]]), 14.0)
        self.assertAlmostEqual(float(offsets[0]), 2.0, delta=0.2)

    def test_a_faint_ridge_beside_a_strong_line_is_ignored(self):
        response = ridge_image([(40.0, 1.0, 1.0), (52.0, 1.0, 0.2)])
        offsets, _ = find_line_offsets(response, np.array([[50.0, 40.0]]), np.array([[1.0, 0.0]]), 14.0)
        self.assertAlmostEqual(float(offsets[0]), -10.0, delta=0.2)

    def test_finds_the_centre_of_a_wide_line(self):
        rng = np.random.default_rng(0)
        response = ridge_image([(63.3, 6.0, 0.6)]) + rng.normal(0, 0.02, (80, 120)).astype(np.float32)
        offsets, _ = find_line_offsets(response, np.array([[60.0, 40.0]]), np.array([[1.0, 0.0]]), 12.0)
        self.assertAlmostEqual(float(offsets[0]), 3.3, delta=0.4)

    def test_a_bright_area_wider_than_the_search_is_not_a_line(self):
        response = np.zeros((80, 120), dtype=np.float32)
        response[:, 30:90] = 0.8
        offsets, _ = find_line_offsets(response, np.array([[60.0, 40.0]]), np.array([[1.0, 0.0]]), 8.0)
        self.assertTrue(np.isnan(offsets[0]))


if __name__ == "__main__":
    unittest.main()
