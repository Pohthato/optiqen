# worker/test_shuttle.py
import unittest
from pathlib import Path

import numpy as np

from geometry.camera import Camera
from shuttle.tracknet import (
    INPUT_SIZE,
    MAX_GAP_FRAMES,
    ShuttleDetection,
    ShuttleDetector,
    decode_heatmap,
    ensemble_weights,
    ensembled_heatmaps,
    gap_mask,
    to_input,
)
from simulation.clip import make_clip
from simulation.rally import Shot, build_rally

WEIGHTS = Path(__file__).resolve().parent / "weights"
HAVE_WEIGHTS = (WEIGHTS / "TrackNet_best.pt").is_file() and (WEIGHTS / "InpaintNet_best.pt").is_file()
WIDTH, HEIGHT = INPUT_SIZE


def blob(x: float, y: float, value: float = 0.9, radius: int = 3) -> np.ndarray:
    heat = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    heat[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = value
    return heat


class DecodeHeatmapTests(unittest.TestCase):
    def test_the_blob_centre_is_scaled_to_the_original_frame(self):
        detection = decode_heatmap(blob(100, 50), (1920, 1080))
        self.assertAlmostEqual(detection.x, (100 + 0.5) * 1920 / WIDTH, delta=1920 / WIDTH)
        self.assertAlmostEqual(detection.y, (50 + 0.5) * 1080 / HEIGHT, delta=1080 / HEIGHT)
        self.assertAlmostEqual(detection.score, 0.9, places=5)
        self.assertFalse(detection.inpainted)

    def test_the_largest_blob_wins(self):
        heat = np.maximum(blob(100, 50, 0.95, radius=1), blob(300, 200, 0.7, radius=4))
        detection = decode_heatmap(heat, (WIDTH, HEIGHT))
        self.assertAlmostEqual(detection.x, 300.5, delta=1.0)
        self.assertAlmostEqual(detection.score, 0.7, places=5)

    def test_nothing_above_half_is_no_shuttle(self):
        self.assertIsNone(decode_heatmap(blob(100, 50, 0.4), (WIDTH, HEIGHT)))


class EnsembleTests(unittest.TestCase):
    def test_positional_weights_favour_the_middle_of_the_window(self):
        np.testing.assert_allclose(ensemble_weights(8), np.array([1, 2, 3, 4, 4, 3, 2, 1]) / 20)

    def fake_predict(self, windows):
        """Heatmap k of a window holds (first frame of the window + k) * 10 + k everywhere."""
        out = []
        for window in windows:
            first = int(window[0][0, 0, 0])
            out.append(np.stack([np.full((HEIGHT, WIDTH), (first + k) * 10.0 + k, np.float32) for k in range(len(window))]))
        return np.stack(out)

    def frames(self, count):
        # Each input carries its own index in its first pixel so the fake network can read it.
        return [np.full((3, HEIGHT, WIDTH), index, np.uint8) for index in range(count)]

    def test_each_frame_is_the_weighted_average_of_every_window_that_saw_it(self):
        seq_len, count = 4, 9
        heat = list(ensembled_heatmaps(self.frames(count), self.fake_predict, seq_len, overlap=True, batch_size=3))
        self.assertEqual(len(heat), count)
        weights = ensemble_weights(seq_len)
        for frame in range(count):
            terms = [(weights[frame - first], frame * 10.0 + (frame - first)) for first in range(max(0, frame - seq_len + 1), min(frame, count - seq_len) + 1)]
            expected = sum(w * v for w, v in terms) / sum(w for w, _ in terms)
            self.assertAlmostEqual(float(heat[frame][0, 0]), expected, places=4)

    def test_batching_does_not_change_the_result(self):
        one = list(ensembled_heatmaps(self.frames(11), self.fake_predict, 4, overlap=True, batch_size=1))
        many = list(ensembled_heatmaps(self.frames(11), self.fake_predict, 4, overlap=True, batch_size=5))
        for a, b in zip(one, many):
            np.testing.assert_allclose(a, b)

    def test_without_overlap_each_frame_comes_from_one_window(self):
        heat = list(ensembled_heatmaps(self.frames(10), self.fake_predict, 4, overlap=False, batch_size=2))
        self.assertEqual(len(heat), 10)
        self.assertEqual([float(h[0, 0]) for h in heat], [frame * 10.0 + frame % 4 for frame in range(10)])

    def test_a_clip_shorter_than_the_window_is_padded_with_its_last_frame(self):
        heat = list(ensembled_heatmaps(self.frames(3), self.fake_predict, 4, overlap=True, batch_size=2))
        self.assertEqual([float(h[0, 0]) for h in heat], [0.0, 11.0, 22.0])


class GapMaskTests(unittest.TestCase):
    def test_a_gap_between_two_sightings_low_in_the_frame_is_filled(self):
        visible = np.array([1, 1, 0, 0, 1, 1], bool)
        ys = np.array([300, 310, 0, 0, 330, 340], float)
        np.testing.assert_array_equal(gap_mask(ys, visible, 1080), [0, 0, 1, 1, 0, 0])

    def test_a_shuttle_that_left_over_the_top_of_the_frame_is_not_invented(self):
        visible = np.array([1, 1, 0, 0, 1], bool)
        ys = np.array([60, 20, 0, 0, 40], float)
        np.testing.assert_array_equal(gap_mask(ys, visible, 1080), [0, 0, 0, 0, 0])

    def test_nothing_is_invented_before_the_first_or_after_the_last_sighting(self):
        visible = np.array([0, 0, 1, 1, 0, 0], bool)
        ys = np.array([0, 0, 400, 410, 0, 0], float)
        np.testing.assert_array_equal(gap_mask(ys, visible, 1080), [0, 0, 0, 0, 0, 0])

    def test_a_long_gap_is_left_empty(self):
        visible = np.array([1] * 3 + [0] * (MAX_GAP_FRAMES + 1) + [1] * 3, bool)
        ys = np.where(visible, 500.0, 0.0)
        self.assertFalse(gap_mask(ys, visible, 1080).any())
        visible = np.array([1] * 3 + [0] * MAX_GAP_FRAMES + [1] * 3, bool)
        ys = np.where(visible, 500.0, 0.0)
        self.assertEqual(int(gap_mask(ys, visible, 1080).sum()), MAX_GAP_FRAMES)


class InputTests(unittest.TestCase):
    def test_frames_become_rgb_at_the_network_size(self):
        frame = np.zeros((90, 160, 3), np.uint8)
        frame[..., 0] = 255  # blue in OpenCV's BGR order
        image = to_input(frame)
        self.assertEqual(image.shape, (3, HEIGHT, WIDTH))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image[2].mean()), 255)
        self.assertEqual(int(image[0].mean()), 0)


class LoadTests(unittest.TestCase):
    def test_missing_weights_are_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "TrackNet_best.pt"):
            ShuttleDetector.load(Path(__file__).resolve().parent / "no-such-weights")


@unittest.skipUnless(HAVE_WEIGHTS, "TrackNetV3 weights are not in worker/weights")
class TrackNetOnSyntheticClipsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ShuttleDetector.load(WEIGHTS, device="cpu")
        size = (640, 360)
        camera = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 1300.0 * 640 / 1920, size)
        rally = build_rally((3.3, 3.6, 1.0), [Shot("clear", (1.8, 11.5), 1.3, None)])
        cls.clip = make_clip(camera, size, rally, fps=30.0, handheld=True, seed=4, tail_s=0.3)
        cls.size = size

    def truth_pixel(self, index):
        position = self.clip.shuttle[index]
        if position is None:
            return None
        pixel = self.clip.cameras[index].project(np.array([position]))[0]
        inside = 0 <= pixel[0] < self.size[0] and 0 <= pixel[1] < self.size[1]
        return pixel if inside else None

    def check(self, detections, max_median_px):
        self.assertEqual(len(detections), len(self.clip.frames))
        errors, missed = [], 0
        for index, detection in enumerate(detections):
            truth = self.truth_pixel(index)
            if truth is None:
                continue
            if detection is None:
                missed += 1
            else:
                errors.append(float(np.hypot(detection.x - truth[0], detection.y - truth[1])))
        self.assertLessEqual(missed, 0.1 * (missed + len(errors)))
        self.assertLess(float(np.median(errors)), max_median_px)

    def test_finds_the_shuttle_in_flight(self):
        background = self.detector.background(self.clip.frames)
        self.check(self.detector.detect(self.clip.frames, background, self.size, overlap=False), 2.5)

    def test_a_short_gap_is_filled_along_the_flight(self):
        background = self.detector.background(self.clip.frames)
        track = self.detector.detect(self.clip.frames, background, self.size, overlap=False, fill_gaps=False)
        seen = [index for index, detection in enumerate(track) if detection is not None]
        middle = seen[len(seen) // 2]
        holed = [None if middle <= index < middle + 3 else detection for index, detection in enumerate(track)]
        filled = self.detector.fill_gaps(holed, self.size)
        for index in range(middle, middle + 3):
            self.assertIsNotNone(filled[index])
            self.assertTrue(filled[index].inpainted)
            truth = self.truth_pixel(index)
            self.assertLess(float(np.hypot(filled[index].x - truth[0], filled[index].y - truth[1])), 6.0)


if __name__ == "__main__":
    unittest.main()
