# worker/test_calibration_adapter.py
import json
import unittest

from geometry.calibration_adapter import corners_to_observations, summarise_calibration
from geometry.camera import Camera
from geometry.court_model import CORNER_LABEL_TO_KEYPOINT, KEYPOINTS
from geometry.synthetic import LANDSCAPE

# High and far enough back that all four singles corners are inside the frame.
WIDE = Camera.look_at((2.59, -5.5, 6.0), (2.59, 6.7, 0.0), 1000.0, LANDSCAPE)


def percent_corners(camera: Camera, size: tuple[int, int]) -> list[dict]:
    corners = []
    for label, name in CORNER_LABEL_TO_KEYPOINT.items():
        x, y = camera.project(KEYPOINTS[name][None, :])[0]
        corners.append({"label": label, "x": x / size[0] * 100, "y": y / size[1] * 100})
    return corners


class CornersToObservationsTests(unittest.TestCase):
    def test_percent_coordinates_become_pixels_keyed_by_keypoint(self):
        observed = corners_to_observations([{"label": "nearLeft", "x": 50, "y": 25}], (1920, 1080))
        self.assertEqual(observed, {"back0_sl": (960.0, 270.0)})

    def test_unknown_labels_and_bad_values_are_skipped(self):
        corners = [
            {"label": "centre", "x": 10, "y": 10},
            {"label": "nearLeft", "x": "bad", "y": 10},
            {"label": "nearRight", "x": None, "y": 10},
            {"label": "farLeft", "x": float("nan"), "y": 10},
            {"label": "farRight", "x": 20, "y": 30},
        ]
        self.assertEqual(corners_to_observations(corners, (1000, 1000)), {"back1_sr": (200.0, 300.0)})


class SummariseCalibrationTests(unittest.TestCase):
    def test_four_corners_give_a_json_ready_camera_summary(self):
        corners = percent_corners(WIDE, LANDSCAPE)
        for corner in corners:
            self.assertTrue(0 <= corner["x"] <= 100 and 0 <= corner["y"] <= 100, corner)
        summary = summarise_calibration(corners, LANDSCAPE)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["cameraTier"], "approximate")
        self.assertLess(abs(summary["focalPx"] / 1000.0 - 1), 0.1)
        self.assertIn(summary["geometryTier"], ("A", "B", "C"))
        self.assertIsNone(summary["looFloorCm"])
        self.assertIsInstance(summary["reasons"], list)
        json.dumps(summary)

    def test_three_corners_are_not_solvable(self):
        corners = percent_corners(WIDE, LANDSCAPE)[:3]
        self.assertIsNone(summarise_calibration(corners, LANDSCAPE))

    def test_empty_corners_are_not_solvable(self):
        self.assertIsNone(summarise_calibration([], LANDSCAPE))


if __name__ == "__main__":
    unittest.main()
