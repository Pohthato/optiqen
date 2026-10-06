# worker/test_track_eval.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluation.track import evaluate_track, floor_error_cm, main
from geometry.camera import Camera
from geometry.tracking import TrackedFrame
from simulation.clip import make_clip, write_clip
from simulation.rally import Shot, build_rally

SIZE = (960, 540)
CAMERA = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 650.0, SIZE)


def turned(camera: Camera, yaw_deg: float) -> Camera:
    delta = cv2.Rodrigues(np.array([0.0, np.radians(yaw_deg), 0.0]).reshape(3, 1))[0]
    rotation = delta @ camera.rotation
    return Camera(camera.focal_px, camera.cx, camera.cy, cv2.Rodrigues(rotation)[0].ravel(), -rotation @ camera.centre, camera.k1)


class FloorErrorTests(unittest.TestCase):
    def test_the_true_camera_has_no_error(self):
        errors = floor_error_cm(CAMERA, CAMERA, SIZE)
        self.assertGreater(len(errors), 40)
        self.assertLess(float(errors.max()), 1e-6)

    def test_a_turned_camera_misplaces_the_floor(self):
        self.assertGreater(float(np.median(floor_error_cm(CAMERA, turned(CAMERA, 0.5), SIZE))), 5.0)

    def test_only_floor_points_the_true_camera_sees_count(self):
        side = Camera.look_at((-4.5, 6.7, 3.5), (2.59, 6.7, 0.0), 1300.0, SIZE)
        self.assertLess(len(floor_error_cm(side, side, SIZE)), len(floor_error_cm(CAMERA, CAMERA, SIZE)))


class EvaluateTrackTests(unittest.TestCase):
    def test_counts_states_errors_and_recovery(self):
        truth = [CAMERA] * 10
        occluded = [False, False, False, True, True, True, False, False, False, False]
        good, wrong = CAMERA, turned(CAMERA, 1.0)
        track = [
            TrackedFrame("anchored", good, 300, 0.2),
            TrackedFrame("tracked", good, 300, 0.2),
            TrackedFrame("tracked", good, 300, 0.2),
            TrackedFrame("lost", None, 0, float("inf")),
            TrackedFrame("lost", None, 0, float("inf")),
            TrackedFrame("lost", None, 0, float("inf")),
            TrackedFrame("reanchored", good, 300, 0.2),
            TrackedFrame("tracked", wrong, 300, 0.2),
            TrackedFrame("lost", None, 10, 3.0),
            TrackedFrame("tracked", good, 300, 0.2),
        ]
        report = evaluate_track(truth, track, SIZE, occluded)
        self.assertEqual(report["frames"], 10)
        self.assertEqual(report["states"], {"anchored": 1, "tracked": 4, "lost": 4, "reanchored": 1})
        self.assertAlmostEqual(report["trackedShareOfVisible"], 6 / 7, places=3)
        self.assertEqual(report["lostWhileVisible"], 1)
        self.assertEqual(report["trackedWhileOccluded"], 0)
        self.assertEqual(report["falseConfident"], 1)
        self.assertEqual(report["recoveryFrames"], [0])
        self.assertLess(report["floorErrorCm"]["median"], 1e-6)
        self.assertGreater(report["floorErrorCm"]["max"], 20.0)

    def test_a_track_that_never_recovers_reports_none(self):
        track = [TrackedFrame("anchored", CAMERA, 300, 0.2), TrackedFrame("lost", None, 0, float("inf")), TrackedFrame("lost", None, 0, float("inf"))]
        report = evaluate_track([CAMERA] * 3, track, SIZE, [False, True, False])
        self.assertEqual(report["recoveryFrames"], [None])
        self.assertEqual(report["lostWhileVisible"], 1)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            evaluate_track([CAMERA] * 3, [TrackedFrame("anchored", CAMERA, 300, 0.2)], SIZE)


class EvaluateClipsCliTests(unittest.TestCase):
    def test_tracks_a_written_synthetic_clip_and_reports_the_gate(self):
        rally = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.0), 1.2, 2.2), Shot("clear", (4.2, 0.8), 1.5, None)])
        clip = make_clip(CAMERA, SIZE, rally, fps=20.0, handheld=True, seed=2, tail_s=0.5, occlusions=[(0.8, 1.6)])
        with tempfile.TemporaryDirectory() as tmp:
            write_clip(clip, Path(tmp) / "clip")
            out = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(Path(tmp) / "clip"), "--out", str(out)])
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(report["clips"][0]["frames"], len(clip.frames))
        self.assertEqual(report["clips"][0]["trackedWhileOccluded"], 0)
        summary = report["summary"]
        self.assertLess(summary["medianFloorErrorCm"], 5.0)
        self.assertEqual(summary["worstRecoveryFrames"], 0)
        self.assertTrue(summary["gatePassed"])


if __name__ == "__main__":
    unittest.main()
