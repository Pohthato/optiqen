# worker/test_track_eval.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluation.track import MISS_CM, evaluate_track, floor_error_cm, main, summarise
from geometry.camera import Camera
from geometry.tracking import TrackedFrame
from simulation.clip import make_clip, write_clip
from simulation.rally import Shot, build_rally

SIZE = (960, 540)
CAMERA = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 650.0, SIZE)


def turned(camera: Camera, yaw_deg: float, pitch_deg: float = 0.0) -> Camera:
    """Turned about the camera's own axes; a negative pitch looks up."""
    delta = cv2.Rodrigues(np.radians([pitch_deg, yaw_deg, 0.0]).reshape(3, 1))[0]
    rotation = delta @ camera.rotation
    return Camera(camera.focal_px, camera.cx, camera.cy, cv2.Rodrigues(rotation)[0].ravel(), -rotation @ camera.centre, camera.k1)


class FloorErrorTests(unittest.TestCase):
    def test_the_true_camera_has_no_error(self):
        errors = floor_error_cm(CAMERA, CAMERA, SIZE)
        self.assertGreater(len(errors), 40)
        self.assertLess(float(errors.max()), 1e-6)

    def test_a_turned_camera_misplaces_the_floor(self):
        self.assertGreater(float(np.median(floor_error_cm(CAMERA, turned(CAMERA, 0.5), SIZE))), 5.0)

    def test_a_camera_looking_at_the_ceiling_counts_misses_not_nan(self):
        errors = floor_error_cm(CAMERA, turned(CAMERA, 0.0, -50.0), SIZE)
        self.assertTrue(np.isfinite(errors).all())
        self.assertIn(MISS_CM, errors)

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

    def test_a_wildly_wrong_camera_is_counted_and_the_report_is_valid_json(self):
        track = [TrackedFrame("anchored", CAMERA, 300, 0.2), TrackedFrame("tracked", turned(CAMERA, 0.0, -50.0), 300, 0.2)]
        report = evaluate_track([CAMERA] * 2, track, SIZE)
        self.assertEqual(report["falseConfident"], 1)
        json.dumps(report, allow_nan=False)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            evaluate_track([CAMERA] * 3, [TrackedFrame("anchored", CAMERA, 300, 0.2)], SIZE)


def clip_report(**changes):
    report = {"clip": "c", "fps": 60.0, "perFrameFloorErrorCm": [0.1, 0.2, 0.3], "falseConfident": 0, "trackedWhileOccluded": 0, "lostWhileVisible": 0, "recoveryFrames": [0]}
    report.update(changes)
    return report


class GateTests(unittest.TestCase):
    def test_passes_when_accurate_honest_and_quick_to_recover(self):
        summary = summarise([clip_report(), clip_report(recoveryFrames=[29])])
        self.assertTrue(summary["gatePassed"])
        self.assertEqual(summary["worstRecoveryFrames"], 29)

    def test_any_confident_but_wrong_frame_fails_the_gate(self):
        self.assertFalse(summarise([clip_report(), clip_report(falseConfident=1)])["gatePassed"])

    def test_a_camera_while_the_lens_was_covered_fails_the_gate(self):
        self.assertFalse(summarise([clip_report(trackedWhileOccluded=1)])["gatePassed"])

    def test_slow_or_no_recovery_fails_the_gate(self):
        self.assertFalse(summarise([clip_report(recoveryFrames=[31])])["gatePassed"])
        summary = summarise([clip_report(recoveryFrames=[None])])
        self.assertFalse(summary["gatePassed"])
        self.assertIsNone(summary["worstRecoveryFrames"])

    def test_a_large_median_error_fails_the_gate(self):
        self.assertFalse(summarise([clip_report(perFrameFloorErrorCm=[4.0, 6.0, 7.0])])["gatePassed"])


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
