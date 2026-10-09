# worker/test_shuttle_eval.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.shuttle import evaluate_shuttle, main

GOLDEN = {
    "schemaVersion": 1,
    "clipId": "c",
    "sourceFps": 30,
    "imageSize": [1920, 1080],
    "courtKeypoints": [],
    "contacts": [],
    "shots": [],
    "shuttlePoints": [
        {"timeMs": 100, "x": 500.0, "y": 300.0},  # found, 5 px off
        {"timeMs": 133, "x": 520.0, "y": 300.0},  # wrong place, 100 px off
        {"timeMs": 167, "x": 540.0, "y": 300.0},  # missed
        {"timeMs": 200, "visible": False},  # false alarm
        {"timeMs": 233, "visible": False},  # correctly empty
        {"timeMs": 267, "x": 580.0, "y": 290.0},  # found by a filled gap, 2 px off
    ],
}
DETECTIONS = {
    "fps": 30.0,
    "imageSize": [1920, 1080],
    "frames": [
        {"timeMs": 66.667},
        {"timeMs": 100.0, "x": 503.0, "y": 304.0, "score": 0.8, "inpainted": False},
        {"timeMs": 133.333, "x": 620.0, "y": 300.0, "score": 0.6, "inpainted": False},
        {"timeMs": 166.667},
        {"timeMs": 200.0, "x": 900.0, "y": 100.0, "score": 0.55, "inpainted": False},
        {"timeMs": 233.333},
        {"timeMs": 266.667, "x": 580.0, "y": 292.0, "score": 0.0, "inpainted": True},
    ],
}


class EvaluateShuttleTests(unittest.TestCase):
    def test_counts_finds_misplacements_misses_and_false_alarms(self):
        report = evaluate_shuttle(GOLDEN, DETECTIONS)
        self.assertEqual(report["labelledFrames"], 6)
        self.assertEqual(report["visible"], 4)
        self.assertEqual(report["found"], 2)
        self.assertEqual(report["wrongPlace"], 1)
        self.assertEqual(report["missed"], 1)
        self.assertEqual(report["falseAlarms"], 1)
        self.assertEqual(report["correctlyEmpty"], 1)
        self.assertEqual(report["foundByFilledGaps"], 1)
        self.assertAlmostEqual(report["recall"], 2 / 4)
        self.assertAlmostEqual(report["precision"], 2 / 4)
        self.assertAlmostEqual(report["medianErrorPx"], 5.0, places=3)
        self.assertAlmostEqual(report["tolerancePx"], 4 * 1920 / 512)

    def test_a_click_without_a_detection_frame_near_it_is_reported(self):
        golden = {**GOLDEN, "shuttlePoints": [{"timeMs": 5000, "x": 1.0, "y": 1.0}]}
        report = evaluate_shuttle(golden, DETECTIONS)
        self.assertEqual(report["unmatchedClicks"], 1)
        self.assertEqual(report["labelledFrames"], 0)

    def test_labels_and_detections_from_different_video_sizes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "1280x720"):
            evaluate_shuttle({**GOLDEN, "imageSize": [1280, 720]}, DETECTIONS)

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden, detections = Path(tmp) / "g.json", Path(tmp) / "d.json"
            golden.write_text(json.dumps(GOLDEN), encoding="utf-8")
            detections.write_text(json.dumps(DETECTIONS), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["--golden", str(golden), "--detections", str(detections)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["found"], 2)


if __name__ == "__main__":
    unittest.main()
