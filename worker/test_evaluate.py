# worker/test_evaluate.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import evaluate_clip, evaluate_directory, main
from geometry.synthetic import observe, standard_scenes


def golden_clip(clip_id: str = "clip-001") -> dict:
    camera, size = standard_scenes()["behind_elevated"]
    keypoints = [{"name": name, "x": x, "y": y} for name, (x, y) in observe(camera, size, noise_px=0.5, seed=1).items()]
    return {
        "schemaVersion": 1,
        "clipId": clip_id,
        "sourceFps": 60,
        "imageSize": list(size),
        "courtKeypoints": keypoints,
        "contacts": [{"timeMs": 1000}, {"timeMs": 2000}, {"timeMs": 3000}],
        "shots": [
            {"timeMs": 1000, "label": "clear", "landing": None},
            {"timeMs": 2000, "label": "smash", "landing": None},
            {"timeMs": 3000, "label": "drop", "landing": None},
        ],
        "rallies": [],
    }


RESULT = {
    "events": [
        {"type": "contact", "timeMs": 1010},
        {"type": "contact", "timeMs": 2050},
        {"type": "contact", "timeMs": 3500},
        {"type": "split_step", "timeMs": 900},
    ],
    "shots": [
        {"timeMs": 1010, "label": "clear", "verified": True},
        {"timeMs": 2050, "label": "drive", "verified": True},
        {"timeMs": 3500, "label": "drop", "verified": True},
    ],
}


class EvaluateClipTests(unittest.TestCase):
    def test_contact_metrics_at_both_tolerances(self):
        report = evaluate_clip(golden_clip(), RESULT)
        tight = report["contacts"]["tol33ms"]
        loose = report["contacts"]["tol100ms"]
        self.assertEqual((tight["truePositives"], tight["falsePositives"], tight["falseNegatives"]), (1, 2, 2))
        self.assertEqual((loose["truePositives"], loose["falsePositives"], loose["falseNegatives"]), (2, 1, 1))
        self.assertEqual(loose["medianAbsErrorMs"], 30.0)

    def test_shot_macro_f1_counts_missed_and_wrong_labels(self):
        shots = evaluate_clip(golden_clip(), RESULT)["shots"]
        self.assertEqual(shots["matched"], 2)
        self.assertEqual(shots["truthCount"], 3)
        self.assertAlmostEqual(shots["macroF1"], 0.25)

    def test_unverified_prediction_is_scored_as_wrong(self):
        result = {
            "events": [],
            "shots": [{"timeMs": 1000, "label": "clear", "verified": False}],
        }
        shots = evaluate_clip(golden_clip(), result)["shots"]
        self.assertEqual(shots["perLabel"]["clear"]["f1"], 0.0)

    def test_calibration_is_scored_from_golden_keypoints(self):
        calibration = evaluate_clip(golden_clip(), RESULT)["calibration"]
        self.assertIn(calibration["tier"], ("validated", "approximate"))
        self.assertIn(calibration["geometryTier"], ("A", "B", "C"))
        self.assertGreater(calibration["redundancy"], 0)

    def test_clip_without_enough_keypoints_has_no_calibration(self):
        clip = golden_clip()
        clip["courtKeypoints"] = clip["courtKeypoints"][:2]
        self.assertIsNone(evaluate_clip(clip, RESULT)["calibration"])


class EvaluateDirectoryTests(unittest.TestCase):
    def test_report_summarises_clips_and_flags_missing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "golden").mkdir()
            (root / "results").mkdir()
            (root / "golden" / "a.json").write_text(json.dumps(golden_clip("a")))
            (root / "golden" / "b.json").write_text(json.dumps(golden_clip("b")))
            (root / "results" / "a.json").write_text(json.dumps(RESULT))
            report = evaluate_directory(root / "golden", root / "results")
            self.assertEqual(report["summary"]["clips"], 2)
            self.assertEqual(report["summary"]["clipsWithResults"], 1)
            by_id = {clip["clipId"]: clip for clip in report["clips"]}
            self.assertIn("error", by_id["b"])
            self.assertAlmostEqual(report["summary"]["meanContactF1At100ms"], by_id["a"]["contacts"]["tol100ms"]["f1"])
            self.assertEqual(sum(report["summary"]["calibrationTiers"].values()), 1)

    def test_cli_writes_the_report_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "golden").mkdir()
            (root / "results").mkdir()
            (root / "golden" / "a.json").write_text(json.dumps(golden_clip("a")))
            (root / "results" / "a.json").write_text(json.dumps(RESULT))
            out = root / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--golden", str(root / "golden"), "--results", str(root / "results"), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["summary"]["clips"], 1)

    def test_cli_returns_nonzero_on_invalid_golden_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "golden").mkdir()
            (root / "results").mkdir()
            (root / "golden" / "bad.json").write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["--golden", str(root / "golden"), "--results", str(root / "results")])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
