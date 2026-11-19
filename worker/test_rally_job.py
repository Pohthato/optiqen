# worker/test_rally_job.py
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.modules.setdefault("runpod", types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a, **k: None, start=lambda *a, **k: None)))
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=object))

import handler  # noqa: E402
import rally_job  # noqa: E402

WORKER = Path(__file__).resolve().parent
CORNERS = [
    {"label": "nearLeft", "x": 20.0, "y": 85.0},
    {"label": "nearRight", "x": 80.0, "y": 85.0},
    {"label": "farRight", "x": 62.0, "y": 25.0},
    {"label": "farLeft", "x": 38.0, "y": 25.0},
]
ANALYSIS = {"video": "/tmp/x/source.mp4", "sources": {"track": "rally.track.json"}, "summary": {"shots": 1}, "shots": [{"shot": 1}], "findings": [{"id": "F1"}]}


def finished(code: int = 0, stderr: str = "", document: dict | None = None):
    """A stand-in for running analyse_rally.py: writes the analysis it was asked for."""

    def run(command, **_):
        if document is not None:
            Path(command[command.index("--out") + 1]).write_text(json.dumps(document), encoding="utf-8")
        return subprocess.CompletedProcess(command, code, "", stderr)

    return run


class CourtPointsTests(unittest.TestCase):
    def test_the_taps_become_the_court_points_of_the_first_frame_in_pixels(self):
        points = rally_job.court_points(CORNERS, (1920, 1080), 30.0, "singles")
        self.assertEqual(points["imageSize"], [1920, 1080])
        self.assertEqual(points["sourceFps"], 30.0)
        placed = {item["name"]: (item["x"], item["y"]) for item in points["courtKeypoints"]}
        self.assertEqual(placed, {"back0_sl": (384.0, 918.0), "back0_sr": (1536.0, 918.0), "back1_sr": (1190.4, 270.0), "back1_sl": (729.6, 270.0)})
        self.assertTrue(all(item["timeMs"] == 0 and item["source"] == "placed" for item in points["courtKeypoints"]))

    def test_doubles_taps_are_the_outer_corners(self):
        names = {item["name"] for item in rally_job.court_points(CORNERS, (1920, 1080), 30.0, "doubles")["courtKeypoints"]}
        self.assertEqual(names, {"back0_dl", "back0_dr", "back1_dr", "back1_dl"})


class AnalyseTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def analyse(self, corners=CORNERS, duration_s=20.0):
        return rally_job.analyse(self.work / "source.mp4", corners, "singles", (1920, 1080), 30.0, duration_s, self.work)

    def test_returns_the_analysis_without_the_jobs_own_paths(self):
        with mock.patch.object(rally_job.subprocess, "run", side_effect=finished(document=ANALYSIS)) as run:
            document, reason = self.analyse()
        self.assertIsNone(reason)
        self.assertEqual(document, {"summary": {"shots": 1}, "shots": [{"shot": 1}], "findings": [{"id": "F1"}]})
        command = run.call_args.args[0]
        self.assertEqual(command[1], str(rally_job.ANALYSE))
        golden = json.loads(Path(command[command.index("--golden") + 1]).read_text(encoding="utf-8"))
        self.assertEqual(len(golden["courtKeypoints"]), 4)
        self.assertEqual(command[command.index("--court") + 1], "singles")

    def test_says_why_when_the_analysis_fails(self):
        stderr = "no sound used: no ffmpeg\nthe court lines could not be found near the labelled points\n"
        with mock.patch.object(rally_job.subprocess, "run", side_effect=finished(code=1, stderr=stderr)):
            document, reason = self.analyse()
        self.assertIsNone(document)
        self.assertEqual(reason, "The rally analysis could not be made: the court lines could not be found near the labelled points.")

    def test_says_why_when_it_runs_too_long(self):
        with mock.patch.object(rally_job.subprocess, "run", side_effect=subprocess.TimeoutExpired("analyse", 900)):
            document, reason = self.analyse()
        self.assertIsNone(document)
        self.assertIn("longer than", reason)

    def test_does_not_try_without_four_corners_or_on_more_than_a_rally(self):
        with mock.patch.object(rally_job.subprocess, "run") as run:
            self.assertIn("all four court corners", self.analyse(corners=CORNERS[:3])[1])
            self.assertIn("one rally", self.analyse(duration_s=rally_job.MAX_SECONDS + 1)[1])
        run.assert_not_called()


class HandlerTests(unittest.TestCase):
    """The handler on a tiny generated clip: the rally analysis is added when it is made, and its
    reason when it is not, and the job completes either way."""

    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        clip = self.temp / "clip.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90))
        for _ in range(6):
            writer.write(np.full((90, 160, 3), 90, dtype=np.uint8))
        writer.release()
        self.download = mock.patch.object(handler, "download_source", lambda _url, destination: shutil.copy(clip, destination))
        self.download.start()

    def tearDown(self):
        self.download.stop()
        shutil.rmtree(self.temp, ignore_errors=True)

    def run_job(self, **analyse):
        job = {"input": {
            "analysisId": "test", "sourceVideoUrl": "https://example.test/clip.mp4", "selectedPlayer": "near", "courtType": "singles",
            "requestedLayers": ["courtMap"], "calibration": {"corners": CORNERS, "confidence": "provisional", "supportsCourtMapping": False, "guidance": "test"},
        }}
        with mock.patch.object(handler.rally_job, "analyse", **analyse) as called:
            return handler.handler(job), called

    def test_a_job_carries_the_rally_analysis(self):
        result, called = self.run_job(return_value=({"findings": [{"id": "F1"}]}, None))
        self.assertNotIn("error", result)
        self.assertEqual(result["rally"], {"findings": [{"id": "F1"}]})
        self.assertIsNone(result["rallyError"])
        video, corners, court, size, fps, duration, _ = called.call_args.args
        self.assertEqual((corners, court, size, fps), (CORNERS, "singles", (160, 90), 30.0))
        self.assertAlmostEqual(duration, 0.2, places=3)
        self.assertIn("rally", result["diagnostics"]["timingsMs"])

    def test_a_job_whose_rally_analysis_fails_still_completes_and_says_why(self):
        for analyse in ({"return_value": (None, "The rally analysis could not be made: no lines.")}, {"side_effect": RuntimeError("boom")}):
            result, _ = self.run_job(**analyse)
            self.assertNotIn("error", result)
            self.assertIsNone(result["rally"])
            self.assertTrue(result["rallyError"].startswith("The rally analysis could not be made"))
            self.assertIn(result["rallyError"], result["diagnostics"]["warnings"])
            self.assertIn("metrics", result)


class ImageTests(unittest.TestCase):
    def test_the_image_holds_every_module_the_handler_and_the_analysis_import(self):
        dockerfile = (WORKER / "Dockerfile").read_text(encoding="utf-8")
        copied = set()
        for sources in re.findall(r"^COPY (.+) /opt/netoval/\S*$", dockerfile, re.MULTILINE):
            copied.update(source.removeprefix("worker/").rstrip("/") for source in sources.split())
        local = {path.stem for path in WORKER.glob("*.py")} | {path.name for path in WORKER.iterdir() if (path / "__init__.py").exists()}
        needed, todo = set(), ["handler", "rally_job", "analyse_rally"]
        while todo:
            module = todo.pop()
            parts = module.split(".")
            path = next((p for p in (WORKER.joinpath(*parts).with_suffix(".py"), WORKER.joinpath(*parts, "__init__.py")) if p.is_file()), None)
            if path is None or module in needed:
                continue
            needed.add(module)
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    todo += [alias.name for alias in node.names if alias.name.split(".")[0] in local]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] in local:
                    todo += [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        needed |= {".".join(module.split(".")[:n]) for module in needed for n in range(1, module.count(".") + 1)}  # their packages
        missing = []
        for module in sorted(needed):
            parts = module.split(".")
            file = "/".join(parts) + (".py" if WORKER.joinpath(*parts).with_suffix(".py").is_file() else "/__init__.py")
            if not any(file == source or file.startswith(source + "/") for source in copied):
                missing.append(file)
        self.assertEqual(missing, [])
        for data in ("rally_value.json", "shot_names.json"):
            self.assertIn(data, copied)


if __name__ == "__main__":
    unittest.main()
