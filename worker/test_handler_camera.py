"""Runs the real RunPod handler on a tiny generated clip, with the GPU-only
dependencies (runpod, ultralytics) stubbed so it imports on a laptop."""
import shutil
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

CORNERS = [
    {"label": "nearLeft", "x": 20.0, "y": 85.0},
    {"label": "nearRight", "x": 80.0, "y": 85.0},
    {"label": "farRight", "x": 62.0, "y": 25.0},
    {"label": "farLeft", "x": 38.0, "y": 25.0},
]


def write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90))
    for _ in range(6):
        writer.write(np.full((90, 160, 3), 90, dtype=np.uint8))
    writer.release()


class HandlerCameraSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.clip = self.temp / "clip.mp4"
        write_clip(self.clip)
        self.download = mock.patch.object(handler, "download_source", lambda _url, destination: shutil.copy(self.clip, destination))
        self.download.start()

    def tearDown(self):
        self.download.stop()
        shutil.rmtree(self.temp, ignore_errors=True)

    def job(self):
        return {"input": {
            "analysisId": "test",
            "sourceVideoUrl": "https://example.test/clip.mp4",
            "selectedPlayer": "near",
            "courtType": "singles",
            "requestedLayers": ["courtMap"],
            "calibration": {"corners": CORNERS, "confidence": "provisional", "supportsCourtMapping": False, "guidance": "test"},
        }}

    def test_completed_result_carries_a_camera_summary(self):
        result = handler.handler(self.job())
        self.assertNotIn("error", result)
        self.assertIsNotNone(result["camera"])
        self.assertIn(result["camera"]["geometryTier"], ("A", "B", "C"))

    def test_doubles_jobs_solve_the_camera_from_the_outer_corners(self):
        job = self.job()
        job["input"]["courtType"] = "doubles"
        result = handler.handler(job)
        self.assertNotIn("error", result)
        self.assertIsNotNone(result["camera"])
        self.assertIn(result["camera"]["geometryTier"], ("A", "B", "C"))

    def test_a_camera_solver_failure_does_not_fail_the_analysis(self):
        with mock.patch.object(handler, "summarise_calibration", side_effect=RuntimeError("solver blew up")):
            result = handler.handler(self.job())
        self.assertNotIn("error", result)
        self.assertIsNone(result["camera"])
        self.assertTrue(any("Camera solution unavailable" in w for w in result["diagnostics"]["warnings"]))


if __name__ == "__main__":
    unittest.main()
