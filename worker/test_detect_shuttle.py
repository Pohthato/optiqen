# worker/test_detect_shuttle.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detect_shuttle import main
from geometry.camera import Camera
from simulation.clip import make_clip, write_clip
from simulation.rally import Shot, build_rally
from test_shuttle import HAVE_WEIGHTS, WEIGHTS

SIZE = (640, 360)


class DetectShuttleCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main(list(args))
        return code, err.getvalue()

    def test_missing_weights_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, SIZE)
            writer.write(np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
            writer.release()
            code, err = self.run_cli(str(video), "--out", str(Path(tmp) / "s.json"), "--weights", tmp)
        self.assertEqual(code, 1)
        self.assertIn("TrackNet_best.pt", err)

    def test_an_unreadable_video_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self.run_cli(str(Path(tmp) / "missing.mp4"), "--out", str(Path(tmp) / "s.json"), "--weights", str(WEIGHTS))
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)

    @unittest.skipUnless(HAVE_WEIGHTS, "TrackNetV3 weights are not in worker/weights")
    def test_writes_a_detection_or_a_miss_for_every_frame(self):
        camera = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 1300.0 * 640 / 1920, SIZE)
        rally = build_rally((3.3, 3.6, 1.0), [Shot("clear", (1.8, 11.5), 1.1, None)])
        clip = make_clip(camera, SIZE, rally, fps=30.0, handheld=False, seed=6, tail_s=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_clip(clip, Path(tmp) / "clip")
            out, overlay = Path(tmp) / "shuttle.json", Path(tmp) / "overlay.mp4"
            code, _ = self.run_cli(str(paths["video"]), "--out", str(out), "--overlay", str(overlay), "--fast", "--weights", str(WEIGHTS))
            result = json.loads(out.read_text(encoding="utf-8"))
            capture = cv2.VideoCapture(str(overlay))
            overlay_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        self.assertEqual(code, 0)
        self.assertEqual(result["model"], "TrackNetV3")
        summary = result["summary"]
        self.assertEqual(summary["detected"] + summary["filled"] + summary["none"], len(clip.frames))
        self.assertEqual(result["imageSize"], list(SIZE))
        self.assertEqual(len(result["frames"]), len(clip.frames))
        self.assertEqual(overlay_frames, len(clip.frames))
        errors = []
        for index, entry in enumerate(result["frames"]):
            self.assertAlmostEqual(entry["timeMs"], index * 1000 / 30.0, delta=1.0)
            position = clip.shuttle[index]
            if position is None or "x" not in entry:
                continue
            truth = clip.cameras[index].project(np.array([position]))[0]
            errors.append(float(np.hypot(entry["x"] - truth[0], entry["y"] - truth[1])))
        self.assertGreater(len(errors), 20)
        self.assertLess(float(np.median(errors)), 3.0)


if __name__ == "__main__":
    unittest.main()
