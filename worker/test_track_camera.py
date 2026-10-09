# worker/test_track_camera.py
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluation.track import floor_error_cm
from geometry.camera import Camera
from geometry.synthetic import observe
from simulation.render import render_frame
from test_tracking import PLAYERS, SIZE, TRUTH, turned
from track_camera import main

FPS = 15.0
PAN = [turned(TRUTH, (0.0, 0.3 * i, 0.0)) for i in range(16)]  # the phone has turned 3 degrees by frame 10
ANCHOR = 10
TAP_NAMES = ["back0_sl", "back0_sr", "short0_sl", "short0_sr", "back1_sl", "back1_sr"]


def labels(taps: dict, frame: int, **extra) -> dict:
    """A labeller export: points placed on one frame, timed the way the labeller times frames."""
    time_ms = round(frame * 1000 / FPS)
    doc = {"courtKeypoints": [{"name": name, "x": x, "y": y, "timeMs": time_ms, "source": "clicked"} for name, (x, y) in taps.items()]}
    doc.update(extra)
    return doc


class TrackCameraCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.video = cls.tmp / "pan.mp4"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
        for index, camera in enumerate(PAN):
            writer.write(render_frame(camera, SIZE, players=PLAYERS, seed=index))
        writer.release()
        cls.taps = observe(PAN[ANCHOR], SIZE, TAP_NAMES, noise_px=1.0, seed=6)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_cli(self, doc: dict, *extra: str) -> tuple[int, str, dict | None]:
        golden, out = self.tmp / "labels.json", self.tmp / "track.json"
        golden.write_text(json.dumps(doc), encoding="utf-8")
        out.unlink(missing_ok=True)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main([str(self.video), "--golden", str(golden), "--out", str(out), *extra])
        result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        return code, err.getvalue(), result

    def test_tracks_every_frame_from_points_placed_on_a_later_frame(self):
        overlay = self.tmp / "overlay.mp4"
        code, _, result = self.run_cli(labels(self.taps, ANCHOR, sourceFps=FPS, imageSize=list(SIZE)), "--overlay", str(overlay))
        capture = cv2.VideoCapture(str(overlay))
        overlay_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        self.assertEqual(code, 0)
        self.assertEqual(result["anchorFrame"], ANCHOR)
        self.assertEqual(result["imageSize"], list(SIZE))
        self.assertEqual(len(result["frames"]), len(PAN))
        self.assertEqual(overlay_frames, len(PAN))
        self.assertEqual(result["summary"]["lost"], 0)
        self.assertLess(abs(result["intrinsics"]["focalPx"] - 650.0) / 650.0, 0.03)
        intrinsics = result["intrinsics"]
        for index, (truth, entry) in enumerate(zip(PAN, result["frames"])):
            self.assertAlmostEqual(entry["timeMs"], index * 1000 / FPS, delta=1.0)
            camera = Camera(intrinsics["focalPx"], intrinsics["cx"], intrinsics["cy"], np.array(entry["rvec"]), np.array(entry["tvec"]), intrinsics["k1"])
            self.assertLess(float(np.median(floor_error_cm(truth, camera, SIZE))), 2.0)

    def test_points_from_one_frame_do_not_anchor_another(self):
        code, err, _ = self.run_cli(labels(self.taps, 0))
        self.assertEqual(code, 1)
        self.assertIn("frame", err)

    def test_mislabelled_points_are_refused(self):
        seen = observe(PAN[ANCHOR], SIZE, ["back0_dl", "back0_dr", "back1_dr", "back1_dl"], noise_px=1.0, seed=7)
        taps = {name.replace("_dl", "_sl").replace("_dr", "_sr"): pixel for name, pixel in seen.items()}
        code, err, result = self.run_cli(labels(taps, ANCHOR))
        self.assertEqual(code, 1)
        self.assertIn("name", err)
        self.assertIsNone(result)

    def test_labels_from_a_different_encode_are_refused(self):
        code, err, _ = self.run_cli(labels(self.taps, ANCHOR, imageSize=[1920, 1080]))
        self.assertEqual(code, 1)
        self.assertIn("1920x1080", err)

    def test_refuses_labels_without_enough_points(self):
        code, err, _ = self.run_cli({"courtKeypoints": [{"name": "back0_sl", "x": 1.0, "y": 2.0, "timeMs": 0}]})
        self.assertEqual(code, 1)
        self.assertIn("court points", err)


if __name__ == "__main__":
    unittest.main()
