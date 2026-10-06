# worker/test_track_camera.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from geometry.camera import Camera
from geometry.synthetic import observe
from simulation.clip import make_clip, write_clip
from simulation.rally import Shot, build_rally
from test_tracking import SIZE, TRUTH, floor_error_cm
from track_camera import main


class TrackingCliTests(unittest.TestCase):
    def test_tracks_a_video_from_labelled_court_points(self):
        rally = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.0), 1.0, None)])
        clip = make_clip(TRUTH, SIZE, rally, fps=15.0, handheld=True, seed=5, tail_s=0.3)
        anchor_frame = 6
        taps = observe(clip.cameras[anchor_frame], SIZE, ["back0_sl", "back0_sr", "short0_sl", "short0_sr", "back1_sl", "back1_sr"], noise_px=1.0, seed=6)
        labels = {"courtKeypoints": [{"name": name, "x": x, "y": y, "timeMs": anchor_frame * 1000 / 15.0, "source": "clicked"} for name, (x, y) in taps.items()]}
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_clip(clip, Path(tmp) / "clip")
            golden = Path(tmp) / "labels.json"
            golden.write_text(json.dumps(labels), encoding="utf-8")
            out, overlay = Path(tmp) / "track.json", Path(tmp) / "overlay.mp4"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(paths["video"]), "--golden", str(golden), "--out", str(out), "--overlay", str(overlay)])
            result = json.loads(out.read_text(encoding="utf-8"))
            capture = cv2.VideoCapture(str(overlay))
            overlay_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        self.assertEqual(code, 0)
        self.assertEqual(result["anchorFrame"], anchor_frame)
        self.assertEqual(result["imageSize"], list(SIZE))
        self.assertEqual(len(result["frames"]), len(clip.frames))
        self.assertEqual(overlay_frames, len(clip.frames))
        self.assertEqual(result["summary"]["lost"], 0)
        self.assertLess(abs(result["intrinsics"]["focalPx"] - 650.0) / 650.0, 0.03)
        errors = []
        for truth, entry in zip(clip.cameras, result["frames"]):
            camera = Camera(result["intrinsics"]["focalPx"], result["intrinsics"]["cx"], result["intrinsics"]["cy"], np.array(entry["rvec"]), np.array(entry["tvec"]), result["intrinsics"]["k1"])
            errors.append(float(np.median(floor_error_cm(truth, camera))))
        self.assertLess(float(np.median(errors)), 3.0)

    def test_refuses_labels_without_enough_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            golden = Path(tmp) / "labels.json"
            golden.write_text(json.dumps({"courtKeypoints": [{"name": "back0_sl", "x": 1.0, "y": 2.0, "timeMs": 0}]}), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = main([str(Path(tmp) / "missing.mp4"), "--golden", str(golden), "--out", str(Path(tmp) / "t.json")])
        self.assertEqual(code, 1)
        self.assertIn("court points", err.getvalue())


if __name__ == "__main__":
    unittest.main()
