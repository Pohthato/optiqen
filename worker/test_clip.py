# worker/test_clip.py
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

from geometry.camera import Camera
from simulation.clip import camera_from_truth, main, make_clip, truth_dict, write_clip
from simulation.rally import Shot, build_rally

SIZE = (320, 180)
CAMERA = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 / 6, SIZE)
SERVE_ONLY = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, None)])


class MakeClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=20.0, handheld=True, seed=1, distractors=1)

    def test_frame_count_and_size_follow_the_duration(self):
        expected = math.ceil((SERVE_ONLY.end_time + 0.5) * 20.0)
        self.assertEqual(len(self.clip.frames), expected)
        self.assertEqual(len(self.clip.cameras), expected)
        self.assertEqual(len(self.clip.shuttle), expected)
        self.assertEqual(self.clip.frames[0].shape, (180, 320, 3))

    def test_shuttle_truth_matches_the_rally_at_each_frame_time(self):
        self.assertIsNone(self.clip.shuttle[0])
        self.assertIsNone(self.clip.shuttle[-1])
        frame = 30  # 1.5 s, mid-flight
        np.testing.assert_allclose(self.clip.shuttle[frame], SERVE_ONLY.shuttle_at(frame / 20.0))

    def test_handheld_cameras_move_and_tripod_ones_do_not(self):
        self.assertFalse(np.allclose(self.clip.cameras[0].rvec, self.clip.cameras[10].rvec))
        still = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, handheld=False)
        self.assertTrue(all(camera is CAMERA for camera in still.cameras))

    def test_audio_covers_the_clip(self):
        duration = SERVE_ONLY.end_time + 0.5
        self.assertEqual(len(self.clip.audio), int(round(duration * self.clip.sample_rate)))
        self.assertEqual(len(self.clip.distractor_times), 1)


class TruthTests(unittest.TestCase):
    def test_truth_is_json_and_reconstructs_each_camera(self):
        clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, seed=2)
        truth = json.loads(json.dumps(truth_dict(clip)))
        self.assertEqual(truth["frameCount"], len(clip.frames))
        self.assertEqual(truth["contacts"][0]["kind"], "serve")
        self.assertIsNone(truth["shuttle"][0])
        point = np.array([[1.0, 9.0, 0.0]])
        rebuilt = camera_from_truth(truth["cameras"][7])
        np.testing.assert_allclose(rebuilt.project(point), clip.cameras[7].project(point), atol=1e-6)


class SoundAndConventionTests(unittest.TestCase):
    def test_each_hit_is_heard_after_sound_travels_to_the_camera(self):
        clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, seed=4)
        truth = truth_dict(clip)
        contact = truth["contacts"][0]
        distance = float(np.linalg.norm(np.array(contact["position"]) - CAMERA.centre))
        self.assertAlmostEqual(contact["audioTimeMs"] - contact["timeMs"], distance / 343.0 * 1000, places=1)
        onset = int(contact["audioTimeMs"] / 1000 * clip.sample_rate)
        self.assertGreater(float(np.max(np.abs(clip.audio[onset : onset + 240]))), 0.2)
        self.assertLess(float(np.max(np.abs(clip.audio[onset - 480 : onset - 48]))), 0.06)

    def test_sound_delay_can_be_switched_off(self):
        clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, sound_delay=False)
        contact = truth_dict(clip)["contacts"][0]
        self.assertEqual(contact["audioTimeMs"], contact["timeMs"])

    def test_truth_states_its_conventions(self):
        truth = truth_dict(make_clip(CAMERA, SIZE, SERVE_ONLY, fps=20.0))
        self.assertAlmostEqual(truth["exposureS"], 0.5 / 20.0)
        self.assertTrue({"shuttle", "audioOffset", "distractorTimes", "imageSize"} <= set(truth["conventions"]))

    def test_odd_frame_sizes_are_refused(self):
        odd = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 200.0, (161, 91))
        with self.assertRaisesRegex(ValueError, "even"):
            make_clip(odd, (161, 91), SERVE_ONLY, fps=10.0)


class WriteClipTests(unittest.TestCase):
    def test_written_files_read_back(self):
        clip = make_clip(CAMERA, SIZE, SERVE_ONLY, fps=10.0, seed=3)
        with tempfile.TemporaryDirectory() as temp:
            paths = write_clip(clip, Path(temp) / "clip")
            capture = cv2.VideoCapture(str(paths["video"]))
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (180, 320, 3))
                frames += 1
            capture.release()
            self.assertEqual(frames, len(clip.frames))
            rate, samples = wavfile.read(paths["audio"])
            self.assertEqual((rate, len(samples)), (clip.sample_rate, len(clip.audio)))
            self.assertEqual(json.loads(paths["truth"].read_text())["frameCount"], len(clip.frames))

    def test_cli_writes_a_canned_rally(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "cli"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--out", str(out), "--fps", "10", "--width", "160", "--height", "90", "--tripod"])
            self.assertEqual(code, 0)
            truth = json.loads((out / "truth.json").read_text())
            self.assertEqual(len(truth["contacts"]), 7)
            self.assertTrue((out / "video.mp4").is_file() and (out / "audio.wav").is_file())


if __name__ == "__main__":
    unittest.main()
