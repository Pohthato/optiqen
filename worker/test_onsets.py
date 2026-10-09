# worker/test_onsets.py
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from audio.extract import read_audio
from audio.onsets import detect_onsets
from simulation.audio import SAMPLE_RATE, render_audio


def sustained_audio(seconds=4.0, loud_from=1.0, loud_to=3.0, hits=()):
    """Steady background with a loud stretch (crowd, reverb) and optional sharp hits inside it."""
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    level = np.where((t >= loud_from) & (t < loud_to), 0.08, 0.01)
    samples = rng.uniform(-1.0, 1.0, len(t)) * level
    for hit in hits:
        start = int(round(hit * SAMPLE_RATE))
        i = np.arange(int(0.03 * SAMPLE_RATE))
        samples[start : start + len(i)] += 0.5 * np.exp(-i / (SAMPLE_RATE * 0.006)) * np.sin(2 * np.pi * 4000 * i / SAMPLE_RATE)
    return samples.astype(np.float32)


class DetectOnsetsTests(unittest.TestCase):
    def test_finds_every_hit_within_5_ms_and_ranks_the_neighbouring_court_below(self):
        hits = [0.5, 1.4, 2.2, 3.05]
        samples, distractors = render_audio(hits, 4.0, distractors=2, seed=11)
        onsets = detect_onsets(samples, SAMPLE_RATE)
        found = [min(onsets, key=lambda onset: abs(onset.time_ms - hit * 1000)) for hit in hits]
        for hit, onset in zip(hits, found):
            self.assertLessEqual(abs(onset.time_ms - hit * 1000), 5.0)
        weakest_hit = min(onset.strength for onset in found)
        for time in distractors:
            for onset in onsets:
                if abs(onset.time_ms - time * 1000) <= 5.0:
                    self.assertLess(onset.strength, weakest_hit)

    def test_quiet_audio_has_no_onsets(self):
        samples, _ = render_audio([], 3.0, seed=2)
        self.assertEqual(detect_onsets(samples, SAMPLE_RATE), [])

    def test_too_short_to_judge_has_no_onsets(self):
        self.assertEqual(detect_onsets(np.zeros(100, dtype=np.float32), SAMPLE_RATE), [])

    def test_a_sustained_loud_stretch_gives_at_most_one_onset(self):
        self.assertLessEqual(len(detect_onsets(sustained_audio(), SAMPLE_RATE)), 1)

    def test_a_hit_inside_a_loud_stretch_is_still_found(self):
        onsets = detect_onsets(sustained_audio(hits=[2.0]), SAMPLE_RATE)
        self.assertTrue(any(abs(onset.time_ms - 2000) <= 5 for onset in onsets))

    def test_works_at_other_sample_rates(self):
        samples, _ = render_audio([0.6, 1.7], 2.5, sample_rate=44_100, seed=3)
        onsets = detect_onsets(samples, 44_100)
        for hit in (0.6, 1.7):
            self.assertTrue(any(abs(onset.time_ms - hit * 1000) <= 5 for onset in onsets))


def completed(returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class ReadAudioTests(unittest.TestCase):
    def test_decodes_mono_float_samples_at_the_requested_rate(self):
        decoded = np.array([0.0, 0.25, -0.5], dtype="<f4").tobytes()
        with mock.patch("audio.extract.shutil.which", return_value="ffmpeg"), mock.patch("audio.extract.subprocess.run", return_value=completed(0, decoded)) as run:
            samples = read_audio(Path("clip.mp4"), 44_100)
        np.testing.assert_array_equal(samples, [0.0, 0.25, -0.5])
        command = run.call_args.args[0]
        for flag in (["-vn"], ["-ac", "1"], ["-ar", "44100"], ["-f", "f32le"]):
            self.assertTrue(any(command[i : i + len(flag)] == flag for i in range(len(command))), flag)

    def test_a_video_without_sound_has_no_audio(self):
        with mock.patch("audio.extract.shutil.which", return_value="ffmpeg"), mock.patch("audio.extract.subprocess.run", return_value=completed(0, b"")):
            self.assertIsNone(read_audio(Path("silent.mp4")))
        failure = completed(1, stderr=b"Output file #0 does not contain any stream")
        with mock.patch("audio.extract.shutil.which", return_value="ffmpeg"), mock.patch("audio.extract.subprocess.run", return_value=failure):
            self.assertIsNone(read_audio(Path("silent.mp4")))

    def test_other_ffmpeg_failures_are_reported_not_taken_for_silence(self):
        failure = completed(1, stderr=b"clip.mp4: Invalid data found when processing input")
        with mock.patch("audio.extract.shutil.which", return_value="ffmpeg"), mock.patch("audio.extract.subprocess.run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "Invalid data"):
                read_audio(Path("clip.mp4"))

    def test_a_missing_ffmpeg_is_reported(self):
        with mock.patch("audio.extract.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                read_audio(Path("clip.mp4"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is installed in the worker image, not necessarily here")
    def test_reads_a_real_file(self):
        samples, _ = render_audio([0.5], 1.0, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hit.wav"
            from scipy.io import wavfile

            wavfile.write(path, SAMPLE_RATE, samples)
            decoded = read_audio(path)
        self.assertEqual(len(decoded), len(samples))
        self.assertTrue(any(abs(onset.time_ms - 500) <= 5 for onset in detect_onsets(decoded, SAMPLE_RATE)))


if __name__ == "__main__":
    unittest.main()
