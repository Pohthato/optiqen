# worker/test_sim_audio.py
import unittest

import numpy as np

from simulation.audio import DISTRACTOR_SPACING_S, SAMPLE_RATE, render_audio

HITS = [0.5, 1.4, 2.2]


def peak(samples: np.ndarray, start_s: float, length_s: float = 0.005) -> float:
    a = int(start_s * SAMPLE_RATE)
    return float(np.max(np.abs(samples[a : a + int(length_s * SAMPLE_RATE)])))


class RenderAudioTests(unittest.TestCase):
    def test_length_type_and_range(self):
        samples, _ = render_audio(HITS, 3.0)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 3 * SAMPLE_RATE)
        self.assertLessEqual(float(np.max(np.abs(samples))), 1.0)

    def test_each_hit_is_a_loud_onset_over_quiet_background(self):
        samples, _ = render_audio(HITS, 3.0)
        for hit in HITS:
            self.assertGreater(peak(samples, hit), 0.2)
            self.assertLess(peak(samples, hit - 0.02, 0.015), 0.06)
        quiet = samples[int(0.1 * SAMPLE_RATE) : int(0.4 * SAMPLE_RATE)]
        self.assertLess(float(np.sqrt(np.mean(quiet**2))), 0.02)

    def test_offset_shifts_every_hit(self):
        samples, _ = render_audio(HITS, 3.0, offset_s=0.05)
        for hit in HITS:
            self.assertGreater(peak(samples, hit + 0.05), 0.2)
            self.assertLess(peak(samples, hit - 0.01, 0.015), 0.06)

    def test_distractors_are_quieter_and_away_from_hits(self):
        samples, distractors = render_audio(HITS, 3.0, distractors=3, seed=5)
        self.assertEqual(len(distractors), 3)
        for time in distractors:
            self.assertTrue(all(abs(time - hit) >= DISTRACTOR_SPACING_S for hit in HITS))
            self.assertTrue(0.05 < peak(samples, time) < 0.2)

    def test_same_seed_same_audio(self):
        a, _ = render_audio(HITS, 3.0, seed=9, distractors=2)
        b, _ = render_audio(HITS, 3.0, seed=9, distractors=2)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
