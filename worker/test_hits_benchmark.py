# worker/test_hits_benchmark.py
import unittest

import numpy as np

from evaluation.hits_benchmark import Scenario, run, score, simulate, standard_scenarios
from hits import Hit


class SimulateTests(unittest.TestCase):
    def test_a_scenario_is_reproducible(self):
        a, b = simulate(Scenario(seed=7)), simulate(Scenario(seed=7))
        self.assertEqual(a.track, b.track)
        self.assertEqual(a.clip.distractor_times, b.clip.distractor_times)

    def test_every_frame_has_a_time_and_a_detection_or_a_miss(self):
        sim = simulate(Scenario(seed=8, fps=60.0))
        self.assertEqual(len(sim.track), len(sim.clip.cameras))
        self.assertEqual(len(sim.times_ms), len(sim.track))
        self.assertTrue(all(b > a for a, b in zip(sim.times_ms, sim.times_ms[1:])))
        self.assertAlmostEqual(sim.times_ms[1] - sim.times_ms[0], 1000 / 60.0)

    def test_false_detections_and_neighbouring_sounds_are_there_when_asked_for(self):
        noisy = simulate(Scenario(seed=9, false_per_s=2.0, neighbour_per_s=1.0))
        clean = simulate(Scenario(seed=9, false_per_s=0.0, neighbour_per_s=0.0, sound=False))
        self.assertGreater(noisy.false_frames, 0)
        self.assertGreater(len(noisy.clip.distractor_times), 0)
        self.assertEqual(clean.false_frames, 0)
        self.assertEqual(clean.clip.distractor_times, [])
        self.assertIsNone(clean.onsets)

    def test_the_standard_set_has_every_view_and_frame_rate_with_sound_and_without(self):
        scenarios = standard_scenarios(18)
        self.assertEqual({(s.view, s.fps, s.sound) for s in scenarios}, {(v, f, s) for v in ("behind", "corner", "side") for f in (30.0, 60.0) for s in (True, False)})
        self.assertNotEqual([s.seed for s in standard_scenarios(6, 5000)], [s.seed for s in standard_scenarios(6)])


class ScoreTests(unittest.TestCase):
    def test_hits_are_matched_one_to_one_within_33_ms(self):
        sim = simulate(Scenario(seed=11, neighbour_per_s=0.0))
        first, second = (contact.time * 1000 for contact in sim.clip.rally.contacts[:2])
        hits = [
            Hit(first + 10, None, first + 10, "flights", (0.0, 0.0, 1.0), 0, 100.0),
            Hit(first + 20, None, first + 20, "flights", (0.0, 0.0, 1.0), 0, 100.0),  # a second claim on the same hit
            Hit(second + 50, None, second + 50, "flights", (0.0, 0.0, 1.0), 0, 100.0),  # too far
        ]
        result = score(sim.clip, hits)
        self.assertEqual(result["found"], 1)
        self.assertEqual(result["claimed"], 3)
        self.assertEqual(result["errorsMs"], [10.0])

    def test_neighbouring_sounds_taken_as_hits_are_counted(self):
        sim = simulate(Scenario(seed=12, neighbour_per_s=1.0))
        heard = sim.clip.distractor_times[0] * 1000
        result = score(sim.clip, [Hit(heard - 25, heard, heard - 25, "sound and flights", (0.0, 0.0, 1.0), 0, 100.0)])
        expected = 0 if any(abs(heard - 25 - c.time * 1000) <= 33 for c in sim.clip.rally.contacts) else 1
        self.assertEqual(result["neighbourHits"], expected)


class RunTests(unittest.TestCase):
    def test_reports_pooled_scores_by_view_frame_rate_and_sound(self):
        report = run(standard_scenarios(4))
        self.assertEqual(len(report["scenarios"]), 4)
        pooled = report["summary"]["pooled"]
        self.assertIn("all", pooled)
        self.assertTrue(any(key.startswith("view=") for key in pooled))
        self.assertTrue(any(key.endswith(",sound=True") for key in pooled))
        self.assertTrue(0.0 <= pooled["all"]["f1"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
