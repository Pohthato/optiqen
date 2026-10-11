# worker/test_shots_benchmark.py
import unittest
from dataclasses import replace

from evaluation.hits_benchmark import Scenario, simulate
from evaluation.shots_benchmark import run
from geometry.court_model import COURT_LENGTH_M, SINGLES_WIDTH_M


class ShotsBenchmarkTests(unittest.TestCase):
    def test_the_last_shot_can_be_aimed_at_a_line(self):
        landings = [simulate(Scenario(seed=seed, final_near_line=True)).clip.rally.landing for seed in range(6)]
        distance = [min(abs(x), abs(x - SINGLES_WIDTH_M), abs(y), abs(y - COURT_LENGTH_M)) for x, y in landings]
        self.assertTrue(all(d <= 0.41 for d in distance))
        self.assertEqual(simulate(Scenario(seed=3)).clip.rally, simulate(replace(Scenario(seed=3), final_near_line=False)).clip.rally)

    def test_reports_shot_errors_landings_and_calls(self):
        report = run([Scenario(seed=7, final_near_line=True, false_per_s=0.0, neighbour_per_s=0.0)])
        summary = report["summary"]["all"]
        self.assertIn("landingErrorM", summary)
        self.assertIn("callsCorrect", summary)
        self.assertTrue(report["scenarios"][0]["matched"])
        self.assertLess(summary["contactErrorM"]["median"], 0.5)


if __name__ == "__main__":
    unittest.main()
