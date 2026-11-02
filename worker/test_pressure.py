# worker/test_pressure.py
import unittest
from dataclasses import replace

import numpy as np

from evaluation.pressure_benchmark import FACTS, score
from pressure import HIT_TIME_SIGMA_MS, rally_pressure, work
from shots import Estimate, Landing, ShotFacts

REACH = Estimate(1.3, 0.02)
CERTAIN = np.zeros((2, 2))


def shot(hit_ms, hitter, contact=(3.0, 4.0, 2.0), serve=False, landing=None) -> ShotFacts:
    return ShotFacts(
        hit_ms, hitter, serve, contact, (0.05, 0.05, 0.1), Estimate(20.0, 1.0), Estimate(10.0, 1.0), Estimate(3.0, 0.1), None, 900.0, landing,
    )


class WorkTests(unittest.TestCase):
    def test_the_way_to_the_shuttle_less_the_reach_over_the_time_given(self):
        done = work((np.array([0.0, 0.0]), CERTAIN), (3.0, 4.0), CERTAIN, Estimate(1.0, 0.0), 800.0, Estimate(1.5, 0.1))
        self.assertAlmostEqual(done.go_m.value, 4.0)
        self.assertAlmostEqual(done.go_m.sigma, 0.0)
        self.assertEqual((done.time_ms.value, done.time_ms.sigma), (800.0, HIT_TIME_SIGMA_MS))
        self.assertAlmostEqual(done.speed_m_s.value, 5.0)
        self.assertEqual(done.height_m, Estimate(1.5, 0.1))

    def test_nothing_to_go_within_reach(self):
        done = work((np.array([0.0, 0.0]), CERTAIN), (0.6, 0.8), CERTAIN, Estimate(1.3, 0.0), 800.0)
        self.assertEqual((done.go_m.value, done.speed_m_s.value), (0.0, 0.0))

    def test_the_uncertainty_comes_from_where_they_stood_the_target_and_the_reach(self):
        along = np.array([[0.04, 0.0], [0.0, 0.0]])  # 20 cm along the way, nothing across it
        for stood, target, reach, sigma in ((along, CERTAIN, 0.0, 0.2), (CERTAIN, along, 0.0, 0.2), (CERTAIN, CERTAIN, 0.2, 0.2), (along, along, 0.2, 0.2 * 3**0.5)):
            done = work((np.array([0.0, 0.0]), stood), (5.0, 0.0), target, Estimate(1.0, reach), 1000.0, rng=np.random.default_rng(1))
            self.assertAlmostEqual(done.go_m.sigma, sigma, delta=0.015)
            self.assertAlmostEqual(done.speed_m_s.sigma, sigma, delta=0.03)


class RallyPressureTests(unittest.TestCase):
    def setUp(self):
        self.stood = {"near": (np.array([3.0, 3.0]), CERTAIN), "far": (np.array([3.0, 10.0]), CERTAIN)}
        self.reaches = {"near": REACH, "far": REACH}

    def test_each_shot_on_the_other_player_up_to_their_contact(self):
        shots = [shot(0.0, "near"), shot(1000.0, "far", (3.0, 12.5, 1.0)), shot(2200.0, "near", (0.5, 3.0, 0.5))]
        first, second, last = rally_pressure(shots, [self.stood] * 3, self.reaches)
        self.assertAlmostEqual(first.go_m.value, 2.5 - 1.3)
        self.assertAlmostEqual(first.time_ms.value, 1000.0)
        self.assertEqual(first.height_m, Estimate(1.0, 0.1))
        self.assertAlmostEqual(second.go_m.value, 2.5 - 1.3)
        self.assertAlmostEqual(second.speed_m_s.value, 1.0)
        self.assertIsNone(last)  # nothing followed it

    def test_a_shot_that_came_down_is_measured_to_where_it_landed(self):
        landing = Landing((3.0, 6.0), ((0.01, 0.0), (0.0, 0.01)), 1500.0, "in", 0.99, 0.99)
        down = rally_pressure([replace(shot(1000.0, "far"), landing=landing)], [self.stood], self.reaches)[0]
        self.assertAlmostEqual(down.go_m.value, 3.0 - 1.3)
        self.assertAlmostEqual(down.time_ms.value, 500.0)
        self.assertIsNone(down.height_m)
        into_net = replace(landing, p_over_net=0.01)
        self.assertEqual(rally_pressure([replace(shot(1000.0, "far"), landing=into_net)], [self.stood], self.reaches), [None])

    def test_none_without_the_player_or_a_reply(self):
        shots = [shot(0.0, "near"), shot(900.0, "far")]
        self.assertEqual(rally_pressure(shots, [{"near": None, "far": None}] * 2, self.reaches), [None, None])
        self.assertEqual(rally_pressure(shots, [None, None], self.reaches), [None, None])
        self.assertEqual(rally_pressure(shots, [self.stood] * 2, {"near": None, "far": None}), [None, None])
        self.assertIsNone(rally_pressure([shot(0.0, "near"), shot(900.0, "near")], [self.stood] * 2, self.reaches)[0])  # a hit was missed
        self.assertIsNone(rally_pressure([shot(0.0, "near"), shot(5000.0, "far", serve=True)], [self.stood] * 2, self.reaches)[0])  # the next rally


class BenchmarkTests(unittest.TestCase):
    def test_the_work_said_against_the_true_work(self):
        result = score(("side", "near", 30.0, 3))
        self.assertLess(abs(result["reach"][0]), 0.05)
        for name in FACTS:
            errors = np.array(result["facts"][name])
            self.assertGreater(len(errors), 2)
            self.assertLess(np.median(np.abs(errors[:, 0])), 0.25, name)


if __name__ == "__main__":
    unittest.main()
