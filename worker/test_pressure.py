# worker/test_pressure.py
import unittest
from dataclasses import replace

import numpy as np

from evaluation.pressure_benchmark import FACTS, score
from geometry.shuttle_physics import GRAVITY
from pressure import HIT_TIME_SIGMA_MS, JUMP_SMASH_M, LEAN_M, PUSH_OFF_MS, REACT_MS, TOP_M_S, contradicts, jump_ms, rally_pressure, work
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
        self.assertAlmostEqual(done.spare_ms.value, 800.0 - REACT_MS - (4.0 - LEAN_M) / TOP_M_S * 1000.0)
        self.assertAlmostEqual(done.spare_ms.sigma, HIT_TIME_SIGMA_MS, delta=1.0)
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
            self.assertAlmostEqual(done.spare_ms.sigma, np.hypot(sigma / TOP_M_S * 1000.0, HIT_TIME_SIGMA_MS), delta=6.0)

    def test_a_jump_takes_its_push_off_and_its_rise(self):
        draws = jump_ms(Estimate(0.3, 0.0))
        topped_out = (2.0 * 0.3 / GRAVITY) ** 0.5 * 1000.0  # a jump that just got there
        hardest = (2.0 / GRAVITY) ** 0.5 * (JUMP_SMASH_M**0.5 - (JUMP_SMASH_M - 0.3) ** 0.5) * 1000.0  # a jump smash's, still rising
        self.assertTrue(np.all((draws >= PUSH_OFF_MS + hardest - 1e-6) & (draws <= PUSH_OFF_MS + topped_out + 1e-6)))
        self.assertAlmostEqual(draws.max(), PUSH_OFF_MS + topped_out, delta=5.0)
        self.assertAlmostEqual(draws.min(), PUSH_OFF_MS + hardest, delta=5.0)
        np.testing.assert_array_equal(jump_ms(Estimate(0.05, 0.01)), 0.0)  # on tiptoe
        np.testing.assert_array_equal(jump_ms(None), 0.0)

    def test_met_in_a_jump_leaves_less_to_spare(self):
        stood = (np.array([0.0, 0.0]), CERTAIN)
        floor = work(stood, (0.6, 0.8), CERTAIN, Estimate(1.3, 0.0), 800.0)
        jumped = work(stood, (0.6, 0.8), CERTAIN, Estimate(1.3, 0.0), 800.0, off_floor=Estimate(0.3, 0.02))
        self.assertAlmostEqual(floor.spare_ms.value - jumped.spare_ms.value, float(np.mean(jump_ms(Estimate(0.3, 0.02)))))
        self.assertGreater(jumped.spare_ms.sigma, 2 * floor.spare_ms.sigma)  # how hard they jumped is not known


class RallyPressureTests(unittest.TestCase):
    def setUp(self):
        self.stood = {"near": (np.array([3.0, 3.0]), CERTAIN), "far": (np.array([3.0, 10.0]), CERTAIN)}
        self.reaches = {"near": REACH, "far": REACH}

    def test_each_shot_on_the_other_player_up_to_their_contact(self):
        shots = [shot(0.0, "near"), shot(1000.0, "far", (3.0, 12.5, 1.0)), shot(2200.0, "near", (0.5, 3.0, 0.5))]
        far_back = {**self.stood, "far": (np.array([3.0, 11.5]), CERTAIN)}
        stood = [self.stood, far_back, {**far_back, "near": (np.array([1.5, 3.0]), CERTAIN)}]  # each a metre from the contact when they meet it
        first, second, last = rally_pressure(shots, stood, self.reaches)
        self.assertAlmostEqual(first.go_m.value, 2.5 - 1.3)
        self.assertAlmostEqual(first.time_ms.value, 1000.0)
        self.assertEqual(first.height_m, Estimate(1.0, 0.1))
        self.assertAlmostEqual(second.go_m.value, 2.5 - 1.3)
        self.assertAlmostEqual(second.speed_m_s.value, 1.0)
        self.assertIsNone(last)  # nothing followed it

    def test_the_reply_met_in_a_jump(self):
        shots = [shot(0.0, "near"), shot(1000.0, "far", (3.0, 11.0, 2.9))]
        floor, jumped = (rally_pressure(shots, [self.stood] * 2, self.reaches, off_floor)[0] for off_floor in (None, [None, Estimate(0.3, 0.02)]))
        self.assertAlmostEqual(floor.spare_ms.value - jumped.spare_ms.value, float(np.mean(jump_ms(Estimate(0.3, 0.02)))))

    def test_a_shot_that_came_down_is_measured_to_where_it_landed(self):
        landing = Landing((3.0, 6.0), ((0.01, 0.0), (0.0, 0.01)), 1500.0, "in", 0.99, 0.99)
        down = rally_pressure([replace(shot(1000.0, "far"), landing=landing)], [self.stood], self.reaches)[0]
        self.assertAlmostEqual(down.go_m.value, 3.0 - 1.3)
        self.assertAlmostEqual(down.time_ms.value, 500.0)
        self.assertIsNone(down.height_m)
        into_net = replace(landing, p_over_net=0.01)
        self.assertEqual(rally_pressure([replace(shot(1000.0, "far"), landing=into_net)], [self.stood], self.reaches), [None])

    def test_none_when_the_measurements_contradict_each_other(self):
        shots = [shot(0.0, "near"), shot(1000.0, "far", (3.0, 12.5, 1.0))]
        self.assertIsNone(rally_pressure(shots, [self.stood] * 2, self.reaches)[0])  # met 2.5 m from the hips
        flew = {**self.stood, "far": (np.array([3.0, 4.0]), CERTAIN)}  # 6 m away at the hit, then a metre from the contact
        self.assertIsNone(rally_pressure(shots, [flew, {**self.stood, "far": (np.array([3.0, 11.5]), CERTAIN)}], self.reaches)[0])

    def test_contradiction_within_the_uncertainty_is_not_one(self):
        loose = np.eye(2) * 0.5**2
        self.assertFalse(contradicts((np.array([3.0, 10.0]), CERTAIN), (np.array([3.0, 10.0]), loose), (3.0, 12.5), CERTAIN, REACH, 1000.0))
        self.assertTrue(contradicts((np.array([3.0, 10.0]), CERTAIN), (np.array([3.0, 10.0]), CERTAIN), (3.0, 12.5), CERTAIN, REACH, 1000.0))
        self.assertFalse(contradicts((np.array([3.0, 4.0]), loose * 16), (np.array([3.0, 11.5]), CERTAIN), (3.0, 12.5), CERTAIN, REACH, 1000.0))

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
            self.assertLess(np.median(np.abs(errors[:, 0])), 75.0 if name == "spare_ms" else 0.25, name)
        self.assertEqual(len(result["pressed"]), len(result["facts"]["go_m"]))
        self.assertLessEqual(result["setAside"], 1)
        self.assertTrue(all(0.0 <= p <= 1.0 for p, _ in result["pressed"]))


if __name__ == "__main__":
    unittest.main()
