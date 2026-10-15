# worker/test_body.py
import unittest
from types import SimpleNamespace

import numpy as np

from body import BASE_FROM_NET_M, RACKET_REACH_M, body_at, centre_of_mass, hit_body, rally_body_facts, readiness, standing_reach
from evaluation.body_benchmark import FACTS, score
from geometry.court_model import NET_Y_M, SINGLES_WIDTH_M
from geometry.skeleton import ANKLE_HEIGHT_M, Body
from shots import Estimate
from simulation.people import SEGMENTS, body

STATURE = 1.75
LENGTH = {name: fraction * STATURE for name, fraction in SEGMENTS.items()}
UP = np.array([0.0, 0.0, 1.0])


def standing(pelvis=(2.0, 3.0, 0.875), turn=0.0, lean=0.0, feet=None, right_hand=None, lift=0.0) -> np.ndarray:
    """A near player's body (facing +y, the net): feet beside the hips, hands down in front."""
    pelvis = np.asarray(pelvis, dtype=float) + lift * UP
    if feet is None:
        feet = (np.array([pelvis[0] - 0.16 * STATURE, pelvis[1], lift]), np.array([pelvis[0] + 0.16 * STATURE, pelvis[1], lift]))
    shoulder = pelvis + 0.29 * STATURE * UP
    hands = (shoulder + np.array([-0.3, 0.2, -0.3]), shoulder + np.array([0.3, 0.2, -0.3]) if right_hand is None else np.asarray(right_hand, dtype=float))
    return body(STATURE, pelvis, np.pi / 2, feet, hands, lean, turn)


def exact(joints: np.ndarray) -> list[Body]:
    return [Body(joints, np.zeros(17), 0.0)]


class AtContactTests(unittest.TestCase):
    def facts(self, joints, contact):
        reach = standing_reach(exact(standing()))
        return hit_body(exact(joints), [0.0], 0.0, "near", contact, (0.0, 0.0, 0.0), reach)

    def test_standing_reach_is_the_player_s_own(self):
        reach = standing_reach(exact(standing()) * 3)
        own = ANKLE_HEIGHT_M + LENGTH["shank"] + LENGTH["thigh"] + LENGTH["torso"] + LENGTH["upper_arm"] + LENGTH["forearm"] + RACKET_REACH_M
        self.assertAlmostEqual(reach.value, own, places=6)
        self.assertGreater(reach.sigma, 0.01)  # the bones may all run a little long

    def test_where_and_how_high_against_the_body(self):
        joints = standing()
        hips = (joints[11] + joints[12]) / 2
        contact = hips + np.array([0.3, 0.4, 1.9])  # 40 cm in front, 30 cm to the right, 1.9 m above the hips
        facts = self.facts(standing(right_hand=contact - np.array([0, 0, 0.5])), contact)
        self.assertEqual(facts.arm, "right")
        self.assertAlmostEqual(facts.ahead_m.value, 0.4, places=6)
        self.assertAlmostEqual(facts.aside_m.value, 0.3, places=6)
        reach = standing_reach(exact(standing())).value
        self.assertAlmostEqual(facts.above_reach_m.value, contact[2] - reach, places=6)
        self.assertAlmostEqual(facts.hips_height_m.value, 0.875, places=6)
        self.assertAlmostEqual(facts.off_floor_m.value, 0.0, places=6)

    def test_the_elbow_s_angle(self):
        joints = standing()
        shoulder = joints[6]
        for want in (60.0, 120.0, 170.0):
            # The wrist this far from the shoulder bends the elbow to this angle.
            reach = np.sqrt(LENGTH["upper_arm"] ** 2 + LENGTH["forearm"] ** 2 - 2 * LENGTH["upper_arm"] * LENGTH["forearm"] * np.cos(np.radians(want)))
            hand = shoulder + reach * np.array([0.0, 0.6, 0.8])
            facts = self.facts(standing(right_hand=hand), hand + np.array([0.0, 0.3, 0.4]))
            self.assertAlmostEqual(facts.elbow_deg.value, want, places=4)

    def test_side_on_is_the_racket_shoulder_back(self):
        for degrees in (-20.0, 0.0, 35.0):
            joints = standing(turn=np.radians(-degrees))  # clockwise from above: the right shoulder back
            facts = self.facts(joints, joints[10] + np.array([0.0, 0.3, 0.5]))
            self.assertAlmostEqual(facts.side_on_deg.value, degrees, places=4)

    def test_a_lunge_s_front_knee_and_a_jump_s_height(self):
        hip_to_ankle = 0.62
        want = np.degrees(np.arccos((LENGTH["thigh"] ** 2 + LENGTH["shank"] ** 2 - hip_to_ankle**2) / (2 * LENGTH["thigh"] * LENGTH["shank"])))
        pelvis = np.array([2.0, 3.0, 0.55])
        right_hip = pelvis + np.array([LENGTH["hip_half"], 0.0, 0.0])
        drop = right_hip[2] - ANKLE_HEIGHT_M
        ahead = np.sqrt(hip_to_ankle**2 - drop**2)
        feet = (np.array([pelvis[0] - 0.16 * STATURE, pelvis[1] - 0.5, 0.0]), np.array([right_hip[0], pelvis[1] + ahead, 0.0]))
        joints = standing(pelvis=pelvis, feet=feet)
        facts = self.facts(joints, joints[10] + np.array([0.0, 0.4, 0.0]))
        self.assertAlmostEqual(facts.front_knee_deg.value, want, places=3)
        jumping = standing(lift=0.35)
        self.assertAlmostEqual(self.facts(jumping, jumping[10] + UP).off_floor_m.value, 0.35, places=6)

    def test_the_centre_of_mass(self):
        joints = standing()
        com = centre_of_mass(joints)
        self.assertAlmostEqual(com[0], 2.0, places=6)  # a left-right symmetric body: on its middle
        self.assertTrue(0.5 * STATURE < com[2] < 0.6 * STATURE)
        moved = centre_of_mass(joints + np.array([1.0, -2.0, 0.5]))
        np.testing.assert_allclose(moved, com + np.array([1.0, -2.0, 0.5]), atol=1e-12)
        leaning = self.facts(standing(lean=0.4), joints[10] + UP).weight_ahead_m.value
        self.assertGreater(leaning, self.facts(joints, joints[10] + UP).weight_ahead_m.value + 0.05)

    def test_uncertainty_follows_the_joints_and_the_contact(self):
        joints = standing()
        sharp = hit_body(exact(joints), [0.0], 0.0, "near", joints[10] + UP, (0.0, 0.0, 0.0), Estimate(2.6, 0.0))
        loose = hit_body([Body(joints, np.full(17, 0.06), 0.0)], [0.0], 0.0, "near", joints[10] + UP, (0.05, 0.05, 0.05), Estimate(2.6, 0.0))
        self.assertLess(sharp.elbow_deg.sigma, 1e-9)
        self.assertGreater(loose.elbow_deg.sigma, 1.0)
        self.assertAlmostEqual(loose.above_reach_m.sigma, 0.05, delta=0.008)

    def test_between_frames(self):
        a, b = standing(), standing(pelvis=(2.0, 3.2, 0.875))
        middle = body_at([Body(a, np.full(17, 0.02), 0.0), Body(b, np.full(17, 0.02), 0.0)], [0.0, 33.3], 16.65)
        np.testing.assert_allclose(middle.joints, (a + b) / 2, atol=1e-9)
        self.assertTrue(np.all(middle.sigma[:5] > 0.02))  # widened by the 10 cm each joint moves in between
        self.assertIsNone(body_at([Body(a, np.zeros(17), 0.0), None], [0.0, 33.3], 16.65))


def hips_moving(profile, times_s, base_offset=(0.3, 0.4)) -> list[Body]:
    """A standing body at base (plus an offset) whose height follows profile(t)."""
    pelvis = np.array([SINGLES_WIDTH_M / 2 + base_offset[0], NET_Y_M - BASE_FROM_NET_M + base_offset[1], 0.875])
    joints = standing(pelvis=pelvis)
    return [Body(joints + profile(t) * UP, np.full(17, 0.002), 0.0) for t in times_s]


class ReadinessTests(unittest.TestCase):
    times = np.arange(0.0, 2.0, 1 / 60)

    def ready(self, profile, hit_s=1.0):
        bodies = hips_moving(profile, self.times)
        return readiness(bodies, self.times * 1000.0, hit_s * 1000.0, "near")

    def test_a_split_step_s_drop_and_timing(self):
        landing = 1.04  # 40 ms after the opponent's hit

        def split(t):
            hop = 0.05 * np.sin(np.pi * (t - (landing - 0.2)) / 0.2) if landing - 0.2 <= t <= landing else 0.0
            dip = -0.07 * np.sin(np.pi * (t - landing) / 0.3) if landing <= t <= landing + 0.3 else 0.0
            return hop + dip

        ready = self.ready(split)
        self.assertAlmostEqual(ready.from_base_m.value, 0.5, places=6)
        self.assertAlmostEqual(ready.split_drop_m.value, 0.12, delta=0.01)
        self.assertAlmostEqual(ready.split_lowest_ms.value, 190.0, delta=10.0)  # the dip's bottom, 150 ms after landing

    def test_no_split_step(self):
        still = self.ready(lambda t: 0.0)
        self.assertLess(still.split_drop_m.value, 0.01)
        self.assertIsNone(still.split_lowest_ms)
        sinking = self.ready(lambda t: -0.1 * min(max(t - 0.9, 0.0), 0.5))  # going low for the next shot, not landing
        self.assertIsNone(sinking.split_lowest_ms)

    def test_needs_the_body_through_the_split_step(self):
        bodies = hips_moving(lambda t: 0.0, self.times)
        bodies[61] = None
        self.assertIsNone(readiness(bodies, self.times * 1000.0, 1000.0, "near"))


class RallyTests(unittest.TestCase):
    def test_each_shot_and_the_readiness_before_it(self):
        times = np.arange(0.0, 3.0, 1 / 60)
        bodies = {"near": hips_moving(lambda t: 0.0, times), "far": [None] * len(times)}
        contact = bodies["near"][0].joints[10] + UP
        shots = [SimpleNamespace(hit_ms=500.0, hitter="near", contact=contact, contact_sigma=(0.02,) * 3),
                 SimpleNamespace(hit_ms=1200.0, hitter="far", contact=(2.0, 10.0, 2.0), contact_sigma=(0.02,) * 3),
                 SimpleNamespace(hit_ms=2000.0, hitter="near", contact=contact, contact_sigma=(0.02,) * 3)]
        facts = rally_body_facts(shots, bodies, times * 1000.0)
        self.assertIsNone(facts[0].ready)  # a serve answers nothing
        self.assertIsNone(facts[1])  # the far player was never seen
        self.assertIsNotNone(facts[2].ready)
        self.assertAlmostEqual(facts[2].ready.from_base_m.value, 0.5, places=6)


class BenchmarkTests(unittest.TestCase):
    def test_fitted_facts_against_the_truth(self):
        result = score(("side", "near", 30.0, 3))
        self.assertGreater(len(result["facts"]["elbow_deg"]), 2)
        for name in FACTS:
            errors = np.abs(np.array(result["facts"][name]))[:, 0]
            limit = 25.0 if name.endswith("_deg") else 0.25
            self.assertLess(np.median(errors), limit, name)
        self.assertLess(abs(result["reach"][0]), 0.05)


if __name__ == "__main__":
    unittest.main()
