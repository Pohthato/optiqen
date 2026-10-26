# worker/test_shot_names.py
import unittest

import numpy as np

from evaluation.hits_benchmark import Scenario
from evaluation.shot_names_benchmark import TEST_FIRST_SEED, learning_set, measure_scenario
from geometry.court_model import NET_Y_M
from geometry.shuttle_physics import simulate
from shot_names import DRAWS, FACTS, facts, measured_facts, nominal_facts, true_facts
from shots import flight_facts
from simulation.rally import SHOT_TYPES, canned_rally
from test_shots import exact_fit

COLUMN = {name: k for k, name in enumerate(FACTS)}


class FactTests(unittest.TestCase):
    def test_the_facts_of_true_flights_are_exact(self):
        rally = canned_rally()
        names, rows = true_facts([rally, rally])
        self.assertEqual(names, [c.kind for c in rally.contacts] * 2)
        np.testing.assert_array_equal(rows[: len(rally.contacts)], rows[len(rally.contacts):])
        for n, (row, contact, flight) in enumerate(zip(rows, rally.contacts, rally.flights)):
            side = 1.0 if contact.hitter == "near" else -1.0
            self.assertAlmostEqual(row[COLUMN["contactHeightM"]], contact.position[2], places=9)
            self.assertAlmostEqual(row[COLUMN["contactFromNetM"]], side * (NET_Y_M - contact.position[1]), places=9)
            self.assertAlmostEqual(row[COLUMN["speedMS"]], float(np.linalg.norm(flight.v0)), places=9)
            path = simulate(flight.p0, flight.v0, np.arange(0.0, flight.end_time - flight.start_time, 0.001))
            self.assertAlmostEqual(row[COLUMN["apexAboveM"]], float(path[:, 2].max()) - contact.position[2], places=2)
            self.assertAlmostEqual(row[COLUMN["flightS"]], flight.end_time - flight.start_time, delta=0.003)
            if n + 1 < len(rally.contacts):  # played again where the next contact is
                following = rally.contacts[n + 1].position
                self.assertAlmostEqual(row[COLUMN["endPastNetM"]], side * (following[1] - NET_Y_M), delta=0.02)
                self.assertAlmostEqual(row[COLUMN["endHeightM"]], following[2], delta=0.02)
            else:  # the last lands
                self.assertAlmostEqual(row[COLUMN["endPastNetM"]], side * (rally.landing[1] - NET_Y_M), delta=0.02)
                self.assertAlmostEqual(row[COLUMN["endHeightM"]], 0.0, places=6)

    def test_a_far_players_shot_reads_like_the_same_shot_from_the_near_half(self):
        p, v = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        mirrored_p, mirrored_v = np.array([p[0], 2 * NET_Y_M - p[1], p[2]]), v * [1.0, -1.0, 1.0]
        near = facts(p, v, True, 0.9, 6.7)
        far = facts(mirrored_p, mirrored_v, False, 0.9, 6.7)
        np.testing.assert_allclose(near, far, atol=1e-9)

    def test_a_flight_on_the_floor_before_it_was_played_again_ends_there(self):
        p, v = np.array([2.0, 5.0, 1.0]), np.array([0.0, 8.0, 2.0])
        row = facts(p, v, True, 2.5, 6.7)[0]
        self.assertAlmostEqual(row[COLUMN["endHeightM"]], 0.0, places=6)
        self.assertLess(row[COLUMN["flightS"]], 2.5)

    def test_a_measured_shot_is_drawn_whole_from_its_uncertainty(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        sure = flight_facts(exact_fit(p0, v0, sigma=1e-4), 0.0, "near", "singles", ends_ms=900.0)
        unsure = flight_facts(exact_fit(p0, v0, sigma=0.1), 0.0, "near", "singles", ends_ms=900.0)
        rng = np.random.default_rng(0)
        nominal = nominal_facts(sure)
        np.testing.assert_allclose(nominal, facts(p0, v0, True, 0.9, 6.7)[0], atol=1e-6)
        drawn = measured_facts(sure, rng)
        np.testing.assert_allclose(drawn.mean(axis=0), nominal, atol=0.05)
        spread = measured_facts(unsure, rng).std(axis=0)
        flown = [COLUMN[name] for name in FACTS if name != "flightS"]  # played again at a known time
        self.assertTrue((spread[flown] > drawn.std(axis=0)[flown]).all())
        self.assertAlmostEqual(spread[COLUMN["flightS"]], 0.0, places=9)
        self.assertAlmostEqual(spread[COLUMN["speedMS"]], unsure.speed_m_s.sigma, delta=0.2 * unsure.speed_m_s.sigma)


class SetTests(unittest.TestCase):
    def test_learning_shots_are_named_repeatably_and_never_serves(self):
        names, rows = learning_set(30)
        self.assertEqual(rows.shape, (len(names), len(FACTS)))
        self.assertNotIn("serve", names)
        self.assertLessEqual(set(names), set(SHOT_TYPES))
        self.assertEqual(learning_set(30)[0], names)
        self.assertNotEqual(learning_set(30, first_seed=30)[0], names)

    def test_a_held_out_scenarios_found_shots_keep_their_truth_and_draws(self):
        rows = measure_scenario(Scenario(seed=TEST_FIRST_SEED, view="behind", fps=30.0))
        self.assertGreater(len(rows), 1)
        for row in rows:
            self.assertIn(row["name"], SHOT_TYPES)
            self.assertEqual(row["draws"].shape, (DRAWS, len(FACTS)))
            self.assertLess(abs(row["nominal"][COLUMN["contactHeightM"]] - row["truth"][COLUMN["contactHeightM"]]), 0.5)


if __name__ == "__main__":
    unittest.main()
