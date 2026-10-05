# worker/test_shuttle_physics.py
import math
import unittest

import numpy as np

from geometry.shuttle_physics import GRAVITY, simulate, solve_launch, time_at_height


class SimulateTests(unittest.TestCase):
    def test_without_drag_it_is_an_exact_parabola(self):
        p0, v0 = np.array([1.0, 2.0, 1.5]), np.array([3.0, 10.0, 8.0])
        times = np.array([0.0, 0.5, 1.0])
        expected = p0 + np.outer(times, v0) - 0.5 * GRAVITY * np.outer(times**2, [0, 0, 1])
        np.testing.assert_allclose(simulate(p0, v0, times, terminal_velocity=None), expected, atol=1e-9)

    def test_vertical_drop_matches_the_analytic_drag_solution(self):
        vt = 6.7
        times = np.array([0.5, 1.0, 2.0])
        expected_z = 20.0 - (vt**2 / GRAVITY) * np.log(np.cosh(GRAVITY * times / vt))
        z = simulate([0, 0, 20.0], [0, 0, 0], times, terminal_velocity=vt)[:, 2]
        np.testing.assert_allclose(z, expected_z, atol=1e-4)

    def test_falling_speed_approaches_terminal_velocity(self):
        z = simulate([0, 0, 50.0], [0, 0, 0], [3.0, 3.01])[:, 2]
        self.assertAlmostEqual((z[0] - z[1]) / 0.01, 6.7, delta=0.07)

    def test_time_zero_is_the_launch_point_and_times_must_increase(self):
        np.testing.assert_allclose(simulate([1, 2, 3], [5, 5, 5], [0.0])[0], [1, 2, 3])
        with self.assertRaises(ValueError):
            simulate([0, 0, 1], [1, 1, 1], [0.5, 0.2])


class TimeAtHeightTests(unittest.TestCase):
    def test_descending_crossing_matches_the_parabola(self):
        expected = (10 + math.sqrt(100 - 4 * (GRAVITY / 2) * 1.0)) / GRAVITY
        t = time_at_height([0, 0, 1.0], [0, 0, 10.0], 2.0, terminal_velocity=None)
        self.assertAlmostEqual(t, expected, places=4)

    def test_height_above_the_apex_is_never_reached(self):
        self.assertIsNone(time_at_height([0, 0, 1.0], [0, 0, 5.0], 10.0))


class SolveLaunchTests(unittest.TestCase):
    def test_a_clear_lands_on_its_target_with_a_realistic_speed(self):
        p0, target = np.array([3.5, 1.4, 2.4]), np.array([1.5, 12.9, 0.0])
        v0 = solve_launch(p0, target, 1.9)
        np.testing.assert_allclose(simulate(p0, v0, [1.9])[0], target, atol=0.01)
        self.assertTrue(15 < np.linalg.norm(v0) < 70)

    def test_non_positive_flight_time_is_refused(self):
        with self.assertRaises(ValueError):
            solve_launch([0, 0, 1], [0, 5, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
