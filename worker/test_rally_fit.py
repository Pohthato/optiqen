# worker/test_rally_fit.py
import unittest

import numpy as np

from geometry.flight_fit import fit_flight
from geometry.rally_fit import CONTACT_SIGMA_M, TERMINAL_PRIOR_SIGMA, Seen, fit_rally, terminal_velocity
from geometry.shuttle_physics import simulate
from test_flight_fit import observed_flights


def solo_fits(flights):
    seen = []
    for flight, terminal, times_ms, pixels, cameras in flights:
        fit = fit_flight(times_ms, pixels, cameras, start_ms=flight.start_time * 1000.0, terminal_velocity=terminal)
        seen.append(Seen(times_ms, pixels, cameras, fit))
    return seen


def start_error(flight, terminal, fit) -> float:
    return float(np.linalg.norm(fit.position_at(flight.start_time * 1000.0) - np.asarray(flight.p0)))


class RallyFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # From behind the near baseline the far player's flights are the poorly seen ones.
        cls.flights = observed_flights(seed=4, view="behind", fps=30.0)
        cls.solo = solo_fits(cls.flights)
        cls.links = [(i, cls.flights[i + 1][0].start_time * 1000.0) for i in range(len(cls.flights) - 1)]
        cls.joint = fit_rally(cls.solo, cls.links)

    def test_neighbouring_flights_meet_at_their_hit(self):
        for i, at in self.links:
            gap = self.joint[i].position_at(at) - self.joint[i + 1].position_at(at)
            self.assertLess(float(np.linalg.norm(gap)), 3 * CONTACT_SIGMA_M)

    def test_together_the_contact_points_are_found_better_than_alone_and_honestly(self):
        alone = [start_error(f[0], f[1], s.fit) for f, s in zip(self.flights[1:], self.solo[1:])]
        together = [start_error(f[0], f[1], fit) for f, fit in zip(self.flights[1:], self.joint[1:])]
        self.assertLess(max(together), 0.4)
        self.assertLess(float(np.median(together)), float(np.median(alone)))
        for flight, fit, error in zip(self.flights[1:], self.joint[1:], together):
            spread = float(np.sqrt(np.trace(fit.position_covariance(flight[0].start_time * 1000.0))))
            self.assertLess(error, 3 * spread)  # no more sure than it should be

    def test_a_well_seen_neighbour_shrinks_a_poorly_seen_flights_uncertainty(self):
        def spread(fit, at):
            return float(np.sqrt(np.trace(fit.position_covariance(at))))

        worst = max(range(1, len(self.flights)), key=lambda i: spread(self.solo[i].fit, self.links[i - 1][1]))
        at = self.links[worst - 1][1]
        self.assertLess(spread(self.joint[worst], at), 0.5 * spread(self.solo[worst].fit, at))

    def test_flights_without_hand_overs_are_left_as_they_were(self):
        alone = fit_rally(self.solo[:2], [])
        for seen, fit in zip(self.solo[:2], alone):
            np.testing.assert_allclose(fit.position_at(seen.times_ms[-1]), seen.fit.position_at(seen.times_ms[-1]), atol=0.02)

    def test_the_landing_of_the_last_flight_is_within_30_cm(self):
        flight, terminal = self.flights[-1][0], self.flights[-1][1]
        landed = simulate(flight.p0, flight.v0, [flight.end_time - flight.start_time], terminal)[0]
        self.assertLess(float(np.linalg.norm(self.joint[-1].position_at(flight.end_time * 1000.0) - landed)), 0.3)


class DragTests(unittest.TestCase):
    def test_a_shuttles_drag_is_measured_from_the_rallys_long_flights(self):
        # A faster shuttle than the usual feather one, fitted first as if it were the usual one.
        flights = observed_flights(seed=9, view="corner", fps=60.0, terminal_velocity=7.2)
        seen = [
            Seen(times_ms, pixels, cameras, fit_flight(times_ms, pixels, cameras, start_ms=flight.start_time * 1000.0, terminal_velocity=6.7))
            for flight, _, times_ms, pixels, cameras in flights
        ]
        value, sigma = terminal_velocity([s for s in seen if s.fit is not None])
        self.assertAlmostEqual(value, 7.2, delta=0.3)
        self.assertLess(sigma, TERMINAL_PRIOR_SIGMA)

    def test_with_nothing_long_to_go_on_the_usual_shuttle_is_assumed(self):
        self.assertEqual(terminal_velocity([]), (6.7, TERMINAL_PRIOR_SIGMA))


if __name__ == "__main__":
    unittest.main()
