# worker/test_rally_fit.py
import unittest

import numpy as np

from geometry.court_model import NET_Y_M
from geometry.flight_fit import fit_flight
from geometry.rally_fit import CONTACT_SIGMA_M, TERMINAL_PRIOR_SIGMA, Reach, Seen, fit_rally, terminal_velocity
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
        cls.result = fit_rally(cls.solo, cls.links)
        cls.joint = cls.result.flights

    def test_neighbouring_flights_meet_where_and_when_the_fit_says(self):
        for i, at in self.result.links:
            met, sigma = self.result.meetings[i]
            gap = self.joint[i].position_at(met) - self.joint[i + 1].position_at(met)
            self.assertLess(float(np.linalg.norm(gap)), 3 * CONTACT_SIGMA_M)
            self.assertLess(abs(met - at), 4 * sigma + 1.0)  # near the measured hit, as sure as it says

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
        alone = fit_rally(self.solo[:2], []).flights
        for seen, fit in zip(self.solo[:2], alone):
            np.testing.assert_allclose(fit.position_at(seen.times_ms[-1]), seen.fit.position_at(seen.times_ms[-1]), atol=0.02)

    def test_the_landing_of_the_last_flight_is_within_30_cm(self):
        flight, terminal = self.flights[-1][0], self.flights[-1][1]
        landed = simulate(flight.p0, flight.v0, [flight.end_time - flight.start_time], terminal)[0]
        self.assertLess(float(np.linalg.norm(self.joint[-1].position_at(flight.end_time * 1000.0) - landed)), 0.3)


class ReachTests(unittest.TestCase):
    """The racket meets the shuttle within reach of its hitter, whose feet put them on the floor
    far more surely than a far flight's own depth."""

    @classmethod
    def setUpClass(cls):
        cls.flights = observed_flights(seed=4, view="behind", fps=30.0)
        cls.solo = solo_fits(cls.flights)

    def at(self, i) -> float:
        return self.flights[i][0].start_time * 1000.0

    def test_a_contact_already_within_reach_is_left_to_its_detections(self):
        # Where its detections put it: with hitters whose reach never binds (the rally fit weighs
        # the sightings just after a hit less than a flight's own fit does).
        free = fit_rally(self.solo, [], reaches=[Reach(self.at(i), seen.fit.position_at(self.at(i))[:2], np.eye(2) * 0.01, 50.0, 0.02) for i, seen in enumerate(self.solo)])
        reaches = [Reach(self.at(i), fit.position_at(self.at(i))[:2] + (0.3, 0.3), np.eye(2) * 0.01, 1.0, 0.02) for i, fit in enumerate(free.flights)]
        held = fit_rally(self.solo, [], reaches=reaches)
        self.assertEqual(held.reaches, reaches)
        for i, (alone, fit) in enumerate(zip(free.flights, held.flights)):  # to well within its own uncertainty
            spread = float(np.sqrt(np.trace(alone.position_covariance(self.at(i)))))
            self.assertLess(float(np.linalg.norm(fit.position_at(self.at(i)) - alone.position_at(self.at(i)))), max(0.02, 0.1 * spread))

    def test_a_contact_the_detections_cannot_place_is_held_within_reach_of_its_hitter(self):
        # The far player's clear, seen from behind the near baseline: its start is metres out alone.
        i = max(range(len(self.solo)), key=lambda k: np.trace(self.solo[k].fit.position_covariance(self.at(k))))
        truth = np.asarray(self.flights[i][0].p0, dtype=float)
        hitter = Reach(self.at(i), truth[:2] + (0.0, 0.4), np.eye(2) * 0.04, 1.2, 0.02)  # the hips just behind the contact
        alone = float(np.linalg.norm(self.solo[i].fit.position_at(self.at(i)) - truth))
        held = fit_rally([self.solo[i]], [], reaches=[hitter])
        contact = held.flights[0].position_at(self.at(i))
        self.assertEqual(held.flights[0].clock, 1.0)  # the flight model is right here: nothing says otherwise
        self.assertGreater(alone, 2.0)
        self.assertLess(float(np.linalg.norm(contact[:2] - hitter.point)), hitter.radius + 0.2)
        self.assertLess(float(np.linalg.norm(contact - truth)), 0.5 * alone)
        spread = float(np.sqrt(np.trace(held.flights[0].position_covariance(self.at(i)))))
        self.assertLess(float(np.linalg.norm(contact - truth)), 3 * spread)  # no more sure than it should be

    def test_a_flight_that_cannot_reach_its_hitter_on_the_clips_clock_runs_on_its_own(self):
        # What the real clips' far flights showed: by the clip's times the flight falls as if
        # under stronger gravity, and its own fit starts it metres short of its hitter. The far
        # player's longest-seen flight: a short one is held in by the reach on any clock.
        flights = observed_flights(seed=7, view="behind", fps=30.0)
        flight, terminal, times_ms, pixels, cameras = max((f for f in flights if f[0].p0[1] > NET_Y_M), key=lambda f: len(f[2]))
        start = flight.start_time * 1000.0
        slow = start + (times_ms - start) / 1.08
        seen = Seen(slow, pixels, cameras, fit_flight(slow, pixels, cameras, start_ms=start, terminal_velocity=terminal))
        truth = np.asarray(flight.p0, dtype=float)
        hitter = Reach(start, truth[:2] + (0.0, 0.4), np.eye(2) * 0.04, 1.2, 0.02)
        held = fit_rally([seen], [], reaches=[hitter])
        fit = held.flights[0]
        alone = float(np.linalg.norm(seen.fit.position_at(start) - truth))
        self.assertGreater(alone, 2.0)
        self.assertIsNotNone(held.reaches[0])
        self.assertGreater(fit.clock, 1.03)  # the truth: 1.08
        self.assertLess(float(np.linalg.norm(fit.position_at(start)[:2] - hitter.point)), hitter.radius + 0.5)
        self.assertLess(float(np.linalg.norm(fit.position_at(start) - truth)), 0.5 * alone)
        self.assertEqual(len(fit.covariance), 7)  # the clock's own doubt is carried
        self.assertLess(float(np.linalg.norm(fit.position_at(start) - truth)), 3 * float(np.sqrt(np.trace(fit.position_covariance(start)))))

    def test_a_hitter_off_the_contacts_line_of_sight_is_let_go(self):
        # The other player, or someone else tracked: wherever along its line of sight the contact
        # was, it was not within their reach.
        i = min(range(len(self.solo)), key=lambda k: np.trace(self.solo[k].fit.position_covariance(self.at(k))))
        start = self.solo[i].fit.position_at(self.at(i))
        elsewhere = Reach(self.at(i), start[:2] + (3.0, 0.0), np.eye(2) * 0.01, 1.0, 0.02)
        held = fit_rally([self.solo[i]], [], reaches=[elsewhere])
        self.assertEqual(held.reaches, [None])
        np.testing.assert_allclose(held.flights[0].position_at(self.at(i)), start, atol=0.02)


class DragTests(unittest.TestCase):
    def test_a_shuttles_drag_is_measured_from_the_rallys_long_flights(self):
        # A faster shuttle than the usual feather one, fitted first as if it were the usual one.
        flights = observed_flights(seed=9, view="corner", fps=60.0, terminal_velocity=7.2)
        seen = [
            Seen(times_ms, pixels, cameras, fit_flight(times_ms, pixels, cameras, start_ms=flight.start_time * 1000.0, terminal_velocity=6.7))
            for flight, _, times_ms, pixels, cameras in flights
        ]
        value, sigma, used = terminal_velocity([s for s in seen if s.fit is not None])
        self.assertGreaterEqual(used, 2)
        self.assertAlmostEqual(value, 7.2, delta=0.3)
        self.assertLess(sigma, TERMINAL_PRIOR_SIGMA)

    def test_with_nothing_long_to_go_on_the_usual_shuttle_is_assumed(self):
        self.assertEqual(terminal_velocity([]), (6.7, TERMINAL_PRIOR_SIGMA, 0))


if __name__ == "__main__":
    unittest.main()
