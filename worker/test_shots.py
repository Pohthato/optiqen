# worker/test_shots.py
import unittest

import numpy as np

from audio.onsets import detect_onsets
from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, LONG_SERVICE_INSET_M, NET_Y_M, SHORT_SERVICE_M, SINGLES_WIDTH_M
from geometry.flight_fit import FlightFit
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, simulate, time_at_height
from hits import ONSET_GAP_MS, find_hits
from shots import call_landing, flight_facts, landing_area, net_height_m, rally_shots
from simulation.audio import SAMPLE_RATE
from test_hits import RALLY, clip_of, seen, times


def exact_fit(p0, v0, start_ms: float = 0.0, sigma: float = 1e-4) -> FlightFit:
    """A fitted flight that is the truth, with a tiny isotropic uncertainty."""
    return FlightFit(start_ms, np.asarray(p0, float), np.asarray(v0, float), FEATHER_TERMINAL_VELOCITY, np.eye(6) * sigma**2, 1.0, 20, 20)


class RuleTests(unittest.TestCase):
    def test_the_net_sags_from_the_posts_to_the_centre(self):
        self.assertAlmostEqual(net_height_m(SINGLES_WIDTH_M / 2), 1.524, places=6)
        self.assertAlmostEqual(net_height_m(-DOUBLES_MARGIN_M), 1.55, places=2)
        self.assertAlmostEqual(net_height_m(SINGLES_WIDTH_M + DOUBLES_MARGIN_M), 1.55, places=2)
        self.assertGreater(net_height_m(0.5), net_height_m(1.5))

    def test_a_rally_shot_must_land_in_the_opponents_half_singles_or_doubles(self):
        self.assertEqual(landing_area("singles", hitter="near"), (0.0, SINGLES_WIDTH_M, NET_Y_M, COURT_LENGTH_M))
        self.assertEqual(landing_area("singles", hitter="far"), (0.0, SINGLES_WIDTH_M, 0.0, NET_Y_M))
        self.assertEqual(landing_area("doubles", hitter="near"), (-DOUBLES_MARGIN_M, SINGLES_WIDTH_M + DOUBLES_MARGIN_M, NET_Y_M, COURT_LENGTH_M))

    def test_a_serve_must_land_in_the_diagonal_service_court(self):
        centre, short = SINGLES_WIDTH_M / 2, NET_Y_M + SHORT_SERVICE_M
        # A near server on the right of the centre line serves to the far court left of it.
        self.assertEqual(landing_area("singles", hitter="near", serve_from_x=3.5), (0.0, centre, short, COURT_LENGTH_M))
        self.assertEqual(landing_area("singles", hitter="far", serve_from_x=1.0), (centre, SINGLES_WIDTH_M, 0.0, NET_Y_M - SHORT_SERVICE_M))
        # Doubles: the wide court, but short of the doubles long service line.
        self.assertEqual(
            landing_area("doubles", hitter="near", serve_from_x=3.5),
            (-DOUBLES_MARGIN_M, centre, short, COURT_LENGTH_M - LONG_SERVICE_INSET_M),
        )

    def test_lines_are_in_and_a_landing_too_close_for_its_uncertainty_is_not_called(self):
        area = (0.0, SINGLES_WIDTH_M, NET_Y_M, COURT_LENGTH_M)
        sure = np.diag([0.01**2, 0.01**2])
        self.assertEqual(call_landing((2.0, 10.0), sure, area)[0], "in")
        self.assertEqual(call_landing((2.0, COURT_LENGTH_M - 0.005), sure * 0.01, area)[0], "in")  # on the line
        self.assertEqual(call_landing((-0.5, 10.0), sure, area)[0], "out")
        self.assertEqual(call_landing((2.0, COURT_LENGTH_M + 0.02), np.diag([0.1**2, 0.1**2]), area)[0], "too close to call")
        verdict, p_in = call_landing((2.0, COURT_LENGTH_M), np.diag([0.1**2, 0.1**2]), area)
        self.assertEqual(verdict, "too close to call")
        self.assertAlmostEqual(p_in, 0.5, delta=0.05)


class FlightFactTests(unittest.TestCase):
    def test_a_flights_facts_match_the_physics(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])  # a clear from the near half
        facts = flight_facts(exact_fit(p0, v0), hit_ms=0.0, hitter="near", court="singles", ends_ms=None)
        self.assertAlmostEqual(facts.speed_m_s.value, float(np.linalg.norm(v0)), places=2)
        self.assertAlmostEqual(facts.elevation_deg.value, float(np.degrees(np.arctan2(6.0, np.hypot(0.5, 18.0)))), places=1)
        path = simulate(p0, v0, np.arange(0.0, 3.0, 0.001))
        self.assertAlmostEqual(facts.apex_m.value, float(path[:, 2].max()), places=2)
        crossing = path[np.argmin(np.abs(path[:, 1] - NET_Y_M))]
        self.assertAlmostEqual(facts.net.height_m.value, float(crossing[2]), places=2)
        self.assertAlmostEqual(facts.net.clearance_m.value, float(crossing[2] - net_height_m(crossing[0])), places=2)
        landed = simulate(p0, v0, [time_at_height(p0, v0, 0.0)])[0]
        np.testing.assert_allclose(facts.landing.point, landed[:2], atol=0.01)
        self.assertEqual(facts.landing.call, "in")
        self.assertLess(facts.speed_m_s.sigma, 0.01)

    def test_a_received_flight_has_no_landing_and_its_uncertainty_follows_the_fit(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        received = flight_facts(exact_fit(p0, v0), hit_ms=0.0, hitter="near", court="singles", ends_ms=900.0)
        self.assertIsNone(received.landing)
        self.assertAlmostEqual(received.flight_ms, 900.0)
        unsure = flight_facts(exact_fit(p0, v0, sigma=0.05), hit_ms=0.0, hitter="near", court="singles", ends_ms=None)
        self.assertGreater(unsure.landing.covariance[1][1], 1e-4)
        self.assertGreater(unsure.speed_m_s.sigma, 0.01)

    def test_a_shot_that_falls_on_its_own_side_did_not_cross_the_net(self):
        p0, v0 = np.array([2.0, 5.0, 1.0]), np.array([0.0, 2.0, 1.0])  # dies before the net
        facts = flight_facts(exact_fit(p0, v0), hit_ms=0.0, hitter="near", court="singles", ends_ms=None)
        self.assertIsNone(facts.net)
        self.assertEqual(facts.landing.call, "out")


class RallyTests(unittest.TestCase):
    def test_a_rallys_shots_from_its_hits(self):
        clip = clip_of()
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        analysis = find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, onsets)
        shots = rally_shots(analysis, seen(clip), times(len(clip.cameras)), clip.cameras, court="singles")
        self.assertEqual(len(shots), len(RALLY.contacts))
        self.assertEqual([s.hitter for s in shots], [c.hitter for c in RALLY.contacts])
        self.assertTrue(shots[0].serve and not any(s.serve for s in shots[1:]))
        self.assertTrue(all(s.landing is None for s in shots[:-1]))
        last = RALLY.flights[-1]
        landed = simulate(last.p0, last.v0, [last.end_time - last.start_time])[0]
        self.assertLess(float(np.hypot(*(np.asarray(shots[-1].landing.point) - landed[:2]))), 0.3)
        self.assertEqual(shots[-1].landing.call, "in")
        for shot, contact in zip(shots, RALLY.contacts):
            self.assertLess(float(np.linalg.norm(np.asarray(shot.contact) - contact.position)), 0.5)
        # A drag known from a whole session is used as given, and makes the facts surer.
        known = rally_shots(analysis, seen(clip), times(len(clip.cameras)), clip.cameras, court="singles", drag=(6.7, 0.02))
        self.assertEqual((known[0].terminal_velocity.value, known[0].terminal_velocity.sigma), (6.7, 0.02))
        self.assertLessEqual(np.trace(np.asarray(known[-1].landing.covariance)), np.trace(np.asarray(shots[-1].landing.covariance)) + 1e-9)


if __name__ == "__main__":
    unittest.main()
