# worker/test_shots.py
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from audio.onsets import detect_onsets
from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, LONG_SERVICE_INSET_M, NET_Y_M, SHORT_SERVICE_M, SINGLES_WIDTH_M
from geometry.flight_fit import FlightFit
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, simulate, solve_launch, time_at_height
from hits import ONSET_GAP_MS, Piece, find_hits
from shots import CALIBRATION_SCALE, LYING_MS, Estimate, Landing, _a_shot, _judged, _last_of_rally, _may_be_on_the_net, _may_be_sent_back, _not_to_be_called, _played_again, _went_on, call_landing, flight_facts, landing_area, net_height_m, rally_end, rally_shots
from simulation.audio import SAMPLE_RATE
from simulation.clip import make_clip
from simulation.rally import Shot, build_rally
from test_hits import CAMERA, FPS, RALLY, SIZE, clip_of, seen, times


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
        self.assertLess(facts.speed_m_s.sigma, CALIBRATION_SCALE * facts.speed_m_s.value + 0.01)  # little beyond the calibration's 1 %

    def test_a_received_flight_has_no_landing_and_its_uncertainty_follows_the_fit(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        received = flight_facts(exact_fit(p0, v0), hit_ms=0.0, hitter="near", court="singles", ends_ms=900.0)
        self.assertIsNone(received.landing)
        self.assertAlmostEqual(received.flight_ms, 900.0)
        unsure = flight_facts(exact_fit(p0, v0, sigma=0.05), hit_ms=0.0, hitter="near", court="singles", ends_ms=None)
        self.assertGreater(unsure.landing.covariance[1][1], 1e-4)
        self.assertGreater(unsure.speed_m_s.sigma, 0.01)

    def test_the_launch_carries_the_contact_and_the_velocity_with_their_uncertainty(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        facts = flight_facts(exact_fit(p0, v0, sigma=0.05), hit_ms=0.0, hitter="near", court="singles", ends_ms=900.0)
        np.testing.assert_allclose(facts.velocity, v0, atol=1e-6)
        covariance = np.asarray(facts.launch_covariance)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.diag(covariance)[:3]), facts.contact_sigma, rtol=1e-6)
        along = v0 / np.linalg.norm(v0)
        self.assertAlmostEqual(float(np.sqrt(along @ covariance[3:, 3:] @ along)), facts.speed_m_s.sigma, delta=1e-3)

    def test_a_shot_that_falls_on_its_own_side_did_not_cross_the_net(self):
        p0, v0 = np.array([2.0, 5.0, 1.0]), np.array([0.0, 2.0, 1.0])  # dies before the net
        facts = flight_facts(exact_fit(p0, v0), hit_ms=0.0, hitter="near", court="singles", ends_ms=None)
        self.assertIsNone(facts.net)
        self.assertEqual(facts.landing.call, "out")


class CallTests(unittest.TestCase):
    def serve(self, x: float, sigma: float):
        """A short serve from the near half, its contact x with this uncertainty, landing well inside
        the far left service court."""
        p0 = np.array([x, 3.6, 1.0])
        target = np.array([1.2, 9.2, 0.0])
        from geometry.shuttle_physics import solve_launch
        v0 = solve_launch(p0, target, 1.1)
        return flight_facts(exact_fit(p0, v0, sigma=sigma), 0.0, "near", "singles", None, serve=True)

    def test_a_serve_is_judged_against_the_diagonal_service_court(self):
        self.assertEqual(self.serve(3.5, 1e-4).landing.call, "in")  # from the right, into the left court
        self.assertEqual(self.serve(1.5, 1e-4).landing.call, "out")  # from the left, so the right court was the target

    def test_a_server_on_the_centre_line_leaves_the_court_unsure(self):
        unsure = self.serve(2.6, 0.08)  # 1 cm from the centre line, known to 8 cm (plus the calibration)
        self.assertEqual(unsure.landing.call, "too close to call")

    def test_a_shot_into_the_net_is_out(self):
        p0, v0 = np.array([2.5, 5.0, 1.0]), np.array([0.0, 9.0, 1.2])  # low and flat: it meets the net
        facts = flight_facts(exact_fit(p0, v0), 0.0, "near", "singles", None)
        self.assertLess(facts.net.clearance_m.value, -0.2)
        self.assertEqual(facts.landing.call, "out")
        self.assertLess(facts.landing.p_over_net, 0.01)

    def test_a_flight_on_the_floor_long_before_the_next_hit_had_landed(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        facts = flight_facts(exact_fit(p0, v0), 0.0, "near", "singles", ends_ms=5000.0)  # the next rally's serve
        self.assertIsNotNone(facts.landing)
        self.assertLess(facts.flight_ms, 2000.0)

    def test_a_hit_long_before_the_fits_start_does_not_blow_up(self):
        p0, v0 = np.array([2.0, 4.0, 1.5]), np.array([-1.0, 20.0, 10.0])
        facts = flight_facts(exact_fit(p0, v0, start_ms=0.0), -250.0, "near", "singles", None)
        self.assertTrue(np.isfinite(facts.contact).all() and np.isfinite(facts.speed_m_s.value))
        self.assertLess(facts.speed_m_s.value, 40.0)


class RallyEndTests(unittest.TestCase):
    def landed(self, p0, v0):
        return flight_facts(exact_fit(p0, v0), 0.0, "near", "singles", None)

    def test_the_end_is_in_out_or_the_net_and_gives_the_winner(self):
        p0 = np.array([2.5, 3.0, 2.5])
        inside = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        end = rally_end([inside])
        self.assertEqual((end.call, end.winner), ("in", "near"))
        self.assertAlmostEqual(sum(end.p.values()), 1.0)
        self.assertEqual(end.time_ms, inside.landing.time_ms)
        out = rally_end([self.landed(p0, solve_launch(p0, [2.5, 14.5, 0.0], 1.4))])
        self.assertEqual((out.call, out.winner), ("out", "far"))
        net = rally_end([self.landed([2.5, 5.0, 1.0], [0.0, 9.0, 1.2])])  # low and flat: it meets the net
        self.assertEqual((net.call, net.winner), ("net", "far"))
        short = rally_end([self.landed([2.0, 5.0, 1.0], [0.0, 2.0, 1.0])])  # it dies before the net
        self.assertEqual((short.call, short.winner), ("net", "far"))
        self.assertGreater(short.p["net"], 0.99)
        self.assertIsNone(rally_end([replace(inside, landing=None)]))  # the end was not seen
        self.assertIsNone(rally_end([]))

    def test_a_shuttle_down_by_the_net_got_over_only_if_it_could_have_cleared_the_tape(self):
        # Both seen to come down 5 cm short of the net, measured to 30 cm either way.
        def loosely(shot):
            return _judged(replace(shot, landing=replace(shot.landing, point=(2.5, NET_Y_M - 0.05), covariance=((0.09, 0.0), (0.0, 0.09)))), "singles")

        flat = loosely(self.landed([2.5, 4.0, 0.6], [0.0, 7.0, 0.5]))  # low and flat: below the tape at the net
        falling = loosely(self.landed([2.5, 6.2, 1.8], [0.0, 0.5, 0.0]))  # dropping steeply by the net
        self.assertIsNone(flat.net)
        self.assertLess(flat.landing.p_over_net, 0.01)
        self.assertGreater(falling.landing.p_over_net, 0.2)

    def test_the_winner_is_too_close_to_call_when_the_shot_s_half_is(self):
        p0 = np.array([2.5, 3.0, 2.5])
        inside = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        unsure = rally_end([replace(inside, contact_sigma=(0.1, 3.0, 0.1))])
        self.assertEqual((unsure.call, unsure.winner), ("in", "too close to call"))
        self.assertTrue(0.5 <= unsure.p_half < 0.95)
        self.assertGreater(rally_end([inside]).p_half, 0.999)

    def test_a_sighting_may_be_on_the_net_only_when_its_ray_passes_through_it(self):
        behind = Camera.look_at((3.05, -6.0, 3.0), (3.05, 6.7, 1.0), 1500.0, (1920, 1080))
        side = Camera.look_at((-9.0, 6.7, 3.0), (3.05, 6.7, 1.0), 1500.0, (1920, 1080))

        def pixel(camera, point):
            return camera.project(np.array([point], dtype=float))[0]

        on_the_net = [3.0, NET_Y_M, 1.0]
        self.assertTrue(_may_be_on_the_net(behind, pixel(behind, on_the_net)))
        self.assertTrue(_may_be_on_the_net(behind, pixel(behind, [3.0, 10.0, 0.8])))  # seen through the net
        self.assertFalse(_may_be_on_the_net(behind, pixel(behind, [3.0, 10.0, 3.5])))  # seen above it
        self.assertFalse(_may_be_on_the_net(side, pixel(side, [3.0, 10.0, 1.0])))  # beyond it, and seen to be

    def test_a_flight_seen_well_beyond_the_net_got_over_it(self):
        shot = self.landed([2.5, 5.0, 1.0], [0.0, 9.0, 1.2])  # fitted below the tape
        self.assertLess(shot.landing.p_over_net, 0.05)
        self.assertEqual(_judged(replace(shot, seen_over=True), "singles").landing.p_over_net, 1.0)

    def test_out_or_the_net_is_too_close_to_call_but_the_winner_is_not(self):
        p0 = np.array([2.5, 3.0, 2.5])
        inside = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        end = rally_end([replace(inside, landing=replace(inside.landing, p_in=0.01, p_over_net=0.5))])
        self.assertEqual(end.p, {"in": 0.01, "out": 0.49, "net": 0.5})
        self.assertEqual((end.call, end.winner), ("too close to call", "far"))
        unsure = rally_end([replace(inside, landing=replace(inside.landing, p_in=0.5, p_over_net=0.9))])
        self.assertEqual((unsure.call, unsure.winner), ("too close to call", "too close to call"))

    def test_what_follows_a_shot_is_a_reply_only_when_it_can_be_one(self):
        p0 = np.array([2.5, 3.0, 2.5])
        clear = exact_fit(p0, solve_launch(p0, [2.5, 12.5, 0.0], 1.8))
        there = clear.position_at(1500.0)  # coming down in the far half
        self.assertTrue(_played_again(clear, 0.0, "near", exact_fit(there, [0.0, -15.0, 5.0], 1500.0), 1500.0))
        for seen in (
            exact_fit([2.5, 4.0, 1.0], [0.0, -15.0, 5.0], 1500.0),  # on the hitter's own side
            exact_fit(there, [0.0, 15.0, 5.0], 1500.0),  # leaving away from the hitter
            exact_fit([2.5, 12.0, 0.03], [0.0, 0.0, 0.0], 1500.0),  # lying on the floor
            exact_fit([47.0, 214.0, -18.0], [-15.0, -26.0, 0.0], 1500.0),  # where no racket can be
            None,
        ):
            self.assertFalse(_played_again(clear, 0.0, "near", seen, 1500.0))
        short = exact_fit([2.0, 5.0, 1.0], [0.0, 2.0, 1.0])  # it never got over the net
        self.assertFalse(_played_again(short, 0.0, "near", exact_fit([2.5, 9.0, 1.0], [0.0, -10.0, 4.0], 800.0), 800.0))

    def test_the_rally_ends_at_the_first_shot_down_unless_its_hitter_played_on(self):
        p0 = np.array([2.5, 3.0, 2.5])
        down = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        again = replace(down, hit_ms=down.hit_ms + 2500.0, landing=None)  # the near player again: the far reply was missed
        replied = replace(again, hitter="far")
        self.assertEqual(_last_of_rally([down, again]), 1)
        self.assertEqual(_last_of_rally([down, replied]), 0)
        self.assertEqual(_last_of_rally([replace(down, landing=replace(down.landing, p_over_net=0.01)), again]), 0)  # into the net
        self.assertEqual(_last_of_rally([down, replace(again, serve=True)]), 0)  # the next rally
        self.assertEqual(_last_of_rally([down, replace(again, hit_ms=down.hit_ms + 4000.0)]), 0)
        self.assertEqual(_last_of_rally([replace(down, landing=None), replied]), 1)
        self.assertIsNone(_last_of_rally([]))
        self.assertEqual(_last_of_rally([down, replace(again, hit_ms=down.hit_ms + 4000.0, hitter="far")]), 0)
        self.assertEqual(_last_of_rally([replace(down, landing=None), replace(again, hit_ms=down.hit_ms + 4000.0, hitter="far")]), 0)  # too long after

    def test_what_ties_a_last_shot_to_the_rally(self):
        p0 = np.array([2.5, 3.0, 2.5])
        down = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        before = replace(down, hit_ms=-1000.0, hitter="far", landing=None)
        self.assertFalse(_may_be_sent_back([before, down], 1, [-1000.0, 0.0]))
        self.assertTrue(_may_be_sent_back([replace(before, hitter="near"), down], 1, [-1000.0, 0.0]))  # from the same half
        self.assertTrue(_may_be_sent_back([before, down], 1, [-1000.0, -400.0, 0.0]))  # a hit between them made no shot
        self.assertTrue(_may_be_sent_back([down], 0, [0.0]))  # the first one found, and no serve
        self.assertFalse(_may_be_sent_back([replace(down, serve=True)], 0, [0.0]))

    def test_a_landing_that_says_too_little_is_not_called(self):
        sure = Landing((2.0, 3.0), ((0.01, 0.0), (0.0, 0.04)), 1000.0, "in", 0.99)
        self.assertFalse(_not_to_be_called(sure, 2.0, 2.0))
        self.assertTrue(_not_to_be_called(sure, 7.0, 2.0))  # one flight does not explain what was seen
        self.assertTrue(_not_to_be_called(replace(sure, covariance=((0.01, 0.0), (0.0, 25.0))), 2.0, 2.0))  # it could be almost anywhere

    def test_what_is_no_shot(self):
        p0 = np.array([2.5, 3.0, 2.5])
        clear = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        short = self.landed([2.0, 5.0, 1.0], [0.0, 2.0, 1.0])
        self.assertTrue(_a_shot(clear))
        self.assertTrue(_a_shot(replace(short, speed_m_s=Estimate(4.0, 0.1))))
        self.assertFalse(_a_shot(replace(clear, flight_ms=6.0)))  # it was on the floor already
        self.assertFalse(_a_shot(replace(short, speed_m_s=Estimate(4.0, 0.1), flight_ms=1900.0)))  # nothing takes so long and stops short of the net

    def test_a_shot_short_of_the_net_ended_the_rally_when_its_own_side_hit_long_after(self):
        short = replace(self.landed([2.0, 5.0, 1.0], [0.0, 2.0, 1.0]), landing=None)
        self.assertIsNone(short.net)
        self.assertEqual(_last_of_rally([short, replace(short, hit_ms=short.hit_ms + 2000.0)]), 0)  # picked up and sent back
        self.assertEqual(_last_of_rally([short, replace(short, hit_ms=short.hit_ms + 200.0)]), 1)  # a split in the middle of a flight

    def test_a_shot_did_not_end_the_rally_when_the_hits_went_on(self):
        p0 = np.array([2.5, 3.0, 2.5])
        down = self.landed(p0, solve_launch(p0, [2.5, 11.0, 0.0], 1.2))
        after = replace(down, hit_ms=down.hit_ms + 1000.0, hitter="far", landing=None)
        two_after = [down.hit_ms + 900.0, down.hit_ms + 1800.0]  # the shuttle falling, then sent back
        self.assertEqual(_last_of_rally([down, after], two_after), 0)
        self.assertEqual(_last_of_rally([down, after], [*two_after, down.hit_ms + 2700.0]), 1)
        self.assertEqual(_last_of_rally([down, after], [*two_after, down.hit_ms + 6000.0]), 0)  # too long after
        self.assertFalse(_went_on(down, [down.hit_ms, *two_after]))  # its own hit does not count
        self.assertTrue(_went_on(down, [*two_after, down.hit_ms + 2700.0]))


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
            # The launch's uncertainty takes in the drag and the hit time as the separate facts do.
            covariance = np.asarray(shot.launch_covariance)
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), -1e-9)
            np.testing.assert_allclose(np.sqrt(np.diag(covariance)[:3]), shot.contact_sigma, rtol=0.1)
            along = np.asarray(shot.velocity) / np.linalg.norm(shot.velocity)
            self.assertAlmostEqual(float(np.sqrt(along @ covariance[3:, 3:] @ along)) / shot.speed_m_s.sigma, 1.0, delta=0.25)
        # A drag known from a whole session is used as given, and makes the facts surer.
        known = rally_shots(analysis, seen(clip), times(len(clip.cameras)), clip.cameras, court="singles", terminal_velocity=(6.7, 0.02))
        self.assertEqual((known[0].terminal_velocity.value, known[0].terminal_velocity.sigma), (6.7, 0.02))
        self.assertLessEqual(np.trace(np.asarray(known[-1].landing.covariance)), np.trace(np.asarray(shots[-1].landing.covariance)) + 1e-9)

    def analysed(self):
        clip = clip_of()
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        return clip, find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, onsets)

    def test_a_hit_that_looks_like_a_serve_with_the_shuttle_still_up_is_a_reply(self):
        clip, analysis = self.analysed()
        analysis.hits[2] = replace(analysis.hits[2], evidence="sound+serve")
        shots = rally_shots(analysis, seen(clip), times(len(clip.cameras)), clip.cameras, court="singles")
        self.assertEqual(len(shots), len(RALLY.contacts))
        self.assertFalse(shots[2].serve)
        self.assertIsNone(shots[1].landing)

    def test_a_flight_seen_to_leave_its_path_while_still_up_did_not_land(self):
        clip, analysis = self.analysed()
        last = len(analysis.hits) - 1
        analysis.hits = analysis.hits[:last]  # the rally's last hit was missed
        with mock.patch("shots._played_again", return_value=False):  # and what followed could not be told for a reply
            shots = rally_shots(analysis, seen(clip), times(len(clip.cameras)), clip.cameras, court="singles")
        self.assertEqual(len(shots), last)
        self.assertIsNone(shots[-1].landing)

    def test_a_shuttle_seen_moving_just_after_it_came_down_did_not_land(self):
        clip, analysis = self.analysed()
        track, stamps = seen(clip), times(len(clip.cameras))
        down = rally_shots(analysis, track, stamps, clip.cameras, court="singles")[-1].landing
        frames = [i for i, t in enumerate(stamps) if down.time_ms + 100.0 < t < down.time_ms + LYING_MS][:4]
        self.assertEqual(len(frames), 4)
        for k, i in enumerate(frames):
            track[i] = (100.0 + 20.0 * k, 100.0)
        analysis.flights.append(Piece(frames, float(stamps[frames[0]])))
        self.assertIsNone(rally_shots(analysis, track, stamps, clip.cameras, court="singles")[-1].landing)

    def test_a_last_shot_from_the_half_of_the_one_before_says_nothing_of_the_end(self):
        clip, analysis = self.analysed()
        last = analysis.hits.pop(-2)  # the reply before the last shot was missed
        analysis.rejected_sounds.append(last.heard_ms)
        shots = rally_shots(analysis, seen(clip), times(len(clip.cameras)), clip.cameras, court="singles")
        self.assertEqual(shots[-2].hitter, shots[-1].hitter)
        self.assertIsNone(shots[-1].landing)

    def test_a_sound_while_the_last_flight_was_out_of_sight_may_be_its_reply(self):
        # A rally ending in a deep clear, lost while still up but seen well beyond the net: surely
        # over it, whatever the fit says of its clearance (a net shot seen from behind is not).
        rally = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, 2.4), Shot("clear", (3.5, 0.6), 1.8, 2.4), Shot("clear", (2.6, 12.4), 1.8, None)])
        clip = make_clip(CAMERA, SIZE, rally, fps=FPS, handheld=True, seed=3, tail_s=0.5, render=False)
        stamps = times(len(clip.cameras))
        lost_ms = rally.contacts[-1].time * 1000.0 + 1000.0
        track = [None if t > lost_ms else point for t, point in zip(stamps, seen(clip))]
        analysis = find_hits(track, stamps, clip.cameras, detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS))
        last = rally_shots(analysis, track, stamps, clip.cameras, court="singles")[-1]
        self.assertTrue(last.seen_over)
        self.assertIsNotNone(last.landing)
        analysis.rejected_sounds.append(lost_ms + 100.0)
        self.assertIsNone(rally_shots(analysis, track, stamps, clip.cameras, court="singles")[-1].landing)


if __name__ == "__main__":
    unittest.main()
