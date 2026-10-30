# worker/test_rally.py
import unittest

import numpy as np

from geometry.court_model import NET_POST_HEIGHT_M, NET_Y_M, SINGLES_WIDTH_M, COURT_LENGTH_M
from geometry.shuttle_physics import simulate
from simulation.rally import BODY_OFFSET_M, NET_HIT_HEIGHTS_M, SHUTTLE_LYING_M, Shot, afterwards, build_rally, canned_rally, player_positions, random_rally


def net_crossing_height(flight, terminal_velocity):
    times = np.arange(0.0, flight.end_time - flight.start_time, 0.002)
    xyz = simulate(flight.p0, flight.v0, times, terminal_velocity)
    side = np.sign(xyz[:, 1] - NET_Y_M)
    index = int(np.where(np.diff(side) != 0)[0][0])
    return float(xyz[index, 2])


class CannedRallyTests(unittest.TestCase):
    def setUp(self):
        self.rally = canned_rally()

    def test_seven_contacts_alternate_from_the_near_server(self):
        hitters = [contact.hitter for contact in self.rally.contacts]
        self.assertEqual(hitters, ["near", "far", "near", "far", "near", "far", "near"])
        kinds = [contact.kind for contact in self.rally.contacts]
        self.assertEqual(kinds, ["serve", "clear", "drop", "lift", "smash", "block", "net"])
        times = [contact.time for contact in self.rally.contacts]
        self.assertEqual(times, sorted(times))
        self.assertTrue(6.0 < self.rally.end_time < 10.0)

    def test_every_flight_clears_the_net(self):
        for flight in self.rally.flights:
            self.assertGreater(net_crossing_height(flight, self.rally.terminal_velocity), NET_POST_HEIGHT_M)

    def test_contacts_are_on_the_hitters_side(self):
        for contact in self.rally.contacts:
            on_near_side = contact.position[1] < NET_Y_M
            self.assertEqual(on_near_side, contact.hitter == "near", contact)

    def test_the_rally_ends_with_a_landing_inside_the_far_half(self):
        x, y = self.rally.landing
        self.assertTrue(0 <= x <= SINGLES_WIDTH_M and NET_Y_M < y <= COURT_LENGTH_M)
        np.testing.assert_allclose(self.rally.shuttle_at(self.rally.end_time), [x, y, 0.0], atol=0.01)

    def test_shuttle_position_is_continuous_and_starts_at_the_serve(self):
        self.assertIsNone(self.rally.shuttle_at(0.0))
        self.assertIsNone(self.rally.shuttle_at(self.rally.end_time + 0.1))
        for contact in self.rally.contacts:
            np.testing.assert_allclose(self.rally.shuttle_at(contact.time), contact.position, atol=1e-6)
            before = self.rally.shuttle_at(contact.time - 1e-4)
            if before is not None:
                self.assertLess(float(np.linalg.norm(before - np.asarray(contact.position))), 0.01)

    def test_players_stand_behind_their_contact_point(self):
        first = self.rally.contacts[0]
        near = player_positions(self.rally, first.time)["near"]
        self.assertAlmostEqual(near[1], first.position[1] - BODY_OFFSET_M[1])
        self.assertEqual(set(player_positions(self.rally, 3.0)), {"near", "far"})


class BuildRallyRefusalTests(unittest.TestCase):
    def test_a_flat_low_shot_into_the_net_is_refused(self):
        with self.assertRaisesRegex(ValueError, "net"):
            build_rally((2.6, 5.0, 0.6), [Shot("drive", (2.6, 8.0), 0.25, None)])

    def test_a_shot_that_stays_on_its_own_side_is_refused(self):
        with self.assertRaisesRegex(ValueError, "net"):
            build_rally((2.6, 3.0, 1.0), [Shot("clear", (2.6, 0.5), 1.5, None)])

    def test_a_receive_height_above_the_apex_is_refused(self):
        with self.assertRaisesRegex(ValueError, "receive height"):
            build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, 9.0), Shot("clear", (3.5, 0.6), 1.8, None)])

    def test_receive_heights_outside_reach_are_refused(self):
        for height in (0.0, -0.002, 4.5):
            with self.assertRaisesRegex(ValueError, "receive height"):
                build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, height), Shot("clear", (3.5, 0.6), 1.8, None)])

    def test_only_the_last_shot_may_land(self):
        with self.assertRaises(ValueError):
            build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, None), Shot("clear", (3.5, 0.6), 1.8, None)])


class RandomRallyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rallies = [random_rally(np.random.default_rng(seed)) for seed in range(40)]

    def test_rallies_are_varied(self):
        kinds = {contact.kind for rally in self.rallies for contact in rally.contacts}
        self.assertTrue({"serve", "clear", "drop", "smash", "lift", "net"} <= kinds, kinds)
        lengths = {len(rally.contacts) for rally in self.rallies}
        self.assertGreaterEqual(len(lengths), 5)
        serve_depths = {rally.flights[0].end_time - rally.flights[0].start_time > 1.5 for rally in self.rallies}
        self.assertEqual(serve_depths, {True, False})  # high and low serves

    def test_every_rally_is_physically_valid(self):
        for rally in self.rallies:
            self.assertEqual(rally.contacts[0].kind, "serve")
            hitters = [contact.hitter for contact in rally.contacts]
            self.assertTrue(all(a != b for a, b in zip(hitters, hitters[1:])))
            self.assertTrue(0.0 <= rally.landing[0] <= SINGLES_WIDTH_M and 0.0 <= rally.landing[1] <= COURT_LENGTH_M)

    def test_smashes_are_hit_from_overhead(self):
        smashes = [c for rally in self.rallies for c in rally.contacts if c.kind == "smash"]
        self.assertTrue(smashes)
        self.assertTrue(all(c.position[2] >= 1.8 for c in smashes))

    def test_the_same_seed_gives_the_same_rally(self):
        a, b = random_rally(np.random.default_rng(3)), random_rally(np.random.default_rng(3))
        self.assertEqual(a.contacts, b.contacts)

    def test_a_rally_in_court_is_won_by_its_last_hitter(self):
        for rally in self.rallies:
            self.assertEqual((rally.end, rally.winner), ("in", rally.contacts[-1].hitter))


def inside(rally) -> bool:
    x, y = rally.landing
    near = rally.contacts[-1].hitter == "near"
    return 0.0 <= x <= SINGLES_WIDTH_M and (NET_Y_M <= y <= COURT_LENGTH_M if near else 0.0 <= y <= NET_Y_M)


class RallyEndTests(unittest.TestCase):
    def test_a_last_shot_into_the_net_stops_there_and_drops_at_its_foot(self):
        rally = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, 2.4), Shot("drop", (2.0, NET_Y_M), 0.8, None, net_height=1.2)])
        self.assertEqual((rally.end, rally.winner), ("net", "near"))
        meets = rally.flights[1]
        np.testing.assert_allclose(rally.shuttle_at(meets.end_time), [2.0, NET_Y_M, 1.2], atol=0.01)
        self.assertEqual(len(rally.flights), len(rally.contacts) + 1)  # the drop down the net
        x, y = rally.landing
        self.assertTrue(NET_Y_M < y < NET_Y_M + 0.5 and abs(x - 2.0) < 0.01)  # on the far hitter's side
        np.testing.assert_allclose(rally.shuttle_at(rally.end_time), [x, y, 0.0], atol=0.01)

    def test_only_the_last_shot_goes_into_the_net_and_only_under_the_tape(self):
        for height in (NET_HIT_HEIGHTS_M[0] - 0.05, NET_HIT_HEIGHTS_M[1] + 0.05):
            with self.assertRaisesRegex(ValueError, "into the net"):
                build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.9), 1.9, 2.4), Shot("drop", (2.0, NET_Y_M), 0.8, None, net_height=height)])
        with self.assertRaisesRegex(ValueError, "into the net"):
            build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, NET_Y_M), 0.6, 2.4, net_height=1.2), Shot("drop", (2.0, 4.0), 0.8, None)])

    def test_random_rallies_end_out_or_in_the_net_when_asked(self):
        for seed in range(8):
            out = random_rally(np.random.default_rng(seed), end="out")
            self.assertEqual(out.end, "out")
            x, y = out.landing
            past = max(-x, x - SINGLES_WIDTH_M, -y, y - COURT_LENGTH_M)
            self.assertTrue(0.29 < past < 1.21, out.landing)
            net = random_rally(np.random.default_rng(seed), end="net")
            hitter = net.contacts[-1].hitter
            self.assertEqual((net.end, net.winner), ("net", "far" if hitter == "near" else "near"))
            self.assertTrue(0.0 < (NET_Y_M - net.landing[1]) * (1 if hitter == "near" else -1) < 0.5)
            line = random_rally(np.random.default_rng(seed), final_near_line=True)
            self.assertEqual(line.end, "in" if inside(line) else "out")
        with self.assertRaises(ValueError):
            random_rally(np.random.default_rng(0), end="in")

    def test_afterwards_the_shuttle_lies_then_goes_back_over_the_net_to_the_winner(self):
        sent = 0
        for seed in range(8):
            for end in (None, "out", "net"):
                rally = random_rally(np.random.default_rng(seed), end=end)
                after = afterwards(rally, np.random.default_rng(seed))
                self.assertEqual((after.contacts, after.flights, after.end), (rally.contacts, rally.flights, rally.end))
                self.assertTrue(0.8 <= after.lying_until - after.end_time <= 2.0)
                np.testing.assert_allclose(after.shuttle_at(after.end_time + 0.5), [*rally.landing, SHUTTLE_LYING_M])
                lies_near = rally.landing[1] < NET_Y_M
                if lies_near == (rally.winner == "near") or after.sent_back is None:
                    self.assertIsNone(after.shuttle_at(after.lying_until + 0.01))
                    self.assertEqual(end == "out", lies_near == (rally.winner == "near"))  # an out lies on the winner's side
                    continue
                sent += 1
                back = after.sent_back
                self.assertIsNone(after.shuttle_at((after.lying_until + back.start_time) / 2))  # in the hand
                self.assertLess(np.hypot(back.p0[0] - rally.landing[0], back.p0[1] - rally.landing[1]), 1.5)
                self.assertGreater(net_crossing_height(back, rally.terminal_velocity), NET_POST_HEIGHT_M)
                caught = after.shuttle_at(back.end_time)
                self.assertEqual(caught[1] < NET_Y_M, rally.winner == "near")
        self.assertGreater(sent, 10)


if __name__ == "__main__":
    unittest.main()
