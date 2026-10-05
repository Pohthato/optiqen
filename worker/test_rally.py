# worker/test_rally.py
import unittest

import numpy as np

from geometry.court_model import NET_POST_HEIGHT_M, NET_Y_M, SINGLES_WIDTH_M, COURT_LENGTH_M
from geometry.shuttle_physics import simulate
from simulation.rally import BODY_OFFSET_M, Shot, build_rally, canned_rally, player_positions


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


if __name__ == "__main__":
    unittest.main()
