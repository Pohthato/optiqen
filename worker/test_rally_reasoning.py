# worker/test_rally_reasoning.py
import unittest
from dataclasses import replace

from pressure import Pressure
from rally_reasoning import p_under, rally_reasons
from shots import Estimate, Landing, ShotFacts, rally_end


def shot(hit_ms, hitter, serve=False) -> ShotFacts:
    return ShotFacts(
        hit_ms, hitter, serve, (3.0, 4.0 if hitter == "near" else 9.4, 2.0), (0.05, 0.05, 0.1),
        Estimate(20.0, 1.0), Estimate(10.0, 1.0), Estimate(3.0, 0.1), None, 900.0, None,
    )


def rally(n, p_in, p_over_net=1.0):
    """`n` shots, the near player serving, the last one coming down in with chance `p_in`."""
    shots = [shot(1000.0 * i, "near" if i % 2 == 0 else "far", serve=i == 0) for i in range(n)]
    shots[-1] = replace(shots[-1], landing=Landing((3.0, 6.0), ((0.01, 0.0), (0.0, 0.01)), 1000.0 * n, "in", p_in, p_over_net))
    return shots, rally_end(shots)


def spared(*ms):
    """Each shot's pressure on its receiver, by the time it left them to spare (None: not known)."""
    return [None if m is None else Pressure(Estimate(1.0, 0.1), Estimate(1000.0, 12.0), Estimate(1.0, 0.1), Estimate(float(m), 10.0), None) for m in ms]


class PUnderTests(unittest.TestCase):
    def test_the_chance_of_less_time_to_spare_than_a_limit(self):
        self.assertAlmostEqual(p_under(Estimate(250.0, 10.0), 250.0), 0.5)
        self.assertGreater(p_under(Estimate(200.0, 10.0), 250.0), 0.999)
        self.assertLess(p_under(Estimate(300.0, 10.0), 250.0), 0.001)
        self.assertEqual(p_under(Estimate(-1.0, 0.0), 0.0), 1.0)
        self.assertEqual(p_under(None, 250.0), 0.5)


class HowItWasWonTests(unittest.TestCase):
    def test_a_shot_that_came_down_in_was_a_winner_when_there_was_no_time_to_get_to_it(self):
        shots, end = rally(3, p_in=1.0)
        won = rally_reasons(shots, end, spared(600, 600, -150))
        self.assertEqual(won.call, "winner")
        self.assertAlmostEqual(sum(won.p.values()), 1.0)
        self.assertEqual(rally_reasons(shots, end, spared(600, 600, 400)).call, "let in")
        unknown = rally_reasons(shots, end, spared(600, 600, None))
        self.assertEqual((unknown.call, unknown.p["winner"]), ("too close to call", 0.5))

    def test_an_error_was_forced_under_pressure_at_its_hit_and_unforced_when_not(self):
        for p_over_net in (1.0, 0.0):  # out, and into the net
            shots, end = rally(4, p_in=0.0, p_over_net=p_over_net)
            self.assertEqual(rally_reasons(shots, end, spared(600, 600, 100, None)).call, "forced error")
            self.assertEqual(rally_reasons(shots, end, spared(600, 600, 800, None)).call, "unforced error")
            unknown = rally_reasons(shots, end, spared(600, 600, None, None))
            self.assertEqual((unknown.call, unknown.p["forced error"]), ("too close to call", 0.5))

    def test_a_serve_is_never_under_pressure(self):
        shots, end = rally(1, p_in=0.0)
        fault = rally_reasons(shots, end, spared(None))
        self.assertEqual((fault.call, fault.p["unforced error"], fault.turn), ("unforced error", 1.0, "no turn"))


class WhereItTurnedTests(unittest.TestCase):
    def test_at_the_first_of_the_winner_s_shots_from_which_all_left_the_loser_under_pressure(self):
        shots, end = rally(7, p_in=1.0)  # the near player's shots are 0, 2, 4 and the winner, 6
        turned = rally_reasons(shots, end, spared(600, 600, 600, 600, 100, 600, -200))
        self.assertEqual(turned.turn, 4)
        self.assertAlmostEqual(sum(turned.p_turn) + turned.p_no_turn, 1.0)
        self.assertEqual(rally_reasons(shots, end, spared(600, 600, 100, 600, 600, 600, -200)).turn, 6)
        self.assertEqual(rally_reasons(shots, end, spared(100, 600, 100, 600, 100, 600, -200)).turn, 0)

    def test_it_did_not_turn_when_the_winner_s_last_shot_left_the_loser_time(self):
        shots, end = rally(4, p_in=0.0)  # the far player's last shot was out: the near player won
        level = rally_reasons(shots, end, spared(100, 100, 800, None))
        self.assertEqual((level.call, level.turn), ("unforced error", "no turn"))

    def test_the_turn_can_be_sure_when_the_winner_is_not(self):
        shots, end = rally(4, p_in=0.5)
        self.assertEqual(end.winner, "too close to call")
        level = rally_reasons(shots, end, spared(800, 800, 800, 800))
        self.assertEqual((level.call, level.turn), ("too close to call", "no turn"))
        self.assertAlmostEqual(sum(level.p_turn) + level.p_no_turn, 1.0)
        unknown = rally_reasons(shots, end, spared(None, None, None, None))
        self.assertAlmostEqual(sum(unknown.p_turn) + unknown.p_no_turn, 1.0)
        self.assertEqual(unknown.turn, "too close to call")

    def test_nothing_is_called_when_the_last_shot_s_half_is_unsure(self):
        shots, end = rally(3, p_in=1.0)
        shots[-1] = replace(shots[-1], contact_sigma=(0.05, 3.0, 0.1))
        unsure = rally_reasons(shots, rally_end(shots), spared(600, 600, -300))
        self.assertGreater(unsure.p["winner"], 0.99)
        self.assertEqual((unsure.call, unsure.turn), ("too close to call", "too close to call"))

    def test_none_without_an_end(self):
        shots, _ = rally(3, p_in=1.0)
        self.assertIsNone(rally_reasons(shots, None, spared(600, 600, -300)))


if __name__ == "__main__":
    unittest.main()
