# worker/test_value_benchmark.py
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

import evaluation.value_benchmark as value_benchmark
from evaluation.shuttleset import Hit, Match, Rally, Spot
from evaluation.value_benchmark import (
    answered_pressed, blurred, by_match_mean, by_match_rate, fit_curve, fit_let_in, flight_times, learn, rally_ats, rally_pressures, rules,
)
from rally_reasoning import left_by_professionals, p_under
from rally_value import fact_rows, named

SURE = np.eye(2) * 1e-4


def spot(x, y):
    return Spot(np.array([x, y], float), SURE)


def hit(hitter, ms, name, stood, receiver, to):
    return Hit(hitter, float(ms), name, spot(*stood), spot(*receiver), spot(*to))


def rally(hits, end="in", how="winner", winner=None):
    last = hits[-1].hitter
    return Rally("match", 1, 1, hits, end, how, winner or (last if end == "in" else {"A": "B", "B": "A"}[last]), (1, 0))


FLIGHTS = {"drop": (500.0, 100.0, 0.0), "smash": (300.0, 20.0, 0.0)}


class FlightTests(unittest.TestCase):
    def test_how_long_answered_shots_took_by_how_far_they_went(self):
        rallies = []
        for k, way in enumerate((2.0, 4.0, 6.0)):
            first = hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, 3.0 + way))
            reply = hit("B", 500 + 100 * way, "drop", (2.6, 3.0 + way), (2.6, 3.0), (2.6, 4.0))
            rallies.append(rally([first, reply]))
        at_none, slope, spread = flight_times(rallies)["serve"]
        self.assertAlmostEqual(at_none, 500.0, places=6)
        self.assertAlmostEqual(slope, 100.0, places=6)
        self.assertAlmostEqual(spread, 0.0, places=6)


class PressureTests(unittest.TestCase):
    def pressures(self, *hits, end="in"):
        return rally_pressures(rally(list(hits), end), 25.0, FLIGHTS, np.random.default_rng(0))

    def test_to_where_the_receiver_met_it(self):
        easy = self.pressures(hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, 11.0)), hit("B", 1000, "clear", (2.6, 10.0), (2.6, 3.0), (2.6, 1.0)), end="out")
        self.assertAlmostEqual(easy[0].go_m.value, 0.0)
        self.assertAlmostEqual(easy[0].spare_ms.value, 1000.0 - 250.0)
        hard = self.pressures(hit("A", 0, "serve", (2.6, 3.0), (1.5, 9.5), (4.5, 10.0)), hit("B", 450, "lift", (3.6, 10.0), (2.6, 3.0), (2.6, 1.0)), end="out")
        go = np.hypot(3.0, 0.5) - 1.4
        self.assertAlmostEqual(hard[0].spare_ms.value, 450.0 - 250.0 - (go - 0.5) / 3.5 * 1000.0, places=6)
        self.assertGreater(p_under(hard[0].spare_ms, 0.0), 0.95)
        self.assertIsNone(easy[1])  # out: no pressure

    def test_none_where_the_positions_contradict_each_other(self):
        far_off = self.pressures(hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, 11.0)), hit("B", 1000, "clear", (2.6, 7.0), (2.6, 3.0), (2.6, 1.0)), end="out")
        self.assertIsNone(far_off[0])

    def test_a_shot_that_came_down_in_is_flown_to_the_floor_as_its_name_is(self):
        serve = hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, 10.0))
        landed = self.pressures(serve, hit("B", 900, "drop", (2.6, 10.0), (0.5, 3.0), (4.6, 7.5)))
        self.assertAlmostEqual(landed[1].time_ms.value, 500.0 + 100.0 * np.hypot(2.0, 2.5))
        self.assertAlmostEqual(landed[1].time_ms.sigma, 40.0)  # a frame at 25 fps, the spread being none
        unnamed = rally_pressures(rally([serve, hit("B", 900, "block", (2.6, 10.0), (0.5, 3.0), (4.6, 7.5))]), 25.0, FLIGHTS, np.random.default_rng(0))
        self.assertIsNone(unnamed[1])

    def test_whether_each_hitter_was_under_pressure_from_the_shot_they_answered(self):
        hits = [hit("A", 0, "serve", (2.6, 3.0), (1.5, 9.5), (4.5, 10.0)), hit("B", 450, "lift", (3.6, 10.0), (2.6, 3.0), (2.6, 1.0)), hit("A", 1650, "clear", (2.6, 1.0), (2.6, 10.0), (2.6, 12.0))]
        pressures = rally_pressures(rally(hits, "out"), 25.0, FLIGHTS, np.random.default_rng(0))
        pressed = answered_pressed(rally(hits, "out"), pressures)
        self.assertEqual(pressed[0], 0.0)  # a serve
        self.assertGreater(pressed[1], 0.95)
        self.assertLess(pressed[2], 0.05)
        self.assertEqual(answered_pressed(rally(hits[1:], "out"), [None, None]), [0.5, 0.5])  # the serve not labelled


class CurveTests(unittest.TestCase):
    def test_the_curve_comes_back(self):
        rng = np.random.default_rng(1)
        x = rng.random(20000)
        won = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 - 1.2 * x)))).astype(float)
        a, b = fit_curve(x, won)
        self.assertAlmostEqual(a, 0.3, delta=0.08)
        self.assertAlmostEqual(b, -1.2, delta=0.12)

    def test_how_often_professionals_let_shots_in_comes_back(self):
        rng = np.random.default_rng(2)
        n = 40000
        unreached, inside = rng.random(n) * 0.5, rng.uniform(0.01, 2.5, n)
        high, push = (rng.random(n) < 0.2).astype(float), np.zeros(n)
        push[high == 0] = (rng.random(int((high == 0).sum())) < 0.1)
        want = (-3.0, -0.5, 2.0, 1.0)
        let_in = (rng.random(n) < (1 - unreached) * left_by_professionals(inside, high, push, want)).astype(float)
        got = fit_let_in(unreached, inside, high, push, let_in)
        np.testing.assert_allclose(got, want, atol=0.25)

    def test_a_rate_pooled_over_matches_with_its_interval(self):
        rate = by_match_rate({"a": (5, 10), "b": (15, 30), "c": (0, 0)}, np.random.default_rng(0))
        self.assertEqual((rate["k"], rate["n"], rate["share"]), (20, 40, 0.5))
        self.assertLessEqual(rate["interval"][0], 0.5)
        self.assertGreaterEqual(rate["interval"][1], 0.5)


class RulesTests(unittest.TestCase):
    def test_on_the_training_matches_only(self):
        def played(video):
            rallies = []
            for short, deep in ((4.5, 10.0), (3.5, 10.5)):
                serve = hit("A", 0, "serve", (2.6, 3.0), (2.6, deep), (2.6, deep))
                drop = hit("B", 900, "drop", (2.6, deep), (2.6, 3.0), (2.6, short))
                winner = hit("A", 900 + 300 * (deep - short), "drop", (2.6, short), (2.6, deep), (0.3, 12.5))
                rallies.append(rally([serve, drop, winner]))
            return Match(video, ("A", "B"), 25.0, rallies)

        report = rules([played("train"), played("test")], ["train"])
        self.assertEqual((report["matches"], report["rallies"]), (1, 2))
        self.assertEqual(report["answered"]["hits"], 4)
        self.assertEqual(report["flightTimesMs"]["drop"], {"atNone": 0, "perMetre": 300, "spread": 0})
        self.assertEqual(sum(report["winnerOrLetIn"]["asFirstWritten"].values()), 2)
        self.assertEqual(sum(report["winnerOrLetIn"]["learnt"]["eachMatchLeftOut"].values()), 2)


class ValueTests(unittest.TestCase):
    def test_what_was_known_at_each_hit_is_never_where_that_shot_went(self):
        hits = [hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, 10.5)), hit("B", 900, "lift", (2.6, 10.5), (2.6, 3.0), (2.6, 1.5)), hit("A", 2300, "smash", (2.6, 1.5), (2.6, 10.5), (4.0, 9.0))]
        first = rally(hits)
        ats = rally_ats(first, rally_pressures(first, 25.0, FLIGHTS, np.random.default_rng(0)))
        np.testing.assert_allclose(ats[2].contact[0], [2.6, 1.5])
        self.assertEqual(ats[2].gap_ms, 1400.0)
        np.testing.assert_allclose(ats[2].before, named("lift"))
        self.assertIsNone(ats[0].before)
        moved = rally([*hits[:-1], replace(hits[-1], goes_to=spot(0.5, 12.5))])
        again = rally_ats(moved, rally_pressures(moved, 25.0, FLIGHTS, np.random.default_rng(0)))
        for i in range(3):
            np.testing.assert_array_equal(fact_rows(ats[i], ats[i - 1] if i else None), fact_rows(again[i], again[i - 1] if i else None))

    def test_positions_as_unsure_as_on_the_users_clips(self):
        match = Match("m", ("A", "B"), 25.0, [rally([hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, 10.5))])])
        once = blurred(match, np.random.default_rng(0)).rallies[0].hits[0]
        np.testing.assert_allclose(once.hitter_at.covariance - SURE, np.eye(2) * 0.13**2)
        np.testing.assert_allclose(once.receiver_at.covariance - SURE, np.eye(2) * 0.25**2)
        np.testing.assert_allclose(once.goes_to.covariance - SURE, np.diag([0.25**2, 0.45**2]))
        moved = np.array([blurred(match, np.random.default_rng(k)).rallies[0].hits[0].hitter_at.point for k in range(2000)])
        np.testing.assert_allclose(moved.std(axis=0), [0.13, 0.13], atol=0.01)
        self.assertEqual(match.rallies[0].hits[0].hitter_at.covariance[0, 0], SURE[0, 0])

    def test_a_mean_over_hits_with_its_interval_by_match(self):
        mean = by_match_mean(np.array([1.0, 1.0, 0.0, 0.0]), np.array(["a", "a", "b", "b"]), np.random.default_rng(0))
        self.assertEqual(mean["mean"], 0.5)
        self.assertEqual(mean["interval"], [0.0, 1.0])

    def test_learnt_on_folds_of_matches_and_refitted(self):
        rng = np.random.default_rng(3)
        matches = []
        for k in range(6):
            rallies = []
            for _ in range(30):
                deep = 10.0 + rng.random()
                serve = hit("A", 0, "serve", (2.6, 3.0), (2.6, 10.0), (2.6, deep))
                reply = hit("B", 900, "lift", (2.6, deep), (2.6, 3.0), (2.6, 1.5))
                if rng.random() < 0.5:
                    rallies.append(rally([serve, reply, hit("A", 2300, "smash", (2.6, 1.5), (2.6, 10.5), (4.0, 9.0))]))
                else:
                    rallies.append(rally([serve, reply, hit("A", 2300, "drop", (2.6, 1.5), (2.6, 10.5), (2.6, 4.0))], "net"))
            matches.append(Match(f"m{k}", ("A", "B"), 25.0, rallies))
        with mock.patch.object(value_benchmark, "L2S", (1e-3,)), mock.patch.object(value_benchmark, "REFITS", 3):
            model, report = learn(matches, [match.video for match in matches])
        self.assertEqual(model.weights.shape[0], 4)
        self.assertLess(report["foldsLogLoss"]["names, l2 0.001"], report["foldsLogLoss"]["rules"])
        smash = rally_ats(next(r for r in matches[0].rallies if r.hits[-1].name == "smash"), [None, None, None])[2]
        self.assertGreater(model.chances(fact_rows(smash, None)).mean(), 0.6)


if __name__ == "__main__":
    unittest.main()
