# worker/test_rally_value.py
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rally_value import FACTS, NAMES, At, Model, chance, fact_rows, fit, named
from shots import Estimate

SURE = np.eye(2) * 1e-6


def place(x, y, covariance=SURE):
    return (np.array([x, y], float), covariance)


def at(hitter, receiver, contact, gap_ms=900.0, spare_ms=Estimate(300.0, 20.0), before="clear", name="smash"):
    return At(place(*hitter), place(*receiver), place(*contact), gap_ms, spare_ms, None if before is None else named(before), named(name))


def fact(rows, name):
    return rows[:, FACTS.index(name)]


class FactTests(unittest.TestCase):
    def test_the_court_is_turned_so_the_hitter_is_on_the_near_half(self):
        then = at((2.0, 3.0), (1.0, 12.0), (2.5, 2.5))
        now = at((1.0, 12.0), (2.0, 3.0), (1.2, 11.5))
        rows = fact_rows(now, then)
        np.testing.assert_allclose([fact(rows, "hitterX")[0], fact(rows, "hitterY")[0]], [5.18 - 1.0, 13.4 - 12.0])
        np.testing.assert_allclose([fact(rows, "receiverX")[0], fact(rows, "receiverY")[0]], [5.18 - 2.0, 13.4 - 3.0])
        np.testing.assert_allclose([fact(rows, "then hitterX")[0], fact(rows, "then hitterY")[0]], [5.18 - 2.0, 13.4 - 3.0])
        near = fact_rows(then, None)
        np.testing.assert_allclose([fact(near, "hitterX")[0], fact(near, "hitterY")[0]], [2.0, 3.0])

    def test_what_is_not_known_is_left_out(self):
        first = at((2.6, 3.0), (2.6, 10.0), (2.6, 3.0), gap_ms=None, spare_ms=None, before=None, name="serve")
        rows = fact_rows(first, None)
        self.assertTrue(np.isnan(fact(rows, "gapS")[0]) and np.isnan(fact(rows, "spareS")[0]))
        self.assertEqual(fact(rows, "spareKnown")[0], 0.0)
        self.assertEqual(sum(fact(rows, f"before {name}")[0] for name in NAMES), 0.0)
        self.assertEqual(fact(rows, "hit serve")[0], 1.0)
        self.assertEqual(fact(rows, "then")[0], 0.0)
        self.assertTrue(np.isnan(fact(rows, "then hitterX")[0]))

    def test_drawn_within_their_uncertainty(self):
        loose = At(place(2.6, 3.0, np.eye(2) * 0.04), place(2.6, 10.0), place(2.6, 3.5), 900.0, Estimate(300.0, 50.0), named("clear"), named("drop"))
        rows = fact_rows(loose, None, np.random.default_rng(0), 4000)
        self.assertAlmostEqual(float(np.std(fact(rows, "hitterX"))), 0.2, delta=0.01)
        self.assertAlmostEqual(float(np.std(fact(rows, "spareS"))), 0.05, delta=0.003)
        self.assertLess(float(np.std(fact(rows, "receiverX"))), 0.002)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.names = rng.choice(["smash", "lift", "drop"], 6000)
        self.rows = np.vstack([fact_rows(at((2.6, 3.0), (2.6, 10.0), (2.6, 3.0), name=name), None) for name in self.names])
        self.won = (rng.random(6000) < np.where(self.names == "smash", 0.7, 0.4)).astype(float)

    def test_who_wins_comes_back(self):
        framed = Model.framed(self.rows, ("hit smash", "hit lift"), False)
        model = replace(framed, weights=fit(framed, self.rows, self.won, 1e-4)[None])
        p = model.chances(self.rows)[:, 0]
        for name in ("smash", "lift", "drop"):
            self.assertAlmostEqual(float(p[self.names == name].mean()), float(self.won[self.names == name].mean()), delta=0.005)

    def test_a_fact_not_known_is_taken_at_its_mean(self):
        framed = Model.framed(np.vstack([self.rows, fact_rows(at((2.6, 3.0), (2.6, 10.0), (2.6, 3.0), gap_ms=None), None)]), ("gapS", "hit smash"), True)
        unknown = framed.terms(fact_rows(at((2.6, 3.0), (2.6, 10.0), (2.6, 3.0), gap_ms=None), None))
        self.assertAlmostEqual(unknown[0, 1], 0.0, places=9)

    def test_saved_and_read_back_with_an_interval_over_refits(self):
        framed = Model.framed(self.rows, ("hit smash", "hit lift"), True)
        weights = fit(framed, self.rows, self.won, 1e-3)
        model = replace(framed, weights=np.vstack([weights, weights + 0.1, weights - 0.1]))
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "value.json")
            again = Model.load(Path(folder) / "value.json")
        np.testing.assert_allclose(again.chances(self.rows[:5]), model.chances(self.rows[:5]), atol=1e-4)
        p, low, high = chance(model, at((2.6, 3.0), (2.6, 10.0), (2.6, 3.0)), None, np.random.default_rng(0))
        self.assertLess(low, p)
        self.assertLess(p, high)


if __name__ == "__main__":
    unittest.main()
