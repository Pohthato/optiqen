# worker/test_shot_names.py
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from body import hit_body, standing_reach
from evaluation.hits_benchmark import Scenario
from evaluation.hand_labels import read_labels
from evaluation.shot_names_benchmark import REAL_HOLDS, TEST_FIRST_SEED, adapt, learning_set, measure_scenario, named_again, real_set, score_cross
from geometry.court_model import NET_Y_M, SINGLES_WIDTH_M
from geometry.shuttle_physics import simulate
from shot_names import (CROSS_M, DRAWS, FACTS, JUMP_M, NAMES, SIMULATED_PATH, SURE, TOO_CLOSE, Detail, Launch, Model, ShotName, backhand, call_of, chances,
                        detail, facts, jump, learn, measured_facts, name_shot, nominal_facts, racket_arms, true_facts)
from shots import Estimate, flight_facts
from simulation.rally import SHOT_TYPES, canned_rally
from test_body import exact, standing
from test_shots import exact_fit

COLUMN = {name: k for k, name in enumerate(FACTS)}


class FactTests(unittest.TestCase):
    def test_the_facts_of_true_flights_are_exact(self):
        rally = canned_rally()
        names, rows, crossed = true_facts([rally, rally])
        self.assertEqual(names, [c.kind for c in rally.contacts] * 2)
        np.testing.assert_array_equal(rows[: len(rally.contacts)], rows[len(rally.contacts):])
        ends = [c.position[0] for c in rally.contacts[1:]] + [rally.landing[0]]
        middle = SINGLES_WIDTH_M / 2
        want = [(c.position[0] - middle) * (x - middle) < 0 and abs(x - c.position[0]) >= CROSS_M for c, x in zip(rally.contacts, ends)]
        self.assertEqual(crossed[: len(rally.contacts)].tolist(), [float(w) for w in want])
        for n, (row, contact, flight) in enumerate(zip(rows, rally.contacts, rally.flights)):
            side = 1.0 if contact.hitter == "near" else -1.0
            self.assertAlmostEqual(row[COLUMN["contactHeightM"]], contact.position[2], places=9)
            self.assertAlmostEqual(row[COLUMN["contactFromNetM"]], side * (NET_Y_M - contact.position[1]), places=9)
            self.assertAlmostEqual(row[COLUMN["speedMS"]], float(np.linalg.norm(flight.v0)), places=9)
            path = simulate(flight.p0, flight.v0, np.arange(0.0, flight.end_time - flight.start_time, 0.001))
            self.assertAlmostEqual(row[COLUMN["apexAboveM"]], float(path[:, 2].max()) - contact.position[2], places=2)
            self.assertAlmostEqual(row[COLUMN["flightS"]], flight.end_time - flight.start_time, delta=0.003)
            if n + 1 < len(rally.contacts):  # played again where the next contact is
                following = rally.contacts[n + 1].position
                self.assertAlmostEqual(row[COLUMN["endPastNetM"]], side * (following[1] - NET_Y_M), delta=0.02)
                self.assertAlmostEqual(row[COLUMN["endHeightM"]], following[2], delta=0.02)
            else:  # the last lands
                self.assertAlmostEqual(row[COLUMN["endPastNetM"]], side * (rally.landing[1] - NET_Y_M), delta=0.02)
                self.assertAlmostEqual(row[COLUMN["endHeightM"]], 0.0, places=6)

    def test_a_far_players_shot_reads_like_the_same_shot_from_the_near_half(self):
        p, v = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        mirrored_p, mirrored_v = np.array([p[0], 2 * NET_Y_M - p[1], p[2]]), v * [1.0, -1.0, 1.0]
        near, near_cross = facts(p, v, True, 0.9, 6.7)
        far, far_cross = facts(mirrored_p, mirrored_v, False, 0.9, 6.7)
        np.testing.assert_allclose(near, far, atol=1e-9)
        np.testing.assert_array_equal(near_cross, far_cross)

    def test_a_flight_on_the_floor_before_it_was_played_again_ends_there(self):
        p, v = np.array([2.0, 5.0, 1.0]), np.array([0.0, 8.0, 2.0])
        row = facts(p, v, True, 2.5, 6.7)[0][0]
        self.assertAlmostEqual(row[COLUMN["endHeightM"]], 0.0, places=6)
        self.assertLess(row[COLUMN["flightS"]], 2.5)

    def test_a_measured_shot_is_drawn_whole_from_its_uncertainty(self):
        p0, v0 = np.array([2.0, 2.5, 2.6]), np.array([0.5, 18.0, 6.0])
        sure = flight_facts(exact_fit(p0, v0, sigma=1e-4), 0.0, "near", "singles", ends_ms=900.0)
        unsure = flight_facts(exact_fit(p0, v0, sigma=0.1), 0.0, "near", "singles", ends_ms=900.0)
        rng = np.random.default_rng(0)
        nominal = nominal_facts(Launch.of(sure))
        np.testing.assert_allclose(nominal, facts(p0, v0, True, 0.9, 6.7)[0][0], atol=1e-6)
        drawn, _ = measured_facts(Launch.of(sure), rng)
        np.testing.assert_allclose(drawn.mean(axis=0), nominal, atol=0.05)
        spread = measured_facts(Launch.of(unsure), rng)[0].std(axis=0)
        flown = [COLUMN[name] for name in FACTS if name != "flightS"]  # played again at a known time
        self.assertTrue((spread[flown] > drawn.std(axis=0)[flown]).all())
        self.assertAlmostEqual(spread[COLUMN["flightS"]], 0.0, places=9)
        self.assertAlmostEqual(spread[COLUMN["speedMS"]], unsure.speed_m_s.sigma, delta=0.2 * unsure.speed_m_s.sigma)


def body_facts(arm: str, aside: float, off_floor: float = 0.0, hitter: str = "near", sigma: float = 0.05):
    """The facts of a body built to have them, standing still: the contact `aside` of the hips'
    middle towards the `arm` hand, which meets it, the feet `off_floor`."""
    joints = standing(lift=off_floor)
    hips = (joints[11] + joints[12]) / 2
    side = 1.0 if arm == "right" else -1.0
    contact = hips + np.array([side * aside, 0.4, 1.0])
    hand = contact - np.array([0.0, 0.0, 0.3])
    joints = standing(lift=off_floor, right_hand=hand) if arm == "right" else standing(lift=off_floor)
    if arm == "left":
        joints[9] = hand
    facts = hit_body(exact(joints), [0.0], 0.0, "near", contact, (0.0, 0.0, 0.0), standing_reach(exact(standing())))
    return replace(facts, hitter=hitter, aside_m=Estimate(facts.aside_m.value, sigma), off_floor_m=Estimate(facts.off_floor_m.value, sigma))


class DetailTests(unittest.TestCase):
    def test_each_detail_is_said_only_when_sure_either_way(self):
        self.assertEqual(detail(SURE, "cross", "straight"), Detail("cross", SURE))
        self.assertEqual(detail(1 - SURE, "cross", "straight").call, "straight")
        self.assertEqual(detail(0.2 + 1e-12, "cross", "straight").call, "straight")  # 0.2 as given
        self.assertEqual(detail(0.5, "cross", "straight").call, TOO_CLOSE)

    def test_cross_is_to_the_other_side_of_the_centre_line_and_far_enough_across(self):
        middle = SINGLES_WIDTH_M / 2
        p = np.array([[middle - 1.5, 2.0, 1.0]] * 3)
        v = np.array([[5.0, 9.0, 3.0], [0.4, 9.0, 3.0], [-0.5, 9.0, 3.0]])  # over the line, straight on, away from it
        _, crossed = facts(p, v, True, 1.0, 6.7)
        self.assertEqual(crossed.tolist(), [1.0, 0.0, 0.0])
        _, drifted = facts([middle - 0.3, 2.0, 1.0], [1.5, 9.0, 3.0], True, 1.0, 6.7)  # over the line, but not CROSS_M across
        self.assertEqual(drifted.tolist(), [0.0])

    def test_a_shot_that_ends_about_the_line_is_too_close_to_call(self):
        model, rng = Model.load(SIMULATED_PATH), np.random.default_rng(0)
        middle = SINGLES_WIDTH_M / 2
        cases = {}
        for vx in (5.0, 2.75, 0.3):  # well over, about CROSS_M across, straight
            p0, v0 = np.array([middle - 1.0, 2.5, 1.0]), np.array([vx, 9.0, 3.0])
            shot = flight_facts(exact_fit(p0, v0, sigma=0.03), 0.0, "near", "singles", ends_ms=1000.0)
            cases[vx] = name_shot(Launch.of(shot), False, model, rng).cross
        self.assertEqual(cases[5.0].call, "cross")
        self.assertEqual(cases[2.75].call, TOO_CLOSE, cases[2.75])
        self.assertEqual(cases[0.3].call, "straight")

    def test_backhand_is_across_the_body_from_the_racket_arm(self):
        right, left = Detail("right", 1.0), Detail("left", 0.0)
        self.assertEqual(backhand(body_facts("right", 0.4), right).call, "forehand")
        self.assertEqual(backhand(body_facts("right", -0.4), right).call, "backhand")
        self.assertEqual(backhand(body_facts("right", -0.02), right).call, TOO_CLOSE)
        # The other wrist was nearer the contact: the contact was on the hips' left all the same.
        self.assertEqual(backhand(body_facts("left", 0.4), right).call, "backhand")
        self.assertEqual(backhand(body_facts("left", 0.4), left).call, "forehand")
        self.assertIsNone(backhand(body_facts("right", 0.4), None))
        self.assertIsNone(backhand(None, right))

    def test_the_racket_arm_is_as_sure_as_the_player_s_shots_agree_on_it(self):
        bodies = [body_facts("right", 0.4)] * 5 + [body_facts("left", 0.4), body_facts("left", 0.4, hitter="far")]
        arms = racket_arms(bodies)
        self.assertEqual((arms["near"], arms["far"]), (Detail("right", 0.833), Detail("left", 0.0)))
        self.assertIsNone(racket_arms(bodies[:-1])["far"])
        # Split three to two, a shot played clearly on one side is too close to call.
        split = racket_arms([body_facts("right", 0.4)] * 2 + [body_facts("left", -0.4)] * 3)["near"]
        self.assertEqual(split, Detail(TOO_CLOSE, 0.4))
        self.assertEqual(backhand(body_facts("right", 0.4), split), Detail(TOO_CLOSE, 0.6))
        self.assertEqual(backhand(body_facts("right", 0.4), arms["near"]).call, "forehand")

    def test_a_jump_is_the_lower_foot_clear_of_tiptoe(self):
        self.assertEqual(jump(body_facts("right", 0.4, off_floor=0.35)).call, "jump")
        self.assertEqual(jump(body_facts("right", 0.4)).call, "no jump")
        self.assertEqual(jump(body_facts("right", 0.4, off_floor=JUMP_M)).call, TOO_CLOSE)
        self.assertIsNone(jump(None))

    def test_the_name_as_a_coach_says_it_holds_only_the_sure_details(self):
        name = ShotName("smash", {"smash": 0.9}, Detail("cross", 0.95), Detail("forehand", 0.01), Detail(TOO_CLOSE, 0.6))
        self.assertEqual(name.said, "cross smash")
        self.assertEqual(ShotName("drop or smash", {}, Detail("straight", 0.0), Detail("backhand", 0.9), Detail("jump", 0.99)).said, "backhand jump drop or smash")
        entry = name.entry()
        self.assertEqual(entry["cross"], {"call": "cross", "p": 0.95})
        self.assertEqual(entry["said"], "cross smash")
        self.assertEqual(ShotName("serve", {"serve": 1.0}).entry(), {"call": "serve", "p": {"serve": 1.0}, "said": "serve", "cross": None, "backhand": None, "jump": None})


class ModelTests(unittest.TestCase):
    def test_learning_tells_apart_names_that_the_facts_tell_apart(self):
        rng = np.random.default_rng(1)
        centres = {"clear": 3.0, "net": -3.0, "drive": 0.0}
        names = [name for name in centres for _ in range(300)]
        rows = np.array([rng.normal(centres[name], 1.0, len(FACTS)) for name in names])
        model = learn(names, rows, 1e-3)
        p = model.probabilities(np.array([np.full(len(FACTS), c) for c in centres.values()]))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertEqual([NAMES[k] for k in p.argmax(axis=1)], list(centres))
        self.assertTrue((p.max(axis=1) > 0.95).all())
        # A fact far beyond anything learnt from reads as the edge of what was.
        far = model.probabilities(np.full((1, len(FACTS)), 50.0))
        np.testing.assert_allclose(far, model.probabilities(rows.max(axis=0)[None]), atol=1e-12)

    def test_a_model_is_kept_and_read_back(self):
        model = Model(np.zeros(len(FACTS)), np.ones(len(FACTS)), np.full(len(FACTS), 0.5), np.ones(len(FACTS)), np.arange(45 * len(NAMES), dtype=float).reshape(45, -1) / 1000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "names.json"
            model.save(path)
            again = Model.load(path)
            np.testing.assert_allclose(again.weights, model.weights)
            stored = json.loads(path.read_text(encoding="utf-8"))
            stored["facts"] = stored["facts"][:-1]
            path.write_text(json.dumps(stored), encoding="utf-8")
            with self.assertRaises(ValueError):
                Model.load(path)

    def test_a_draw_that_could_not_be_flown_is_left_out(self):
        model = Model.load(SIMULATED_PATH)
        rng = np.random.default_rng(2)
        drawn = model.mean + rng.normal(0.0, 0.3, (5, len(FACTS))) * model.scale
        spoilt = np.vstack([drawn, np.full(len(FACTS), np.nan)])
        np.testing.assert_allclose(chances(model, spoilt), model.probabilities(drawn).mean(axis=0))
        np.testing.assert_allclose(chances(model, np.stack([spoilt, spoilt])), np.stack([chances(model, drawn)] * 2))

    def test_a_name_is_given_alone_only_when_sure(self):
        p = np.zeros(len(NAMES))
        p[NAMES.index("drop")], p[NAMES.index("smash")] = SURE, 1 - SURE
        self.assertEqual(call_of(p), "drop")
        p[NAMES.index("drop")], p[NAMES.index("smash")] = 0.3, 0.7
        self.assertEqual(call_of(p), "smash or drop")

    def test_the_model_learnt_from_simulation_names_the_simulators_canned_rally(self):
        model, rally, rng = Model.load(SIMULATED_PATH), canned_rally(), np.random.default_rng(0)
        _, _, crossed = true_facts([rally])
        calls = []
        for n, (contact, flight) in enumerate(zip(rally.contacts, rally.flights)):
            ends = flight.end_time * 1000.0 if n + 1 < len(rally.flights) else None
            shot = flight_facts(exact_fit(flight.p0, flight.v0, start_ms=flight.start_time * 1000.0, sigma=0.01), flight.start_time * 1000.0, contact.hitter, "singles", ends, serve=n == 0)
            calls.append(name_shot(Launch.of(shot), shot.serve, model, rng))
        self.assertEqual(calls[0].call, "serve")
        right = [call.call.split(" or ")[0] == contact.kind for call, contact in zip(calls[1:], rally.contacts[1:])]
        self.assertGreaterEqual(sum(right), len(right) - 1, [c.call for c in calls])
        self.assertTrue(all(abs(sum(call.p.values()) - 1.0) < 0.01 for call in calls))
        self.assertIsNone(calls[0].cross)
        self.assertEqual([call.cross.call for call in calls[1:]], ["cross" if x else "straight" for x in crossed[1:]])


def canned_clip() -> tuple[str, dict]:
    """The canned rally named by hand, its block and net shot named the other way round from the
    simulator, and as an analysis file (only what names need of each shot)."""
    rally = canned_rally()
    shots = []
    for n, (contact, flight) in enumerate(zip(rally.contacts, rally.flights)):
        ends = flight.end_time * 1000.0 if n + 1 < len(rally.flights) else None
        shot = flight_facts(exact_fit(flight.p0, flight.v0, start_ms=flight.start_time * 1000.0, sigma=0.01), flight.start_time * 1000.0, contact.hitter, "singles", ends, serve=n == 0)
        shots.append({
            "shot": n + 1, "hitMs": shot.hit_ms, "hitter": shot.hitter, "serve": shot.serve, "flightMs": shot.flight_ms,
            "landing": None if shot.landing is None else {}, "launch": {"value": [*shot.contact, *shot.velocity], "covariance": [list(row) for row in shot.launch_covariance]},
            "name": {"call": contact.kind, "p": {contact.kind: 1.0}, "backhand": {"call": "forehand", "p": 0.01}, "jump": None},  # as first named
        })
    swapped = {"block": "net", "net": "block"}
    lines = [f"Serve time({rally.contacts[0].hitter} player serve):{rally.contacts[0].time}"]
    for half in ("near", "far"):
        lines.append(f"{half} player hit times:" + ",".join(f"{c.time}({swapped.get(c.kind, c.kind)})" for c in rally.contacts[1:] if c.hitter == half))
    analysis = {"events": [{"type": "contact", "timeMs": shot["hitMs"], "evidence": "flights"} for shot in shots], "shots": shots, "summary": {"terminalVelocity": {"value": 6.7, "sigma": 0.05}}}
    return "\n".join(lines), analysis


class RealTests(unittest.TestCase):
    def test_a_strong_hold_keeps_the_prior_but_for_its_sureness_and_a_weak_one_learns_the_new_shots(self):
        prior = Model.load(SIMULATED_PATH)
        rows = np.array([np.full(len(FACTS), 0.5), np.full(len(FACTS), -0.5)]) * prior.scale + prior.mean
        names = [("push", "lift"), "net"]
        held = learn(names, rows, 1e6, prior).weights
        scale = float((held * prior.weights).sum() / (prior.weights**2).sum())
        np.testing.assert_allclose(held, scale * prior.weights, atol=1e-3)
        p = learn(names, rows, 1e-6, prior).probabilities(rows)
        self.assertAlmostEqual(p[0, NAMES.index("push")], p[0, NAMES.index("lift")], delta=0.05)
        self.assertGreater(p[0, NAMES.index("push")] + p[0, NAMES.index("lift")], 0.95)
        self.assertGreater(p[1, NAMES.index("net")], 0.95)

    def test_clips_named_by_hand_move_the_names_and_each_is_named_by_a_model_learnt_without_it(self):
        text, analysis = canned_clip()
        labelled = {f"Clip {n}": read_labels(f"Clip {n}:\n" + text)[f"Clip {n}"] for n in (1, 2, 3)}
        analyses = {clip: analysis for clip in labelled}
        rows = real_set(labelled, analyses)
        self.assertEqual(len(rows), 3 * (len(analysis["shots"]) - 1))  # every shot but the serve
        prior = Model.load(SIMULATED_PATH)
        model, report, named = adapt(prior, labelled, analyses)
        self.assertEqual(report["shots"], len(rows))
        self.assertLess(report["hold"], max(REAL_HOLDS))  # the clips say more than the simulator
        kinds = [c.kind for c in canned_rally().contacts]
        for clip in labelled:
            calls = [shot["name"]["call"] for shot in named[clip]["shots"]]
            self.assertTrue(all(shot["name"]["backhand"] == {"call": "forehand", "p": 0.01} for shot in named[clip]["shots"]))
            self.assertTrue(all(shot["name"]["cross"]["call"] in ("cross", "straight") for shot in named[clip]["shots"][1:]))
            self.assertEqual(calls[0], "serve")
            self.assertEqual(calls[kinds.index("block")], "net", calls)
            self.assertEqual(calls[kinds.index("net")], "block", calls)
        self.assertEqual(named_again(analysis, prior)["shots"][kinds.index("block")]["name"]["call"].split(" or ")[0], "block")


class SetTests(unittest.TestCase):
    def test_learning_shots_are_named_repeatably_and_never_serves(self):
        names, rows = learning_set(30)
        self.assertEqual(rows.shape, (len(names), len(FACTS)))
        self.assertNotIn("serve", names)
        self.assertLessEqual(set(names), set(SHOT_TYPES))
        self.assertEqual(learning_set(30)[0], names)
        self.assertNotEqual(learning_set(30, first_seed=30)[0], names)

    def test_a_held_out_scenarios_found_shots_keep_their_truth_and_draws(self):
        rows = measure_scenario(Scenario(seed=TEST_FIRST_SEED, view="behind", fps=30.0))
        self.assertGreater(len(rows), 1)
        for row in rows:
            self.assertIn(row["name"], SHOT_TYPES)
            self.assertEqual(row["draws"].shape, (DRAWS, len(FACTS)))
            self.assertLess(abs(row["nominal"][COLUMN["contactHeightM"]] - row["truth"][COLUMN["contactHeightM"]]), 0.5)
            self.assertIn(row["cross"], (0.0, 1.0))
            self.assertTrue(0.0 <= row["crossDrawn"] <= 1.0)

    def test_cross_or_straight_is_scored_by_each_call(self):
        report = score_cross({"cross": np.array([1.0, 1.0, 0.0, 0.0, 1.0]), "crossDrawn": np.array([0.9, 0.85, 0.85, 0.1, 0.5])})
        self.assertEqual((report["shots"], report["trulyCross"]), (5, 3))
        self.assertEqual(report["cross"], {"given": 3, "meanP": 0.867, "trulyCross": 2})
        self.assertEqual(report["straight"], {"given": 1, "meanP": 0.1, "trulyCross": 0})
        self.assertEqual(report[TOO_CLOSE], {"given": 1, "meanP": 0.5, "trulyCross": 1})


if __name__ == "__main__":
    unittest.main()
