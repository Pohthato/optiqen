import unittest

from findings import TOO_CLOSE, findings


def estimate(value, sigma):
    return {"value": value, "sigma": sigma}


def shot(number, hitter, hit_ms, landing=None, pressure=None):
    return {
        "shot": number, "hitMs": hit_ms, "hitter": hitter, "serve": number == 1,
        "contact": [3.0, 2.0 if hitter == "near" else 11.0, 1.2], "contactSigma": [0.1, 0.2, 0.05],
        "speedMS": estimate(30.0, 2.0), "elevationDeg": estimate(10.0, 3.0), "apexM": estimate(2.5, 0.1),
        "net": {"xM": 3.0, "heightM": estimate(1.9, 0.1), "clearanceM": estimate(0.35, 0.08)},
        "name": {"call": "drop or smash", "p": {"drop": 0.5, "smash": 0.4, "clear": 0.1}, "said": "cross drop or smash",
                 "cross": {"call": "cross", "p": 0.9}, "backhand": None, "jump": {"call": "no jump", "p": 0.0}},
        "landing": landing, "pressure": pressure,
        "body": {"arm": "right", "elbowDeg": estimate(120.0, 10.0), "ready": {"fromBaseM": estimate(1.0, 0.2)}},
        "players": {"near": {"point": [3.0, 2.0], "covariance": [[0.04, 0.0], [0.0, 0.09]]}, "far": None},
        "value": {"near": {"p": 0.5, "interval": [0.45, 0.55]}, "far": {"p": 0.5, "interval": [0.45, 0.55]}},
    }


def analysis(end=None, reasons=None, value_turn=None, last_landing=None, players=None):
    shots = [
        shot(1, "near", 1000.0, pressure={"spareMs": estimate(400.0, 50.0)}),
        shot(2, "far", 2000.0, pressure={"spareMs": estimate(100.0, 40.0)}),
        shot(3, "near", 3000.0, landing=last_landing),
    ]
    events = [{"type": "contact", "timeMs": s["hitMs"], "frame": int(s["hitMs"] * 30 / 1000) + 1} for s in shots]
    return {
        "fps": 30.0, "events": events, "shots": shots,
        "summary": {"end": end, "reasons": reasons, "valueTurn": value_turn, "players": players},
    }


LANDING_OUT = {"point": [0.2, 13.6], "covariance": [[0.01, 0.0], [0.0, 0.04]], "timeMs": 3900.0, "call": "out", "pIn": 0.01, "pOverNet": 1.0}
END_OUT = {"shot": 3, "hitter": "near", "timeMs": 3900.0, "p": {"in": 0.01, "out": 0.99, "net": 0.0}, "call": "out", "winner": "far", "pHitterWon": 0.01}


def kinds(found, kind):
    return [f for f in found if f["kind"] == kind]


def measurements(facts):
    """Every measured number in a finding's facts, as the dict that holds it."""
    if isinstance(facts, dict):
        if "value" in facts and isinstance(facts["value"], (int, float)):
            yield facts
        for item in facts.values():
            yield from measurements(item)


class FindingTests(unittest.TestCase):
    def test_every_finding_has_an_id_its_frames_and_every_number_its_uncertainty(self):
        reasons = {"p": {"winner": 0.0, "let in": 0.0, "forced error": 0.97, "unforced error": 0.03}, "call": "forced error",
                   "turn": {"call": 2, "pAt": [0.0, 0.96, 0.0], "pNoTurn": 0.04}}
        players = {"near": {"frames": 90, "standingReachM": estimate(2.6, 0.02), "armReachM": estimate(1.39, 0.01), "racketArm": {"call": "right", "p": 0.9}}, "far": None}
        found = findings(analysis(END_OUT, reasons, {"call": "too close to call", "pAt": [0.0, 0.5, 0.3], "pNoTurn": 0.2}, LANDING_OUT, players))
        self.assertEqual([f["id"] for f in found], [f"F{i}" for i in range(1, len(found) + 1)])
        self.assertEqual([f["kind"] for f in found], ["end", "how", "turn", "valueTurn", "shot", "shot", "shot", "player"])
        for f in found:
            self.assertTrue(f["evidence"], f["id"])
            self.assertTrue(all(isinstance(e["frame"], int) for e in f["evidence"]))
            for measured in measurements(f["facts"]):
                self.assertTrue("sigma" in measured or "interval" in measured, (f["id"], measured))

    def test_frames_are_where_each_hit_was_seen_and_where_the_last_shot_came_down(self):
        found = findings(analysis(END_OUT, last_landing=LANDING_OUT))
        self.assertEqual([e["frame"] for e in kinds(found, "end")[0]["evidence"]], [91, 117])
        self.assertEqual([e["frame"] for e in kinds(found, "shot")[1]["evidence"]], [61])

    def test_a_call_left_too_close_stays_too_close(self):
        end = {**END_OUT, "p": {"in": 0.6, "out": 0.4, "net": 0.0}, "call": "too close to call", "winner": "too close to call", "pHitterWon": 0.6}
        reasons = {"p": {"winner": 0.6, "let in": 0.0, "forced error": 0.1, "unforced error": 0.3}, "call": "too close to call",
                   "turn": {"call": "too close to call", "pAt": [0.0, 0.3, 0.0], "pNoTurn": 0.7}}
        found = findings(analysis(end, reasons, last_landing={**LANDING_OUT, "call": "too close to call", "pIn": 0.6}))
        for f in found[:3]:
            self.assertEqual(f["call"], TOO_CLOSE, f["text"])
            self.assertIsNone(f["p"])
            self.assertIn("too close to call", f["text"])
        self.assertNotIn("won the point", found[0]["text"])
        self.assertEqual(found[2]["facts"]["leans"], "no turn")

    def test_who_each_reason_is_about(self):
        let_in = {"p": {"winner": 0.02, "let in": 0.98, "forced error": 0.0, "unforced error": 0.0}, "call": "let in", "turn": {"call": "no turn", "pAt": [0.0, 0.0, 0.0], "pNoTurn": 1.0}}
        end_in = {**END_OUT, "p": {"in": 0.99, "out": 0.01, "net": 0.0}, "call": "in", "winner": "near", "pHitterWon": 0.99}
        how = kinds(findings(analysis(end_in, let_in, last_landing=LANDING_OUT)), "how")[0]
        self.assertEqual((how["about"], how["shots"]), ("far", [3]))
        forced = {"p": {"winner": 0.0, "let in": 0.0, "forced error": 0.97, "unforced error": 0.03}, "call": "forced error", "turn": {"call": 2, "pAt": [0.0, 0.96, 0.0], "pNoTurn": 0.04}}
        found = findings(analysis(END_OUT, forced, last_landing=LANDING_OUT))
        how, turn = kinds(found, "how")[0], kinds(found, "turn")[0]
        self.assertEqual((how["about"], how["shots"], how["facts"]["spareMs"]["value"]), ("near", [2, 3], 100.0))
        self.assertEqual((turn["about"], turn["call"], turn["p"]), ("far", "shot 2", 0.96))

    def test_a_shot_says_its_name_as_a_coach_would_and_not_how_the_last_one_ended(self):
        found = findings(analysis(END_OUT, last_landing=LANDING_OUT))
        first, last = kinds(found, "shot")[0], kinds(found, "shot")[-1]
        self.assertEqual((first["call"], first["p"]), ("drop or smash", 0.9))
        self.assertIn("cross drop or smash", first["text"])
        self.assertNotIn("came down", last["text"])
        self.assertEqual(last["facts"]["landing"]["call"], "out")
        self.assertEqual(first["facts"]["receiver"]["player"], "far")

    def test_a_shot_that_did_not_get_over_the_net_says_so(self):
        found = findings(analysis(last_landing={**LANDING_OUT, "pOverNet": 0.01}))
        self.assertIn("did not get over the net", kinds(found, "shot")[-1]["text"])

    def test_without_an_end_there_are_only_the_shots(self):
        found = findings(analysis())
        self.assertEqual([f["kind"] for f in found], ["shot"] * 3)


if __name__ == "__main__":
    unittest.main()
