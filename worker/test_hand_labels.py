# worker/test_hand_labels.py
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evaluation.hand_labels import labels_for, main, match_clip, read_labels, summarise

LABELS = """// notes: push vs lift is subjective
Clip 3:
Serve time(far player serve):1.17(serve)
Near player hit times: 1.82(net), 3.38(cross push/lift),5.46 (cross block(near body)), 7.26(push, out long end of rally),
Far Player hit times: 2.51(chop net), 4.88(half smash: in, end of rally), 6.4(backhand cut off cross)

clip 4:
Serve time(near player serve):0.5
"""


def named(call):
    """A shot's name as the analysis gives it: alone at 0.9, or two names at 0.6 and 0.4."""
    first, *second = call.split(" or ")
    return {"call": call, "p": {first: 0.6, second[0]: 0.4} if second else {first: 0.9, "other": 0.1}}


def analysis(contacts, shots):
    return {
        "events": [{"type": "contact", "timeMs": t, "evidence": "flights"} for t in contacts],
        "shots": [{"shot": n, "hitMs": t, "hitter": who, **({"name": named(call)} if call else {})} for n, (t, who, call) in enumerate(shots, start=1)],
    }


class ReadLabelsTests(unittest.TestCase):
    def test_every_hit_of_every_clip_with_its_hitter_and_name(self):
        clips = read_labels(LABELS)
        self.assertEqual(sorted(clips), ["Clip 3", "Clip 4"])
        hits = clips["Clip 3"]
        self.assertEqual([h.time_ms for h in hits], [1170, 1820, 2510, 3380, 4880, 5460, 6400, 7260])
        self.assertEqual([h.hitter for h in hits], ["far", "near", "far", "near", "far", "near", "far", "near"])
        self.assertEqual([h.name for h in hits], ["serve", "net", "chop net", "cross push/lift", "half smash", "cross block", "backhand cut off cross", "push"])
        self.assertEqual(clips["Clip 4"][0].hitter, "near")

    def test_names_stand_for_the_labels_they_may_be(self):
        self.assertEqual(labels_for("cross push/lift"), ("push", "lift"))
        self.assertEqual(labels_for("lift/push"), ("push", "lift"))
        self.assertEqual(labels_for("backhand cut off cross"), ())
        self.assertEqual(labels_for("jump out stop drop: into the net"), ("drop",))
        self.assertEqual(labels_for("cross flat slice/drop"), ("drop",))
        self.assertEqual(labels_for("net from low"), ("net",))
        self.assertEqual(labels_for("lift from low"), ("lift",))
        self.assertEqual(labels_for("long net"), ("net",))
        self.assertEqual(labels_for("backhand save clear"), ("clear",))
        self.assertEqual(labels_for("waved at the crowd"), ())


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.hits = read_labels(LABELS)["Clip 3"]

    def test_hits_hitters_and_names_against_the_labels(self):
        found = analysis(
            [1150, 1850, 2600, 3300, 5500, 6400, 7200, 9000],
            [(1150, "far", "serve"), (1850, "near", "net"), (2600, "far", "drop or net"), (3300, "near", "lift"), (5500, "near", "net"), (7200, "far", "push"), (9000, "near", "clear")],
        )
        report = summarise(*match_clip(self.hits, found))
        self.assertEqual((report["labelled"], report["found"], report["matched"]), (8, 8, 7))
        self.assertEqual([row["timeMs"] for row in report["missed"]], [4880])
        self.assertEqual([row["timeMs"] for row in report["extra"]], [9000])
        self.assertEqual(report["timingMs"]["maxAbs"], 90.0)
        # 6.4 s was found with no shot: no hitter and no name. 7.26 s was given to the far player.
        self.assertEqual((report["hitterRight"], report["hitterKnown"]), (5, 6))
        self.assertEqual([row["timeMs"] for row in report["wrongHitter"]], [7260])
        names = report["names"]
        # Named: serve, net, drop for a net, lift for push or lift (right), net for a block, push.
        # The cut is not scored, so end to end adds only the missed smash.
        self.assertEqual((names["found"]["shots"], names["found"]["right"]), (6, 4))
        self.assertEqual((names["endToEnd"]["shots"], names["endToEnd"]["right"]), (7, 4))
        self.assertEqual(names["confusion"]["lift"], {"lift": 1})
        self.assertEqual(names["confusion"]["net"], {"net": 1, "drop": 1})
        self.assertLess(names["endToEnd"]["macroF1"], names["found"]["macroF1"])
        low, high = names["found"]["interval"]
        self.assertTrue(low <= names["found"]["macroF1"] <= high)
        # The model's own calls, the serve left out: net, lift, net (for the block) and push alone,
        # each said at 0.9; "drop or net" for the chop net has its label among the two.
        self.assertEqual(names["calls"], {"alone": {"given": 4, "right": 3, "meanP": 0.9}, "twoNames": {"given": 1, "labelAmong": 1}})

    def test_no_names_in_the_analysis_scores_hits_only(self):
        found = analysis([1170, 1820], [(1170, "far", None), (1820, "near", None)])
        report = summarise(*match_clip(self.hits, found))
        self.assertIsNone(report["names"])
        self.assertEqual(report["matched"], 2)

    def test_the_tolerance_decides_a_match(self):
        found = analysis([4880 + 240], [])
        self.assertEqual(summarise(*match_clip(self.hits, found))["matched"], 1)
        self.assertEqual(summarise(*match_clip(self.hits, found, tolerance_ms=200))["matched"], 0)

    def test_the_command_scores_each_clip_and_pools_them(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "hittimes.txt").write_text(LABELS, encoding="utf-8")
            (folder / "3.json").write_text(json.dumps(analysis([1170, 1820], [(1820, "near", "net")])), encoding="utf-8")
            (folder / "4.json").write_text(json.dumps(analysis([500, 3000], [])), encoding="utf-8")
            with redirect_stdout(StringIO()) as printed:
                main([str(folder / "hittimes.txt"), f"Clip 3={folder / '3.json'}", f"Clip 4={folder / '4.json'}", "--out", str(folder / "report.json")])
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        self.assertEqual((report["pooled"]["labelled"], report["pooled"]["found"], report["pooled"]["matched"]), (9, 4, 3))
        self.assertIn("All: labelled 9, found 4, matched 3", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
