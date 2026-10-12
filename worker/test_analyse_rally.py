# worker/test_analyse_rally.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from analyse_rally import main
from evaluation.hits_benchmark import SIZE, Scenario, simulate
from test_find_hits import shuttle_doc, track_doc

SIM = simulate(Scenario(seed=22, jitter_px=1.0, false_per_s=0.0, neighbour_per_s=0.3, final_near_line=True))


class AnalyseRallyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.video = root / "clip.mp4"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"mp4v"), SIM.clip.fps, SIZE)
        for _ in SIM.track:
            writer.write(np.full((SIZE[1], SIZE[0], 3), 90, np.uint8))
        writer.release()
        (root / "clip.track.json").write_text(json.dumps(track_doc(SIM)), encoding="utf-8")
        (root / "clip.shuttle.json").write_text(json.dumps(shuttle_doc(SIM)), encoding="utf-8")
        args = [
            str(cls.video), "--track", str(root / "clip.track.json"), "--shuttle", str(root / "clip.shuttle.json"),
            "--out", str(root / "clip.analysis.json"), "--overlay", str(root / "overlay.mp4"), "--diagram", str(root / "court.png"),
        ]
        with mock.patch("find_hits.read_audio", return_value=SIM.clip.audio), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cls.code = main(args)
        cls.result = json.loads((root / "clip.analysis.json").read_text(encoding="utf-8"))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_shot_with_its_facts_and_the_last_one_called(self):
        self.assertEqual(self.code, 0)
        shots = self.result["shots"]
        truth = SIM.clip.rally.contacts
        self.assertGreaterEqual(len(shots), len(truth) - 1)
        self.assertTrue(shots[0]["serve"])
        for shot in shots:
            self.assertTrue({"hitMs", "hitter", "contact", "speedMS", "elevationDeg", "apexM", "net", "flightMs", "landing", "path"} <= set(shot))
            self.assertGreater(len(shot["path"]), 2)
        last = shots[-1]["landing"]
        self.assertIn(last["call"], ("in", "out", "too close to call"))
        self.assertLess(float(np.hypot(*(np.asarray(last["point"]) - np.asarray(SIM.clip.rally.landing)))), 0.5)
        self.assertEqual(self.result["court"], "singles")
        drag = self.result["summary"]["terminalVelocity"]
        self.assertIn("long flights", drag["from"])
        self.assertAlmostEqual(drag["value"], 6.7, delta=0.5)
        self.assertEqual(self.result["summary"]["shots"], len(shots))

    def test_the_overlay_has_every_frame_and_the_diagram_the_whole_court(self):
        overlay = cv2.VideoCapture(str(self.root / "overlay.mp4"))
        count = 0
        while overlay.read()[0]:
            count += 1
        overlay.release()
        self.assertEqual(count, len(SIM.track))
        diagram = cv2.imread(str(self.root / "court.png"))
        self.assertIsNotNone(diagram)
        self.assertGreater(diagram.shape[0], diagram.shape[1])  # the court is long

    def test_it_never_writes_over_its_inputs(self):
        shuttle = self.root / "clip.shuttle.json"
        before = shuttle.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = main([str(self.video), "--track", str(self.root / "clip.track.json"), "--shuttle", str(shuttle), "--out", str(shuttle)])
        self.assertEqual(code, 1)
        self.assertIn("overwrite", err.getvalue())
        self.assertEqual(shuttle.read_bytes(), before)

    def test_without_court_points_or_a_track_it_says_what_it_needs(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = main([str(self.video), "--out", str(self.root / "x.json"), "--shuttle", str(self.root / "clip.shuttle.json")])
        self.assertEqual(code, 1)
        self.assertIn("--golden", err.getvalue())


if __name__ == "__main__":
    unittest.main()
