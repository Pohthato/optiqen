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

from analyse_rally import find_players, main
from evaluation.hits_benchmark import SIZE, Scenario, simulate
from geometry.skeleton import JOINTS
from shot_names import Launch
from simulation.people import model_pose, playing, skeleton
from test_find_hits import shuttle_doc, track_doc

SIM = simulate(Scenario(seed=22, jitter_px=1.0, false_per_s=0.0, neighbour_per_s=0.3, final_near_line=True))


def poses_doc(sim, seed=3) -> dict:
    """What a pose detector reports: both players' joints with 3 px of noise and a learned 3D
    pose, and a spectator beside the court."""
    rng = np.random.default_rng(seed)
    frames = []
    for camera, time_ms in zip(sim.clip.cameras, sim.times_ms):
        people = []
        for side in ("near", "far"):
            joints = playing(sim.clip.rally, side, time_ms / 1000.0)
            pixels = camera.project(joints) + rng.normal(0, 3.0, (17, 2))
            people.append({
                "keypoints": np.column_stack([pixels, np.full(17, 0.9)]).tolist(),
                "box": [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()],
                "world": model_pose(camera, joints[list(JOINTS)], rng).tolist(),
            })
        spectator = camera.project(skeleton((-3.0, 9.0), 0.0))
        people.append({"keypoints": np.column_stack([spectator, np.full(17, 0.9)]).tolist(), "box": None, "world": None})
        frames.append({"timeMs": round(time_ms, 3), "people": people})
    return {"frames": frames}


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
        (root / "clip.poses.json").write_text(json.dumps(poses_doc(SIM)), encoding="utf-8")
        args = [
            str(cls.video), "--track", str(root / "clip.track.json"), "--shuttle", str(root / "clip.shuttle.json"), "--poses", str(root / "clip.poses.json"),
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
            self.assertTrue({"hitMs", "hitter", "contact", "speedMS", "elevationDeg", "apexM", "net", "flightMs", "landing", "path", "name"} <= set(shot))
            self.assertGreater(len(shot["path"]), 2)
            self.assertAlmostEqual(sum(shot["name"]["p"].values()), 1.0, delta=0.01)
            launch = Launch.entry(shot, self.result["summary"]["terminalVelocity"])
            np.testing.assert_allclose(launch.state[:3], shot["contact"], atol=1e-3)
            self.assertGreater(float(np.linalg.eigvalsh(launch.covariance).min()), 0.0)
        self.assertEqual(shots[0]["name"], {"call": "serve", "p": {"serve": 1.0}})
        self.assertEqual(sum(self.result["summary"]["names"].values()), len(shots))
        last = shots[-1]["landing"]
        self.assertIn(last["call"], ("in", "out", "too close to call"))
        self.assertLess(float(np.hypot(*(np.asarray(last["point"]) - np.asarray(SIM.clip.rally.landing)))), 0.5)
        self.assertEqual(self.result["court"], "singles")
        drag = self.result["summary"]["terminalVelocity"]
        self.assertIn("long flights", drag["from"])
        self.assertAlmostEqual(drag["value"], 6.7, delta=0.5)
        self.assertEqual(self.result["summary"]["shots"], len(shots))

    def test_every_shot_with_the_hitter_s_body_and_where_both_players_stood(self):
        shots = self.result["shots"]
        players = self.result["summary"]["players"]
        self.assertIsNone(self.result["summary"]["playersError"])
        for half in ("near", "far"):
            self.assertEqual(players[half]["frames"], len(SIM.track))
            self.assertGreater(players[half]["standingReachM"]["value"], 2.0)
        with_body = [shot for shot in shots if shot["body"] is not None]
        self.assertGreaterEqual(len(with_body), len(shots) - 1)
        for shot in with_body:
            body = shot["body"]
            self.assertIn(body["arm"], ("left", "right"))
            for name in ("aboveReachM", "aheadM", "asideM", "elbowDeg", "sideOnDeg", "hipsHeightM", "frontKneeDeg", "offFloorM", "weightAheadM"):
                self.assertGreater(body[name]["sigma"], 0.0, name)
            self.assertTrue(0.0 <= body["elbowDeg"]["value"] <= 180.0)
            self.assertTrue(0.4 <= body["hipsHeightM"]["value"] <= 1.3)
        self.assertTrue(any(shot["body"]["ready"] is not None for shot in with_body[1:]))
        misses = []
        for shot in shots:
            for half in ("near", "far"):
                hips = playing(SIM.clip.rally, half, shot["hitMs"] / 1000.0)[[11, 12], :2].mean(axis=0)
                stood = shot["players"][half]
                miss = np.asarray(stood["point"]) - hips
                misses.append(float(np.hypot(*miss)))
                self.assertLess(float(miss @ np.linalg.solve(np.asarray(stood["covariance"]), miss)), 13.8, (shot["shot"], half))  # 99.9 %
        self.assertLess(np.median(misses), 0.15)

    def test_without_the_pose_model_or_with_poses_of_other_frames_there_are_no_players(self):
        cameras, times_ms = [None] * 3, [0.0, 33.3, 66.7]
        bodies, why = find_players(self.video, self.root / "clip.track.json", self.root / "none.poses.json", True, self.root / "missing.task", cameras, times_ms)
        self.assertIsNone(bodies)
        self.assertIn("no pose model", why)
        self.assertFalse((self.root / "none.poses.json").exists())
        (self.root / "short.poses.json").write_text(json.dumps({"frames": [{"timeMs": 0.0, "people": []}]}), encoding="utf-8")
        bodies, why = find_players(self.video, self.root / "clip.track.json", self.root / "short.poses.json", False, self.root / "missing.task", cameras, times_ms)
        self.assertIsNone(bodies)
        self.assertIn("not of the same frames", why)

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
