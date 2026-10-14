# worker/test_lift_players.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import lift_players
import track_players
from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from geometry.skeleton import JOINTS
from simulation.camera_path import stable_handheld
from simulation.people import model_pose, playing
from simulation.rally import random_rally


def documents(seed=6, view="side", count=36):
    """A detector's people (each with a model's 3D pose), the camera track and the true bodies."""
    rng = np.random.default_rng(seed)
    rally = random_rally(rng)
    position, target = VIEWS[view]
    base = Camera.look_at(position, target, FOCAL_PX, SIZE)
    cameras = stable_handheld(base, count, 30.0, seed=seed)
    times = [i * 1000 / 30 for i in range(count)]
    frames, truth = [], {"near": [], "far": []}
    for i, (camera, t) in enumerate(zip(cameras, times)):
        people = []
        for side in ("far", "near"):
            joints = playing(rally, side, t / 1000)
            truth[side].append(joints)
            pixels = camera.project(joints) + rng.normal(0, 2.0, (17, 2))
            people.append({
                "keypoints": np.column_stack([pixels, np.full(17, 0.9)]).tolist(),
                "box": [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()],
                "world": model_pose(camera, joints[list(JOINTS)], rng).tolist(),
            })
        frames.append({"timeMs": t, "people": people})
    track = {
        "intrinsics": {"focalPx": base.focal_px, "cx": base.cx, "cy": base.cy, "k1": base.k1},
        "frames": [{"timeMs": t, "rvec": np.asarray(c.rvec).ravel().tolist(), "tvec": np.asarray(c.tvec).ravel().tolist()} for c, t in zip(cameras, times)],
    }
    return {"frames": frames}, track, {side: np.array(bodies) for side, bodies in truth.items()}


class LiftTests(unittest.TestCase):
    def test_from_a_detector_s_people_to_the_players_bodies(self):
        poses, track, truth = documents()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "poses.json").write_text(json.dumps(poses), encoding="utf-8")
            (root / "track.json").write_text(json.dumps(track), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(track_players.main(["--poses", str(root / "poses.json"), "--track", str(root / "track.json"), "--out", str(root / "players.json")]), 0)
                self.assertEqual(lift_players.main(["--players", str(root / "players.json"), "--track", str(root / "track.json"), "--out", str(root / "bodies.json")]), 0)
            result = json.loads((root / "bodies.json").read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["with3dPose"], 2 * len(track["frames"]))
        for side in ("near", "far"):
            bodies = result["bodies"][side]
            self.assertTrue(all(body is not None for body in bodies))
            self.assertTrue(all(body["joints"][1] is None and body["joints"][0] is not None for body in bodies))  # eyes not fitted
            fitted = np.array([[body["joints"][j] for j in JOINTS] for body in bodies], dtype=float)
            error = np.linalg.norm(fitted - truth[side][:, list(JOINTS)], axis=2)
            self.assertLess(np.median(error), 0.12, side)

    def test_refuses_players_of_other_frames(self):
        poses, track, _ = documents(count=8)
        players = {"players": {"near": [None] * 7, "far": [None] * 7}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "players.json").write_text(json.dumps(players), encoding="utf-8")
            (root / "track.json").write_text(json.dumps(track), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                code = lift_players.main(["--players", str(root / "players.json"), "--track", str(root / "track.json"), "--out", str(root / "bodies.json")])
            self.assertEqual(code, 1)
            self.assertFalse((root / "bodies.json").exists())


if __name__ == "__main__":
    unittest.main()
