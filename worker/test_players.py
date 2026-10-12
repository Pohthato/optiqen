# worker/test_players.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from players import Person, floor_position, half_of, stands_there, track_players
from simulation.camera_path import stable_handheld
from simulation.people import bystanders, detect, skeleton
from simulation.rally import player_positions, random_rally
from track_players import main


def scene(seed: int, view: str, fps: float, miss: float = 0.05):
    """A random rally's two players and the people around the court, as a detector sees them
    from a steadily held phone: per frame the detections (shuffled), which of them is each
    player (or None when missed), the cameras and the times."""
    rng = np.random.default_rng(seed)
    rally = random_rally(rng)
    position, target = VIEWS[view]
    base = Camera.look_at(tuple(np.asarray(position) + rng.uniform(-0.5, 0.5, 3) * (1, 1, 0.4)), target, FOCAL_PX, SIZE)
    count = int(rally.end_time * fps)
    cameras = stable_handheld(base, count, fps, seed=seed)
    others = bystanders(rng)
    frames, truth = [], []
    for index in range(count):
        time = index / fps
        players = player_positions(rally, time)
        bodies = [("near", skeleton(players["near"], np.pi / 2)), ("far", skeleton(players["far"], -np.pi / 2, stature=1.8))]
        bodies += [(None, skeleton(where, facing, stature, raised)) for where, facing, stature, raised in others]
        seen = []
        for who, joints in bodies:
            if who is not None and rng.random() < miss:
                continue
            found = detect(joints, cameras[index], SIZE, rng)
            if found is not None:
                seen.append((who, Person(found.joints, found.scores, found.box)))
        order = rng.permutation(len(seen))
        people = [seen[k][1] for k in order]
        whose = {seen[k][0]: n for n, k in enumerate(order) if seen[k][0] is not None}
        frames.append(people)
        truth.append(whose)
    return frames, truth, cameras, [i * 1000.0 / fps for i in range(count)]


class FloorTests(unittest.TestCase):
    def test_a_standing_person_is_put_where_they_stand(self):
        camera = Camera.look_at(*VIEWS["corner"], FOCAL_PX, SIZE)
        found = detect(skeleton((2.0, 3.0), np.pi / 2), camera, SIZE, np.random.default_rng(0), noise_px=0.5)
        floor = floor_position(Person(found.joints, found.scores, found.box), camera)
        self.assertLess(float(np.hypot(floor[0] - 2.0, floor[1] - 3.0)), 0.1)

    def test_a_person_seated_high_is_not_where_their_feet_put_them(self):
        # From the side, the umpire's raised feet land in the court; their head is far too high above that.
        camera = Camera.look_at(*VIEWS["side"], FOCAL_PX, SIZE)
        rng = np.random.default_rng(1)
        umpire = detect(skeleton((-2.4, 6.6), 0.0, 1.7, raised=1.2), camera, SIZE, rng, noise_px=0.5)
        person = Person(umpire.joints, umpire.scores, umpire.box)
        floor = floor_position(person, camera)
        self.assertIsNotNone(half_of(floor))  # the feet alone would make the umpire a player
        self.assertFalse(stands_there(person, camera, floor))
        player = detect(skeleton((2.0, 4.0), np.pi / 2), camera, SIZE, rng, noise_px=0.5)
        player = Person(player.joints, player.scores, player.box)
        self.assertTrue(stands_there(player, camera, floor_position(player, camera)))

    def test_only_the_court_and_the_room_to_chase_around_it_are_play(self):
        self.assertEqual(half_of((2.5, 3.0)), "near")
        self.assertEqual(half_of((2.5, 14.5)), "far")  # chasing a clear behind the baseline
        self.assertIsNone(half_of((-2.4, 6.7)))  # the umpire's chair
        self.assertIsNone(half_of((9.7, 5.0)))  # the next court
        self.assertIsNone(half_of((2.0, 17.5)))  # spectators


class IdentityTests(unittest.TestCase):
    def test_each_player_is_found_among_everyone_else(self):
        right, total, strangers = 0, 0, 0
        for seed, view, fps in [(s, v, f) for s in (3, 4) for v in VIEWS for f in (30.0, 60.0)]:
            frames, truth, cameras, times = scene(seed, view, fps)
            tracks = track_players(frames, cameras, times)
            for half in ("near", "far"):
                for chosen, whose in zip(tracks[half], truth):
                    if half not in whose:
                        strangers += chosen is not None and chosen.detection not in whose.values()
                        continue
                    total += 1
                    right += chosen is not None and chosen.detection == whose[half]
        self.assertGreaterEqual(right / total, 0.99)
        self.assertEqual(strangers, 0)

    def test_without_a_camera_nobody_is_placed(self):
        frames, _, cameras, times = scene(5, "behind", 30.0)
        tracks = track_players(frames[:5], [None] * 5, times[:5])
        self.assertEqual(tracks, {"near": [None] * 5, "far": [None] * 5})


class CliTests(unittest.TestCase):
    def test_writes_each_halfs_player_per_frame(self):
        frames, truth, cameras, times = scene(7, "corner", 30.0)
        poses = {"frames": [
            {"timeMs": t, "people": [{"keypoints": np.column_stack([p.joints, p.scores]).tolist(), "box": p.box.tolist()} for p in people]}
            for people, t in zip(frames, times)
        ]}
        base = cameras[0]
        track = {
            "intrinsics": {"focalPx": base.focal_px, "cx": base.cx, "cy": base.cy, "k1": base.k1},
            "frames": [{"timeMs": t, "rvec": np.asarray(c.rvec).ravel().tolist(), "tvec": np.asarray(c.tvec).ravel().tolist()} for c, t in zip(cameras, times)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "p.json").write_text(json.dumps(poses), encoding="utf-8")
            (root / "t.json").write_text(json.dumps(track), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--poses", str(root / "p.json"), "--track", str(root / "t.json"), "--out", str(root / "out.json")])
            result = json.loads((root / "out.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        found = [entry for entry in result["players"]["near"] if entry is not None]
        self.assertGreater(len(found), 0.9 * len(times))
        self.assertTrue(all(entry["floor"][1] < 6.7 and len(entry["keypoints"]) == 17 for entry in found))
        self.assertTrue(all(entry is None or entry["floor"][1] > 6.7 for entry in result["players"]["far"]))


if __name__ == "__main__":
    unittest.main()
