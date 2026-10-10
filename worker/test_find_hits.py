# worker/test_find_hits.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation.evaluate import evaluate_clip
from evaluation.hits_benchmark import SIZE, Scenario, simulate
from find_hits import main

SIM = simulate(Scenario(seed=22, jitter_px=1.0, false_per_s=0.0, neighbour_per_s=0.3))


def shuttle_doc(sim, inpainted=()) -> dict:
    frames = []
    for index, (point, time_ms) in enumerate(zip(sim.track, sim.times_ms)):
        entry = {"timeMs": round(time_ms, 3)}
        if point is not None:
            entry.update(x=round(point[0], 2), y=round(point[1], 2), score=0.8, inpainted=index in inpainted)
        frames.append(entry)
    return {"fps": 0.0, "imageSize": list(SIZE), "model": "TrackNetV3", "frames": frames}  # fps metadata missing


def track_doc(sim, lost=()) -> dict:
    camera = sim.clip.cameras[0]
    frames = []
    for index, (frame_camera, time_ms) in enumerate(zip(sim.clip.cameras, sim.times_ms)):
        entry = {"timeMs": round(time_ms, 3), "state": "lost" if index in lost else "tracked"}
        if index not in lost:
            entry.update(rvec=np.asarray(frame_camera.rvec).ravel().tolist(), tvec=np.asarray(frame_camera.tvec).ravel().tolist())
        frames.append(entry)
    return {"intrinsics": {"focalPx": camera.focal_px, "cx": camera.cx, "cy": camera.cy, "k1": camera.k1}, "frames": frames}


class FindHitsCliTests(unittest.TestCase):
    def run_cli(self, shuttle, track, audio):
        with tempfile.TemporaryDirectory() as tmp:
            shuttle_path, track_path, out = Path(tmp) / "s.json", Path(tmp) / "t.json", Path(tmp) / "hits.json"
            shuttle_path.write_text(json.dumps(shuttle), encoding="utf-8")
            track_path.write_text(json.dumps(track), encoding="utf-8")
            with mock.patch("find_hits.read_audio", return_value=audio), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main([str(Path(tmp) / "clip.mp4"), "--shuttle", str(shuttle_path), "--track", str(track_path), "--out", str(out)])
            return code, json.loads(out.read_text(encoding="utf-8")) if out.exists() else None

    def test_writes_contact_events_the_evaluation_scores_and_the_flights_between_them(self):
        code, result = self.run_cli(shuttle_doc(SIM), track_doc(SIM), SIM.clip.audio)
        self.assertEqual(code, 0)
        self.assertTrue(result["summary"]["sound"])
        contacts = [event for event in result["events"] if event["type"] == "contact"]
        self.assertTrue(all({"timeMs", "heardMs", "evidence", "position", "frame"} <= set(event) for event in contacts))
        self.assertTrue(result["flights"])
        self.assertTrue(all(len(flight["p0"]) == 3 and len(flight["v0"]) == 3 for flight in result["flights"]))
        golden = {"clipId": "c", "imageSize": list(SIZE), "contacts": [{"timeMs": round(c.time * 1000)} for c in SIM.clip.rally.contacts], "shots": []}
        self.assertGreaterEqual(evaluate_clip(golden, result)["contacts"]["tol33ms"]["f1"], 0.85)

    def test_filled_gaps_and_frames_without_a_camera_are_not_sightings(self):
        everything = set(range(len(SIM.track)))
        code, result = self.run_cli(shuttle_doc(SIM, inpainted=everything), track_doc(SIM), SIM.clip.audio)
        self.assertEqual((code, result["events"]), (0, []))
        code, result = self.run_cli(shuttle_doc(SIM), track_doc(SIM, lost=everything), SIM.clip.audio)
        self.assertEqual((code, result["events"]), (0, []))

    def test_a_track_of_another_video_is_refused(self):
        track = track_doc(SIM)
        track["frames"] = track["frames"][:-5]
        code, result = self.run_cli(shuttle_doc(SIM), track, SIM.clip.audio)
        self.assertEqual((code, result), (1, None))

    def test_a_video_without_sound_still_gives_hits_from_the_flights(self):
        code, result = self.run_cli(shuttle_doc(SIM), track_doc(SIM), None)
        self.assertEqual(code, 0)
        self.assertFalse(result["summary"]["sound"])
        self.assertTrue(result["events"])
        self.assertTrue(all(event["heardMs"] is None for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
