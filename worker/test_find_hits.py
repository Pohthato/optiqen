# worker/test_find_hits.py
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.evaluate import evaluate_clip
from evaluation.hits_benchmark import SIZE, Scenario, simulate
from find_hits import main


def shuttle_doc(sim, inpainted=()) -> dict:
    frames = []
    for index, (point, time_ms) in enumerate(zip(sim.track, sim.times_ms)):
        entry = {"timeMs": round(time_ms, 3)}
        if point is not None:
            entry.update(x=round(point[0], 2), y=round(point[1], 2), score=0.8, inpainted=index in inpainted)
        frames.append(entry)
    return {"fps": 0.0, "imageSize": list(SIZE), "model": "TrackNetV3", "frames": frames}  # fps metadata missing


class FindHitsCliTests(unittest.TestCase):
    def run_cli(self, doc, audio):
        with tempfile.TemporaryDirectory() as tmp:
            shuttle, out = Path(tmp) / "s.json", Path(tmp) / "hits.json"
            shuttle.write_text(json.dumps(doc), encoding="utf-8")
            with mock.patch("find_hits.read_audio", return_value=audio), contextlib.redirect_stdout(io.StringIO()):
                code = main([str(Path(tmp) / "clip.mp4"), "--shuttle", str(shuttle), "--out", str(out)])
            return code, json.loads(out.read_text(encoding="utf-8"))

    def test_writes_contact_events_the_evaluation_scores(self):
        sim = simulate(Scenario(seed=22, jitter_px=1.0, false_per_s=0.0, neighbour_per_s=0.3))
        code, result = self.run_cli(shuttle_doc(sim), sim.clip.audio)
        self.assertEqual(code, 0)
        self.assertTrue(result["summary"]["sound"])
        contacts = [event for event in result["events"] if event["type"] == "contact"]
        self.assertTrue(all({"timeMs", "heardMs", "evidence", "x", "y", "point"} <= set(event) for event in contacts))
        golden = {"clipId": "c", "imageSize": list(SIZE), "contacts": [{"timeMs": round(c.time * 1000)} for c in sim.clip.rally.contacts], "shots": []}
        self.assertGreaterEqual(evaluate_clip(golden, result)["contacts"]["tol33ms"]["f1"], 0.75)

    def test_filled_gaps_are_not_taken_as_sightings(self):
        sim = simulate(Scenario(seed=22, jitter_px=1.0, false_per_s=0.0, neighbour_per_s=0.0))
        everything_filled = set(range(len(sim.track)))
        code, result = self.run_cli(shuttle_doc(sim, inpainted=everything_filled), sim.clip.audio)
        self.assertEqual(code, 0)
        self.assertEqual(result["events"], [])

    def test_a_video_without_sound_still_gives_hits_from_the_flight(self):
        sim = simulate(Scenario(seed=22, jitter_px=1.0, false_per_s=0.0, neighbour_per_s=0.0, sound=False))
        code, result = self.run_cli(shuttle_doc(sim), None)
        self.assertEqual(code, 0)
        self.assertFalse(result["summary"]["sound"])
        self.assertTrue(all(event["heardMs"] is None for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
