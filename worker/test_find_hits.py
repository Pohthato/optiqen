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
from find_hits import main
from test_hits import FPS, SIZE, clip_and_track


def shuttle_doc(track) -> dict:
    frames = []
    for index, point in enumerate(track):
        entry = {"timeMs": round(index * 1000 / FPS, 3)}
        if point is not None:
            entry.update(x=round(point[0], 2), y=round(point[1], 2), score=0.8, inpainted=False)
        frames.append(entry)
    return {"fps": FPS, "imageSize": list(SIZE), "model": "TrackNetV3", "frames": frames}


class FindHitsCliTests(unittest.TestCase):
    def run_cli(self, clip, track, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            shuttle, out = Path(tmp) / "s.json", Path(tmp) / "hits.json"
            shuttle.write_text(json.dumps(shuttle_doc(track)), encoding="utf-8")
            with mock.patch("find_hits.read_audio", return_value=clip.audio if clip is not None else None):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main([str(Path(tmp) / "clip.mp4"), "--shuttle", str(shuttle), "--out", str(out), *extra])
            return code, json.loads(out.read_text(encoding="utf-8"))

    def test_writes_contact_events_the_evaluation_scores(self):
        clip, track = clip_and_track(22)
        code, result = self.run_cli(clip, track)
        self.assertEqual(code, 0)
        self.assertTrue(result["summary"]["sound"])
        contacts = [event for event in result["events"] if event["type"] == "contact"]
        self.assertTrue(all({"timeMs", "heardMs", "evidence", "x", "y"} <= set(event) for event in contacts))
        golden = {"clipId": "c", "imageSize": list(SIZE), "contacts": [{"timeMs": round(c.time * 1000)} for c in clip.rally.contacts], "shots": []}
        report = evaluate_clip(golden, result)
        self.assertGreaterEqual(report["contacts"]["tol33ms"]["f1"], 0.9)

    def test_a_video_without_sound_still_gives_hits_from_the_flight(self):
        clip, track = clip_and_track(22, distractors=0)
        code, result = self.run_cli(None, track)
        self.assertEqual(code, 0)
        self.assertFalse(result["summary"]["sound"])
        self.assertTrue(result["events"])
        self.assertTrue(all(event["heardMs"] is None for event in result["events"]))


if __name__ == "__main__":
    unittest.main()
