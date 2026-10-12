# worker/track_players.py
"""The two players of a rally video, frame by frame, from a pose detector's people and the
camera track.

    python track_players.py --poses clip.poses.json --track clip.track.json --out clip.players.json

clip.poses.json lists, per frame, every person a pose detector found:
    {"frames": [{"timeMs": 0.0, "people": [{"keypoints": [[x, y, score], ... 17 COCO joints], "box": [x0, y0, x1, y1]}]}]}
Any detector will do; which one ships is a licensing decision (see the players-in-3D plan).
clip.track.json comes from track_camera.py. The output gives, per frame, each half's player:
where they stand on the court and their joints, or null where they were not found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from find_hits import _cameras, rounded
from players import HALVES, Person, track_players


def load_people(poses_doc: dict) -> tuple[list[list[Person]], list[float]]:
    frames, times = [], []
    for frame in poses_doc["frames"]:
        people = []
        for item in frame.get("people", []):
            joints = np.asarray(item["keypoints"], dtype=float).reshape(17, 3)
            box = np.asarray(item["box"], dtype=float) if item.get("box") is not None else None
            people.append(Person(joints[:, :2], joints[:, 2], box))
        frames.append(people)
        times.append(float(frame["timeMs"]))
    return frames, times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find the two players in every frame from a pose detector's people and the camera track.")
    parser.add_argument("--poses", type=Path, required=True, help="every person a pose detector found, per frame")
    parser.add_argument("--track", type=Path, required=True, help="track_camera.py output for the same video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    frames, times_ms = load_people(json.loads(args.poses.read_text(encoding="utf-8")))
    track_doc = json.loads(args.track.read_text(encoding="utf-8"))
    cameras = _cameras(track_doc)
    if len(cameras) != len(frames) or any(abs(f["timeMs"] - t) > 1.0 for f, t in zip(track_doc["frames"], times_ms)):
        print("the poses and the camera track are not of the same frames", file=sys.stderr)
        return 1
    tracks = track_players(frames, cameras, times_ms)
    result = {
        "summary": {"frames": len(frames), **{half: sum(entry is not None for entry in tracks[half]) for half in HALVES}},
        "conventions": {"floor": "court metres where the player stands (x across, y along)", "keypoints": "pixels and score per COCO joint"},
        "players": {
            half: [
                None if entry is None else {
                    "timeMs": round(time_ms, 3),
                    "floor": rounded(entry.floor, 3),
                    "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(s), 3)] for (x, y), s in zip(entry.joints, entry.scores)],
                }
                for entry, time_ms in zip(tracks[half], times_ms)
            ]
            for half in HALVES
        },
    }
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
