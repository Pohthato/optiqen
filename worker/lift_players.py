# worker/lift_players.py
"""The two players' bodies in 3D, frame by frame, from their tracked joints and the camera track.

    python lift_players.py --players clip.players.json --track clip.track.json --out clip.bodies.json

clip.players.json comes from track_players.py (with each player's 3D pose from the detector, when
it gave one); clip.track.json from track_camera.py. The output gives, per frame, each half's
player as 17 COCO joints in court metres (null for the eyes and ears, which are not fitted),
each joint's standard deviation in metres, and how far the fitted joints miss the detected ones;
null where the player was not found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from find_hits import _cameras
from geometry.camera import Camera
from geometry.skeleton import Body, fit_bodies
from players import HALVES

FACING = {"near": np.pi / 2, "far": -np.pi / 2}  # towards the net


def fit_players(players_doc: dict, cameras: Sequence[Camera | None], times_ms: Sequence[float]) -> tuple[dict[str, list[Body | None]], int]:
    """Each half's body per frame (None where the player was not found), and how many of the
    players' frames had the detector's 3D pose."""
    bodies, with_pose = {}, 0
    for half in HALVES:
        frames, priors = [], []
        for entry in players_doc["players"][half]:
            if entry is None:
                frames.append(None)
                priors.append(None)
                continue
            keypoints = np.asarray(entry["keypoints"], dtype=float).reshape(17, 3)
            frames.append((keypoints[:, :2], keypoints[:, 2]))
            world = entry.get("world")
            priors.append(np.asarray(world, dtype=float) if world is not None else None)
            with_pose += world is not None
        bodies[half] = fit_bodies(frames, cameras, times_ms, facing=FACING[half], priors=priors)
    return bodies, with_pose


def lift(players_doc: dict, track_doc: dict) -> dict:
    times_ms = [float(frame["timeMs"]) for frame in track_doc["frames"]]
    bodies, with_pose = fit_players(players_doc, _cameras(track_doc), times_ms)
    return {
        "summary": {
            "frames": len(times_ms),
            **{half: sum(body is not None for body in bodies[half]) for half in HALVES},
            "with3dPose": with_pose,
            "medianSigmaM": {
                half: round(float(np.nanmedian([b.sigma for b in bodies[half] if b is not None])), 3) if any(bodies[half]) else None
                for half in HALVES
            },
        },
        "conventions": {
            "joints": "court metres (x across, y along, z up) per COCO joint; null where not fitted",
            "sigma": "one standard deviation of each joint's position, metres",
            "rmsPx": "how far the fitted joints miss the detected ones",
        },
        "bodies": {
            half: [
                None if body is None else {
                    "timeMs": round(time_ms, 3),
                    "joints": [None if not np.isfinite(j).all() else [round(float(v), 3) for v in j] for j in body.joints],
                    "sigma": [None if not np.isfinite(s) else round(float(s), 3) for s in body.sigma],
                    "rmsPx": round(body.rms_px, 2),
                }
                for body, time_ms in zip(bodies[half], times_ms)
            ]
            for half in HALVES
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the two players' bodies in 3D from their tracked joints.")
    parser.add_argument("--players", type=Path, required=True, help="track_players.py output")
    parser.add_argument("--track", type=Path, required=True, help="track_camera.py output for the same video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    players_doc = json.loads(args.players.read_text(encoding="utf-8"))
    track_doc = json.loads(args.track.read_text(encoding="utf-8"))
    if any(len(players_doc["players"][half]) != len(track_doc["frames"]) for half in HALVES):
        print("the players and the camera track are not of the same frames", file=sys.stderr)
        return 1
    result = lift(players_doc, track_doc)
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
