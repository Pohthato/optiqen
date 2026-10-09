# worker/find_hits.py
"""Racket hits in a video, from its sound track and its shuttle detections.

    python find_hits.py clip.mp4 --shuttle clip.shuttle.json [--track clip.track.json] --out clip.hits.json

clip.shuttle.json comes from detect_shuttle.py; clip.track.json (optional) from track_camera.py,
which tells which half of the court each hit was on so sound delays are measured per half. The
output lists contact events in the worker's result format, so evaluation.evaluate scores them
against labelled hits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from audio.extract import SAMPLE_RATE, read_audio
from audio.onsets import detect_onsets
from geometry.camera import Camera
from hits import ONSET_GAP_MS, court_side, find_hits


def _cameras(track_doc: dict) -> list[Camera | None]:
    intrinsics = track_doc["intrinsics"]
    return [
        Camera(intrinsics["focalPx"], intrinsics["cx"], intrinsics["cy"], np.array(f["rvec"]), np.array(f["tvec"]), intrinsics["k1"])
        if "rvec" in f
        else None
        for f in track_doc["frames"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find racket hits from a video's sound and shuttle detections.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--shuttle", type=Path, required=True, help="detect_shuttle.py output for this video")
    parser.add_argument("--track", type=Path, help="track_camera.py output for this video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    shuttle = json.loads(args.shuttle.read_text(encoding="utf-8"))
    size = tuple(shuttle["imageSize"])
    # Filled gaps are InpaintNet's guesses, not sightings: a hit is judged on what was seen.
    track = [(f["x"], f["y"]) if "x" in f and not f.get("inpainted") else None for f in shuttle["frames"]]
    times_ms = [f["timeMs"] for f in shuttle["frames"]]
    sound_error = None
    try:
        audio = read_audio(args.video, SAMPLE_RATE)
    except RuntimeError as error:
        audio, sound_error = None, str(error)
        print(f"no sound used: {error}", file=sys.stderr)
    onsets = detect_onsets(audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS) if audio is not None else None
    side = None
    if args.track:
        cameras = _cameras(json.loads(args.track.read_text(encoding="utf-8")))

        def side(pixel: tuple[float, float], frame: int) -> str | None:
            camera = cameras[frame] if frame < len(cameras) else None
            return court_side(camera, pixel) if camera is not None else None

    hits = find_hits(track, times_ms, onsets, side=side, frame_size=size)
    events = [
        {
            "type": "contact",
            "timeMs": round(hit.time_ms, 1),
            "heardMs": None if hit.heard_ms is None else round(hit.heard_ms, 1),
            "evidence": hit.evidence,
            "x": round(hit.pixel[0], 1),
            "y": round(hit.pixel[1], 1),
            # A serve found from its sound and the flight that follows: x, y is the first sighting.
            "point": "first sighting" if hit.evidence == "sound and start" else "contact",
        }
        for hit in hits
    ]
    evidence = [hit.evidence for hit in hits]
    result = {
        "video": args.video.name,
        "fps": shuttle["fps"],
        "imageSize": list(size),
        "summary": {
            "hits": len(hits),
            "sound": onsets is not None,
            "soundError": sound_error,
            "byEvidence": {kind: evidence.count(kind) for kind in sorted(set(evidence))},
        },
        "events": events,
    }
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
