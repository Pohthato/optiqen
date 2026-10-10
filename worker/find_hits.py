# worker/find_hits.py
"""Racket hits in a video, from its shuttle detections, its camera track and its sound.

    python find_hits.py clip.mp4 --shuttle clip.shuttle.json --track clip.track.json --out clip.hits.json

clip.shuttle.json comes from detect_shuttle.py and clip.track.json from track_camera.py: the
camera of every frame is what lets each flight be fitted in 3D. The output lists contact events
in the worker's result format, so evaluation.evaluate scores them against labelled hits, plus the
fitted flights between them.
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
from hits import ONSET_GAP_MS, find_hits


def _cameras(track_doc: dict) -> list[Camera | None]:
    intrinsics = track_doc["intrinsics"]
    return [
        Camera(intrinsics["focalPx"], intrinsics["cx"], intrinsics["cy"], np.array(f["rvec"]), np.array(f["tvec"]), intrinsics["k1"])
        if "rvec" in f
        else None
        for f in track_doc["frames"]
    ]


def _rounded(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find racket hits from a video's shuttle detections, camera track and sound.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--shuttle", type=Path, required=True, help="detect_shuttle.py output for this video")
    parser.add_argument("--track", type=Path, required=True, help="track_camera.py output for this video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    shuttle = json.loads(args.shuttle.read_text(encoding="utf-8"))
    cameras = _cameras(json.loads(args.track.read_text(encoding="utf-8")))
    if len(cameras) != len(shuttle["frames"]):
        print(f"the camera track has {len(cameras)} frames and the shuttle detections {len(shuttle['frames'])}", file=sys.stderr)
        return 1
    # Filled gaps are InpaintNet's guesses, not sightings, and a frame without a camera cannot be
    # placed in 3D: a hit is judged on what was seen through a known camera.
    track = [
        (f["x"], f["y"]) if "x" in f and not f.get("inpainted") and camera is not None else None
        for f, camera in zip(shuttle["frames"], cameras)
    ]
    known = next((camera for camera in cameras if camera is not None), None)
    cameras = [camera or known for camera in cameras]  # placeholders where nothing is seen
    times_ms = [f["timeMs"] for f in shuttle["frames"]]
    sound_error = None
    try:
        audio = read_audio(args.video, SAMPLE_RATE)
    except RuntimeError as error:
        audio, sound_error = None, str(error)
        print(f"no sound used: {error}", file=sys.stderr)
    onsets = detect_onsets(audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS) if audio is not None else None

    result = find_hits(track, times_ms, cameras, onsets) if known is not None else None
    hits = result.hits if result is not None else []
    events = [
        {
            "type": "contact",
            "timeMs": round(hit.time_ms, 1),
            "heardMs": None if hit.heard_ms is None else round(hit.heard_ms, 1),
            "evidence": hit.evidence,
            "position": None if hit.position is None else _rounded(hit.position, 3),  # court metres
            "frame": hit.frame,
        }
        for hit in hits
    ]
    flights = [
        {
            "startMs": round(piece.fit.start_ms, 1),
            "firstMs": round(times_ms[piece.indices[0]], 1),
            "lastMs": round(times_ms[piece.indices[-1]], 1),
            "p0": _rounded(piece.fit.p0, 3),
            "v0": _rounded(piece.fit.v0, 3),
            "rmsPx": round(piece.fit.rms_px, 2),
        }
        for piece in (result.flights if result is not None else [])
        if piece.fit is not None
    ]
    evidence = [hit.evidence for hit in hits]
    output = {
        "video": args.video.name,
        "fps": shuttle["fps"],
        "imageSize": shuttle["imageSize"],
        "summary": {
            "hits": len(hits),
            "sound": onsets is not None,
            "soundError": sound_error,
            "audioOffsetMs": None if result is None or result.audio_offset_ms is None else round(result.audio_offset_ms, 1),
            "noisePx": None if result is None else round(result.noise_px, 2),
            "byEvidence": {kind: evidence.count(kind) for kind in sorted(set(evidence))},
        },
        "events": events,
        "flights": flights,
    }
    args.out.write_text(json.dumps(output, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
