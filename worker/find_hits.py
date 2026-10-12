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
from hits import ONSET_GAP_MS, RallyAnalysis, find_hits


def _cameras(track_doc: dict) -> list[Camera | None]:
    intrinsics = track_doc["intrinsics"]
    return [
        Camera(intrinsics["focalPx"], intrinsics["cx"], intrinsics["cy"], np.array(f["rvec"]), np.array(f["tvec"]), intrinsics["k1"])
        if "rvec" in f
        else None
        for f in track_doc["frames"]
    ]


def rounded(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def load_detections(shuttle_doc: dict, track_doc: dict) -> tuple[list, list[float], list[Camera | None]]:
    """The shuttle sightings (pixel or None), their times and every frame's camera (or None).
    Filled gaps are InpaintNet's guesses, not sightings, and a frame without a camera cannot be
    placed in 3D: a hit is judged on what was seen through a known camera. Raises ValueError when
    the two documents are not of the same frames."""
    cameras = _cameras(track_doc)
    times_ms = [f["timeMs"] for f in shuttle_doc["frames"]]
    if len(cameras) != len(times_ms) or any(abs(f["timeMs"] - t) > 1.0 for f, t in zip(track_doc["frames"], times_ms)):
        raise ValueError("the camera track and the shuttle detections are not of the same frames")
    track = [
        (f["x"], f["y"]) if "x" in f and not f.get("inpainted") and camera is not None else None
        for f, camera in zip(shuttle_doc["frames"], cameras)
    ]
    return track, times_ms, cameras


def hear(video: Path) -> tuple[list | None, str | None]:
    """The video's sound onsets for hit finding, or None and why there are none."""
    try:
        audio = read_audio(video, SAMPLE_RATE)
    except RuntimeError as error:
        return None, str(error)
    return (detect_onsets(audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS) if audio is not None else None), None


def hits_document(video: Path, shuttle_doc: dict, result: RallyAnalysis | None, onsets, sound_error: str | None, times_ms) -> dict:
    """The hits and flights in the worker's result format."""
    hits = result.hits if result is not None else []
    events = [
        {
            "type": "contact",
            "timeMs": round(hit.time_ms, 1),
            "heardMs": None if hit.heard_ms is None else round(hit.heard_ms, 1),
            "evidence": hit.evidence,
            "position": None if hit.position is None else rounded(hit.position, 3),  # court metres
            "frame": hit.frame,
        }
        for hit in hits
    ]
    flights = [
        {
            "startMs": round(piece.fit.start_ms, 1),
            "firstMs": round(times_ms[piece.indices[0]], 1),
            "lastMs": round(times_ms[piece.indices[-1]], 1),
            "p0": rounded(piece.fit.p0, 3),
            "v0": rounded(piece.fit.v0, 3),
            "rmsPx": round(piece.fit.rms_px, 2),
        }
        for piece in (result.flights if result is not None else [])
        if piece.fit is not None
    ]
    evidence = [hit.evidence for hit in hits]
    return {
        "video": video.name,
        "fps": shuttle_doc["fps"],
        "imageSize": shuttle_doc["imageSize"],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find racket hits from a video's shuttle detections, camera track and sound.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--shuttle", type=Path, required=True, help="detect_shuttle.py output for this video")
    parser.add_argument("--track", type=Path, required=True, help="track_camera.py output for this video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    shuttle = json.loads(args.shuttle.read_text(encoding="utf-8"))
    try:
        track, times_ms, cameras = load_detections(shuttle, json.loads(args.track.read_text(encoding="utf-8")))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    onsets, sound_error = hear(args.video)
    if sound_error:
        print(f"no sound used: {sound_error}", file=sys.stderr)
    result = find_hits(track, times_ms, cameras, onsets) if any(camera is not None for camera in cameras) else None
    output = hits_document(args.video, shuttle, result, onsets, sound_error, times_ms)
    args.out.write_text(json.dumps(output, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
