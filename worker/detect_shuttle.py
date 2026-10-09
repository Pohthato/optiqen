# worker/detect_shuttle.py
"""The shuttle in every frame of a video, with TrackNetV3.

    python detect_shuttle.py clip.mp4 --out clip.shuttle.json [--overlay clip.shuttle.mp4] [--fast]

Writes, per frame, the video timestamp and the shuttle's pixel position with a score, or only the
timestamp when it is not seen. Filled gaps are marked "inpainted". --fast runs each frame through
one window instead of eight overlapping ones: about 8x quicker, a little less accurate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from shuttle.tracknet import BACKGROUND_SAMPLES, ShuttleDetection, ShuttleDetector
from video import VideoFile

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
TRAIL_FRAMES = 10


def _entry(time_ms: float, detection: ShuttleDetection | None) -> dict:
    entry = {"timeMs": round(time_ms, 3)}
    if detection is not None:
        entry.update(x=round(detection.x, 2), y=round(detection.y, 2), score=round(detection.score, 3), inpainted=detection.inpainted)
    return entry


def _write_overlay(video: VideoFile, track: list[ShuttleDetection | None], path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), video.fps, video.size)
    for index, (_, frame) in enumerate(video.frames()):
        for back in range(TRAIL_FRAMES):
            if index - back >= 0 and track[index - back] is not None:
                detection = track[index - back]
                colour = (255, 160, 0) if detection.inpainted else (0, 255, 255)
                cv2.circle(frame, (int(round(detection.x)), int(round(detection.y))), max(2, 6 - back // 2), colour, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find the shuttle in every frame of a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, help="also write a video with the shuttle's recent path drawn on it")
    parser.add_argument("--fast", action="store_true", help="one window per frame instead of eight overlapping ones")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args(argv)

    video = VideoFile(args.video)
    if not video.opened:
        print(f"cannot read {args.video}", file=sys.stderr)
        return 1
    try:
        detector = ShuttleDetector.load(args.weights)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    every = max(1, video.frame_count // BACKGROUND_SAMPLES)
    background = detector.background([frame for index, (_, frame) in enumerate(video.frames()) if index % every == 0])
    times: list[float] = []

    def frames():
        for time_ms, frame in video.frames():
            times.append(time_ms)
            yield frame

    track = detector.detect(frames(), background, video.size, overlap=not args.fast)
    if args.overlay:
        _write_overlay(video, track, args.overlay)
    seen = sum(detection is not None for detection in track)
    result = {
        "video": args.video.name,
        "fps": video.fps,
        "imageSize": list(video.size),
        "model": "TrackNetV3",
        "mode": "fast" if args.fast else "overlapping windows",
        "summary": {"frames": len(track), "seen": seen, "inpainted": sum(bool(d and d.inpainted) for d in track)},
        "frames": [_entry(time_ms, detection) for time_ms, detection in zip(times, track)],
    }
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
