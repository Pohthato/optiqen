# worker/evaluation/shuttle.py
"""The shuttle detector against shuttle clicks from the labeller, frame by frame.

    python -m evaluation.shuttle --golden clip.json --detections clip.shuttle.json

A clicked shuttle is found when a detection is within the TrackNet tolerance (4 px at the
network's 512 px width, scaled to the video); a detection further away is in the wrong place.
A detection on a frame marked "can't see it" is a false alarm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOLERANCE_AT_512_PX = 4.0


def evaluate_shuttle(golden: dict[str, Any], detections: dict[str, Any]) -> dict[str, Any]:
    size = list(golden["imageSize"])
    if list(detections["imageSize"]) != size:
        found = detections["imageSize"]
        raise ValueError(f"the labels are for a {size[0]}x{size[1]} video but the detections for {found[0]}x{found[1]}")
    tolerance = TOLERANCE_AT_512_PX * size[0] / 512
    frames = detections["frames"]
    times = np.array([frame["timeMs"] for frame in frames], dtype=float)
    half_frame = 500.0 / (detections.get("fps") or golden["sourceFps"])
    counts = {"found": 0, "wrongPlace": 0, "missed": 0, "falseAlarms": 0, "correctlyEmpty": 0, "foundByFilledGaps": 0}
    errors: list[float] = []
    labelled = unmatched = visible = 0
    for point in golden.get("shuttlePoints", []):
        nearest = int(np.argmin(np.abs(times - point["timeMs"]))) if len(times) else -1
        if nearest < 0 or abs(times[nearest] - point["timeMs"]) > half_frame:
            unmatched += 1
            continue
        labelled += 1
        detection = frames[nearest]
        seen = "x" in detection
        if point.get("visible") is False:
            counts["falseAlarms" if seen else "correctlyEmpty"] += 1
            continue
        visible += 1
        if not seen:
            counts["missed"] += 1
            continue
        error = math.hypot(detection["x"] - point["x"], detection["y"] - point["y"])
        errors.append(error)
        if error <= tolerance:
            counts["found"] += 1
            counts["foundByFilledGaps"] += bool(detection.get("inpainted"))
        else:
            counts["wrongPlace"] += 1
    detected = counts["found"] + counts["wrongPlace"] + counts["falseAlarms"]
    return {
        "labelledFrames": labelled,
        "unmatchedClicks": unmatched,
        "visible": visible,
        **counts,
        "recall": counts["found"] / visible if visible else None,
        "precision": counts["found"] / detected if detected else None,
        "tolerancePx": tolerance,
        "medianErrorPx": round(float(np.median(errors)), 3) if errors else None,
        "p90ErrorPx": round(float(np.percentile(errors, 90)), 3) if errors else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score shuttle detections against shuttle clicks.")
    parser.add_argument("--golden", type=Path, required=True, help="labeller export with shuttlePoints")
    parser.add_argument("--detections", type=Path, required=True, help="detect_shuttle.py output for the same video")
    args = parser.parse_args(argv)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_shuttle(golden, detections), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
