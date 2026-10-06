# worker/evaluation/track.py
"""How well per-frame camera tracking holds on synthetic clips with known cameras.

    python -m evaluation.track runs/clip-a runs/clip-b --out track-report.json

Each clip directory is a simulation.clip export (video.mp4 + truth.json). The tracker is
anchored the way a person would: a few singles-court points tapped (with 1 px error) on the
first frame where the lens is clear. The gate: median floor error under 5 cm and the court
found again within half a second of the lens being uncovered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from geometry.calibrate import solve_camera
from geometry.camera import Camera
from geometry.court_model import COLUMNS, COURT_LENGTH_M
from geometry.synthetic import observe
from geometry.tracking import TrackedFrame, iter_track
from simulation.clip import camera_from_truth

FALSE_CONFIDENT_CM = 20.0
GATE_MEDIAN_CM = 5.0
GATE_RECOVERY_S = 0.5
TAP_NAMES = [f"{row}_{column}" for row in ("back0", "short0", "short1", "back1") for column in ("sl", "sr")]
FLOOR_GRID = np.array(
    [[x, y, 0.0] for x in np.linspace(COLUMNS["dl"], COLUMNS["dr"], 7) for y in np.linspace(0.0, COURT_LENGTH_M, 14)]
)


def floor_error_cm(truth: Camera, estimate: Camera, image_size: tuple[int, int]) -> np.ndarray:
    """For floor points the true camera sees: how far from each point the estimate puts the
    pixel it appears at (what every floor measurement downstream would be off by)."""
    pixels = truth.project(FLOOR_GRID)
    width, height = image_size
    with np.errstate(invalid="ignore"):
        seen = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    estimated = estimate.pixel_to_plane(pixels[seen], 0.0)
    return np.linalg.norm(estimated - FLOOR_GRID[seen, :2], axis=1) * 100.0


def evaluate_track(
    truth: Sequence[Camera],
    track: Sequence[TrackedFrame],
    image_size: tuple[int, int],
    occluded: Sequence[bool] | None = None,
) -> dict[str, Any]:
    if len(truth) != len(track):
        raise ValueError(f"{len(track)} tracked frames for {len(truth)} true cameras")
    occluded = list(occluded) if occluded is not None else [False] * len(truth)
    states: dict[str, int] = {}
    for step in track:
        states[step.state] = states.get(step.state, 0) + 1
    errors = [
        float(np.median(floor_error_cm(camera, step.camera, image_size)))
        for camera, step in zip(truth, track)
        if step.camera is not None
    ]
    visible = [step for step, covered in zip(track, occluded) if not covered]
    recovery: list[int | None] = []
    for index in range(1, len(track)):
        if occluded[index - 1] and not occluded[index]:
            back = next((later for later in range(index, len(track)) if track[later].camera is not None), None)
            recovery.append(None if back is None else back - index)
    return {
        "frames": len(track),
        "states": states,
        "trackedShareOfVisible": round(sum(step.camera is not None for step in visible) / len(visible), 4) if visible else None,
        "lostWhileVisible": sum(step.camera is None for step in visible),
        "trackedWhileOccluded": sum(step.camera is not None for step, covered in zip(track, occluded) if covered),
        "floorErrorCm": {
            "median": round(float(np.median(errors)), 3) if errors else None,
            "p90": round(float(np.percentile(errors, 90)), 3) if errors else None,
            "max": round(float(np.max(errors)), 3) if errors else None,
        },
        "falseConfident": sum(error > FALSE_CONFIDENT_CM for error in errors),
        "recoveryFrames": recovery,
        "perFrameFloorErrorCm": [round(error, 3) for error in errors],
    }


def _anchor(truth: Camera, image_size: tuple[int, int], seed: int) -> Camera:
    """The camera a person's taps give: visible singles-court points, 1 px of tap error."""
    taps = observe(truth, image_size, TAP_NAMES, noise_px=1.0, seed=seed)
    solution = solve_camera(taps, image_size)
    if solution is None:
        raise ValueError("fewer than 4 singles-court points are in view to anchor on")
    return solution.camera


def evaluate_clip_dir(directory: Path, seed: int = 0) -> dict[str, Any]:
    truth = json.loads((directory / "truth.json").read_text(encoding="utf-8"))
    size = (truth["imageSize"][0], truth["imageSize"][1])
    cameras = [camera_from_truth(entry) for entry in truth["cameras"]]
    occluded = truth.get("occluded") or [False] * len(cameras)
    first_clear = occluded.index(False)
    capture = cv2.VideoCapture(str(directory / "video.mp4"))

    def frames():
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame

    track = list(iter_track(frames(), _anchor(cameras[first_clear], size, seed)))
    capture.release()
    report = evaluate_track(cameras[: len(track)], track, size, occluded[: len(track)])
    report["clip"] = directory.name
    report["fps"] = truth["fps"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate per-frame camera tracking on synthetic clips.")
    parser.add_argument("clips", type=Path, nargs="+", help="simulation.clip output directories")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    clips = [evaluate_clip_dir(directory) for directory in args.clips]
    errors = [error for clip in clips for error in clip["perFrameFloorErrorCm"]]
    recoveries = [frames for clip in clips for frames in clip["recoveryFrames"]]
    worst_recovery_s = max(
        [float("inf") if frames is None else frames / clip["fps"] for clip in clips for frames in clip["recoveryFrames"]],
        default=0.0,
    )
    median = round(float(np.median(errors)), 3) if errors else None
    summary = {
        "aggregation": "per-frame median floor errors pooled over clips",
        "clips": len(clips),
        "medianFloorErrorCm": median,
        "falseConfident": sum(clip["falseConfident"] for clip in clips),
        "trackedWhileOccluded": sum(clip["trackedWhileOccluded"] for clip in clips),
        "worstRecoveryFrames": None if None in recoveries else max(recoveries, default=0),
        "gate": f"median floor error < {GATE_MEDIAN_CM} cm, court found within {GATE_RECOVERY_S} s of the lens clearing",
        "gatePassed": bool(median is not None and median < GATE_MEDIAN_CM and worst_recovery_s <= GATE_RECOVERY_S),
    }
    report = {"clips": clips, "summary": summary}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
