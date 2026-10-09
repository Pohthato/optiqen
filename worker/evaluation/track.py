# worker/evaluation/track.py
"""How well per-frame camera tracking holds on synthetic clips with known cameras.

    python -m evaluation.track runs/clip-a runs/clip-b --out track-report.json

Each clip directory is a simulation.clip export (video.mp4 + truth.json). The tracker is
anchored the way a person would: a few singles-court points tapped (with 1 px error) on the
first frame where the lens is clear. The gate: median floor error under 5 cm, no frame given a
camera that is more than 20 cm wrong, no camera while the lens is covered, and the court found
again within half a second of the lens being uncovered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from geometry.camera import Camera
from geometry.court_model import COLUMNS, COURT_LENGTH_M
from geometry.synthetic import observe
from geometry.tracking import TrackedFrame, anchor_on_points, iter_track
from simulation.clip import camera_from_truth

FALSE_CONFIDENT_CM = 20.0
MISS_CM = 10_000.0  # a floor point the estimate cannot put on the floor at all (its ray misses it)
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
    errors = np.linalg.norm(estimated - FLOOR_GRID[seen, :2], axis=1) * 100.0
    return np.where(np.isfinite(errors), np.minimum(errors, MISS_CM), MISS_CM)


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


def _video_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def evaluate_clip_dir(directory: Path, seed: int = 0) -> dict[str, Any]:
    truth = json.loads((directory / "truth.json").read_text(encoding="utf-8"))
    size = (truth["imageSize"][0], truth["imageSize"][1])
    cameras = [camera_from_truth(entry) for entry in truth["cameras"]]
    occluded = truth.get("occluded") or [False] * len(cameras)
    first_clear = occluded.index(False)
    frame = next(frame for index, frame in enumerate(_video_frames(directory / "video.mp4")) if index == first_clear)
    taps = observe(cameras[first_clear], size, TAP_NAMES, noise_px=1.0, seed=seed)
    anchor = anchor_on_points(frame, taps)
    track = list(iter_track(_video_frames(directory / "video.mp4"), anchor.camera, fit_intrinsics=False))
    report = evaluate_track(cameras[: len(track)], track, size, occluded[: len(track)])
    report["clip"] = directory.name
    report["fps"] = truth["fps"]
    return report


def summarise(clips: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [error for clip in clips for error in clip["perFrameFloorErrorCm"]]
    recoveries = [frames for clip in clips for frames in clip["recoveryFrames"]]
    worst_recovery_s = max(
        [float("inf") if frames is None else frames / clip["fps"] for clip in clips for frames in clip["recoveryFrames"]],
        default=0.0,
    )
    median = round(float(np.median(errors)), 3) if errors else None
    false_confident = sum(clip["falseConfident"] for clip in clips)
    while_covered = sum(clip["trackedWhileOccluded"] for clip in clips)
    return {
        "aggregation": "per-frame median floor errors pooled over clips",
        "clips": len(clips),
        "medianFloorErrorCm": median,
        "falseConfident": false_confident,
        "trackedWhileOccluded": while_covered,
        "lostWhileVisible": sum(clip["lostWhileVisible"] for clip in clips),
        "worstRecoveryFrames": None if None in recoveries else max(recoveries, default=0),
        "gate": (
            f"median floor error < {GATE_MEDIAN_CM} cm, no camera more than {FALSE_CONFIDENT_CM:.0f} cm wrong, "
            f"no camera while covered, court found within {GATE_RECOVERY_S} s of the lens clearing"
        ),
        "gatePassed": bool(
            median is not None
            and median < GATE_MEDIAN_CM
            and false_confident == 0
            and while_covered == 0
            and worst_recovery_s <= GATE_RECOVERY_S
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate per-frame camera tracking on synthetic clips.")
    parser.add_argument("clips", type=Path, nargs="+", help="simulation.clip output directories")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    clips = [evaluate_clip_dir(directory) for directory in args.clips]
    report = {"clips": clips, "summary": summarise(clips)}
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
