# worker/track_camera.py
"""Per-frame camera for a real court video, anchored on court points placed in the labeller.

    python track_camera.py clip.mp4 --golden clip.json --out clip.track.json [--overlay clip.track.mp4]

The labelled points of one frame give a first camera (geometry.calibrate); the court lines in
that frame refine it and fix the focal length; then every frame is tracked from the start of
the video (geometry.tracking). Lost frames get no camera.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from evaluation.golden import keypoint_frame
from geometry.calibrate import solve_camera
from geometry.lines import line_response
from geometry.tracking import TrackedFrame, acquire, iter_track, model_polylines

STATE_COLOURS = {"anchored": (0, 255, 0), "tracked": (0, 255, 255), "reanchored": (255, 160, 0), "lost": (0, 0, 255)}


def _frames(capture: cv2.VideoCapture, keep: list[np.ndarray]) -> Iterator[np.ndarray]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        keep[:] = [frame]
        yield frame


def _draw(frame: np.ndarray, index: int, step: TrackedFrame) -> np.ndarray:
    image = frame.copy()
    colour = STATE_COLOURS[step.state]
    if step.camera is not None:
        height, width = image.shape[:2]
        for line in model_polylines(step.camera, (width, height)):
            cv2.polylines(image, [np.round(line).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
    cv2.putText(image, f"{index} {step.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
    return image


def _frame_entry(step: TrackedFrame) -> dict:
    entry = {"state": step.state, "inliers": step.inliers, "rmsPx": round(step.rms_px, 3) if np.isfinite(step.rms_px) else None}
    if step.camera is not None:
        entry["rvec"] = [round(float(v), 7) for v in np.asarray(step.camera.rvec).ravel()]
        entry["tvec"] = [round(float(v), 6) for v in np.asarray(step.camera.tvec).ravel()]
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track the camera through a court video from labelled court points.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--golden", type=Path, required=True, help="labeller export with court points placed on one frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, help="also write a video with the tracked court drawn on it")
    args = parser.parse_args(argv)

    placed = keypoint_frame(json.loads(args.golden.read_text(encoding="utf-8")))
    if placed is None or len(placed[1]) < 4:
        print("the labels need at least 4 court points placed on one frame", file=sys.stderr)
        return 1
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"cannot read {args.video}", file=sys.stderr)
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    time_ms, points = placed
    anchor_frame = int(round((time_ms or 0) / 1000 * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
    ok, frame = capture.read()
    solution = solve_camera(points, size)
    if not ok or solution is None or solution.tier == "unavailable":
        print("the labelled court points do not give a usable camera: check their names and positions", file=sys.stderr)
        return 1
    anchor = acquire(line_response(frame), solution.camera, fit_intrinsics=True)
    if anchor is None or not anchor.confident:
        print("the court lines could not be found near the labelled points", file=sys.stderr)
        return 1

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current: list[np.ndarray] = []
    writer = None
    if args.overlay:
        writer = cv2.VideoWriter(str(args.overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frames = []
    for index, step in enumerate(iter_track(_frames(capture, current), anchor.camera, fit_intrinsics=False)):
        frames.append(_frame_entry(step))
        if writer is not None:
            writer.write(_draw(current[0], index, step))
    capture.release()
    if writer is not None:
        writer.release()

    camera = anchor.camera
    states = [entry["state"] for entry in frames]
    rms = [entry["rmsPx"] for entry in frames if entry["state"] != "lost"]
    result = {
        "video": args.video.name,
        "fps": fps,
        "imageSize": list(size),
        "anchorFrame": anchor_frame,
        "intrinsics": {"focalPx": round(camera.focal_px, 3), "cx": camera.cx, "cy": camera.cy, "k1": round(camera.k1, 6)},
        "conventions": {
            "rvec/tvec": "world (court metres: x across, y along, z up) to camera (OpenCV: x right, y down, z forward)",
            "lost": "no camera: the court could not be found with confidence in that frame",
        },
        "summary": {
            "frames": len(frames),
            "lost": states.count("lost"),
            "reanchored": states.count("reanchored"),
            "medianRmsPx": round(float(np.median(rms)), 3) if rms else None,
        },
        "frames": frames,
    }
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
