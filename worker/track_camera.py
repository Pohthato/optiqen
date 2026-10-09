# worker/track_camera.py
"""Per-frame camera for a real court video, anchored on court points placed in the labeller.

    python track_camera.py clip.mp4 --golden clip.json --out clip.track.json [--overlay clip.track.mp4]

The labelled points of one frame give a first camera; the court lines in that frame refine it,
fix the focal length and must agree with the points (geometry.tracking.anchor_on_points). Then
every frame is tracked from the start of the video. Lost frames get no camera.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from evaluation.golden import keypoint_frame
from geometry.tracking import AnchorError, TrackedFrame, anchor_on_points, iter_track, model_polylines
from video import VideoFile

STATE_COLOURS = {"anchored": (0, 255, 0), "tracked": (0, 255, 255), "reanchored": (255, 160, 0), "lost": (0, 0, 255)}


def _draw(frame: np.ndarray, index: int, step: TrackedFrame) -> np.ndarray:
    image = frame.copy()
    colour = STATE_COLOURS[step.state]
    if step.camera is not None:
        for line in model_polylines(step.camera):
            cv2.polylines(image, [np.round(line).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
    cv2.putText(image, f"{index} {step.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
    return image


def _frame_entry(time_ms: float, step: TrackedFrame) -> dict:
    entry = {
        "timeMs": round(time_ms, 3),
        "state": step.state,
        "inliers": step.inliers,
        "rmsPx": round(step.rms_px, 3) if np.isfinite(step.rms_px) else None,
    }
    if step.camera is not None:
        entry["rvec"] = [round(float(v), 7) for v in np.asarray(step.camera.rvec).ravel()]
        entry["tvec"] = [round(float(v), 6) for v in np.asarray(step.camera.tvec).ravel()]
    return entry


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track the camera through a court video from labelled court points.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--golden", type=Path, required=True, help="labeller export with court points placed on one frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, help="also write a video with the tracked court drawn on it")
    args = parser.parse_args(argv)

    labels = json.loads(args.golden.read_text(encoding="utf-8"))
    placed = keypoint_frame(labels)
    if placed is None or len(placed[1]) < 4:
        return _fail("the labels need at least 4 court points placed on one frame")
    video = VideoFile(args.video)
    if not video.opened:
        return _fail(f"cannot read {args.video}")
    labelled_size = labels.get("imageSize")
    if labelled_size and tuple(labelled_size) != video.size:
        return _fail(
            f"the labels were placed on a {labelled_size[0]}x{labelled_size[1]} video but this one is "
            f"{video.size[0]}x{video.size[1]}: label the same file you track"
        )
    # The labeller times a frame as index * 1000 / fps with the clip's frame rate.
    fps = labels.get("sourceFps") or video.fps
    if not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
        return _fail("the video has no frame rate: add sourceFps to the labels")
    time_ms, points = placed
    anchor_frame = int(round((time_ms or 0) / 1000 * fps))
    frame = next((frame for index, (_, frame) in enumerate(video.frames()) if index == anchor_frame), None)
    if frame is None:
        return _fail(f"the labelled frame ({anchor_frame}) is past the end of the video")
    try:
        anchor = anchor_on_points(frame, points)
    except AnchorError as error:
        return _fail(str(error))

    writer = cv2.VideoWriter(str(args.overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, video.size) if args.overlay else None
    current: list[tuple[float, np.ndarray]] = []

    def remembered() -> Iterator[np.ndarray]:
        for item in video.frames():
            current[:] = [item]
            yield item[1]

    frames = []
    for index, step in enumerate(iter_track(remembered(), anchor.camera, fit_intrinsics=False)):
        frames.append(_frame_entry(current[0][0], step))
        if writer is not None:
            writer.write(_draw(current[0][1], index, step))
    if writer is not None:
        writer.release()

    camera = anchor.camera
    states = [entry["state"] for entry in frames]
    rms = [entry["rmsPx"] for entry in frames if entry["state"] != "lost"]
    result = {
        "video": args.video.name,
        "fps": fps,
        "imageSize": list(video.size),
        "anchorFrame": anchor_frame,
        "intrinsics": {"focalPx": round(camera.focal_px, 3), "cx": camera.cx, "cy": camera.cy, "k1": round(camera.k1, 6)},
        "conventions": {
            "rvec/tvec": "world (court metres: x across, y along, z up) to camera (OpenCV: x right, y down, z forward)",
            "timeMs": "the frame's timestamp in the video file",
            "lost": "no camera: the court could not be found with confidence in that frame",
            "rmsPx": "RMS distance of the lines that agree with the camera (within the inlier tolerance)",
        },
        "summary": {
            "frames": len(frames),
            "lost": states.count("lost"),
            "reanchored": states.count("reanchored"),
            "medianRmsPx": round(float(np.median(rms)), 3) if rms else None,
        },
        "frames": frames,
    }
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
