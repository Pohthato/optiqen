# worker/rally_job.py
"""The rally analysis (analyse_rally.py) for an app job: the job's video and its four corner taps.
The taps become court points placed on the first frame, as the handler's own calibration takes
them; the camera is tracked from there, and the shuttle and the players are found in the video.
What comes back is the analysis as analyse_rally.py writes it, findings and all, or the reason it
could not be made, so a job never fails for want of it.

A job's video is taken to be one rally: in a longer one the analysis ends where the first rally
does. It runs in a process of its own, so a crash or a stuck model costs only this analysis.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from geometry.calibration_adapter import corners_to_observations

ANALYSE = Path(__file__).resolve().with_name("analyse_rally.py")
MAX_SECONDS = float(os.environ.get("RALLY_MAX_SECONDS", "90"))  # longer than any rally
TIMEOUT_S = float(os.environ.get("RALLY_TIMEOUT_SECONDS", "900"))
POSE_MODEL = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "weights")) / "pose_landmarker_heavy.task"
LEFT_OUT = ("video", "sources")  # the job's temporary paths say nothing about the rally


def court_points(corners: list[dict[str, Any]], image_size: tuple[int, int], fps: float, court_type: str) -> dict[str, Any]:
    """The corner taps as the labeller exports court points (track_camera.py's --golden): in
    pixels, named by the court model, placed on the first frame."""
    observed = corners_to_observations(corners, image_size, court_type)
    return {
        "imageSize": list(image_size),
        "sourceFps": fps,
        "courtKeypoints": [{"name": name, "x": x, "y": y, "timeMs": 0, "source": "placed"} for name, (x, y) in observed.items()],
    }


def analyse(
    video: Path, corners: list[dict[str, Any]], court_type: str, image_size: tuple[int, int], fps: float, duration_s: float, work: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """The rally analysis of a job's video, or None and why there is none."""
    if duration_s > MAX_SECONDS:
        return None, f"The video is {duration_s:.0f} s long; the rally analysis takes one rally, up to {MAX_SECONDS:.0f} s. Trim the video to one rally."
    points = court_points(corners, image_size, fps, court_type)
    if len(points["courtKeypoints"]) < 4:
        return None, "The rally analysis needs all four court corners tapped; only three were, so the fourth would be a guess."
    golden, out = work / "rally.points.json", work / "rally.analysis.json"
    golden.write_text(json.dumps(points), encoding="utf-8")
    command = [sys.executable, str(ANALYSE), str(video), "--golden", str(golden), "--out", str(out), "--court", court_type, "--pose-model", str(POSE_MODEL)]
    try:
        done = subprocess.run(command, cwd=ANALYSE.parent, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, f"The rally analysis took longer than {TIMEOUT_S:.0f} s and was stopped."
    if done.returncode != 0 or not out.is_file():
        said = [line.strip() for line in done.stderr.splitlines() if line.strip()]
        return None, f"The rally analysis could not be made: {said[-1][:300] if said else f'it stopped with code {done.returncode}'}."
    document = json.loads(out.read_text(encoding="utf-8"))
    return {key: value for key, value in document.items() if key not in LEFT_OUT}, None
