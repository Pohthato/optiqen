# worker/evaluation/golden.py
"""Schema and loader for hand-labelled golden clips (see docs/golden-set.md)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, KEYPOINTS, SINGLES_WIDTH_M

SCHEMA_VERSION = 1
GOLDEN_SHOT_LABELS = ("serve", "clear", "drop", "net", "lift", "drive", "push", "smash", "block", "other")
WINNERS = ("near", "far", "unknown")
PLAYERS = ("near", "far")
# Where a court point came from: clicked by a person, a geometric proposal accepted as-is,
# or a proposal a person moved. Only clicked and adjusted points are independent evidence.
KEYPOINT_SOURCES = ("clicked", "adjusted", "proposed")
# COCO keypoint order, shared with pose models and tools/labeller.
COCO_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
LANDING_MARGIN_M = 1.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_golden(doc: Any) -> list[str]:
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    errors: list[str] = []
    if doc.get("schemaVersion") != SCHEMA_VERSION:
        errors.append(f"schemaVersion must be {SCHEMA_VERSION}")
    clip_id = doc.get("clipId")
    if not isinstance(clip_id, str) or not clip_id.strip():
        errors.append("clipId must be a non-empty string")
    if not _is_number(doc.get("sourceFps")) or doc["sourceFps"] <= 0:
        errors.append("sourceFps must be a positive number")
    size = doc.get("imageSize")
    size_ok = (
        isinstance(size, list)
        and len(size) == 2
        and all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in size)
    )
    if not size_ok:
        errors.append("imageSize must be [width, height] positive integers")
    for index, item in enumerate(doc.get("courtKeypoints", [])):
        where = f"courtKeypoints[{index}]"
        if not isinstance(item, dict) or item.get("name") not in KEYPOINTS:
            errors.append(f"{where}: keypoint name must be one of the court model names")
        elif not (_is_number(item.get("x")) and _is_number(item.get("y"))):
            errors.append(f"{where}: x and y must be numbers")
        elif size_ok and not (0 <= item["x"] < size[0] and 0 <= item["y"] < size[1]):
            errors.append(f"{where}: pixel is outside the image")
        if isinstance(item, dict) and "timeMs" in item and (not _is_number(item["timeMs"]) or item["timeMs"] < 0):
            errors.append(f"{where}: timeMs (the frame the point was clicked on) must be a non-negative number")
        if isinstance(item, dict) and "source" in item and item["source"] not in KEYPOINT_SOURCES:
            errors.append(f"{where}: source must be one of {', '.join(KEYPOINT_SOURCES)}")
    for key in ("contacts", "shots"):
        if not isinstance(doc.get(key), list):
            errors.append(f"{key} must be a list")
    for index, item in enumerate(doc.get("contacts") or []):
        if not isinstance(item, dict) or not _is_number(item.get("timeMs")) or item["timeMs"] < 0:
            errors.append(f"contacts[{index}]: timeMs must be a non-negative number")
    for index, item in enumerate(doc.get("shots") or []):
        where = f"shots[{index}]"
        if not isinstance(item, dict) or not _is_number(item.get("timeMs")) or item["timeMs"] < 0:
            errors.append(f"{where}: timeMs must be a non-negative number")
            continue
        if item.get("label") not in GOLDEN_SHOT_LABELS:
            errors.append(f"{where}: label must be one of {', '.join(GOLDEN_SHOT_LABELS)}")
        landing = item.get("landing")
        if landing is not None:
            in_court = (
                isinstance(landing, dict)
                and _is_number(landing.get("x"))
                and _is_number(landing.get("y"))
                and -DOUBLES_MARGIN_M - LANDING_MARGIN_M <= landing["x"] <= SINGLES_WIDTH_M + DOUBLES_MARGIN_M + LANDING_MARGIN_M
                and -LANDING_MARGIN_M <= landing["y"] <= COURT_LENGTH_M + LANDING_MARGIN_M
            )
            if not in_court:
                errors.append(f"{where}: landing must be null or court metres near the court")
    for index, item in enumerate(doc.get("rallies", [])):
        where = f"rallies[{index}]"
        if not isinstance(item, dict) or not (_is_number(item.get("startMs")) and _is_number(item.get("endMs"))):
            errors.append(f"{where}: startMs and endMs must be numbers")
            continue
        if item["startMs"] >= item["endMs"]:
            errors.append(f"{where}: startMs must be before endMs")
        if item.get("winner") not in WINNERS:
            errors.append(f"{where}: winner must be one of {', '.join(WINNERS)}")
    if "selectedPlayer" in doc and doc["selectedPlayer"] not in PLAYERS:
        errors.append("selectedPlayer must be near or far when present")
    contact_times = {item["timeMs"] for item in doc.get("contacts") or [] if isinstance(item, dict) and _is_number(item.get("timeMs"))}
    seen_pose_times: set[float] = set()
    for index, item in enumerate(doc.get("poses") or []):
        where = f"poses[{index}]"
        problems = _pose_errors(where, item, size if size_ok else None)
        errors.extend(problems)
        if problems and not isinstance(item, dict):
            continue
        time = item.get("timeMs")
        if _is_number(time):
            if time not in contact_times:
                errors.append(f"{where}: timeMs must be the time of a labelled contact")
            if time in seen_pose_times:
                errors.append(f"{where}: duplicate pose for the contact at {time} ms")
            seen_pose_times.add(time)
        if doc.get("selectedPlayer") in PLAYERS and item.get("player") in PLAYERS and item["player"] != doc["selectedPlayer"]:
            errors.append(f"{where}: player must be the selectedPlayer ({doc['selectedPlayer']})")
    errors.extend(_shuttle_errors(doc.get("shuttlePoints", []), size if size_ok else None))
    return errors


def _shuttle_errors(points: Any, size: list[int] | None) -> list[str]:
    """Shuttle clicks: per frame, where the shuttle is, or visible false when it cannot be seen."""
    if not isinstance(points, list):
        return ["shuttlePoints must be a list"]
    errors: list[str] = []
    seen: set[float] = set()
    for index, item in enumerate(points):
        where = f"shuttlePoints[{index}]"
        if not isinstance(item, dict) or not _is_number(item.get("timeMs")) or item["timeMs"] < 0:
            errors.append(f"{where}: timeMs must be a non-negative number")
            continue
        if "visible" in item and item["visible"] is not False:
            errors.append(f"{where}: visible, when given, must be false (the shuttle was not seen)")
        elif item.get("visible") is False:
            if "x" in item or "y" in item:
                errors.append(f"{where}: a shuttle not seen has no x and y")
        elif not (_is_number(item.get("x")) and _is_number(item.get("y"))):
            errors.append(f"{where}: x and y must be numbers")
        elif size is not None and not (0 <= item["x"] < size[0] and 0 <= item["y"] < size[1]):
            errors.append(f"{where}: shuttle is outside the image")
        if item["timeMs"] in seen:
            errors.append(f"{where}: duplicate frame at {item['timeMs']} ms")
        seen.add(item["timeMs"])
    return errors


def _pose_errors(where: str, item: Any, size: list[int] | None) -> list[str]:
    """A pose is one player's 17 COCO joints at one frame: [x, y, visibility 0|1|2] each."""
    if not isinstance(item, dict) or not _is_number(item.get("timeMs")) or item["timeMs"] < 0:
        return [f"{where}: timeMs must be a non-negative number"]
    errors = []
    if item.get("player") not in PLAYERS:
        errors.append(f"{where}: player must be near or far")
    joints = item.get("keypoints")
    if not isinstance(joints, list) or len(joints) != len(COCO_JOINTS):
        return errors + [f"{where}: keypoints must list the {len(COCO_JOINTS)} COCO joints as [x, y, visibility]"]
    for index, joint in enumerate(joints):
        name = COCO_JOINTS[index]
        if not (isinstance(joint, list) and len(joint) == 3 and all(_is_number(v) for v in joint) and joint[2] in (0, 1, 2)):
            errors.append(f"{where}.{name}: must be [x, y, visibility] with visibility 0, 1 or 2")
        elif joint[2] > 0 and size is not None and not (0 <= joint[0] < size[0] and 0 <= joint[1] < size[1]):
            errors.append(f"{where}.{name}: labelled joint is outside the image")
    return errors


def keypoint_frame(doc: dict[str, Any]) -> tuple[Any, dict[str, tuple[float, float]]] | None:
    """The court points placed on one frame: the frame (timeMs) with the most of them. A handheld
    camera moves, so points placed on different frames cannot be mixed; proposed points are
    projections of the placed ones, not evidence, so they are left out."""
    frames: dict[Any, list[dict[str, Any]]] = {}
    for item in doc.get("courtKeypoints", []):
        if item.get("source") != "proposed":
            frames.setdefault(item.get("timeMs"), []).append(item)
    if not frames:
        return None
    time_ms, keypoints = max(frames.items(), key=lambda entry: len(entry[1]))
    return time_ms, {item["name"]: (item["x"], item["y"]) for item in keypoints}


def load_golden_dir(directory: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: invalid JSON ({exc.msg})")
            continue
        errors = validate_golden(doc)
        if errors:
            problems.extend(f"{path.name}: {error}" for error in errors)
        else:
            documents.append(doc)
    if problems:
        raise ValueError("\n".join(problems))
    return documents
