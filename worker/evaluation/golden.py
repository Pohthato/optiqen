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
    return errors


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
