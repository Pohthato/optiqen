# worker/evaluation/evaluate.py
"""Compare worker results with hand-labelled golden clips.

    python -m evaluation.evaluate --golden golden/ --results results/ --out report.json

`results/<clipId>.json` is the worker job output for that clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.golden import GOLDEN_SHOT_LABELS, load_golden_dir
from evaluation.metrics import macro_f1, match_events
from geometry.calibrate import solve_camera
from geometry.quality import assess_geometry

CONTACT_TOLERANCES_MS = (33, 100)
SHOT_MATCH_TOLERANCE_MS = 100


def evaluate_calibration(golden: dict[str, Any]) -> dict[str, Any] | None:
    """Solve the camera from the golden keypoints of one frame (the frame with the most points).
    Handheld clips move the camera, so points clicked on different frames cannot be mixed.
    Proposed points are projections of the clicked ones, not evidence, so they are left out."""
    size = (golden["imageSize"][0], golden["imageSize"][1])
    frames: dict[Any, list[dict[str, Any]]] = {}
    for item in golden.get("courtKeypoints", []):
        if item.get("source") == "proposed":
            continue
        frames.setdefault(item.get("timeMs"), []).append(item)
    if not frames:
        return None
    frame_time, keypoints = max(frames.items(), key=lambda entry: len(entry[1]))
    observations = {item["name"]: (item["x"], item["y"]) for item in keypoints}
    solution = solve_camera(observations, size)
    if solution is None:
        return None
    quality = assess_geometry(solution.camera, size)
    return {
        "source": "golden keypoints",
        "frameTimeMs": frame_time,
        "tier": solution.tier,
        "rmsPx": round(solution.rms_px, 3),
        "floorRmsCm": round(solution.floor_rms_cm, 2),
        "looFloorCm": None if solution.loo_floor_cm is None else round(solution.loo_floor_cm, 2),
        "redundancy": solution.redundancy,
        "outliers": list(solution.outliers),
        "geometryTier": quality.tier,
    }


def _contact_report(predicted: list[float], truth: list[float], tolerance_ms: float) -> dict[str, Any]:
    match = match_events(predicted, truth, tolerance_ms)
    return {
        "truePositives": match.true_positives,
        "falsePositives": match.false_positives,
        "falseNegatives": match.false_negatives,
        "precision": round(match.precision, 4),
        "recall": round(match.recall, 4),
        "f1": round(match.f1, 4),
        "medianAbsErrorMs": match.median_abs_error_ms,
    }


def _shot_labels(golden: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], list[str], int, int]:
    """Aligned (truth, prediction) labels. Missed truth shots score as "missed"; a verified
    prediction with no labelled shot near it scores against "none", so false claims lower
    that label's precision instead of vanishing from the report."""
    predicted_shots = result.get("shots", [])
    truth_shots = golden["shots"]
    match = match_events(
        [shot["timeMs"] for shot in predicted_shots], [shot["timeMs"] for shot in truth_shots], SHOT_MATCH_TOLERANCE_MS
    )
    matched = {truth_index: predicted_index for predicted_index, truth_index in match.pairs}
    y_true = [shot["label"] for shot in truth_shots]
    y_pred: list[str] = []
    for truth_index in range(len(truth_shots)):
        predicted_index = matched.get(truth_index)
        if predicted_index is None:
            y_pred.append("missed")
            continue
        shot = predicted_shots[predicted_index]
        y_pred.append(shot["label"] if shot.get("verified") else "unverified")
    matched_predictions = set(matched.values())
    unmatched = 0
    for predicted_index, shot in enumerate(predicted_shots):
        if predicted_index not in matched_predictions and shot.get("verified"):
            y_true.append("none")
            y_pred.append(shot["label"])
            unmatched += 1
    return y_true, y_pred, len(match.pairs), unmatched


def evaluate_clip(golden: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    predicted_contacts = [event["timeMs"] for event in result.get("events", []) if event.get("type") == "contact"]
    truth_contacts = [contact["timeMs"] for contact in golden["contacts"]]
    contacts = {f"tol{tol}ms": _contact_report(predicted_contacts, truth_contacts, tol) for tol in CONTACT_TOLERANCES_MS}
    y_true, y_pred, matched, unmatched = _shot_labels(golden, result)
    scored = macro_f1(y_true, y_pred, GOLDEN_SHOT_LABELS)
    return {
        "clipId": golden["clipId"],
        "calibration": evaluate_calibration(golden),
        "workerCamera": result.get("camera"),
        "contacts": contacts,
        "shots": {
            "matched": matched,
            "truthCount": len(golden["shots"]),
            "unmatchedVerifiedPredictions": unmatched,
            "macroF1": round(scored["macroF1"], 4),
            "perLabel": scored["perLabel"],
        },
    }


def evaluate_directory(golden_dir: Path, results_dir: Path) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    pooled_true: list[str] = []
    pooled_pred: list[str] = []
    for golden in load_golden_dir(golden_dir):
        result_path = Path(results_dir) / f"{golden['clipId']}.json"
        if not result_path.is_file():
            clips.append({"clipId": golden["clipId"], "error": f"no result file {result_path.name}"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        clips.append(evaluate_clip(golden, result))
        y_true, y_pred, _, _ = _shot_labels(golden, result)
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)
    scored = [clip for clip in clips if "error" not in clip]
    tiers: dict[str, int] = {}
    for clip in scored:
        if clip["calibration"] is not None:
            tier = clip["calibration"]["tier"]
            tiers[tier] = tiers.get(tier, 0) + 1

    # Gate numbers pool the counts over clips instead of averaging per-clip scores: a clip with
    # nothing to find (and nothing found) is correct, and a ten-shot clip is too noisy alone.
    def pooled_contact_f1(key: str) -> float | None:
        if not scored:
            return None
        tp = sum(clip["contacts"][key]["truePositives"] for clip in scored)
        fp = sum(clip["contacts"][key]["falsePositives"] for clip in scored)
        fn = sum(clip["contacts"][key]["falseNegatives"] for clip in scored)
        return round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None

    return {
        "clips": clips,
        "summary": {
            "aggregation": "counts pooled over clips",
            "clips": len(clips),
            "clipsWithResults": len(scored),
            "contactF1At33ms": pooled_contact_f1("tol33ms"),
            "contactF1At100ms": pooled_contact_f1("tol100ms"),
            "shotMacroF1": round(macro_f1(pooled_true, pooled_pred, GOLDEN_SHOT_LABELS)["macroF1"], 4) if scored else None,
            "calibrationTiers": tiers,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate worker results against golden clips.")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_directory(args.golden, args.results)
    except ValueError as exc:
        print(f"Invalid golden set:\n{exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
