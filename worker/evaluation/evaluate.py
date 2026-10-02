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
    size = (golden["imageSize"][0], golden["imageSize"][1])
    observations = {item["name"]: (item["x"], item["y"]) for item in golden.get("courtKeypoints", [])}
    solution = solve_camera(observations, size)
    if solution is None:
        return None
    quality = assess_geometry(solution.camera, size)
    return {
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


def evaluate_clip(golden: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    predicted_contacts = [event["timeMs"] for event in result.get("events", []) if event.get("type") == "contact"]
    truth_contacts = [contact["timeMs"] for contact in golden["contacts"]]
    contacts = {f"tol{tol}ms": _contact_report(predicted_contacts, truth_contacts, tol) for tol in CONTACT_TOLERANCES_MS}

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
    scored = macro_f1(y_true, y_pred, GOLDEN_SHOT_LABELS)
    return {
        "clipId": golden["clipId"],
        "calibration": evaluate_calibration(golden),
        "contacts": contacts,
        "shots": {
            "matched": len(match.pairs),
            "truthCount": len(truth_shots),
            "macroF1": round(scored["macroF1"], 4),
            "perLabel": scored["perLabel"],
        },
    }


def evaluate_directory(golden_dir: Path, results_dir: Path) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    for golden in load_golden_dir(golden_dir):
        result_path = Path(results_dir) / f"{golden['clipId']}.json"
        if not result_path.is_file():
            clips.append({"clipId": golden["clipId"], "error": f"no result file {result_path.name}"})
            continue
        clips.append(evaluate_clip(golden, json.loads(result_path.read_text(encoding="utf-8"))))
    scored = [clip for clip in clips if "error" not in clip]
    tiers: dict[str, int] = {}
    for clip in scored:
        if clip["calibration"] is not None:
            tier = clip["calibration"]["tier"]
            tiers[tier] = tiers.get(tier, 0) + 1

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "clips": clips,
        "summary": {
            "clips": len(clips),
            "clipsWithResults": len(scored),
            "meanContactF1At33ms": mean([clip["contacts"]["tol33ms"]["f1"] for clip in scored]),
            "meanContactF1At100ms": mean([clip["contacts"]["tol100ms"]["f1"] for clip in scored]),
            "meanShotMacroF1": mean([clip["shots"]["macroF1"] for clip in scored]),
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
