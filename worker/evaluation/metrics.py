# worker/evaluation/metrics.py
"""Event-timing and classification metrics used by every capability gate."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class EventMatch:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    median_abs_error_ms: float | None
    pairs: tuple[tuple[int, int], ...]  # (predicted index, truth index)


def match_events(predicted_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> EventMatch:
    """One-to-one matching, closest pairs first, within an inclusive tolerance."""
    candidates = sorted(
        (abs(p - t), pi, ti)
        for pi, p in enumerate(predicted_ms)
        for ti, t in enumerate(truth_ms)
        if abs(p - t) <= tolerance_ms
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    pairs: list[tuple[int, int]] = []
    errors: list[float] = []
    for distance, pi, ti in candidates:
        if pi in used_predicted or ti in used_truth:
            continue
        used_predicted.add(pi)
        used_truth.add(ti)
        pairs.append((pi, ti))
        errors.append(distance)
    true_positives = len(pairs)
    false_positives = len(predicted_ms) - true_positives
    false_negatives = len(truth_ms) - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventMatch(
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
        float(median(errors)) if errors else None,
        tuple(sorted(pairs)),
    )


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] | None = None) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    classes = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    per_label: dict[str, dict[str, float]] = {}
    for label in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "predicted": tp + fp,
        }
    scored = [row["f1"] for row in per_label.values() if row["support"] > 0 or row["predicted"] > 0]
    return {"macroF1": sum(scored) / len(scored) if scored else 0.0, "perLabel": per_label}
