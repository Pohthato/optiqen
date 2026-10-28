# worker/evaluation/hand_labels.py
"""Hits and shot names labelled by hand in the hittimes format, and analyses scored against them.

The format has one block per clip, with times in seconds and each shot's name in brackets:

    Clip 9:
    Serve time(far player serve):0.34
    Near player hit times:1.07(middle net),2.6(lift)
    Far Player hit times:1.75(middle net),3.98(drop)

A found hit is matched to a labelled one, closest pairs first, within the tolerance. Its hitter
and name are those of the analysis's shot that starts within 100 ms of it (the name: the most
likely one). A labelled name stands for the golden labels it may be (`labels_for`).

    python -m evaluation.hand_labels Clips/hittimes.txt "Clip 9=clip9.analysis.json" "Clip 8=clip8.analysis.json"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.golden import GOLDEN_SHOT_LABELS
from evaluation.metrics import macro_f1, macro_f1_interval, match_events

TOLERANCE_MS = 250.0  # hits timed by eye, some with the racket blurred or hidden
SHOT_WITHIN_MS = 100.0
# The golden labels a hand-written name may be, from the first of these words it contains. A
# name with both "push" and "lift" may be either: the two are a matter of judgement.
NAME_WORDS = (
    ("cut", ()),  # a cut may be a drop or a smash: not scored until it is named
    ("serve", ("serve",)),
    ("smash", ("smash",)),
    ("drop", ("drop",)),
    ("clear", ("clear",)),
    ("block", ("block",)),
    ("drive", ("drive",)),
    ("lift", ("lift",)),
    ("push", ("push",)),
    ("net", ("net",)),
)


@dataclass(frozen=True)
class LabelledHit:
    time_ms: float
    hitter: str  # "near" | "far"
    name: str  # as written, without its notes
    labels: tuple[str, ...]  # the golden labels it may be; () when it is not scored


def _clean(name: str) -> str:
    """The name alone, in lower case: notes after a colon or comma, or in brackets, dropped."""
    return " ".join(re.split(r"[:,(]", name.lower())[0].split())


def labels_for(name: str) -> tuple[str, ...]:
    words = _clean(name)
    if "push" in words and "lift" in words:
        return ("push", "lift")
    return next((labels for word, labels in NAME_WORDS if word in words), ())


def read_labels(text: str) -> dict[str, list[LabelledHit]]:
    """Every clip's labelled hits, in time order, by clip name ("Clip 9")."""
    clips: dict[str, list[LabelledHit]] = {}
    hits: list[LabelledHit] | None = None
    for line in text.splitlines():
        line = line.strip()
        head = re.fullmatch(r"clip\s+(\d+)\s*:", line, re.IGNORECASE)
        if head:
            hits = clips.setdefault(f"Clip {head.group(1)}", [])
            continue
        lower = line.lower()
        if hits is None or ":" not in line:
            continue
        label, times = line.split(":", 1)
        if lower.startswith("serve time"):
            time = re.match(r"\s*(\d+(?:\.\d+)?)", times)
            if time:
                hits.append(LabelledHit(float(time.group(1)) * 1000.0, "far" if "far" in label.lower() else "near", "serve", ("serve",)))
        elif "hit times" in lower:
            hitter = "near" if lower.startswith("near") else "far"
            for time, name in re.findall(r"(\d+(?:\.\d+)?)\s*(?:\(([^)]*)\))?", times):
                hits.append(LabelledHit(float(time) * 1000.0, hitter, _clean(name), labels_for(name)))
    return {clip: sorted(found, key=lambda hit: hit.time_ms) for clip, found in clips.items()}


def _top_name(shot: dict[str, Any] | None) -> str | None:
    name = (shot or {}).get("name")
    return max(name["p"], key=name["p"].get) if name else None


def _sureness(shot: dict[str, Any] | None) -> float | None:
    name = (shot or {}).get("name")
    return max(name["p"].values()) if name else None


def match_clip(hits: list[LabelledHit], analysis: dict[str, Any], tolerance_ms: float = TOLERANCE_MS) -> tuple[list[dict], list[dict]]:
    """Each labelled hit with what was found for it, and the found hits that match none."""
    shots = analysis.get("shots") or []
    found = [event for event in analysis["events"] if event["type"] == "contact"]

    def shot_at(time_ms: float) -> dict[str, Any] | None:
        near = [shot for shot in shots if abs(shot["hitMs"] - time_ms) <= SHOT_WITHIN_MS]
        return min(near, key=lambda shot: abs(shot["hitMs"] - time_ms)) if near else None

    match = match_events([event["timeMs"] for event in found], [hit.time_ms for hit in hits], tolerance_ms)
    by_truth = {t: p for p, t in match.pairs}
    rows = []
    for t, hit in enumerate(hits):
        event = found[by_truth[t]] if t in by_truth else None
        shot = shot_at(event["timeMs"]) if event is not None else None
        rows.append({
            "timeMs": hit.time_ms, "hitter": hit.hitter, "name": hit.name, "labels": list(hit.labels),
            "foundMs": None if event is None else event["timeMs"],
            "shot": None if shot is None else shot.get("shot"),
            "foundHitter": None if shot is None else shot["hitter"],
            "predicted": _top_name(shot),
            "call": (shot or {}).get("name", {}).get("call"),
            "sure": _sureness(shot),
        })
    matched = set(by_truth.values())
    extras = []
    for p, event in enumerate(found):
        if p not in matched:
            shot = shot_at(event["timeMs"])
            extras.append({"timeMs": event["timeMs"], "hitter": None if shot is None else shot["hitter"], "evidence": event.get("evidence")})
    return rows, extras


def _names(rows: list[dict], end_to_end: bool) -> dict[str, Any] | None:
    """Macro-F1 of the names on the scored labelled hits: those named by the analysis, or with
    `end_to_end` all of them, a missed or unnamed hit counting as a wrong name. A labelled name
    that may be two labels takes the one named when it is either."""
    scored = [row for row in rows if row["labels"] and (end_to_end or row["predicted"] is not None)]
    if not any(row["predicted"] is not None for row in scored):
        return None
    truth = [row["predicted"] if row["predicted"] in row["labels"] else row["labels"][0] for row in scored]
    predicted = [row["predicted"] or "none" for row in scored]
    result = macro_f1(truth, predicted, GOLDEN_SHOT_LABELS)
    low, high = macro_f1_interval(truth, predicted, GOLDEN_SHOT_LABELS)
    return {
        "shots": len(scored), "right": sum(t == p for t, p in zip(truth, predicted)),
        "macroF1": round(result["macroF1"], 4), "interval": [round(low, 3), round(high, 3)], "perLabel": result["perLabel"],
    }


def _calls(rows: list[dict]) -> dict[str, Any]:
    """How the model's calls fared (the rally's serves left out): a name given alone, against how
    sure it said it was, and two names, against whether the label was one of them."""
    called = [row for row in rows if row["labels"] and row["call"] not in (None, "serve")]
    alone = [row for row in called if " or " not in row["call"]]
    two = [row for row in called if " or " in row["call"]]
    return {
        "alone": {"given": len(alone), "right": sum(row["call"] in row["labels"] for row in alone), "meanP": round(float(np.mean([row["sure"] for row in alone])), 3) if alone else None},
        "twoNames": {"given": len(two), "labelAmong": sum(bool(set(row["call"].split(" or ")) & set(row["labels"])) for row in two)},
    }


def summarise(rows: list[dict], extras: list[dict]) -> dict[str, Any]:
    """Hits, hitters and names over labelled hits from one clip or many."""
    matched = [row for row in rows if row["foundMs"] is not None]
    errors = np.array([row["foundMs"] - row["timeMs"] for row in matched])
    known = [row for row in matched if row["foundHitter"] is not None]
    named = [row for row in matched if row["labels"] and row["predicted"] is not None]
    confusion: dict[str, dict[str, int]] = {}
    for row in named:
        truth = row["predicted"] if row["predicted"] in row["labels"] else "/".join(row["labels"])
        confusion.setdefault(truth, {}).setdefault(row["predicted"], 0)
        confusion[truth][row["predicted"]] += 1
    found = len(matched) + len(extras)
    return {
        "labelled": len(rows), "found": found, "matched": len(matched),
        "recall": round(len(matched) / len(rows), 4) if rows else None,
        "precision": round(len(matched) / found, 4) if found else None,
        "timingMs": None if not len(errors) else {
            "medianAbs": round(float(np.median(np.abs(errors))), 1), "maxAbs": round(float(np.abs(errors).max()), 1), "mean": round(float(errors.mean()), 1),
        },
        "hitterRight": sum(row["foundHitter"] == row["hitter"] for row in known), "hitterKnown": len(known),
        "missed": [row for row in rows if row["foundMs"] is None],
        "extra": extras,
        "wrongHitter": [row for row in known if row["foundHitter"] != row["hitter"]],
        "names": None if not named else {"found": _names(rows, False), "endToEnd": _names(rows, True), "confusion": confusion, "calls": _calls(rows)},
    }


def _line(title: str, report: dict[str, Any]) -> str:
    timing = report["timingMs"]
    text = (f"{title}: labelled {report['labelled']}, found {report['found']}, matched {report['matched']}"
            f" (recall {report['recall']:.0%}, precision {report['precision'] or 0:.0%})")
    if timing:
        text += f", timing {timing['medianAbs']:.0f} / {timing['maxAbs']:.0f} ms, hitter {report['hitterRight']} of {report['hitterKnown']}"
    if report["names"]:
        found, end, alone = report["names"]["found"], report["names"]["endToEnd"], report["names"]["calls"]["alone"]
        low, high = found["interval"]
        text += f", names {found['right']} of {found['shots']} (macro-F1 {found['macroF1']:.2f}, {low:.2f}-{high:.2f}; end to end {end['macroF1']:.2f})"
        if alone["given"]:
            text += f", alone right {alone['right']} of {alone['given']} at mean p {alone['meanP']:.2f}"
    for row in report["missed"]:
        text += f"\n  missed {row['timeMs'] / 1000:6.2f} s {row['hitter']:4s} {row['name']}"
    for row in report["extra"]:
        text += f"\n  extra  {row['timeMs'] / 1000:6.2f} s {row['hitter'] or '-':4s} ({row['evidence']})"
    for row in report["wrongHitter"]:
        text += f"\n  wrong hitter at {row['timeMs'] / 1000:.2f} s: said {row['foundHitter']}, was {row['hitter']}"
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score analyses against hits and shot names labelled by hand.")
    parser.add_argument("labels", type=Path, help="the hittimes file")
    parser.add_argument("clips", nargs="+", help='"Clip 9=clip9.analysis.json", one per clip')
    parser.add_argument("--tolerance-ms", type=float, default=TOLERANCE_MS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    labelled = read_labels(args.labels.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"toleranceMs": args.tolerance_ms, "clips": {}}
    all_rows: list[dict] = []
    all_extras: list[dict] = []
    for item in args.clips:
        clip, _, path = item.partition("=")
        if clip not in labelled:
            parser.error(f"{clip} is not in {args.labels}")
        rows, extras = match_clip(labelled[clip], json.loads(Path(path).read_text(encoding="utf-8")), args.tolerance_ms)
        report["clips"][clip] = summarise(rows, extras)
        all_rows += rows
        all_extras += extras
        print(_line(clip, report["clips"][clip]))
    if len(args.clips) > 1:
        report["pooled"] = summarise(all_rows, all_extras)
        print(_line("All", {**report["pooled"], "missed": [], "extra": [], "wrongHitter": []}))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
