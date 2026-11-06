# worker/evaluation/value_benchmark.py
"""What a position is worth, on ShuttleSet's professional rallies (evaluation.shuttleset).

The 6f rules on professionals. Each hit's time to spare (pressure.work) runs from where the
receiver stood at the hit to where they met the shuttle, in the time between the two hits, with
a reach of REACH and each hit's frame FRAME_SIGMA frames out. A hit whose positions contradict
each other (pressure.contradicts) has none. ShuttleSet does not say when a shot that came down in
reached the floor, so its flight takes as long as answered shots of its name took over the same
way in the training matches (`flight_times`), and is as unsure as they spread. A shot met in the
air is met before it would land, so that time is short and leans towards a winner.

The rules' value: at each hit, the chance its hitter goes on to win the rally, from the chance
that the shot they answered left them under pressure (rally_reasoning.p_under; never on a serve,
even odds when not known), through one logistic curve fitted on the training matches.

Only the training matches are used here; the test matches are for the learned value, once.
Rates come with 95 % intervals from resampling whole matches.

    python -m evaluation.value_benchmark ShuttleSet --out rules.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.shuttleset import Match, Rally, held_out, read
from pressure import Pressure, contradicts, work
from rally_reasoning import PRESSED_SPARE_MS, p_under
from shots import CALL_CONFIDENCE, Estimate

REACH = Estimate(1.4, 0.1)  # the app finds 1.37 to 1.39 m on the user's clips
FRAME_SIGMA = 1.0  # a hit's frame, picked by hand
RESAMPLES = 2000


def contact(rally: Rally, i: int) -> np.ndarray:
    """Where hit `i` met the shuttle: where the shot before went, or for the rally's first hit
    where its hitter stood."""
    return rally.hits[i - 1].goes_to.point if i else rally.hits[i].hitter_at.point


def flight_times(rallies: list[Rally]) -> dict[str, tuple[float, float, float]]:
    """For each name, how long an answered shot took by how far it went: (ms at none, ms per
    metre, the spread), by least squares."""
    ways: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rally in rallies:
        for i, hit in enumerate(rally.hits[:-1]):
            ways[hit.name].append((float(np.linalg.norm(hit.goes_to.point - contact(rally, i))), rally.hits[i + 1].time_ms - hit.time_ms))
    fitted = {}
    for name, pairs in ways.items():
        way, ms = np.array(pairs).T
        slope, at_none = np.polyfit(way, ms, 1)
        fitted[name] = (float(at_none), float(slope), float(np.std(ms - (at_none + slope * way))))
    return fitted


def rally_pressures(rally: Rally, fps: float, flights: dict[str, tuple[float, float, float]], rng: np.random.Generator) -> list[Pressure | None]:
    """Each hit's pressure on its receiver: to where they met it, or, on a shot that came down in,
    to the floor."""
    frame_ms = 1000.0 / fps * FRAME_SIGMA
    out: list[Pressure | None] = []
    for i, hit in enumerate(rally.hits):
        stood = (hit.receiver_at.point, hit.receiver_at.covariance)
        if i + 1 < len(rally.hits):
            after = rally.hits[i + 1]
            time_ms = after.time_ms - hit.time_ms
            there = (after.hitter_at.point, after.hitter_at.covariance)
            if contradicts(stood, there, hit.goes_to.point, hit.goes_to.covariance, REACH, time_ms):
                out.append(None)
            else:
                out.append(work(stood, hit.goes_to.point, hit.goes_to.covariance, REACH, time_ms, None, rng, frame_ms * 2**0.5))
        elif rally.end == "in" and hit.name in flights:
            at_none, slope, spread = flights[hit.name]
            time_ms = at_none + slope * float(np.linalg.norm(hit.goes_to.point - contact(rally, i)))
            out.append(work(stood, hit.goes_to.point, hit.goes_to.covariance, REACH, time_ms, None, rng, float(np.hypot(spread, frame_ms))))
        else:
            out.append(None)
    return out


def answered_pressed(rally: Rally, pressures: list[Pressure | None]) -> list[float]:
    """At each hit, the chance its hitter was under pressure from the shot they answered."""
    out = []
    for i, hit in enumerate(rally.hits):
        if i == 0:
            out.append(0.0 if hit.name == "serve" else 0.5)  # a serve not labelled may have been anything
        else:
            out.append(p_under(None if pressures[i - 1] is None else pressures[i - 1].spare_ms, PRESSED_SPARE_MS))
    return out


def fit_curve(x: np.ndarray, won: np.ndarray, steps: int = 50) -> tuple[float, float]:
    """The logistic curve P(won) = 1 / (1 + exp(-(a + b x))), by Newton's method."""
    design, coef = np.column_stack([np.ones_like(x), x]), np.zeros(2)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-design @ coef))
        step = np.linalg.solve(design.T @ (design * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2), design.T @ (won - p))
        coef += step
        if np.abs(step).max() < 1e-10:
            break
    return float(coef[0]), float(coef[1])


def log_loss(p: np.ndarray, won: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(won * np.log(p) + (1 - won) * np.log(1 - p)))


def by_match_rate(counts: dict[str, tuple[int, int]], rng: np.random.Generator) -> dict[str, Any]:
    """k of n pooled over matches, with a 95 % interval from resampling whole matches."""
    k, n = np.array(list(counts.values()), dtype=float).T if counts else (np.zeros(0), np.zeros(0))
    if n.sum() == 0:
        return {"k": 0, "n": 0}
    picks = rng.integers(0, len(n), (RESAMPLES, len(n)))
    shares = k[picks].sum(axis=1) / np.maximum(n[picks].sum(axis=1), 1)
    low, high = np.percentile(shares, [2.5, 97.5])
    return {"k": int(k.sum()), "n": int(n.sum()), "share": round(float(k.sum() / n.sum()), 3), "interval": [round(float(low), 3), round(float(high), 3)]}


def rules(matches: list[Match], train: list[str]) -> dict[str, Any]:
    """The 6f rules on the training matches."""
    rng = np.random.default_rng(0)
    learn = [match for match in matches if match.video in set(train)]
    flights = flight_times([rally for match in learn for rally in match.rallies])
    rows = []  # match, chance the hitter was under pressure, whether they won
    answered: dict[str, Counter] = defaultdict(Counter)
    landed, errors = Counter(), Counter()
    for match in learn:
        for rally in match.rallies:
            pressures = rally_pressures(rally, match.fps, flights, rng)
            for i, pressure in enumerate(pressures[:-1]):
                tally = answered[match.video]
                tally["hits"] += 1
                if pressure is None:
                    tally["contradict"] += 1
                    continue
                unreached = p_under(pressure.spare_ms, 0.0)
                tally["unreached"] += unreached >= CALL_CONFIDENCE
                tally["pressed"] += p_under(pressure.spare_ms, PRESSED_SPARE_MS) >= CALL_CONFIDENCE
            pressed = answered_pressed(rally, pressures)
            rows += [(match.video, x, hit.hitter == rally.winner) for x, hit in zip(pressed, rally.hits)]
            if rally.end == "in" and pressures[-1] is not None:
                unreached = p_under(pressures[-1].spare_ms, 0.0)
                said = "winner" if unreached >= CALL_CONFIDENCE else "let in" if unreached <= 1 - CALL_CONFIDENCE else "too close to call"
                landed[(rally.how, said)] += 1
            elif rally.end != "in":
                forced = pressed[-1]
                errors["forced" if forced >= CALL_CONFIDENCE else "unforced" if forced <= 1 - CALL_CONFIDENCE else "too close to call"] += 1

    videos, x, won = np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows], float)
    a, b = fit_curve(x, won)
    curve = 1.0 / (1.0 + np.exp(-(a + b * x)))

    def lost(mask: np.ndarray) -> dict[str, Any]:
        return by_match_rate({video: (int((~won[mask & (videos == video)].astype(bool)).sum()), int((mask & (videos == video)).sum())) for video in train}, rng)

    return {
        "matches": len(learn),
        "rallies": sum(len(match.rallies) for match in learn),
        "flightTimesMs": {name: {"atNone": round(f[0]), "perMetre": round(f[1]), "spread": round(f[2])} for name, f in sorted(flights.items())},
        "answered": {
            "hits": sum(t["hits"] for t in answered.values()),
            "contradict": by_match_rate({v: (t["contradict"], t["hits"]) for v, t in answered.items()}, rng),
            "unreached": by_match_rate({v: (t["unreached"], t["hits"] - t["contradict"]) for v, t in answered.items()}, rng),
            "pressed": by_match_rate({v: (t["pressed"], t["hits"] - t["contradict"]) for v, t in answered.items()}, rng),
        },
        "hitterLoses": {
            "pressed": lost(x >= CALL_CONFIDENCE),
            "notPressed": lost(x <= 1 - CALL_CONFIDENCE),
            "tooCloseToCall": lost((x > 1 - CALL_CONFIDENCE) & (x < CALL_CONFIDENCE)),
        },
        "curve": {"a": round(a, 4), "b": round(b, 4), "logLoss": round(log_loss(curve, won), 4), "logLossAtTheRate": round(log_loss(np.full_like(won, won.mean()), won), 4)},
        "winnerOrLetIn": {f"{how}, said {said}": count for (how, said), count in sorted(landed.items())},
        "errors": dict(errors),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The 6f rules on ShuttleSet's training matches.")
    parser.add_argument("root", type=Path, help="the folder holding ShuttleSet's set/")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    matches = read(args.root)
    report = rules(matches, held_out(matches)["train"])
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
