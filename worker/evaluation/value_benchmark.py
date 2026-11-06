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

Winner or let in, as the rules were first written: a winner when the receiver had no time to get
to it. As learnt here (`fit_let_in`, rally_reasoning.LET_IN): let in with the chance the receiver
could get there times the chance a professional left such a shot, by how far inside the lines it
came down and its name. It is scored leaving each training match out of its own fit.

The learned value (rally_value), from the training matches only (`learn`): the model, and how
much of it, is chosen by leaving folds of whole matches out (`FOLDS`), then fitted on them all
and again on `REFITS` draws of them with replacement.

The test matches are opened once (`test`), with the model and the rules as fitted: the held-out
matches and the held-out players' matches, as ShuttleSet clicked them and again with positions as
uncertain as the app's on the user's clips (`blurred`). Who wins is told by log loss a hit, and
the value's chances are checked in tenths and by refitting the share won on their log odds
(intercept 0 and slope 1 when they are calibrated). The learnt let-in rule is scored there too.

Rates and differences come with 95 % intervals from resampling whole matches.

    python -m evaluation.value_benchmark ShuttleSet rules --out rules.json
    python -m evaluation.value_benchmark ShuttleSet learn --out learn.json  # writes rally_value.json
    python -m evaluation.value_benchmark ShuttleSet test --out test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from evaluation.shuttleset import Match, Rally, Spot, held_out, read
from geometry.court_model import NET_Y_M
from pressure import Pressure, contradicts, work
from rally_reasoning import PRESSED_SPARE_MS, inside_by, left_by_professionals, p_under
from rally_value import FACTS, NAMES, At, Model, chance, fact_rows, fit, named
from shots import CALL_CONFIDENCE, Estimate, landing_area

REACH = Estimate(1.4, 0.1)  # the app finds 1.37 to 1.39 m on the user's clips
FRAME_SIGMA = 1.0  # a hit's frame, picked by hand
RESAMPLES = 2000
FOLDS = 5
L2S = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
REFITS = 20
_NAMED = tuple(fact for fact in FACTS if fact == "then" or fact.split(" ")[-1] in NAMES)
CANDIDATES = {"names": (_NAMED, False), "names and their products": (_NAMED, True), "linear": (FACTS, False), "products": (FACTS, True)}
# How unsure the app was on the user's five clips, medians over their hits: where a player stood
# (the longer axis, taken both ways) and where the shuttle was met (across, along), by half.
CLIPS_PLAYER_SIGMA_M = {"near": 0.13, "far": 0.25}
CLIPS_CONTACT_SIGMA_M = {"near": (0.14, 0.16), "far": (0.25, 0.45)}


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


def fit_let_in(unreached: np.ndarray, inside: np.ndarray, high: np.ndarray, push: np.ndarray, let_in: np.ndarray) -> tuple[float, ...]:
    """rally_reasoning.LET_IN by maximum likelihood: a shot that came down in was let in with the
    chance (1 - unreached) * left_by_professionals(inside, high, push)."""
    def loss(weights: np.ndarray) -> float:
        p = np.clip((1.0 - unreached) * left_by_professionals(inside, high, push, weights), 1e-9, 1 - 1e-9)
        return float(-np.sum(let_in * np.log(p) + (1 - let_in) * np.log(1 - p)))

    return tuple(round(float(w), 2) for w in minimize(loss, np.zeros(4), method="BFGS").x)


def _came_down(rally: Rally, pressures: list[Pressure | None]) -> tuple[float, float, bool, bool, bool] | None:
    """For a rally that ended with the shuttle down in: the chance the receiver could not get
    there, how far inside the lines it came down, whether it was a clear or a lift, and a push,
    and whether it was let in. None for other ends, or with no pressure."""
    if rally.end != "in" or pressures[-1] is None:
        return None
    last = rally.hits[-1]
    half = "near" if last.hitter_at.point[1] < NET_Y_M else "far"
    area = landing_area("singles", half, float(last.hitter_at.point[0]) if last.name == "serve" else None)
    inside = float(inside_by(last.goes_to.point, area)[0])
    return p_under(pressures[-1].spare_ms, 0.0), inside, last.name in ("clear", "lift"), last.name == "push", rally.how == "let in"


def _said(p_let_in: float) -> str:
    return "let in" if p_let_in >= CALL_CONFIDENCE else "winner" if p_let_in <= 1 - CALL_CONFIDENCE else "too close to call"


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
    landed = []  # match, chance the receiver could not get there, inside the lines, clear or lift, push, let in
    answered: dict[str, Counter] = defaultdict(Counter)
    first_written, errors = Counter(), Counter()
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
            came_down = _came_down(rally, pressures)
            if came_down is not None:
                first_written[(rally.how, _said(1.0 - came_down[0]))] += 1
                landed.append((match.video, *came_down))
            elif rally.end != "in":
                forced = pressed[-1]
                errors["forced" if forced >= CALL_CONFIDENCE else "unforced" if forced <= 1 - CALL_CONFIDENCE else "too close to call"] += 1

    videos, x, won = np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows], float)
    a, b = fit_curve(x, won)
    curve = 1.0 / (1.0 + np.exp(-(a + b * x)))

    where, *features, let_in = (np.array(column, dtype=float if k else object) for k, column in enumerate(zip(*landed)))
    left_out = np.zeros(len(let_in))
    for video in train:
        out = where == video
        if out.any():
            weights = fit_let_in(*(f[~out] for f in features), let_in[~out])
            left_out[out] = (1.0 - features[0][out]) * left_by_professionals(*(f[out] for f in features[1:]), weights)
    learnt = Counter(("let in" if let else "winner", _said(p)) for p, let in zip(left_out, let_in))

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
        "winnerOrLetIn": {
            "asFirstWritten": {f"{how}, said {said}": count for (how, said), count in sorted(first_written.items())},
            "learnt": {
                "weights": fit_let_in(*features, let_in),
                "eachMatchLeftOut": {f"{how}, said {said}": count for (how, said), count in sorted(learnt.items())},
                "logLoss": round(log_loss(left_out, let_in), 4),
                "logLossAtTheRate": round(log_loss(np.full_like(let_in, let_in.mean()), let_in), 4),
            },
        },
        "errors": dict(errors),
    }


def rally_ats(rally: Rally, pressures: list[Pressure | None]) -> list[At]:
    """What was known at each hit: the hitter met the shuttle where the shot before went (the
    rally's first hit, where its hitter stood)."""
    ats = []
    for i, hit in enumerate(rally.hits):
        before = rally.hits[i - 1] if i else None
        met = hit.hitter_at if before is None else before.goes_to
        ats.append(At(
            (hit.hitter_at.point, hit.hitter_at.covariance), (hit.receiver_at.point, hit.receiver_at.covariance), (met.point, met.covariance),
            None if before is None else hit.time_ms - before.time_ms,
            None if before is None or pressures[i - 1] is None else pressures[i - 1].spare_ms,
            None if before is None else named(before.name), named(hit.name),
        ))
    return ats


@dataclass
class Hits:
    """Every hit of some matches: what was known at it and at the hit before, whether its hitter
    won the rally, and the chance the shot they answered left them under pressure."""

    videos: np.ndarray
    ats: list[tuple[At, At | None]]
    won: np.ndarray
    pressed: np.ndarray

    def rows(self) -> np.ndarray:
        return np.vstack([fact_rows(now, then) for now, then in self.ats])


def hits_of(matches: list[Match], flights: dict[str, tuple[float, float, float]], rng: np.random.Generator) -> Hits:
    videos, ats, won, pressed = [], [], [], []
    for match in matches:
        for rally in match.rallies:
            pressures = rally_pressures(rally, match.fps, flights, rng)
            at = rally_ats(rally, pressures)
            for i, (hit, x) in enumerate(zip(rally.hits, answered_pressed(rally, pressures))):
                videos.append(match.video)
                ats.append((at[i], at[i - 1] if i else None))
                won.append(hit.hitter == rally.winner)
                pressed.append(x)
    return Hits(np.array(videos), ats, np.array(won, float), np.array(pressed))


def _curve(coefficients: tuple[float, float], x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(coefficients[0] + coefficients[1] * x)))


def _training(matches: list[Match], train: list[str], rng: np.random.Generator) -> tuple[dict[str, tuple[float, float, float]], Hits]:
    learn_on = [match for match in matches if match.video in set(train)]
    flights = flight_times([rally for match in learn_on for rally in match.rallies])
    return flights, hits_of(learn_on, flights, rng)


def learn(matches: list[Match], train: list[str]) -> tuple[Model, dict[str, Any]]:
    """The learned value: each of CANDIDATES at each of L2S scored on folds of the training
    matches, each fold's hits told by a model fitted on the others; the best is fitted on them all
    and on REFITS draws of them with replacement."""
    rng = np.random.default_rng(0)
    _, hits = _training(matches, train, rng)
    rows, won = hits.rows(), hits.won
    fold_of = {video: k % FOLDS for k, video in enumerate(rng.permutation(sorted(set(hits.videos))))}
    fold = np.array([fold_of[video] for video in hits.videos])
    out_of_fold = {"rules": np.zeros(len(won))}
    for k in range(FOLDS):
        out_of_fold["rules"][fold == k] = _curve(fit_curve(hits.pressed[fold != k], won[fold != k]), hits.pressed[fold == k])
    for label, (facts, products) in CANDIDATES.items():
        for l2 in L2S:
            p = np.zeros(len(won))
            for k in range(FOLDS):
                framed = Model.framed(rows[fold != k], facts, products)
                weights = fit(framed, rows[fold != k], won[fold != k], l2)
                p[fold == k] = replace(framed, weights=weights[None]).chances(rows[fold == k])[:, 0]
            out_of_fold[(label, l2)] = p
    scores = {key: log_loss(p, won) for key, p in out_of_fold.items()}
    label, l2 = min((key for key in scores if key != "rules"), key=scores.get)
    framed = Model.framed(rows, *CANDIDATES[label])
    weights = [fit(framed, rows, won, l2)]
    videos = sorted(set(hits.videos))
    for _ in range(REFITS):
        index = np.concatenate([np.flatnonzero(hits.videos == video) for video in rng.choice(videos, len(videos))])
        weights.append(fit(framed, rows[index], won[index], l2, start=weights[0]))
    lift = _losses(out_of_fold["rules"], won) - _losses(out_of_fold[(label, l2)], won)
    report = {
        "hits": len(won),
        "foldsLogLoss": {"rules": round(scores["rules"], 4), **{f"{key[0]}, l2 {key[1]:g}": round(score, 4) for key, score in scores.items() if key != "rules"}},
        "chosen": {"model": label, "l2": l2, "terms": len(weights[0])},
        "liftOverRulesOnFolds": by_match_mean(lift, hits.videos, rng),
    }
    return replace(framed, weights=np.array(weights)), report


def _losses(p: np.ndarray, won: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(won * np.log(p) + (1 - won) * np.log(1 - p))


def by_match_mean(values: np.ndarray, videos: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """The mean over hits, with a 95 % interval from resampling whole matches."""
    names = sorted(set(videos))
    sums = np.array([values[videos == video].sum() for video in names])
    counts = np.array([(videos == video).sum() for video in names])
    picks = rng.integers(0, len(names), (RESAMPLES, len(names)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(sums.sum() / counts.sum()), 4), "interval": [round(float(low), 4), round(float(high), 4)]}


def blurred(match: Match, rng: np.random.Generator) -> Match:
    """The match with every position moved, and as unsure, as the app's on the user's clips."""
    def blur(spot: Spot, sigmas: dict[str, Any]) -> Spot:
        sigma = np.broadcast_to(sigmas["near" if spot.point[1] < NET_Y_M else "far"], 2)
        return Spot(spot.point + rng.normal(0.0, sigma), spot.covariance + np.diag(np.square(sigma)))

    rallies = []
    for rally in match.rallies:
        hits = [replace(hit, hitter_at=blur(hit.hitter_at, CLIPS_PLAYER_SIGMA_M), receiver_at=blur(hit.receiver_at, CLIPS_PLAYER_SIGMA_M), goes_to=blur(hit.goes_to, CLIPS_CONTACT_SIGMA_M)) for hit in rally.hits]
        rallies.append(replace(rally, hits=hits))
    return replace(match, rallies=rallies)


def told(hits: Hits, model: Model, curve: tuple[float, float], rng: np.random.Generator) -> dict[str, Any]:
    """How well the learned value and the rules' value tell who wins, on hits neither learnt from."""
    value = np.array([chance(model, now, then, rng)[0] for now, then in hits.ats])
    rules_value, won, videos = _curve(curve, hits.pressed), hits.won, hits.videos
    tenths = {}
    for k in range(10):
        inside = np.minimum((value * 10).astype(int), 9) == k
        if inside.any():
            share = by_match_rate({video: (int(won[inside & (videos == video)].sum()), int((inside & (videos == video)).sum())) for video in sorted(set(videos))}, rng)
            tenths[f"{k / 10:.1f}-{(k + 1) / 10:.1f}"] = {"hits": int(inside.sum()), "chance": round(float(value[inside].mean()), 3), "won": share.get("share"), "interval": share.get("interval")}
    log_odds = np.log(np.clip(value, 1e-9, 1 - 1e-9) / np.clip(1 - value, 1e-9, 1 - 1e-9))
    names = sorted(set(videos))
    index = [np.flatnonzero(videos == video) for video in names]
    refits = []
    for picks in rng.integers(0, len(names), (RESAMPLES, len(names))):
        chosen = np.concatenate([index[i] for i in picks])
        refits.append(fit_curve(log_odds[chosen], won[chosen]))
    intercept, slope = fit_curve(log_odds, won)
    (intercept_low, slope_low), (intercept_high, slope_high) = np.percentile(np.array(refits), [2.5, 97.5], axis=0)
    return {
        "matches": len(names),
        "hits": len(won),
        "logLoss": {"value": round(float(_losses(value, won).mean()), 4), "rules": round(float(_losses(rules_value, won).mean()), 4), "evenOdds": round(float(np.log(2.0)), 4)},
        "liftOverRules": by_match_mean(_losses(rules_value, won) - _losses(value, won), videos, rng),
        "tenths": tenths,
        "calibration": {
            "intercept": round(intercept, 3), "interceptInterval": [round(float(intercept_low), 3), round(float(intercept_high), 3)],
            "slope": round(slope, 3), "slopeInterval": [round(float(slope_low), 3), round(float(slope_high), 3)],
        },
    }


def let_in_told(matches: list[Match], flights: dict[str, tuple[float, float, float]], rate: float, rng: np.random.Generator) -> dict[str, Any]:
    """The learnt let-in rule (rally_reasoning.LET_IN) on rallies it never saw."""
    landed = [came_down for match in matches for rally in match.rallies if (came_down := _came_down(rally, rally_pressures(rally, match.fps, flights, rng))) is not None]
    unreached, inside, high, push, let_in = (np.array(column, dtype=float) for column in zip(*landed))
    p = (1.0 - unreached) * left_by_professionals(inside, high, push)
    said = Counter(("let in" if let else "winner", _said(chance)) for chance, let in zip(p, let_in))
    return {
        "said": {f"{how}, said {call}": count for (how, call), count in sorted(said.items())},
        "logLoss": round(log_loss(p, let_in), 4),
        "logLossAtTheTrainingRate": round(log_loss(np.full_like(let_in, rate), let_in), 4),
    }


def test(matches: list[Match], held: dict[str, Any], model: Model) -> dict[str, Any]:
    """The test matches, opened once: the learned value against the rules on the held-out matches
    and the held-out players', as clicked and as uncertain as the user's clips."""
    rng = np.random.default_rng(0)
    flights, hits = _training(matches, held["train"], rng)
    curve = fit_curve(hits.pressed, hits.won)
    learnt_on = [rally for match in matches if match.video in set(held["train"]) for rally in match.rallies]
    rate = sum(rally.how == "let in" for rally in learnt_on) / sum(rally.end == "in" for rally in learnt_on)
    report: dict[str, Any] = {}
    for label, videos in (("heldOutMatches", held["test"]), ("heldOutPlayers", held["stressed"])):
        chosen = [match for match in matches if match.video in set(videos)]
        report[label] = {
            "asClicked": told(hits_of(chosen, flights, rng), model, curve, rng),
            "asUncertainAsYourClips": told(hits_of([blurred(match, rng) for match in chosen], flights, rng), model, curve, rng),
            "letIn": let_in_told(chosen, flights, rate, rng),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The 6f rules and the learned value on ShuttleSet.")
    parser.add_argument("root", type=Path, help="the folder holding ShuttleSet's set/")
    parser.add_argument("what", nargs="?", default="rules", choices=("rules", "learn", "test"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    matches = read(args.root)
    held = held_out(matches)
    if args.what == "learn":
        model, report = learn(matches, held["train"])
        model.save()
    elif args.what == "test":
        report = test(matches, held, Model.load())
    else:
        report = rules(matches, held["train"])
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
