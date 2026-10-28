# worker/evaluation/shot_names_benchmark.py
"""Shots to learn names from, the models learnt from them, and shots to test the names on.

Learning from simulation: the true facts of simulated rallies' shots
(simulation.rally.random_rally), drawn from a random stream of their own, so no test rally is
among them. The weights' hold towards zero is chosen on further learning rallies the fit never
saw. Testing: the hits benchmark's scenarios (every view at 30 and 60 fps, with sound and
without, jitter, dropout, false detections, neighbouring courts), from seeds the learning never
uses (from TEST_FIRST_SEED; the first sixty are the hits benchmark's held-out rallies), through
the real hit finding and flight fitting. Each shot matched to a true hit (within ±33 ms) keeps its
true name and facts, its measured facts, and draws of them from their uncertainty, and whether it
went cross-court, truly and over the draws. Serves are left out: the rally names them.

Learning from real play: shots named by hand (evaluation.hand_labels) move the simulated model as
far as they can show (shot_names.learn), held towards it. Each clip is named by a model learnt
without it, its hold chosen the same way on the other clips alone, so its score is honest; the
model kept is learnt from every clip.

    python -m evaluation.shot_names_benchmark measure --scenarios 120 --workers 12 --out held_out.npz
    python -m evaluation.shot_names_benchmark measure --scenarios 120 --workers 12 --stress --out stressed.npz
    python -m evaluation.shot_names_benchmark learn --rallies 3000 --workers 12
    python -m evaluation.shot_names_benchmark adapt Clips/hittimes.txt "Clip 9=clip9.analysis.json" ... --named named/
    python -m evaluation.shot_names_benchmark score held_out.npz stressed.npz --out names.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.hand_labels import LabelledHit, match_clip, read_labels, summarise
from evaluation.hits_benchmark import TOLERANCE_MS, simulate, standard_scenarios
from evaluation.metrics import macro_f1, macro_f1_interval, match_events
from hits import find_hits
from shot_names import MODEL_PATH, NAMES, SIMULATED_PATH, TOO_CLOSE, Detail, Launch, Model, call_of, chances, detail, learn, measured_facts, name_shot, nominal_facts, true_facts
from shots import rally_shots
from simulation.rally import random_rally

LEARNING_STREAM = 5
TEST_FIRST_SEED = 5000
HOLDS = (1e-7, 1e-6, 1e-5, 1e-4)  # the weights' hold towards zero, chosen among these
REAL_HOLDS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # real shots' hold towards the simulated weights
CHECK_SHARE = 0.2  # further learning rallies, as a share of those learnt from, to choose it on
CONFIDENCE_BINS = (0.0, 0.5, 0.65, 0.8, 0.9, 1.0001)


def _learning_rally(seed: int):
    return random_rally(np.random.default_rng([LEARNING_STREAM, seed]))


def learning_set(rallies: int, first_seed: int = 0, workers: int = 1) -> tuple[list[str], np.ndarray]:
    """The names and true facts of every shot but the serve of `rallies` random rallies."""
    seeds = range(first_seed, first_seed + rallies)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            built = list(pool.map(_learning_rally, seeds, chunksize=16))
    else:
        built = [_learning_rally(seed) for seed in seeds]
    names, rows, _ = true_facts(built)
    keep = [k for k, name in enumerate(names) if name != "serve"]
    return [names[k] for k in keep], rows[keep]


def learn_model(rallies: int, workers: int = 1) -> tuple[Model, dict[str, Any]]:
    """The model learnt from `rallies` rallies, its hold chosen on the next ones' log loss."""
    names, rows = learning_set(rallies, workers=workers)
    check_names, check_rows = learning_set(max(1, int(CHECK_SHARE * rallies)), first_seed=rallies, workers=workers)
    truth = [NAMES.index(name) for name in check_names]
    tried = []
    for hold in HOLDS:
        model = learn(names, rows, hold)
        p = model.probabilities(check_rows)
        tried.append((float(-np.mean(np.log(p[np.arange(len(p)), truth] + 1e-12))), hold, model))
    loss, hold, model = min(tried, key=lambda t: t[0])
    return model, {"rallies": rallies, "shots": len(names), "hold": hold, "checkLogLoss": round(loss, 4), "tried": {h: round(l, 4) for l, h, _ in tried}}


def measure_scenario(scenario) -> list[dict[str, Any]]:
    """Every shot of one scenario that was found, with its true and measured facts."""
    sim = simulate(scenario)
    analysis = find_hits(sim.track, sim.times_ms, sim.cameras, sim.onsets)
    shots = rally_shots(analysis, sim.track, sim.times_ms, sim.cameras)
    names, truth, crossed = true_facts([sim.clip.rally])
    match = match_events([shot.hit_ms for shot in shots], [c.time * 1000.0 for c in sim.clip.rally.contacts], TOLERANCE_MS)
    rng = np.random.default_rng(scenario.seed)
    rows = []
    for p, t in match.pairs:
        if names[t] == "serve":
            continue
        draws, cross = measured_facts(Launch.of(shots[p]), rng)
        rows.append({
            "name": names[t], "truth": truth[t], "nominal": nominal_facts(Launch.of(shots[p])), "draws": draws,
            "cross": crossed[t], "crossDrawn": float(np.nanmean(cross)), "view": scenario.view, "fps": scenario.fps, "seed": scenario.seed,
        })
    return rows


def measure(scenarios, workers: int = 1) -> dict[str, np.ndarray]:
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            found = list(pool.map(measure_scenario, scenarios))
    else:
        found = [measure_scenario(s) for s in scenarios]
    rows = [row for scenario in found for row in scenario]
    return {key: np.array([row[key] for row in rows]) for key in ("name", "truth", "nominal", "draws", "cross", "crossDrawn", "view", "fps", "seed")}


def _f1(names: list[str], p: np.ndarray, seeds: np.ndarray) -> dict[str, float]:
    """Macro-F1 of the likeliest names, with its 95 % interval over resamples of whole scenarios
    (one rally's shots are not independent of each other)."""
    calls = [NAMES[k] for k in p.argmax(axis=1)]
    low, high = macro_f1_interval(names, calls, NAMES, seeds)
    return {"value": round(macro_f1(names, calls, NAMES)["macroF1"], 4), "low": round(low, 3), "high": round(high, 3)}


def score_cross(held_out: dict[str, np.ndarray]) -> dict[str, Any]:
    """Cross or straight over the draws against the true flight: per call, how often it was given,
    the chance of cross it said on average, and the share truly cross."""
    truth, p = held_out["cross"].astype(bool), held_out["crossDrawn"]
    calls = np.array([detail(chance, "cross", "straight").call for chance in p])
    out: dict[str, Any] = {"shots": len(truth), "trulyCross": int(truth.sum())}
    for call in ("cross", "straight", TOO_CLOSE):
        given = calls == call
        out[call] = {"given": int(given.sum()), "meanP": round(float(p[given].mean()), 3) if given.any() else None, "trulyCross": int(truth[given].sum())}
    return out


def score(model: Model, held_out: dict[str, np.ndarray]) -> dict[str, Any]:
    """The names on held-out shots: macro-F1 from the true facts (how far the families overlap by
    themselves), from the measured facts alone, and over their draws (as the pipeline names
    them), each with its interval. For the last: how often a name given alone is right, by name
    and by how sure."""
    names = held_out["name"].tolist()
    draws, seeds = held_out["draws"], held_out["seed"]
    p = chances(model, draws)
    calls = [call_of(row) for row in p]
    alone = {name: {"given": 0, "right": 0} for name in NAMES}
    for name, call in zip(names, calls):
        if " or " not in call:
            alone[call]["given"] += 1
            alone[call]["right"] += call == name
    pairs = [(name, call) for name, call in zip(names, calls) if " or " in call]
    top = p.max(axis=1)
    right = p.argmax(axis=1) == np.array([NAMES.index(name) for name in names])
    bins = []
    for low, high in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        inside = (top >= low) & (top < high)
        if inside.any():
            bins.append({"from": low, "to": min(high, 1.0), "shots": int(inside.sum()), "meanP": round(float(top[inside].mean()), 3), "right": round(float(right[inside].mean()), 3)})
    views = {}
    for view, fps in sorted(set(zip(held_out["view"].tolist(), held_out["fps"].tolist()))):
        rows = (held_out["view"] == view) & (held_out["fps"] == fps)
        views[f"{view} {fps:.0f} fps"] = {"shots": int(rows.sum()), "macroF1": _f1([n for n, r in zip(names, rows) if r], p[rows], seeds[rows])}
    return {
        "shots": len(names), "byName": dict(Counter(names)),
        "macroF1": {
            "trueFacts": _f1(names, model.probabilities(held_out["truth"]), seeds),
            "measured": _f1(names, model.probabilities(held_out["nominal"]), seeds),
            "drawn": _f1(names, p, seeds),
        },
        "givenAlone": {"share": round(1 - len(pairs) / len(names), 3), "byName": alone},
        "twoNames": {"shots": len(pairs), "truthAmongThem": sum(name in call.split(" or ") for name, call in pairs)},
        "sureness": bins,
        "byView": views,
    }


def real_set(labelled: dict[str, list[LabelledHit]], analyses: dict[str, dict]) -> list[dict[str, Any]]:
    """Every hit named by hand that was found as a shot, serves left out (the rally names them):
    its clip, the names it may be, and its measured facts and draws of them."""
    rng = np.random.default_rng(0)
    rows = []
    for clip, analysis in analyses.items():
        drag = analysis["summary"]["terminalVelocity"]
        for row in match_clip(labelled[clip], analysis)[0]:
            if row["shot"] is not None and row["labels"] and "serve" not in row["labels"]:
                launch = Launch.entry(analysis["shots"][row["shot"] - 1], drag)
                rows.append({"clip": clip, "names": tuple(row["labels"]), "nominal": nominal_facts(launch), "draws": measured_facts(launch, rng)[0]})
    return rows


def _learnt(prior: Model, rows: list[dict[str, Any]], hold: float) -> Model:
    return learn([row["names"] for row in rows], np.array([row["nominal"] for row in rows]), hold, prior)


def _log_loss(model: Model, rows: list[dict[str, Any]]) -> float:
    """Over the shots' draws, as the pipeline names them; a shot that may be either of two names
    is right with either."""
    right = [sum(chances(model, row["draws"])[NAMES.index(name)] for name in row["names"]) for row in rows]
    return -float(np.sum(np.log(np.array(right) + 1e-12)))


def choose_hold(prior: Model, rows: list[dict[str, Any]], clips: list[str]) -> tuple[float, dict[float, float]]:
    """The hold under which each of `clips` is best named by a model learnt from the others."""
    tried = {}
    for hold in REAL_HOLDS:
        tried[hold] = round(sum(
            _log_loss(_learnt(prior, [r for r in rows if r["clip"] in clips and r["clip"] != clip], hold), [r for r in rows if r["clip"] == clip])
            for clip in clips
        ), 3)
    return min(tried, key=tried.get), tried


def named_again(analysis: dict[str, Any], model: Model) -> dict[str, Any]:
    """The analysis with every shot named by `model`, the details from the hitter's body kept."""
    rng = np.random.default_rng(0)
    drag = analysis["summary"]["terminalVelocity"]
    shots = []
    for shot in analysis["shots"]:
        name = name_shot(Launch.entry(shot, drag), shot["serve"], model, rng)
        kept = {key: Detail(**shot["name"][key]) if shot["name"].get(key) else None for key in ("backhand", "jump")}
        shots.append({**shot, "name": replace(name, **kept).entry()})
    return {**analysis, "shots": shots}


def adapt(prior: Model, labelled: dict[str, list[LabelledHit]], analyses: dict[str, dict]) -> tuple[Model, dict[str, Any], dict[str, dict]]:
    """The model learnt from every clip's named shots, a report, and each clip named by a model
    learnt without it."""
    rows = real_set(labelled, analyses)
    clips = list(analyses)
    named, holds = {}, {}
    for clip in clips:
        holds[clip], _ = choose_hold(prior, rows, [c for c in clips if c != clip])
        named[clip] = named_again(analyses[clip], _learnt(prior, [r for r in rows if r["clip"] != clip], holds[clip]))
    hold, tried = choose_hold(prior, rows, clips)
    report = {"shots": len(rows), "byClip": dict(Counter(row["clip"] for row in rows)), "holdWithout": holds, "hold": hold, "tried": tried}
    return _learnt(prior, rows, hold), report, named


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Learn shot names from simulated shots and test them on held-out ones.")
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="write held-out shots, measured through the pipeline")
    measuring.add_argument("--scenarios", type=int, default=120)
    measuring.add_argument("--first-seed", type=int, default=TEST_FIRST_SEED)
    measuring.add_argument("--stress", action="store_true", help="a world that differs from the method's models")
    measuring.add_argument("--workers", type=int, default=1)
    measuring.add_argument("--out", type=Path, required=True)
    learning = commands.add_parser("learn", help="learn the model from simulated rallies' true facts")
    learning.add_argument("--rallies", type=int, default=3000)
    learning.add_argument("--workers", type=int, default=1)
    learning.add_argument("--out", type=Path, default=SIMULATED_PATH)
    adapting = commands.add_parser("adapt", help="move the simulated model by shots named by hand")
    adapting.add_argument("labels", type=Path, help="the hittimes file")
    adapting.add_argument("clips", nargs="+", help='"Clip 9=clip9.analysis.json", one per clip')
    adapting.add_argument("--prior", type=Path, default=SIMULATED_PATH)
    adapting.add_argument("--out", type=Path, default=MODEL_PATH)
    adapting.add_argument("--named", type=Path, help="a folder for each clip's analysis named by the model learnt without it")
    adapting.add_argument("--report", type=Path)
    scoring = commands.add_parser("score", help="score the model on held-out shots")
    scoring.add_argument("held_out", type=Path, nargs="+")
    scoring.add_argument("--model", type=Path, default=MODEL_PATH)
    scoring.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "measure":
        scenarios = [replace(s, stress=args.stress) for s in standard_scenarios(args.scenarios, args.first_seed)]
        held_out = measure(scenarios, args.workers)
        np.savez_compressed(args.out, **held_out)
        print(f"{len(held_out['name'])} shots from {len(scenarios)} scenarios:", dict(Counter(held_out["name"].tolist())))
    elif args.command == "learn":
        model, report = learn_model(args.rallies, args.workers)
        model.save(args.out)
        print(json.dumps(report))
    elif args.command == "adapt":
        labelled = read_labels(args.labels.read_text(encoding="utf-8"))
        paths = dict(item.partition("=")[::2] for item in args.clips)
        analyses = {clip: json.loads(Path(path).read_text(encoding="utf-8")) for clip, path in paths.items()}
        model, report, named = adapt(Model.load(args.prior), labelled, analyses)
        model.save(args.out)
        everything = [], []
        for clip, analysis in named.items():
            rows, extras = match_clip(labelled[clip], analysis)
            report.setdefault("clips", {})[clip] = summarise(rows, extras)["names"]
            everything[0].extend(rows)
            everything[1].extend(extras)
            if args.named:
                args.named.mkdir(parents=True, exist_ok=True)
                (args.named / Path(paths[clip]).name).write_text(json.dumps(analysis), encoding="utf-8")
        report["pooled"] = summarise(*everything)["names"]
        text = json.dumps(report, indent=2)
        if args.report:
            args.report.write_text(text, encoding="utf-8")
        print(text)
    else:
        model = Model.load(args.model)
        report = {path.name: {**score(model, dict(np.load(path))), "crossOrStraight": score_cross(dict(np.load(path)))} for path in args.held_out}
        text = json.dumps(report, indent=2)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
