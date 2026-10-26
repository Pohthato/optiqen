# worker/evaluation/shot_names_benchmark.py
"""Shots to learn names from, and shots to test the names on.

Learning: the true facts of simulated rallies' shots (simulation.rally.random_rally), drawn from a
random stream of their own, so no test rally is among them. Testing: the hits benchmark's
scenarios (every view at 30 and 60 fps, with sound and without, jitter, dropout, false
detections, neighbouring courts), from seeds no other benchmark uses, through the real hit
finding and flight fitting. Each shot matched to a true hit (within ±33 ms) keeps its true name
and facts, its measured facts, and draws of them from their uncertainty. Serves are left out:
the rally names them.

    python -m evaluation.shot_names_benchmark --scenarios 120 --workers 12 --out held_out.npz
    python -m evaluation.shot_names_benchmark --scenarios 120 --workers 12 --stress --out stressed.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.hits_benchmark import TOLERANCE_MS, simulate, standard_scenarios
from evaluation.metrics import match_events
from hits import find_hits
from shot_names import measured_facts, nominal_facts, true_facts
from shots import rally_shots
from simulation.rally import random_rally

LEARNING_STREAM = 5
TEST_FIRST_SEED = 5000


def learning_set(rallies: int, first_seed: int = 0) -> tuple[list[str], np.ndarray]:
    """The names and true facts of every shot but the serve of `rallies` random rallies."""
    names, rows = true_facts([random_rally(np.random.default_rng([LEARNING_STREAM, seed])) for seed in range(first_seed, first_seed + rallies)])
    keep = [k for k, name in enumerate(names) if name != "serve"]
    return [names[k] for k in keep], rows[keep]


def measure_scenario(scenario) -> list[dict[str, Any]]:
    """Every shot of one scenario that was found, with its true and measured facts."""
    sim = simulate(scenario)
    analysis = find_hits(sim.track, sim.times_ms, sim.cameras, sim.onsets)
    shots = rally_shots(analysis, sim.track, sim.times_ms, sim.cameras)
    names, truth = true_facts([sim.clip.rally])
    match = match_events([shot.hit_ms for shot in shots], [c.time * 1000.0 for c in sim.clip.rally.contacts], TOLERANCE_MS)
    rng = np.random.default_rng(scenario.seed)
    return [
        {
            "name": names[t], "truth": truth[t], "nominal": nominal_facts(shots[p]), "draws": measured_facts(shots[p], rng),
            "view": scenario.view, "fps": scenario.fps, "seed": scenario.seed,
        }
        for p, t in match.pairs
        if names[t] != "serve"
    ]


def measure(scenarios, workers: int = 1) -> dict[str, np.ndarray]:
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            found = list(pool.map(measure_scenario, scenarios))
    else:
        found = [measure_scenario(s) for s in scenarios]
    rows = [row for scenario in found for row in scenario]
    return {key: np.array([row[key] for row in rows]) for key in ("name", "truth", "nominal", "draws", "view", "fps", "seed")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure held-out simulated shots for testing shot names.")
    parser.add_argument("--scenarios", type=int, default=120)
    parser.add_argument("--first-seed", type=int, default=TEST_FIRST_SEED)
    parser.add_argument("--stress", action="store_true", help="a world that differs from the method's models")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    scenarios = [replace(s, stress=args.stress) for s in standard_scenarios(args.scenarios, args.first_seed)]
    held_out = measure(scenarios, args.workers)
    np.savez_compressed(args.out, **held_out)
    print(f"{len(held_out['name'])} shots from {len(scenarios)} scenarios:", dict(Counter(held_out["name"].tolist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
