# worker/evaluation/hit_timing.py
"""How well each way of timing a hit does, on the hard benchmark's rallies (the table in the
shuttle-in-3D plan): for every boundary between two fitted flights near a true hit, the error of
the 3D closest approach, the meet of the two flights in the image (what hits.py uses), the image
quadratics, and the sound less its travel from the true contact.

    python -m evaluation.hit_timing --scenarios 60 --workers 12
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import hits
from evaluation.hits_benchmark import simulate, standard_scenarios

NEAR_MS = 150.0  # a boundary this close to a true hit is that hit's


def _errors(scenario) -> list[tuple[float, float, float, float | None]]:
    sim = simulate(scenario)
    times = np.asarray(sim.times_ms, dtype=float)
    analysis = hits.find_hits(sim.track, sim.times_ms, sim.cameras, sim.onsets)
    truth = np.array([c.time * 1000 for c in sim.clip.rally.contacts])
    sounds = sorted(o.time_ms for o in sim.onsets) if sim.onsets is not None else []
    rows = []
    for a, b in zip(analysis.flights, analysis.flights[1:]):
        if a.fit is None or b.fit is None or times[b.indices[0]] - times[a.indices[-1]] > hits.HARD_GAP_MS:
            continue
        low, high = times[a.indices[-1]], times[b.indices[0]]
        k = int(np.argmin(np.abs(truth - 0.5 * (low + high))))
        if abs(truth[k] - 0.5 * (low + high)) > NEAR_MS:
            continue
        span = np.linspace(low, high, 41)
        closest = span[int(np.argmin(np.linalg.norm(a.fit.positions_at(span) - b.fit.positions_at(np.maximum(span, b.fit.start_ms)), axis=1)))]
        projected = hits._projected_meet(a.fit, b.fit, sim.cameras[b.indices[0]], low, high)
        image = hits._image_meet(sim.track, times, a.indices, b.indices, 2.0)
        travel = float(np.linalg.norm(np.asarray(sim.clip.rally.contacts[k].position) - sim.clip.cameras[b.indices[0]].centre)) / hits.SPEED_OF_SOUND_M_S * 1000
        heard = [s for s in sounds if 0 <= s - truth[k] <= 120]
        rows.append((closest - truth[k], projected - truth[k], None if image is None else image - truth[k], heard[0] - travel - truth[k] if heard else None))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ways of timing a hit on the hard benchmark.")
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    scenarios = standard_scenarios(args.scenarios, args.first_seed)
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [row for result in pool.map(_errors, scenarios) for row in result]
    else:
        rows = [row for scenario in scenarios for row in _errors(scenario)]
    for label, k in (("3D closest approach", 0), ("projected meet (used)", 1), ("image quadratics", 2), ("sound less travel", 3)):
        errors = np.abs([row[k] for row in rows if row[k] is not None])
        print(f"{label:22s} n={len(errors):3d}  median {np.median(errors):5.1f} ms  p90 {np.percentile(errors, 90):5.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
