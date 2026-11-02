# worker/evaluation/pressure_benchmark.py
"""The pressure on a receiver at each hit against the truth on synthetic rallies.

Per scenario, as in the body benchmark: a random rally's near or far player, seen from one of the
three views at 30 or 60 fps, with 3 px of joint noise, 5 % of joints missed and an emulated
learned 3D pose model, the bodies fitted over the whole rally. Each of the opponent's hits that
the player answered gets its pressure (pressure.work), with the contact point as the shuttle's
analysis gives it: the true one off by CONTACT_SIGMA_M on each axis, the hit times off by
HIT_TIME_SIGMA_MS, and said to be. The truth is the same on the true body, contact and times.
Per fact: the median and 90th percentile error, and how often the error is within twice the
stated uncertainty. And whether the receiver was under pressure (rally_reasoning): how many hits
were called so at CALL_CONFIDENCE, how many of those were right, and how the rest lean.

    python -m evaluation.pressure_benchmark --seeds 3 --first-seed 3000 --workers 12 --out pressure.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from body import arm_reach, standing_at
from evaluation.body_benchmark import CONTACT_SIGMA_M, _errors, fitted_player
from evaluation.hits_benchmark import VIEWS
from geometry.skeleton import Body
from pressure import HIT_TIME_SIGMA_MS, work
from rally_reasoning import PRESSED_SPARE_MS, p_under
from shots import CALL_CONFIDENCE, Estimate
from simulation.people import playing

FACTS = ("go_m", "speed_m_s", "spare_ms", "height_m")


def score(job: tuple[str, str, float, int]) -> dict:
    view, side, fps, seed = job
    rally, times, bodies, truth = fitted_player(view, side, fps, seed)
    reach, true_reach = arm_reach(bodies), arm_reach([Body(t, np.zeros(17), 0.0) for t in truth])
    noise = np.random.default_rng([seed, list(VIEWS).index(view), side == "far", int(fps), 1])
    contacts = sorted(rally.contacts, key=lambda c: c.time)
    out = {"view": view, "side": side, "fps": fps, "seed": seed, "reach": [reach.value - true_reach.value, reach.sigma], "facts": {name: [] for name in FACTS}, "pressed": []}
    for before, after in zip(contacts, contacts[1:]):
        if after.hitter != side or before.hitter == side or not times[0] < before.time * 1000.0 < times[-1]:
            continue
        stood = standing_at(bodies, times, before.time * 1000.0)
        if stood is None:
            continue
        contact = np.asarray(after.position) + noise.normal(0, CONTACT_SIGMA_M, 3)
        flight_ms = (after.time - before.time) * 1000.0
        said = work(
            stood, contact[:2], np.eye(2) * CONTACT_SIGMA_M**2, reach, flight_ms + noise.normal(0, HIT_TIME_SIGMA_MS), Estimate(float(contact[2]), CONTACT_SIGMA_M),
            rng=noise, time_sigma_ms=HIT_TIME_SIGMA_MS * 2**0.5,
        )
        hips = playing(rally, side, before.time)[[11, 12], :2].mean(axis=0)
        true = work((hips, np.zeros((2, 2))), after.position[:2], np.zeros((2, 2)), true_reach, flight_ms, Estimate(float(after.position[2]), 0.0), rng=noise, time_sigma_ms=0.0)
        for name in FACTS:
            got, want = getattr(said, name), getattr(true, name)
            out["facts"][name].append([got.value - want.value, got.sigma])
        out["pressed"].append([p_under(said.spare_ms, PRESSED_SPARE_MS), bool(true.spare_ms.value < PRESSED_SPARE_MS)])
    return out


def _calls(pairs: list) -> dict:
    """Under pressure or not, called at CALL_CONFIDENCE, against the truth."""
    if not pairs:
        return {"n": 0}
    p, truth = np.array([pair[0] for pair in pairs]), np.array([pair[1] for pair in pairs], bool)
    called, right = (p >= CALL_CONFIDENCE) | (p <= 1.0 - CALL_CONFIDENCE), (p >= 0.5) == truth
    return {"n": len(p), "underPressure": int(truth.sum()), "called": int(called.sum()), "right": int((called & right).sum()), "tooCloseLeansRight": int((~called & right).sum())}


def summarise(results: list[dict]) -> dict:
    groups = {"all": results} | {view: [r for r in results if r["view"] == view] for view in VIEWS}
    return {
        key: {
            **{name: _errors([pair for r in group for pair in r["facts"][name]]) for name in FACTS}, "armReachM": _errors([r["reach"] for r in group]),
            "pressed": _calls([pair for r in group for pair in r["pressed"]]),
        }
        for key, group in groups.items() if group
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the pressure on the receiver at each hit on synthetic rallies.")
    parser.add_argument("--seeds", type=int, default=3, help="rallies per view, half and frame rate")
    parser.add_argument("--first-seed", type=int, default=3000)
    parser.add_argument("--fps", type=float, nargs="+", default=[30.0, 60.0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    jobs = [(view, side, fps, args.first_seed + k) for view in VIEWS for side in ("near", "far") for fps in args.fps for k in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(score, jobs))
    else:
        results = [score(job) for job in jobs]
    summary = {f"{fps:g} fps": summarise([r for r in results if r["fps"] == fps]) for fps in args.fps}
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")
    print(json.dumps({key: value["all"] for key, value in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
