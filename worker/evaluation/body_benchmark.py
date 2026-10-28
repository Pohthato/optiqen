# worker/evaluation/body_benchmark.py
"""Body facts at each hit against the truth on synthetic rallies.

Per scenario, as in the skeleton benchmark: a random rally's near or far player (stepping,
lunging, reaching, jumping, split-stepping), seen from one of the three views at 30 or 60 fps
through a steadily held phone, with 3 px of joint noise, 5 % of joints missed and an emulated
learned 3D pose model. The bodies are fitted over the whole rally. Each of the player's hits
gets its body facts (body.hit_body), with the contact point as the shuttle's analysis gives it:
the true one off by CONTACT_SIGMA_M on each axis, and said to be. Before each hit the player's
readiness at the opponent's hit is taken (body.readiness).

The truth is the same facts on the true body at the exact moment of contact, and the
readiness from the true bodies 240 times a second. Per fact: the median and 90th percentile
error, and how often the error is within twice the stated uncertainty. Per split step: found
when there was one, found when there was none, and the error in its timing. Per hit, the jump
and backhand calls (shot_names), each against the same call on the true body: how often each
call was given and right. The simulated players are right-handed, and nearly every contact is
on the racket side of the hips.

    python -m evaluation.body_benchmark --seeds 3 --first-seed 100 --workers 12 --out body.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from body import hit_body, readiness, standing_reach
from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from geometry.skeleton import JOINTS, Body, fit_bodies
from shot_names import TOO_CLOSE, backhand, jump, racket_arms
from simulation.camera_path import stable_handheld
from simulation.people import model_pose, playing
from simulation.rally import random_rally

CONTACT_SIGMA_M = 0.05
DETAILS = {"jump": ("jump", "no jump"), "backhand": ("backhand", "forehand")}
FACTS = ("above_reach_m", "ahead_m", "aside_m", "elbow_deg", "side_on_deg", "hips_height_m", "front_knee_deg", "off_floor_m", "weight_ahead_m")
TRUE_RATE = 240.0


def _exact(rally, side: str, time_s: float) -> Body:
    return Body(playing(rally, side, time_s), np.zeros(17), 0.0)


def score(job: tuple[str, str, float, int]) -> dict:
    view, side, fps, seed = job
    rng = np.random.default_rng(seed)
    rally = random_rally(rng)
    position, target = VIEWS[view]
    base = Camera.look_at(tuple(np.asarray(position) + rng.uniform(-0.5, 0.5, 3) * (1, 1, 0.4)), target, FOCAL_PX, SIZE)
    count = int((rally.end_time + 0.5) * fps)
    cameras = stable_handheld(base, count, fps, seed=seed)
    idx = list(JOINTS)
    times = np.arange(count) * 1000.0 / fps
    truth = [playing(rally, side, t / 1000.0) for t in times]
    frames = [(c.project(t) + rng.normal(0, 3.0, (17, 2)), np.where(rng.random(17) < 0.05, 0.0, 0.9)) for c, t in zip(cameras, truth)]
    model_rng = np.random.default_rng(seed + 100)
    priors = [model_pose(c, t[idx], model_rng) for c, t in zip(cameras, truth)]
    bodies = fit_bodies(frames, cameras, times, facing=np.pi / 2 if side == "near" else -np.pi / 2, priors=priors)
    reach = standing_reach(bodies)
    true_reach = standing_reach([Body(t, np.zeros(17), 0.0) for t in truth])
    contacts = sorted(rally.contacts, key=lambda c: c.time)
    contact_rng = np.random.default_rng([seed, list(VIEWS).index(view), side == "far", int(fps)])
    out = {"view": view, "side": side, "fps": fps, "seed": seed, "facts": {name: [] for name in FACTS}, "reach": [reach.value - true_reach.value, reach.sigma],
           "fromBase": [], "splits": [], "details": []}
    hits = []
    for k, contact in enumerate(contacts):
        if contact.hitter != side or not times[0] < contact.time * 1000.0 < times[-1]:
            continue
        noisy = np.asarray(contact.position) + contact_rng.normal(0, CONTACT_SIGMA_M, 3)
        fitted = hit_body(bodies, times, contact.time * 1000.0, side, noisy, (CONTACT_SIGMA_M,) * 3, reach)
        true = hit_body([_exact(rally, side, contact.time)], [contact.time * 1000.0], contact.time * 1000.0, side, contact.position, (0.0, 0.0, 0.0), true_reach)
        if fitted is None or true is None:
            continue
        for name in FACTS:
            got, want = getattr(fitted, name), getattr(true, name)
            out["facts"][name].append([got.value - want.value, got.sigma])
        hits.append((fitted, true))
        if k == 0 or contacts[k - 1].hitter == side:
            continue
        opponent = contacts[k - 1].time
        ready = readiness(bodies, times, opponent * 1000.0, side)
        fine = np.arange(opponent - 0.6, opponent + 0.6, 1.0 / TRUE_RATE)
        true_ready = readiness([_exact(rally, side, t) for t in fine], fine * 1000.0, opponent * 1000.0, side)
        if ready is None or true_ready is None:
            continue
        out["fromBase"].append([ready.from_base_m.value - true_ready.from_base_m.value, ready.from_base_m.sigma])
        timing = None
        if ready.split_lowest_ms is not None and true_ready.split_lowest_ms is not None:
            timing = [ready.split_lowest_ms.value - true_ready.split_lowest_ms.value, ready.split_lowest_ms.sigma]
        out["splits"].append({"true": true_ready.split_lowest_ms is not None, "found": ready.split_lowest_ms is not None, "timing": timing,
                              "drop": [ready.split_drop_m.value - true_ready.split_drop_m.value, ready.split_drop_m.sigma]})
    arm, true_arm = (racket_arms([hit[k] for hit in hits])[side] for k in (0, 1))
    for fitted, true in hits:
        said = {"jump": jump(fitted), "backhand": backhand(fitted, arm)}
        truth = {"jump": jump(true), "backhand": backhand(true, true_arm)}
        out["details"].append({key: [truth[key].call, None if said[key] is None else said[key].call, None if said[key] is None else said[key].p] for key in DETAILS})
    out["racketArm"] = None if arm is None else arm.call
    return out


def _calls(details: list[dict], key: str) -> dict:
    """How often each call was given and right, and how sure it was."""
    out = {"hits": len(details), "true": sum(d[key][0] == DETAILS[key][0] for d in details)}
    for call in (*DETAILS[key], TOO_CLOSE, None):
        given = [d[key] for d in details if d[key][1] == call]
        out[str(call)] = {"given": len(given), "right": sum(g[0] == call for g in given)} if call in DETAILS[key] else len(given)
    p = np.array([d[key][2] for d in details if d[key][2] is not None])
    out["meanP"] = round(float(p.mean()), 3) if len(p) else None
    return out


def _errors(pairs: list) -> dict:
    if not pairs:
        return {"n": 0}
    error, sigma = np.abs(np.array(pairs)).T
    return {"n": len(error), "median": round(float(np.median(error)), 3), "p90": round(float(np.percentile(error, 90)), 3), "within2Sigma": round(float(np.mean(error <= 2 * sigma)), 3)}


def summarise(results: list[dict]) -> dict:
    out: dict = {}
    groups = {"all": results} | {f"{v}, {s}": [r for r in results if (r["view"], r["side"]) == (v, s)] for v in VIEWS for s in ("near", "far")}
    for key, group in groups.items():
        if not group:
            continue
        splits = [split for r in group for split in r["splits"]]
        real = [s for s in splits if s["true"]]
        none = [s for s in splits if not s["true"]]
        details = [d for r in group for d in r["details"]]
        out[key] = {
            **{name: _errors([pair for r in group for pair in r["facts"][name]]) for name in FACTS},
            **{name: _calls(details, name) for name in DETAILS},
            "racketArmRight": f"{sum(r['racketArm'] == 'right' for r in group)} of {len(group)}",
            "standingReachM": _errors([r["reach"] for r in group]),
            "fromBaseM": _errors([pair for r in group for pair in r["fromBase"]]),
            "splitStep": {
                "found": f"{sum(s['found'] for s in real)} of {len(real)}", "foundWhereNone": f"{sum(s['found'] for s in none)} of {len(none)}",
                "timingMs": _errors([s["timing"] for s in real if s["timing"] is not None]), "dropM": _errors([s["drop"] for s in real]),
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark body facts at each hit on synthetic rallies.")
    parser.add_argument("--seeds", type=int, default=3, help="rallies per view, half and frame rate")
    parser.add_argument("--first-seed", type=int, default=100)
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
