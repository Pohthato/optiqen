# worker/evaluation/shots_benchmark.py
"""Shot facts against the truth on the hard synthetic rallies.

The hits benchmark's scenarios (every view at 30 and 60 fps, with sound and without, jitter,
dropout, false detections, neighbouring courts), with each rally's last shot aimed within 40 cm
of a boundary line, in or out. Per scenario the hits are found (hits.find_hits), the flights
refitted together and each shot's facts taken (shots.rally_shots). Shots matched to a true hit
(within ±33 ms) are compared on contact point, launch speed, apex and net clearance; the rally's
last shot on its landing and call, and on whether the stated uncertainty covers the truth.

    python -m evaluation.shots_benchmark --scenarios 60 --workers 12 --out shots.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from evaluation.hits_benchmark import TOLERANCE_MS, simulate, standard_scenarios
from geometry.court_model import NET_Y_M
from geometry.shuttle_physics import simulate as fly
from hits import find_hits
from shots import landing_area, net_height_m, rally_shots

COVERAGE = 0.95


def _truth(rally) -> list[dict[str, Any]]:
    """Per true hit: contact, launch speed, apex and net crossing of the flight that leaves it."""
    out = []
    for contact, flight in zip(rally.contacts, rally.flights):
        times = np.arange(0.0, flight.end_time - flight.start_time + 1e-9, 0.002)
        path = fly(flight.p0, flight.v0, times, rally.terminal_velocity)
        side = np.sign(path[:, 1] - NET_Y_M)
        cross = np.nonzero(side[1:] != side[:-1])[0]
        clearance = None
        if len(cross):
            i = int(cross[0])
            f = (NET_Y_M - path[i, 1]) / (path[i + 1, 1] - path[i, 1])
            at = path[i] + f * (path[i + 1] - path[i])
            clearance = float(at[2] - net_height_m(at[0]))
        out.append({
            "timeMs": contact.time * 1000.0, "kind": contact.kind, "hitter": contact.hitter, "contact": np.asarray(contact.position),
            "speed": float(np.linalg.norm(flight.v0)), "apex": float(path[:, 2].max()), "clearance": clearance,
        })
    return out


def score_scenario(scenario, session_error: float | None = None, court: str = "singles") -> dict[str, Any]:
    """One scenario scored. With `session_error`, the shuttle's drag is given as a session of many
    rallies would measure it (emulated: the true terminal velocity off by that relative error,
    drawn per scenario, with that standard deviation). Without it each rally measures its own.
    The rally's landing is judged on the call the method itself made on its last shot, whatever
    that shot was; a rally where it made none counts as such."""
    sim = simulate(scenario)
    analysis = find_hits(sim.track, sim.times_ms, sim.cameras, sim.onsets)
    drag = None
    if session_error is not None:
        true_drag = sim.clip.rally.terminal_velocity
        drag = (true_drag * (1 + np.random.default_rng(scenario.seed + 31).normal(0.0, session_error)), true_drag * session_error)
    shots = rally_shots(analysis, sim.track, sim.times_ms, sim.cameras, court=court, terminal_velocity=drag)
    truth = _truth(sim.clip.rally)
    pairs = sorted((abs(s.hit_ms - t["timeMs"]), a, b) for a, s in enumerate(shots) for b, t in enumerate(truth) if abs(s.hit_ms - t["timeMs"]) <= TOLERANCE_MS)
    matched, used_s, used_t = [], set(), set()
    for _, a, b in pairs:
        if a not in used_s and b not in used_t:
            used_s.add(a), used_t.add(b)
            matched.append((shots[a], truth[b], b))
    rows = []
    for shot, true, _ in matched:
        rows.append({
            "kind": true["kind"],
            "contactErrorM": float(np.linalg.norm(np.asarray(shot.contact) - true["contact"])),
            "hitterRight": shot.hitter == true["hitter"],
            "speedErrorMS": abs(shot.speed_m_s.value - true["speed"]),
            "apexErrorM": abs(shot.apex_m.value - true["apex"]),
            "clearanceErrorM": abs(shot.net.clearance_m.value - true["clearance"]) if shot.net is not None and true["clearance"] is not None else None,
        })
    landing = None
    last = len(truth) - 1
    final = shots[-1] if shots else None
    if final is not None and final.landing is not None:
        rally = sim.clip.rally
        point = np.asarray(rally.landing)
        x_low, x_high, y_low, y_high = landing_area(court, rally.contacts[-1].hitter)
        truly_in = bool(x_low <= point[0] <= x_high and y_low <= point[1] <= y_high)
        miss = np.asarray(final.landing.point) - point
        covariance = np.asarray(final.landing.covariance)
        inside = float(miss @ np.linalg.pinv(covariance) @ miss) <= chi2.ppf(COVERAGE, 2)
        landing = {
            "errorM": float(np.linalg.norm(miss)), "call": final.landing.call, "trulyIn": truly_in,
            "covered": bool(inside), "distanceToLineM": float(min(point[0] - x_low, x_high - point[0], point[1] - y_low, y_high - point[1])),
            "isTrueLastShot": any(shot is final and b == last for shot, _, b in matched),
        }
    return {**asdict(scenario), "truthShots": len(truth), "shots": len(shots), "matched": rows, "landing": landing}


def _lower_bound(right: int, total: int, z: float = 1.96) -> float | None:
    """The 95 % Wilson lower bound of a proportion: what `right of total` supports."""
    if total == 0:
        return None
    p = right / total
    centre = p + z * z / (2 * total)
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return round(float((centre - spread) / (1 + z * z / total)), 3)


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for r in results for row in r["matched"]]
    landings = [r["landing"] for r in results if r["landing"] is not None]
    called = [l for l in landings if l["call"] != "too close to call"]
    wrong = [l for l in called if (l["call"] == "in") != l["trulyIn"]]
    own = [l for l in landings if l["isTrueLastShot"]]

    def stats(values: list[float]) -> dict[str, float] | None:
        values = [v for v in values if v is not None]
        return {"median": round(float(np.median(values)), 3), "p90": round(float(np.percentile(values, 90)), 3), "n": len(values)} if values else None

    return {
        "shotsMatched": f"{len(rows)} of {sum(r['truthShots'] for r in results)}",
        "contactErrorM": stats([r["contactErrorM"] for r in rows]),
        "hitterRight": round(float(np.mean([r["hitterRight"] for r in rows])), 3) if rows else None,
        "speedErrorMS": stats([r["speedErrorMS"] for r in rows]),
        "apexErrorM": stats([r["apexErrorM"] for r in rows]),
        "netClearanceErrorM": stats([r["clearanceErrorM"] for r in rows]),
        "landingErrorM": stats([l["errorM"] for l in own]),
        "landingCoverage": round(float(np.mean([l["covered"] for l in own])), 3) if own else None,
        "landings": len(landings),
        "rallyEndsWithoutACall": f"{len(results) - len(landings)} of {len(results)}",
        "lastShotNotTheTrueOne": len(landings) - len(own),
        "callsCorrect": f"{len(called) - len(wrong)} of {len(called)}",
        "callsCorrectLowerBound95": _lower_bound(len(called) - len(wrong), len(called)),
        "tooCloseToCall": f"{len(landings) - len(called)} of {len(landings)}",
        "wrongCalls": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in l.items()} for l in wrong],
        "tooCloseDistancesM": sorted(round(l["distanceToLineM"], 3) for l in landings if l["call"] == "too close to call"),
    }


def run(scenarios, workers: int = 1, session_error: float | None = None, court: str = "singles") -> dict[str, Any]:
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(score_scenario, scenarios, [session_error] * len(scenarios), [court] * len(scenarios)))
    else:
        results = [score_scenario(s, session_error, court) for s in scenarios]
    by = {"all": summarise(results)}
    for view in sorted({r["view"] for r in results}):
        by[f"view={view}"] = summarise([r for r in results if r["view"] == view])
    for sound in (True, False):
        by[f"sound={sound}"] = summarise([r for r in results if r["sound"] == sound])
    return {"scenarios": results, "summary": by}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark shot facts on hard synthetic rallies.")
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stress", action="store_true", help="a world that differs from the method's models (see hits_benchmark.simulate)")
    parser.add_argument("--session-drag", type=float, metavar="ERROR", help="emulate a drag measured over a session: the truth to this relative error (0.01)")
    parser.add_argument("--court", choices=("singles", "doubles"), default="singles", help="the boundaries calls are made and judged against")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    scenarios = [replace(s, final_near_line=True, stress=args.stress) for s in standard_scenarios(args.scenarios, args.first_seed)]
    report = run(scenarios, workers=args.workers, session_error=args.session_drag, court=args.court)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
