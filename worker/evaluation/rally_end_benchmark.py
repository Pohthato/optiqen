# worker/evaluation/rally_end_benchmark.py
"""How each rally ended and who won, called against the truth on the hard synthetic rallies.

The hits benchmark's scenarios, each rally ended by turn in one of four ways: in court, within
40 cm of a line (in or out), 0.3-1.2 m out, or into the net. Every other four go on past the end:
the shuttle lies where it came down and, on the loser's side, is sent back over the net
(simulation.rally.afterwards). Per scenario the hits are found, the shots taken (shots.rally_shots,
which ends the rally where the shuttle came down) and the end called (shots.rally_end). Scored:
whether the rally's last shot is the true last hit, the end and the winner against the truth, how
sure each call said it was, and how close the truth was to a line or under the tape.

    python -m evaluation.rally_end_benchmark --scenarios 120 --first-seed 9000 --workers 12 [--stress] --out ends.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from evaluation.hits_benchmark import TOLERANCE_MS, simulate, standard_scenarios
from evaluation.shots_benchmark import _lower_bound
from geometry.court_model import COURT_LENGTH_M, SINGLES_WIDTH_M
from geometry.shuttle_physics import simulate as fly
from hits import find_hits
from shots import CALL_CONFIDENCE, net_height_m, rally_end, rally_shots

ENDINGS = {"court": (False, None), "line": (True, None), "out": (False, "out"), "net": (False, "net")}
ENDS = ("in", "out", "net")
TOO_CLOSE = "too close to call"


def scenarios(count: int, first_seed: int, stress: bool = False) -> list:
    """The standard scenarios with each ending in turn, and every other four of them going on
    past the end."""
    out = []
    for index, scenario in enumerate(standard_scenarios(count, first_seed)):
        near_line, end = list(ENDINGS.values())[index % len(ENDINGS)]
        out.append(replace(scenario, final_near_line=near_line, end=end, afterwards=(index // len(ENDINGS)) % 2 == 1, stress=stress))
    return out


def margin_m(rally) -> float:
    """How far the truth was from the other end: for a landing, from the nearest line of the
    half it was aimed at (positive inside, negative outside); for the net, how far under the tape
    the shuttle met it."""
    if rally.end == "net":
        meets = rally.flights[-2]
        at = fly(meets.p0, meets.v0, [meets.end_time - meets.start_time], rally.terminal_velocity)[0]
        return float(net_height_m(at[0]) - at[2])
    x, y = rally.landing
    depth = COURT_LENGTH_M - y if rally.contacts[-1].hitter == "near" else y
    return float(min(x, SINGLES_WIDTH_M - x, depth))


def score_scenario(scenario) -> dict[str, Any]:
    sim = simulate(scenario)
    rally = sim.clip.rally
    analysis = find_hits(sim.track, sim.times_ms, sim.cameras, sim.onsets)
    shots = rally_shots(analysis, sim.track, sim.times_ms, sim.cameras)
    end = rally_end(shots)
    last = rally.contacts[-1]
    ending = next(name for name, (near_line, kind) in ENDINGS.items() if (near_line, kind) == (scenario.final_near_line, scenario.end))
    return {
        **asdict(scenario), "ending": ending, "trueEnd": rally.end, "trueWinner": rally.winner, "marginM": round(margin_m(rally), 3),
        "shots": len(shots), "lastShotRight": bool(shots) and abs(shots[-1].hit_ms - last.time * 1000.0) <= TOLERANCE_MS,
        "end": None if end is None else {"call": end.call, "p": {k: round(v, 4) for k, v in end.p.items()}, "winner": end.winner, "pHitterWon": round(end.p_hitter_won, 4)},
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per true end: how often the end was not seen, called right or wrong, or too close to call;
    per call given: how often it was right, and the 95 % lower bound of that; the same for the
    winner; and the margins of the ends left too close to call."""
    seen = [r for r in results if r["end"] is not None]
    by_truth = {}
    for kind in ENDS:
        rows = [r for r in results if r["trueEnd"] == kind]
        calls = [r["end"]["call"] for r in rows if r["end"] is not None]
        by_truth[kind] = {
            "rallies": len(rows), "notSeen": sum(r["end"] is None for r in rows), "right": calls.count(kind),
            "tooClose": calls.count(TOO_CLOSE), "wrong": sum(call not in (kind, TOO_CLOSE) for call in calls),
        }
    given = {}
    for kind in ENDS:
        called = [r for r in seen if r["end"]["call"] == kind]
        right = sum(r["trueEnd"] == kind for r in called)
        given[kind] = {"given": len(called), "right": right, "rightLowerBound95": _lower_bound(right, len(called))}
    winners = [r for r in seen if r["end"]["winner"] != TOO_CLOSE]
    winners_right = sum(r["end"]["winner"] == r["trueWinner"] for r in winners)
    wrong = [r for r in seen if r["end"]["call"] not in (r["trueEnd"], TOO_CLOSE) or r["end"]["winner"] not in (r["trueWinner"], TOO_CLOSE)]
    return {
        "rallies": len(results), "lastShotRight": sum(r["lastShotRight"] for r in results),
        "byTrueEnd": by_truth, "byCall": given,
        "winner": {"given": len(winners), "right": winners_right, "rightLowerBound95": _lower_bound(winners_right, len(winners)), "tooClose": len(seen) - len(winners)},
        "calledAtLeast": CALL_CONFIDENCE,
        "tooCloseMarginsM": {kind: sorted(r["marginM"] for r in seen if r["trueEnd"] == kind and r["end"]["call"] == TOO_CLOSE) for kind in ENDS},
        "wrong": [{k: r[k] for k in ("seed", "view", "fps", "ending", "afterwards", "trueEnd", "trueWinner", "marginM", "end")} for r in wrong],
    }


def run(scenario_list, workers: int = 1) -> dict[str, Any]:
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(score_scenario, scenario_list))
    else:
        results = [score_scenario(s) for s in scenario_list]
    by = {"all": summarise(results)}
    for name in ENDINGS:
        by[f"ending={name}"] = summarise([r for r in results if r["ending"] == name])
    for afterwards in (False, True):
        by[f"afterwards={afterwards}"] = summarise([r for r in results if r["afterwards"] == afterwards])
    for view in sorted({r["view"] for r in results}):
        by[f"view={view}"] = summarise([r for r in results if r["view"] == view])
    return {"scenarios": results, "summary": by}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark how rallies end, and who won, on hard synthetic rallies.")
    parser.add_argument("--scenarios", type=int, default=120)
    parser.add_argument("--first-seed", type=int, default=9000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stress", action="store_true", help="a world that differs from the method's models (see hits_benchmark.simulate)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = run(scenarios(args.scenarios, args.first_seed, args.stress), workers=args.workers)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"]["all"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
