# worker/rally_reasoning.py
"""Why the rally was won, by rules written down, one of them learnt, each with its chance.

How it was won, from its end (shots.rally_end) and the pressure at each hit (pressure.py):
- a shot that came down in was a "winner" unless the receiver let it in: could get to it, and
  left it. Time to get there is not enough to say so: most professional winners leave the
  receiver time to get there but not to play the shot. How often a receiver who could get there
  leaves a shot is learnt from professionals (ShuttleSet's training matches,
  evaluation.value_benchmark), by how far inside the lines it came down and whether it was a
  clear or a lift, a push or another shot (`LET_IN`), over the landing's uncertainty;
- an error, out or into the net, was "forced" when its hitter was under pressure at that hit
  (less than PRESSED_SPARE_MS to spare after the shot before it), and "unforced" when not. A
  serve is never under pressure.
Where it turned: at the first of the winner's shots from which every one of theirs left the
loser under pressure. It did not turn when the winner's last shot did not.

The chances take the end's and each hit's pressure as they are, and as independent; a hit whose
pressure is not known was under it at even odds. A call is made at CALL_CONFIDENCE, and none
when it is unsure which half the last shot came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Sequence

import numpy as np

from geometry.court_model import NET_Y_M
from pressure import Pressure
from shots import CALL_CONFIDENCE, Estimate, RallyEnd, ShotFacts, landing_area

PRESSED_SPARE_MS = 250.0  # under pressure: less than a quarter of a second to spare
# The log odds that a professional left a shot they could get to: at none of the below, per
# natural log of the metres it came down inside the lines, for a clear or a lift, for a push.
LET_IN = (-4.79, -0.48, 3.79, 2.60)
ON_THE_LINE_M = 0.01  # a landing on a line counts as this far inside it
_UNIT_DRAWS = np.random.default_rng(0).standard_normal((2000, 2))


@dataclass(frozen=True)
class Reasons:
    p: dict[str, float]  # the chance it was a "winner", "let in", a "forced error" or an "unforced error"
    call: str  # the one of those CALL_CONFIDENCE sure, or "too close to call"
    p_turn: list[float]  # the chance it turned at each shot
    p_no_turn: float  # and that it did not turn
    turn: int | str  # the index of the shot it turned at, "no turn", or "too close to call"


def p_under(spare: Estimate | None, limit_ms: float) -> float:
    """The chance the time to spare was under `limit_ms`; even odds when it is not known."""
    if spare is None:
        return 0.5
    return 0.5 * (1.0 + erf((limit_ms - spare.value) / (max(spare.sigma, 1e-6) * sqrt(2.0))))


def inside_by(points, area) -> np.ndarray:
    """How far inside the lines of `area` (shots.landing_area) each point is, negative outside.
    The net is not a line."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    x_low, x_high, y_low, y_high = area
    ways = [points[:, 0] - x_low, x_high - points[:, 0]]
    ways += [] if y_low == NET_Y_M else [points[:, 1] - y_low]
    ways += [] if y_high == NET_Y_M else [y_high - points[:, 1]]
    return np.min(ways, axis=0)


def left_by_professionals(inside_m, high, push, weights=LET_IN) -> np.ndarray:
    """The chance a professional left a shot they could get to that came down `inside_m` inside
    the lines, `high` and `push` being the chances it was a clear or a lift, and a push."""
    at, per_log_m, if_high, if_push = weights
    base = at + per_log_m * np.log(np.maximum(np.asarray(inside_m, dtype=float), ON_THE_LINE_M))
    return high * _chance(base + if_high) + push * _chance(base + if_push) + (1.0 - high - push) * _chance(base)


def _chance(log_odds):
    return 1.0 / (1.0 + np.exp(-log_odds))


def p_let_in(shot: ShotFacts, name: dict[str, float], court: str, unreached: float) -> float:
    """The chance the last shot, given that it came down in, was let in: that the receiver could
    get to it, times that a professional would have left it, over its landing inside the lines."""
    landing = shot.landing
    root = np.linalg.cholesky(np.asarray(landing.covariance, dtype=float) + np.eye(2) * 1e-12)
    inside = inside_by(np.asarray(landing.point, dtype=float) + _UNIT_DRAWS @ root.T, landing_area(court, shot.hitter, shot.contact[0] if shot.serve else None))
    inside = inside[inside >= 0.0] if np.any(inside >= 0.0) else np.zeros(1)
    left = left_by_professionals(inside, name.get("clear", 0.0) + name.get("lift", 0.0), name.get("push", 0.0))
    return (1.0 - unreached) * float(np.mean(left))


def rally_reasons(
    shots: Sequence[ShotFacts], end: RallyEnd | None, pressures: Sequence[Pressure | None], names: Sequence[dict[str, float]], court: str = "singles"
) -> Reasons | None:
    """How the rally in `shots` was won and where it turned, or None when its end was not seen.
    `pressures` is each shot's pressure on its receiver (pressure.rally_pressure), and `names`
    each shot's chance of each name (shot_names.ShotName.p)."""
    if end is None:
        return None
    spare = [None if pressure is None else pressure.spare_ms for pressure in pressures]
    pressed = [p_under(s, PRESSED_SPARE_MS) for s in spare]
    last = len(shots) - 1
    let_in = p_let_in(shots[last], names[last], court, p_under(spare[last], 0.0)) if shots[last].landing is not None else 0.0
    forced = 0.0 if shots[last].serve else pressed[last - 1] if last > 0 else 0.5
    error = end.p["out"] + end.p["net"]
    p = {"winner": end.p["in"] * (1.0 - let_in), "let in": end.p["in"] * let_in, "forced error": error * forced, "unforced error": error * (1.0 - forced)}

    p_turn, p_no_turn = [0.0] * len(shots), 0.0
    hitter = shots[last].hitter
    for winner, p_won in ((hitter, end.p_hitter_won), ("far" if hitter == "near" else "near", 1.0 - end.p_hitter_won)):
        theirs = [i for i, shot in enumerate(shots) if shot.hitter == winner]
        from_here = 1.0  # the chance every one of their shots from this one on left the loser under pressure
        for k in range(len(theirs) - 1, -1, -1):
            from_here *= pressed[theirs[k]]
            p_turn[theirs[k]] += p_won * from_here * (1.0 - pressed[theirs[k - 1]] if k else 1.0)
        p_no_turn += p_won * (1.0 - pressed[theirs[-1]] if theirs else 1.0)

    sure = end.p_half >= CALL_CONFIDENCE
    how = max(p, key=p.get)
    at = max(range(len(shots)), key=p_turn.__getitem__)
    turn, p_most = ("no turn", p_no_turn) if p_no_turn >= p_turn[at] else (at, p_turn[at])
    return Reasons(
        p, how if sure and p[how] >= CALL_CONFIDENCE else "too close to call",
        p_turn, p_no_turn, turn if sure and p_most >= CALL_CONFIDENCE else "too close to call",
    )
