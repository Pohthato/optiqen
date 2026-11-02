# worker/rally_reasoning.py
"""Why the rally was won, by rules written down rather than learnt, each with its chance.

How it was won, from its end (shots.rally_end) and the pressure at each hit (pressure.py):
- a shot that came down in was a "winner" when the receiver had no time to spare to get to it,
  and was "let in" when they had;
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

from pressure import Pressure
from shots import CALL_CONFIDENCE, Estimate, RallyEnd, ShotFacts

PRESSED_SPARE_MS = 250.0  # under pressure: less than a quarter of a second to spare


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


def rally_reasons(shots: Sequence[ShotFacts], end: RallyEnd | None, pressures: Sequence[Pressure | None]) -> Reasons | None:
    """How the rally in `shots` was won and where it turned, or None when its end was not seen.
    `pressures` is each shot's pressure on its receiver (pressure.rally_pressure)."""
    if end is None:
        return None
    spare = [None if pressure is None else pressure.spare_ms for pressure in pressures]
    pressed = [p_under(s, PRESSED_SPARE_MS) for s in spare]
    last = len(shots) - 1
    unreached = p_under(spare[last], 0.0)
    forced = 0.0 if shots[last].serve else pressed[last - 1] if last > 0 else 0.5
    error = end.p["out"] + end.p["net"]
    p = {"winner": end.p["in"] * unreached, "let in": end.p["in"] * (1.0 - unreached), "forced error": error * forced, "unforced error": error * (1.0 - forced)}

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
