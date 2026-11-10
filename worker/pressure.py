# worker/pressure.py
"""How hard each shot made the other player work, from the receiver's side.

For a shot: how far they had to go from where they stood at the hit to get the racket to the
shuttle, less their reach; the time the flight gave them; the speed that needed; the time that
left them to spare, for a player who starts REACT_MS after the hit, covers the way at TOP_M_S
and leans LEAN_M further at a stretch, and, when they met it in a jump, pushes off for
PUSH_OFF_MS and rises to it; and how high they met it. How fast they rose is not known: anywhere
between a jump that just got there and one as hard as a professional's jump smash. A jump the
player chose costs the same time as one they needed. A shot that came down unmet is measured to
where it landed. A shot into the net, or one followed by neither a reply nor a
landing, has none, and so has one whose measurements contradict each other: the reply's contact
out of the receiver's reach at a stretch from where they stood when they met it, or a way between
the two hits faster than MAX_RUN_M_S. Each number comes with its uncertainty, from draws of where
they stood (body.standing_at), the contact or the landing, their reach (body.arm_reach), how
high they were off the floor, how fast they rose and the time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import ndtri

from geometry.shuttle_physics import GRAVITY
from shot_names import JUMP_M
from shots import MIN_FLIGHT_MS, Estimate, ShotFacts

DRAWS = 2000
HIT_TIME_SIGMA_MS = 12.0  # either end of a flight's time, about a third of a frame at 30 fps
REACT_MS = 250.0  # a player starts to move about a quarter of a second after the other's hit
TOP_M_S = 3.5  # and covers the way no faster than this on average
LEAN_M = 0.5  # at a stretch the shoulders are up to this far ahead of the hips
MAX_RUN_M_S = 5.0  # no one covers the court faster: a player seen to was put in the wrong place
AGREE_SIGMAS = 2.0  # measurements further apart than this many standard deviations contradict each other
# A jump's time on the floor before take-off: about 0.22 s in professionals' jump smashes, under
# 0.2 s in a quick rebound (Rambely and Wan Abas, ISBS 2008), so the shorter.
PUSH_OFF_MS = 200.0
JUMP_SMASH_M = 0.54  # how high professionals' centre of mass rose in those jump smashes
_JUMP_DRAWS = np.random.default_rng(0).random((DRAWS, 2))  # the height off the floor, and how hard they jumped


@dataclass(frozen=True)
class Pressure:
    go_m: Estimate  # how far they had to go, beyond their reach
    time_ms: Estimate  # how long the flight gave them
    speed_m_s: Estimate  # how fast that made them go
    spare_ms: Estimate  # the time left once there; below zero they could not get there in time
    height_m: Estimate | None  # how high they met it; None when it came down unmet


def jump_ms(off_floor: Estimate | None) -> np.ndarray:
    """Draws of the time a jump took from the push-off to `off_floor` (the lower foot's height):
    none below JUMP_M."""
    if off_floor is None:
        return np.zeros(DRAWS)
    height = np.maximum(off_floor.value + off_floor.sigma * ndtri(_JUMP_DRAWS[:, 0]), 0.0)
    least = np.sqrt(2.0 * GRAVITY * height)  # the take-off speed that just gets there
    speed = least + _JUMP_DRAWS[:, 1] * np.maximum(np.sqrt(2.0 * GRAVITY * JUMP_SMASH_M) - least, 0.0)
    rise = (speed - np.sqrt(speed**2 - least**2)) / GRAVITY
    return np.where(height > JUMP_M, PUSH_OFF_MS + rise * 1000.0, 0.0)


def work(
    stood: tuple[np.ndarray, np.ndarray], target, target_covariance, reach: Estimate, time_ms: float, height: Estimate | None = None,
    rng: np.random.Generator | None = None, time_sigma_ms: float = HIT_TIME_SIGMA_MS, off_floor: Estimate | None = None,
) -> Pressure:
    """The work of getting from `stood` (where, and its covariance) to reach `target` in `time_ms`,
    `off_floor` when they met it (None: on the floor)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    point, target = np.asarray(stood[0], float), np.asarray(target, float)
    go = max(float(np.linalg.norm(target - point)) - reach.value, 0.0)
    from_draws = rng.multivariate_normal(point, np.asarray(stood[1], float), DRAWS, check_valid="ignore")
    to_draws = rng.multivariate_normal(target, np.asarray(target_covariance, float), DRAWS, check_valid="ignore")
    gos = np.maximum(np.linalg.norm(to_draws - from_draws, axis=1) - rng.normal(reach.value, reach.sigma, DRAWS), 0.0)
    times = np.maximum(rng.normal(time_ms, time_sigma_ms, DRAWS), MIN_FLIGHT_MS)
    jumps = jump_ms(off_floor)
    spares = times - REACT_MS - np.maximum(gos - LEAN_M, 0.0) / TOP_M_S * 1000.0 - jumps
    return Pressure(
        Estimate(go, float(np.std(gos))), Estimate(float(time_ms), float(time_sigma_ms)),
        Estimate(go / max(time_ms, MIN_FLIGHT_MS) * 1000.0, float(np.std(gos / times)) * 1000.0),
        Estimate(time_ms - REACT_MS - max(go - LEAN_M, 0.0) / TOP_M_S * 1000.0 - float(np.mean(jumps)), float(np.std(spares))), height,
    )


def _apart(a, b, covariance) -> tuple[float, float]:
    """How far `b` is from `a`, and the standard deviation of that along the way between them."""
    way = np.asarray(b, float) - np.asarray(a, float)
    length = float(np.linalg.norm(way))
    unit = way / max(length, 1e-9)
    return length, float(np.sqrt(unit @ np.asarray(covariance, float) @ unit))


def contradicts(stood, there, contact, contact_covariance, reach: Estimate, time_ms: float) -> bool:
    """Whether where the receiver stood at the hit (`stood`) and when they met the shuttle
    (`there`, each where and its covariance) cannot both be right with the contact."""
    reached, sigma = _apart(there[0], contact, np.asarray(there[1]) + np.asarray(contact_covariance))
    if reached - reach.value - LEAN_M > AGREE_SIGMAS * float(np.hypot(sigma, reach.sigma)):
        return True
    moved, sigma = _apart(stood[0], there[0], np.asarray(stood[1]) + np.asarray(there[1]))
    return moved - MAX_RUN_M_S * time_ms / 1000.0 > AGREE_SIGMAS * sigma


def rally_pressure(
    shots: Sequence[ShotFacts], stood: Sequence[dict | None], reaches: dict[str, Estimate | None], off_floor: Sequence[Estimate | None] | None = None,
) -> list[Pressure | None]:
    """Each shot's pressure on the receiver. `stood` is both players at each hit (each half's
    body.standing_at, or None), `reaches` each half's body.arm_reach, and `off_floor` how high
    each shot's hitter was off the floor (body.BodyFacts.off_floor_m, or None)."""
    rng = np.random.default_rng(0)
    out: list[Pressure | None] = []
    for i, shot in enumerate(shots):
        receiver = "far" if shot.hitter == "near" else "near"
        where, reach = (stood[i] or {}).get(receiver), reaches.get(receiver)
        after = shots[i + 1] if i + 1 < len(shots) else None
        if where is None or reach is None:
            out.append(None)
        elif after is not None and after.hitter == receiver and not after.serve:
            covariance, time_ms = np.diag(np.asarray(after.contact_sigma[:2]) ** 2), after.hit_ms - shot.hit_ms
            there = (stood[i + 1] or {}).get(receiver)
            if there is not None and contradicts(where, there, after.contact[:2], covariance, reach, time_ms):
                out.append(None)
            else:
                jumped = None if off_floor is None else off_floor[i + 1]
                out.append(work(where, after.contact[:2], covariance, reach, time_ms, Estimate(after.contact[2], after.contact_sigma[2]), rng, off_floor=jumped))
        elif after is None and shot.landing is not None and shot.landing.p_over_net >= 0.5:
            out.append(work(where, shot.landing.point, shot.landing.covariance, reach, shot.landing.time_ms - shot.hit_ms, None, rng))
        else:
            out.append(None)
    return out
