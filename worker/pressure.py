# worker/pressure.py
"""How hard each shot made the other player work, from the receiver's side.

For a shot: how far they had to go from where they stood at the hit to get the racket to the
shuttle, less their reach; the time the flight gave them; the speed that needed; the time that
left them to spare, for a player who starts REACT_MS after the hit and covers the way at TOP_M_S;
and how high they met it. A shot that came down unmet is measured to where it landed. A shot
into the net, or one followed by neither a reply nor a landing, has none. Each number comes with
its uncertainty, from draws of where they stood (body.standing_at), the contact or the landing,
their reach (body.arm_reach) and the time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shots import MIN_FLIGHT_MS, Estimate, ShotFacts

DRAWS = 2000
HIT_TIME_SIGMA_MS = 12.0  # either end of a flight's time, about a third of a frame at 30 fps
REACT_MS = 250.0  # a player starts to move about a quarter of a second after the other's hit
TOP_M_S = 3.5  # and covers the way no faster than this on average


@dataclass(frozen=True)
class Pressure:
    go_m: Estimate  # how far they had to go, beyond their reach
    time_ms: Estimate  # how long the flight gave them
    speed_m_s: Estimate  # how fast that made them go
    spare_ms: Estimate  # the time left once there; below zero they could not get there in time
    height_m: Estimate | None  # how high they met it; None when it came down unmet


def work(
    stood: tuple[np.ndarray, np.ndarray], target, target_covariance, reach: Estimate, time_ms: float, height: Estimate | None = None,
    rng: np.random.Generator | None = None, time_sigma_ms: float = HIT_TIME_SIGMA_MS,
) -> Pressure:
    """The work of getting from `stood` (where, and its covariance) to reach `target` in `time_ms`."""
    rng = rng if rng is not None else np.random.default_rng(0)
    point, target = np.asarray(stood[0], float), np.asarray(target, float)
    go = max(float(np.linalg.norm(target - point)) - reach.value, 0.0)
    from_draws = rng.multivariate_normal(point, np.asarray(stood[1], float), DRAWS, check_valid="ignore")
    to_draws = rng.multivariate_normal(target, np.asarray(target_covariance, float), DRAWS, check_valid="ignore")
    gos = np.maximum(np.linalg.norm(to_draws - from_draws, axis=1) - rng.normal(reach.value, reach.sigma, DRAWS), 0.0)
    times = np.maximum(rng.normal(time_ms, time_sigma_ms, DRAWS), MIN_FLIGHT_MS)
    spares = times - REACT_MS - gos / TOP_M_S * 1000.0
    return Pressure(
        Estimate(go, float(np.std(gos))), Estimate(float(time_ms), float(time_sigma_ms)),
        Estimate(go / max(time_ms, MIN_FLIGHT_MS) * 1000.0, float(np.std(gos / times)) * 1000.0),
        Estimate(time_ms - REACT_MS - go / TOP_M_S * 1000.0, float(np.std(spares))), height,
    )


def rally_pressure(shots: Sequence[ShotFacts], stood: Sequence[dict | None], reaches: dict[str, Estimate | None]) -> list[Pressure | None]:
    """Each shot's pressure on the receiver. `stood` is both players at each hit (each half's
    body.standing_at, or None) and `reaches` each half's body.arm_reach."""
    rng = np.random.default_rng(0)
    out: list[Pressure | None] = []
    for i, shot in enumerate(shots):
        receiver = "far" if shot.hitter == "near" else "near"
        where, reach = (stood[i] or {}).get(receiver), reaches.get(receiver)
        after = shots[i + 1] if i + 1 < len(shots) else None
        if where is None or reach is None:
            out.append(None)
        elif after is not None and after.hitter == receiver and not after.serve:
            sigma = np.asarray(after.contact_sigma[:2])
            height = Estimate(after.contact[2], after.contact_sigma[2])
            out.append(work(where, after.contact[:2], np.diag(sigma**2), reach, after.hit_ms - shot.hit_ms, height, rng))
        elif after is None and shot.landing is not None and shot.landing.p_over_net >= 0.5:
            out.append(work(where, shot.landing.point, shot.landing.covariance, reach, shot.landing.time_ms - shot.hit_ms, None, rng))
        else:
            out.append(None)
    return out
