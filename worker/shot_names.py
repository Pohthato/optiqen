# worker/shot_names.py
"""What a shot is called (clear, drop, smash, lift, net, push, drive, block), from what its
flight did.

A name rests on these facts, each seen from the hitter's side, so a far player's shot reads like
a near player's: where it was hit from (the contact's height and distance from the net), how it
left the racket (speed, and angle above the horizontal), how high it rose above the contact, and
where it went (how far past the net it ended, how high, and after how long). A shot ends where it
was played again, or on the floor.

The facts come from the shuttle's state leaving the racket, flown under gravity and drag. A
measured shot's state is uncertain (`shots.ShotFacts.launch_covariance`), so its facts are taken
over many draws of it, each a whole flight.
"""
from __future__ import annotations

import numpy as np

from geometry.court_model import NET_Y_M
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY
from shots import MAX_FLIGHT_S, PATH_STEP_S, ShotFacts, _integrate

FACTS = ("contactHeightM", "contactFromNetM", "speedMS", "elevationDeg", "apexAboveM", "endPastNetM", "endHeightM", "flightS")
DRAWS = 200
CHUNK = 2000  # launches flown at once (each keeps its whole path in memory)


def facts(contact, velocity, near, duration_s, terminal_velocity) -> np.ndarray:
    """The facts (B, len(FACTS)) of B launches: contacts and velocities (B, 3), each flown until
    its `duration_s` (inf: until the floor) or the floor, whichever comes first. `near` (the
    hitter's half), `duration_s` and `terminal_velocity` are one for all or one per launch."""
    p = np.asarray(contact, dtype=float).reshape(-1, 3)
    v = np.asarray(velocity, dtype=float).reshape(-1, 3)
    count = len(p)
    near = np.broadcast_to(np.asarray(near, dtype=bool), (count,))
    durations = np.minimum(np.broadcast_to(np.asarray(duration_s, dtype=float), (count,)), MAX_FLIGHT_S)
    drag = np.broadcast_to(np.asarray(terminal_velocity, dtype=float), (count,))
    out = np.empty((count, len(FACTS)))
    for start in range(0, count, CHUNK):
        rows = slice(start, start + CHUNK)
        span = float(durations[rows].max())
        path, _ = _integrate(p[rows], v[rows], span, PATH_STEP_S, drag[rows, None])  # (n + 1, b, 3)
        step = span / (len(path) - 1)
        last = np.minimum(np.round(durations[rows] / step).astype(int), len(path) - 1)
        z = path[:, :, 2]
        k = np.arange(len(path))[:, None]
        down = (z[1:] <= 0.0) & (z[:-1] > 0.0) & (k[1:] <= last)
        landed = down.any(axis=0)
        i = np.where(landed, down.argmax(axis=0), last)  # the step before the floor, or the end
        b = np.arange(len(i))
        nxt = np.minimum(i + 1, len(path) - 1)
        f = np.where(landed, z[i, b] / np.where(landed, z[i, b] - z[nxt, b], 1.0), 0.0)
        end = path[i, b] + f[:, None] * (path[nxt, b] - path[i, b])
        apex = np.where(k <= np.where(landed, nxt, i), z, -np.inf).max(axis=0)
        side = np.where(near[rows], 1.0, -1.0)
        pr, vr = p[rows], v[rows]
        out[rows] = np.column_stack([
            pr[:, 2],
            side * (NET_Y_M - pr[:, 1]),
            np.linalg.norm(vr, axis=1),
            np.degrees(np.arctan2(vr[:, 2], np.hypot(vr[:, 0], vr[:, 1]))),
            apex - pr[:, 2],
            side * (end[:, 1] - NET_Y_M),
            end[:, 2],
            (i + f) * step,
        ])
    return out


def true_facts(rallies) -> tuple[list[str], np.ndarray]:
    """Every shot of simulated rallies (simulation.rally.Rally), serves included: its name, and its
    facts from the true flight, all flown at once."""
    names, p, v, near, durations, drag = [], [], [], [], [], []
    for rally in rallies:
        for n, (contact, flight) in enumerate(zip(rally.contacts, rally.flights)):
            names.append(contact.kind)
            p.append(flight.p0)
            v.append(flight.v0)
            near.append(contact.hitter == "near")
            durations.append(flight.end_time - flight.start_time if n + 1 < len(rally.flights) else np.inf)  # the last one lands
            drag.append(rally.terminal_velocity)
    return names, facts(p, v, near, durations, drag) if names else np.empty((0, len(FACTS)))


def _duration_s(shot: ShotFacts) -> float:
    """How long a measured shot flew before it was played again; inf when it went to the floor."""
    return np.inf if shot.landing is not None or shot.flight_ms >= MAX_FLIGHT_S * 1000.0 else shot.flight_ms / 1000.0


def nominal_facts(shot: ShotFacts) -> np.ndarray:
    drag = shot.terminal_velocity.value if shot.terminal_velocity is not None else FEATHER_TERMINAL_VELOCITY
    return facts(shot.contact, shot.velocity, shot.hitter == "near", _duration_s(shot), drag)[0]


def measured_facts(shot: ShotFacts, rng: np.random.Generator, draws: int = DRAWS) -> np.ndarray:
    """The facts of `draws` flights drawn from a measured shot's launch and its uncertainty, each
    flown with its own draw of the shuttle's drag."""
    launches = rng.multivariate_normal([*shot.contact, *shot.velocity], np.asarray(shot.launch_covariance), size=draws, method="eigh")
    drag = shot.terminal_velocity
    drags = FEATHER_TERMINAL_VELOCITY if drag is None else np.maximum(rng.normal(drag.value, drag.sigma, draws), 1.0)
    return facts(launches[:, :3], launches[:, 3:], shot.hitter == "near", _duration_s(shot), drags)
