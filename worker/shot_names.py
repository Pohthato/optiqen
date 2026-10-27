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

The model (`shot_names.json`, written by `evaluation.shot_names_benchmark learn`) is a
multinomial logistic regression on the facts and their pairwise products, learnt from the true
facts of simulated shots. A shot's chance of each name is the model's averaged over the draws,
so it is as unsure as the measurement. The likeliest name is given alone when it is at least
`SURE`; otherwise the two likeliest are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from geometry.court_model import NET_Y_M
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY
from shots import MAX_FLIGHT_S, PATH_STEP_S, ShotFacts, _integrate

FACTS = ("contactHeightM", "contactFromNetM", "speedMS", "elevationDeg", "apexAboveM", "endPastNetM", "endHeightM", "flightS")
NAMES = ("clear", "drop", "smash", "drive", "push", "lift", "net", "block")
DRAWS = 200
CHUNK = 2000  # launches flown at once (each keeps its whole path in memory)
SURE = 0.8
MODEL_PATH = Path(__file__).with_name("shot_names.json")


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


@dataclass(frozen=True)
class ShotName:
    call: str  # a name, or "drop or smash" when neither is sure enough alone
    p: dict[str, float]  # each name's chance


@dataclass(frozen=True)
class Model:
    """Each fact is held to the range the model learnt from: past it the products of facts run
    wild, so a smash faster than any simulated one reads as the fastest simulated."""

    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # (terms, names)

    def terms(self, rows: np.ndarray) -> np.ndarray:
        z = (np.clip(np.atleast_2d(rows), self.low, self.high) - self.mean) / self.scale
        i, j = np.triu_indices(z.shape[1])
        return np.column_stack([np.ones(len(z)), z, z[:, i] * z[:, j]])

    def probabilities(self, rows: np.ndarray) -> np.ndarray:
        return softmax(self.terms(rows) @ self.weights, axis=1)

    def save(self, path: Path) -> None:
        fields = {key: np.round(getattr(self, key), 6).tolist() for key in ("low", "high", "mean", "scale", "weights")}
        path.write_text(json.dumps({"facts": FACTS, "names": NAMES, **fields}, indent=1) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "Model":
        stored = json.loads(path.read_text(encoding="utf-8"))
        if tuple(stored["facts"]) != FACTS or tuple(stored["names"]) != NAMES:
            raise ValueError(f"{path} was learnt on other facts or names")
        return Model(*(np.array(stored[key]) for key in ("low", "high", "mean", "scale", "weights")))


def learn(names: list[str], rows: np.ndarray, l2: float) -> Model:
    """The model that best predicts these names from these facts, every weight but the
    intercepts held towards zero by `l2`."""
    base = Model(rows.min(axis=0), rows.max(axis=0), rows.mean(axis=0), rows.std(axis=0), np.zeros(0))
    terms = base.terms(rows)
    truth = np.eye(len(NAMES))[[NAMES.index(name) for name in names]]
    shape = (terms.shape[1], len(NAMES))

    def loss(flat: np.ndarray) -> tuple[float, np.ndarray]:
        weights = flat.reshape(shape)
        logits = terms @ weights
        log_p = logits - logsumexp(logits, axis=1, keepdims=True)
        held = weights.copy()
        held[0] = 0.0
        value = -(truth * log_p).sum() / len(terms) + 0.5 * l2 * (held**2).sum()
        gradient = terms.T @ (np.exp(log_p) - truth) / len(terms) + l2 * held
        return value, gradient.ravel()

    fit = minimize(loss, np.zeros(shape).ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    return Model(base.low, base.high, base.mean, base.scale, fit.x.reshape(shape))


def call_of(p: np.ndarray) -> str:
    first, second = np.argsort(p)[::-1][:2]
    return NAMES[first] if p[first] >= SURE else f"{NAMES[first]} or {NAMES[second]}"


def name_shot(shot: ShotFacts, model: Model, rng: np.random.Generator) -> ShotName:
    """A measured shot's name. A serve is named by the rally, not the model."""
    if shot.serve:
        return ShotName("serve", {"serve": 1.0})
    p = model.probabilities(measured_facts(shot, rng)).mean(axis=0)
    return ShotName(call_of(p), {name: round(float(chance), 3) for name, chance in zip(NAMES, p)})
