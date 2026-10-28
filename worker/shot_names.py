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

The model is a multinomial logistic regression on the facts and their pairwise products. It is
first learnt from the true facts of simulated shots (`shot_names_simulated.json`, written by
`evaluation.shot_names_benchmark learn`). Shots named by hand in real play then move it as far
as they can show (`shot_names.json`, written by `... adapt`; see `learn`): the simulator's
families are not real play's. A shot's chance of each name is the model's averaged over the
draws, so it is as unsure as the measurement. The likeliest name is given alone when it is at
least `SURE`; otherwise the two likeliest are.

Then the details a coach adds, each with its chance, said when it is at least `SURE` either way
and "too close to call" between:
- Cross or straight: cross when the shot ended on the other side of the court's centre line from
  where it was hit, at least `CROSS_M` across from it; over the same draws of the flight.
- Backhand or forehand: backhand when the contact was on the side of the hips away from the
  racket arm. Each shot's arm is the one whose wrist was nearer the contact (body.hit_body), and
  a pose can take one wrist for the other, so the racket arm is only as sure as the player's
  shots agree on it: right with the share of their shots hit with the right (`racket_arms`).
- From a jump or not: the lower foot more than `JUMP_M` off the floor at contact.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr, softmax

from body import BodyFacts
from geometry.court_model import NET_Y_M, SINGLES_WIDTH_M
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY
from shots import MAX_FLIGHT_S, PATH_STEP_S, Estimate, ShotFacts, _integrate

FACTS = ("contactHeightM", "contactFromNetM", "speedMS", "elevationDeg", "apexAboveM", "endPastNetM", "endHeightM", "flightS")
NAMES = ("clear", "drop", "smash", "drive", "push", "lift", "net", "block")
DRAWS = 200
CHUNK = 2000  # launches flown at once (each keeps its whole path in memory)
SURE = 0.8
MODEL_PATH = Path(__file__).with_name("shot_names.json")
SIMULATED_PATH = Path(__file__).with_name("shot_names_simulated.json")
CROSS_M = SINGLES_WIDTH_M / 4  # so a shot from the middle that drifts over the line is not cross
JUMP_M = 0.15  # past what an ankle rises on tiptoe, or in a split step's hop
TOO_CLOSE = "too close to call"


def facts(contact, velocity, near, duration_s, terminal_velocity) -> tuple[np.ndarray, np.ndarray]:
    """The facts (B, len(FACTS)) of B launches, and whether each went cross-court (B: 1 or 0, NaN
    for one that could not be flown): contacts and velocities (B, 3), each flown until its
    `duration_s` (inf: until the floor) or the floor, whichever comes first. `near` (the hitter's
    half), `duration_s` and `terminal_velocity` are one for all or one per launch."""
    p = np.asarray(contact, dtype=float).reshape(-1, 3)
    v = np.asarray(velocity, dtype=float).reshape(-1, 3)
    count = len(p)
    near = np.broadcast_to(np.asarray(near, dtype=bool), (count,))
    durations = np.minimum(np.broadcast_to(np.asarray(duration_s, dtype=float), (count,)), MAX_FLIGHT_S)
    drag = np.broadcast_to(np.asarray(terminal_velocity, dtype=float), (count,))
    out = np.empty((count, len(FACTS)))
    cross = np.empty(count)
    for start in range(0, count, CHUNK):
        rows = slice(start, start + CHUNK)
        span = float(durations[rows].max())
        with np.errstate(over="ignore", invalid="ignore"):  # a launch too fast to fly gives NaN facts
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
        middle = SINGLES_WIDTH_M / 2
        across = ((pr[:, 0] - middle) * (end[:, 0] - middle) < 0) & (np.abs(end[:, 0] - pr[:, 0]) >= CROSS_M)
        cross[rows] = np.where(np.isfinite(end[:, 0]), across, np.nan)
    return out, cross


def true_facts(rallies) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Every shot of simulated rallies (simulation.rally.Rally), serves included: its name, and its
    facts and whether it went cross-court from the true flight, all flown at once."""
    names, p, v, near, durations, drag = [], [], [], [], [], []
    for rally in rallies:
        for n, (contact, flight) in enumerate(zip(rally.contacts, rally.flights)):
            names.append(contact.kind)
            p.append(flight.p0)
            v.append(flight.v0)
            near.append(contact.hitter == "near")
            durations.append(flight.end_time - flight.start_time if n + 1 < len(rally.flights) else np.inf)  # the last one lands
            drag.append(rally.terminal_velocity)
    return (names, *facts(p, v, near, durations, drag)) if names else (names, np.empty((0, len(FACTS))), np.empty(0))


def _duration_s(landed: bool, flight_ms: float) -> float:
    """How long a measured shot flew before it was played again; inf when it went to the floor."""
    return np.inf if landed or flight_ms >= MAX_FLIGHT_S * 1000.0 else flight_ms / 1000.0


@dataclass(frozen=True)
class Launch:
    """A measured shot as its name sees it: the shuttle leaving the racket (the contact and the
    velocity, with their 6 x 6 covariance), from which half, how long it flew before it was played
    again (inf: to the floor), and the shuttle's drag (None: a feather shuttle's)."""

    state: np.ndarray
    covariance: np.ndarray
    near: bool
    duration_s: float
    drag: Estimate | None

    @staticmethod
    def of(shot: ShotFacts) -> "Launch":
        state = np.array([*shot.contact, *shot.velocity])
        return Launch(state, np.asarray(shot.launch_covariance), shot.hitter == "near", _duration_s(shot.landing is not None, shot.flight_ms), shot.terminal_velocity)

    @staticmethod
    def entry(shot: dict, drag: dict | None) -> "Launch":
        """A shot of an analysis file (analyse_rally), with the rally's drag from its summary."""
        terminal = None if drag is None else Estimate(drag["value"], drag["sigma"])
        landed = shot["landing"] is not None
        return Launch(np.array(shot["launch"]["value"]), np.array(shot["launch"]["covariance"]), shot["hitter"] == "near", _duration_s(landed, shot["flightMs"]), terminal)


def nominal_facts(launch: Launch) -> np.ndarray:
    drag = launch.drag.value if launch.drag is not None else FEATHER_TERMINAL_VELOCITY
    return facts(launch.state[:3], launch.state[3:], launch.near, launch.duration_s, drag)[0][0]


def measured_facts(launch: Launch, rng: np.random.Generator, draws: int = DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """The facts of `draws` flights drawn from a measured launch and its uncertainty, each flown
    with its own draw of the shuttle's drag, and whether each went cross-court."""
    launches = rng.multivariate_normal(launch.state, launch.covariance, size=draws, method="eigh")
    drag = launch.drag
    drags = FEATHER_TERMINAL_VELOCITY if drag is None else np.maximum(rng.normal(drag.value, drag.sigma, draws), 1.0)
    return facts(launches[:, :3], launches[:, 3:], launch.near, launch.duration_s, drags)


@dataclass(frozen=True)
class Detail:
    call: str  # "cross" or "straight", "backhand" or "forehand", "jump" or "no jump", "right" or "left", or TOO_CLOSE
    p: float  # the chance of the first of the two


def detail(p: float, first: str, second: str) -> Detail:
    """The call follows the chance as given, to three places."""
    p = round(float(p), 3)
    return Detail(first if p >= SURE else second if 1 - p >= SURE else TOO_CLOSE, p)


def _beyond(fact: Estimate, limit: float) -> float:
    """The chance that a measured fact is truly past `limit`."""
    return float(ndtr((fact.value - limit) / max(fact.sigma, 1e-9)))


def racket_arms(bodies: list[BodyFacts | None]) -> dict[str, Detail | None]:
    """Each half's racket arm, right with the share of its player's shots hit with the right arm
    (None: no body seen)."""
    arms = {}
    for half in ("near", "far"):
        hit = [body.arm for body in bodies if body is not None and body.hitter == half]
        arms[half] = detail(hit.count("right") / len(hit), "right", "left") if hit else None
    return arms


def backhand(body: BodyFacts | None, racket_arm: Detail | None) -> Detail | None:
    """None when the body or the racket arm is not known. The body's contact aside is towards the
    arm whose wrist was nearer it; here it is turned to the hips' right."""
    if body is None or racket_arm is None:
        return None
    right = Estimate((1.0 if body.arm == "right" else -1.0) * body.aside_m.value, body.aside_m.sigma)
    left = _beyond(Estimate(-right.value, right.sigma), 0.0)
    return detail(racket_arm.p * left + (1.0 - racket_arm.p) * (1.0 - left), "backhand", "forehand")


def jump(body: BodyFacts | None) -> Detail | None:
    return None if body is None else detail(_beyond(body.off_floor_m, JUMP_M), "jump", "no jump")


@dataclass(frozen=True)
class ShotName:
    call: str  # a name, or "drop or smash" when neither is sure enough alone
    p: dict[str, float]  # each name's chance
    cross: Detail | None = None  # None for a serve: the rules send it cross
    backhand: Detail | None = None  # None with the body or the racket arm unknown
    jump: Detail | None = None

    @property
    def said(self) -> str:
        """The name as a coach says it, with each detail that is sure: "cross jump smash"."""
        sure = [d.call for d in (self.cross, self.backhand, self.jump) if d is not None and d.call in ("cross", "backhand", "jump")]
        return " ".join([*sure, self.call])

    def entry(self) -> dict:
        """As an analysis reports it (analyse_rally): the call with each family's chance, the name as
        a coach says it, and each detail's call and chance (None: not known)."""
        details = {"cross": self.cross, "backhand": self.backhand, "jump": self.jump}
        return {"call": self.call, "p": self.p, "said": self.said, **{key: None if d is None else asdict(d) for key, d in details.items()}}


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


def learn(names: list[str | tuple[str, ...]], rows: np.ndarray, l2: float, prior: Model | None = None) -> Model:
    """The model that best predicts these names from these facts, every weight but the
    intercepts held by `l2` towards zero.

    Or, from a `prior` (whose fact ranges and scales it keeps): the prior's weights scaled as a
    whole, so the shots may say it is surer than it should be, and the intercepts and the single
    facts' weights moved from there, held by `l2`. The products of facts stay the prior's: a few
    shots cannot place curved boundaries, and products moved to fit them swing the names of shots
    unlike any of them.

    A shot named by a tuple of names may be any of them, each as likely."""
    base = prior or Model(rows.min(axis=0), rows.max(axis=0), rows.mean(axis=0), rows.std(axis=0), np.zeros(0))
    terms = base.terms(rows)
    truth = np.array([np.isin(NAMES, name) for name in names], dtype=float)
    truth /= truth.sum(axis=1, keepdims=True)
    centre = np.zeros((terms.shape[1], len(NAMES))) if prior is None else prior.weights
    free = len(centre) if prior is None else 1 + len(FACTS)  # the rows of weights that move
    held = np.ones((free, 1))
    if prior is None:
        held[0] = 0.0

    def weights_of(flat: np.ndarray) -> np.ndarray:
        weights = flat[0] * centre
        weights[:free] += flat[1:].reshape(free, len(NAMES))
        return weights

    def loss(flat: np.ndarray) -> tuple[float, np.ndarray]:
        moved = flat[1:].reshape(free, len(NAMES))
        logits = terms @ weights_of(flat)
        log_p = logits - logsumexp(logits, axis=1, keepdims=True)
        gradient = terms.T @ (np.exp(log_p) - truth) / len(terms)
        value = -(truth * log_p).sum() / len(terms) + 0.5 * l2 * (held * moved**2).sum()
        return value, np.concatenate([[(gradient * centre).sum()], (gradient[:free] + l2 * held * moved).ravel()])

    start = np.concatenate([[1.0], np.zeros(free * len(NAMES))])
    bounds = [(0.0, None)] + [(None, None)] * (free * len(NAMES))  # the prior's scale is not turned over
    fit = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
    return Model(base.low, base.high, base.mean, base.scale, weights_of(fit.x))


def chances(model: Model, drawn: np.ndarray) -> np.ndarray:
    """The chance of each name over draws of a shot's facts (..., draws, facts). A draw that could
    not be flown is left out: from a hopeless measurement it may leave the racket thousands of
    metres a second, past what the flight's integration can follow."""
    p = model.probabilities(drawn.reshape(-1, len(FACTS))).reshape(*drawn.shape[:-1], len(NAMES))
    return np.nanmean(p, axis=-2)


def call_of(p: np.ndarray) -> str:
    first, second = np.argsort(p)[::-1][:2]
    return NAMES[first] if p[first] >= SURE else f"{NAMES[first]} or {NAMES[second]}"


def name_shot(launch: Launch, serve: bool, model: Model, rng: np.random.Generator, body: BodyFacts | None = None, racket_arm: Detail | None = None) -> ShotName:
    """A measured shot's name and details, from its launch and the hitter's body at contact. A
    serve is named by the rally, not the model."""
    if serve:
        return ShotName("serve", {"serve": 1.0}, None, backhand(body, racket_arm), jump(body))
    drawn, cross = measured_facts(launch, rng)
    p = chances(model, drawn)
    names = {name: round(float(chance), 3) for name, chance in zip(NAMES, p)}
    return ShotName(call_of(p), names, detail(np.nanmean(cross), "cross", "straight"), backhand(body, racket_arm), jump(body))
