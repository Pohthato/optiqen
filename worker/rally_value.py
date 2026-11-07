# worker/rally_value.py
"""At each hit, the chance its hitter goes on to win the rally, learnt from ShuttleSet's
professional rallies (evaluation.value_benchmark).

What it sees at a hit (`At`) is what the app has measured by then: where the hitter and the
receiver stood and where the hitter met the shuttle, the time the shot before gave the hitter and
their time to spare, and the chances of the names of the shot before and of the shot being hit.
And the same at the hit before. Never where the shot being hit went, so a rally's end cannot leak
in. The court is turned so that the hitter is always on the near half.

Where the rally turned, by the value (`turn`): at the first hit from which the winner's chance
stayed above even. It did not turn when that is the first hit, or when the winner was not ahead
at the last one. Its chance is the share of the fits that put it there, weighted by the chance
each player won.

A logistic model on these facts, or on them and their pairwise products (as shot_names), each
fact held to the range it was learnt on and a fact not known taken at its mean. Which facts, and
whether their products, is chosen on folds of the training matches. The weights are
fitted on the training matches and again on training matches drawn with replacement, so a
chance's interval spans those refits and draws of its inputs within their uncertainty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from geometry.court_model import COURT_LENGTH_M, NET_Y_M, SINGLES_WIDTH_M
from pressure import Pressure
from shot_names import NAMES as SHOT_NAMES
from shots import CALL_CONFIDENCE, Estimate, ShotFacts

NAMES = ("serve", *SHOT_NAMES)
CONTINUOUS = ("hitterX", "hitterY", "receiverX", "receiverY", "contactX", "contactY", "gapS", "spareS")
_ONE_HIT = (*CONTINUOUS, "spareKnown", *(f"before {name}" for name in NAMES))
FACTS = (*_ONE_HIT, *(f"hit {name}" for name in NAMES), "then", *(f"then {fact}" for fact in _ONE_HIT))
MODEL_PATH = Path(__file__).with_name("rally_value.json")
DRAWS = 50

Place = tuple[np.ndarray, np.ndarray]  # a point on the floor (x, y) and its covariance


@dataclass(frozen=True)
class At:
    """What is known at a hit. Where a player stood is None when not seen. `gap_ms` is the time
    since the shot before and `spare_ms` the time that shot left the hitter to spare, each None
    when not known; `before` and `name` are the chances of each of NAMES for the shot before (None
    when not known) and the one being hit."""

    hitter: Place | None
    receiver: Place | None
    contact: Place
    gap_ms: float | None
    spare_ms: Estimate | None
    before: np.ndarray | None
    name: np.ndarray


def _drawn(place: Place, rng: np.random.Generator | None, draws: int) -> np.ndarray:
    point = np.asarray(place[0], float)
    if rng is None:
        return np.tile(point, (draws, 1))
    return rng.multivariate_normal(point, np.asarray(place[1], float), draws, check_valid="ignore")


def fact_rows(now: At, then: At | None, rng: np.random.Generator | None = None, draws: int = 1) -> np.ndarray:
    """FACTS at a hit, as measured (no `rng`), or in `draws` draws within their uncertainty. A fact
    not known is NaN."""
    rows = np.full((draws, len(FACTS)), np.nan)
    far = float((now.hitter or now.contact)[0][1]) > NET_Y_M
    rows[:, FACTS.index("then")] = then is not None
    for prefix, at in (("", now), ("then ", then)):
        column = {fact: FACTS.index(prefix + fact) for fact in _ONE_HIT}
        names = [column[f"before {name}"] for name in NAMES]
        if at is None:
            rows[:, column["spareKnown"]] = 0.0
            rows[:, names] = 0.0
            continue
        for who in ("hitter", "receiver", "contact"):
            if getattr(at, who) is None:
                continue
            xy = _drawn(getattr(at, who), rng, draws)
            if far:
                xy = np.array([SINGLES_WIDTH_M, COURT_LENGTH_M]) - xy
            rows[:, column[f"{who}X"]], rows[:, column[f"{who}Y"]] = xy[:, 0], xy[:, 1]
        if at.gap_ms is not None:
            rows[:, column["gapS"]] = at.gap_ms / 1000.0
        if at.spare_ms is not None:
            noise = 0.0 if rng is None else rng.standard_normal(draws) * at.spare_ms.sigma
            rows[:, column["spareS"]] = (at.spare_ms.value + noise) / 1000.0
        rows[:, column["spareKnown"]] = at.spare_ms is not None
        rows[:, names] = 0.0 if at.before is None else np.asarray(at.before, float)
    rows[:, [FACTS.index(f"hit {name}") for name in NAMES]] = np.asarray(now.name, float)
    return rows


def named(name: str) -> np.ndarray:
    """A name known for sure, as chances over NAMES."""
    return (np.array(NAMES) == name).astype(float)


@dataclass(frozen=True)
class Model:
    facts: tuple[str, ...]  # those of FACTS it uses
    products: bool
    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # (fits, terms): the fit on all the training matches, then its refits

    @staticmethod
    def framed(rows: np.ndarray, facts: tuple[str, ...], products: bool) -> "Model":
        """A model without weights, its ranges, means and scales taken from these rows."""
        x = rows[:, [FACTS.index(fact) for fact in facts]]
        mean, scale = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        return Model(facts, products, np.nanmin(x, axis=0), np.nanmax(x, axis=0), mean, np.where(scale > 1e-9, scale, 1.0), np.zeros((0, 0)))

    def terms(self, rows: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(rows)[:, [FACTS.index(fact) for fact in self.facts]]
        z = (np.clip(np.where(np.isnan(x), self.mean, x), self.low, self.high) - self.mean) / self.scale
        if not self.products:
            return np.column_stack([np.ones(len(z)), z])
        i, j = np.triu_indices(z.shape[1])
        return np.column_stack([np.ones(len(z)), z, z[:, i] * z[:, j]])

    def chances(self, rows: np.ndarray) -> np.ndarray:
        """The chance the hitter wins, by each fit: (rows, fits)."""
        return 1.0 / (1.0 + np.exp(-(self.terms(rows) @ self.weights.T)))

    def save(self, path: Path = MODEL_PATH) -> None:
        fields = {key: np.round(getattr(self, key), 5).tolist() for key in ("low", "high", "mean", "scale", "weights")}
        path.write_text(json.dumps({"facts": self.facts, "products": self.products, **fields}) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "Model":
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not set(stored["facts"]) <= set(FACTS):
            raise ValueError(f"{path} was learnt on other facts")
        return Model(tuple(stored["facts"]), stored["products"], *(np.array(stored[key]) for key in ("low", "high", "mean", "scale", "weights")))


def fit(model: Model, rows: np.ndarray, won: np.ndarray, l2: float, start: np.ndarray | None = None) -> np.ndarray:
    """The weights that best tell who won from these rows, all but the intercept held by `l2`
    towards zero."""
    terms, won = model.terms(rows), np.asarray(won, float)
    held = np.ones(terms.shape[1])
    held[0] = 0.0

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        log_odds = terms @ w
        value = np.mean(np.logaddexp(0.0, log_odds) - won * log_odds) + 0.5 * l2 * np.sum(held * w**2)
        return float(value), terms.T @ (1.0 / (1.0 + np.exp(-log_odds)) - won) / len(won) + l2 * held * w

    begin = np.zeros(terms.shape[1]) if start is None else start
    return minimize(loss, begin, jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x


def chances_at(model: Model, now: At, then: At | None, rng: np.random.Generator, draws: int = DRAWS) -> np.ndarray:
    """The chance the hitter wins the rally, by draws of the facts and fits: (draws, fits)."""
    return model.chances(fact_rows(now, then, rng, draws))


def chance(model: Model, now: At, then: At | None, rng: np.random.Generator, draws: int = DRAWS) -> tuple[float, float, float]:
    """The chance the hitter wins the rally, and its 95 % interval over the refits and draws of the
    facts."""
    p = chances_at(model, now, then, rng, draws)
    low, high = np.percentile(p, [2.5, 97.5])
    return float(p.mean()), float(low), float(high)


def shot_ats(shots: Sequence[ShotFacts], stood: Sequence[dict | None], pressures: Sequence[Pressure | None], names: Sequence[dict[str, float]]) -> list[At]:
    """What the app knew at each of a rally's shots: where each half's player stood
    (body.standing_at, or None), each shot's pressure on its receiver (pressure.rally_pressure)
    and its chance of each name (shot_names.ShotName.p)."""
    ats = []
    for i, shot in enumerate(shots):
        other = "far" if shot.hitter == "near" else "near"
        at = stood[i] or {}
        before = shots[i - 1] if i else None
        ats.append(At(
            at.get(shot.hitter), at.get(other), (np.array(shot.contact[:2]), np.diag(np.square(shot.contact_sigma[:2]))),
            None if before is None else shot.hit_ms - before.hit_ms,
            None if before is None or pressures[i - 1] is None else pressures[i - 1].spare_ms,
            None if before is None else np.array([names[i - 1].get(name, 0.0) for name in NAMES]),
            np.array([names[i].get(name, 0.0) for name in NAMES]),
        ))
    return ats


def turn(by_fit: np.ndarray, hitters: Sequence[str], won: dict[str, float], sure: bool = True) -> tuple[list[float], float, int | str]:
    """The chance the rally turned at each hit and that it did not, from each hit's chance for its
    hitter by fit (hits, fits) and the chance each player won the rally; and the call: the index of
    the hit, "no turn", or "too close to call" when neither is CALL_CONFIDENCE sure or the winner's
    half is not (`sure`)."""
    p_at, p_no_turn = np.zeros(len(hitters)), 0.0
    for winner, p_won in won.items():
        ahead = np.where((np.asarray(hitters) == winner)[:, None], by_fit, 1.0 - by_fit) > 0.5
        stays = np.flip(np.cumprod(np.flip(ahead, axis=0), axis=0), axis=0).astype(bool)
        first = np.where(stays.any(axis=0), stays.argmax(axis=0), 0)  # 0: ahead from the first hit, or never at the end
        p_at += p_won * np.array([np.mean(first == k) if k else 0.0 for k in range(len(hitters))])
        p_no_turn += p_won * float(np.mean(first == 0))
    at = int(np.argmax(p_at))
    call, p_call = ("no turn", p_no_turn) if p_no_turn >= p_at[at] else (at, float(p_at[at]))
    return p_at.tolist(), p_no_turn, call if sure and p_call >= CALL_CONFIDENCE else "too close to call"
