# worker/shots.py
"""What each shot of a rally did, from its fitted 3D flight.

Per shot: where it was hit and how high, by which half, how fast and at what angle it left the
racket, how high it went, how high it crossed the net and by how much it cleared it, how long it
flew, and, for the shot that ends the rally, where it landed and whether that was in. Every
number carries a standard deviation propagated from the flight fit's covariance, and a landing
too close to a line for its uncertainty is "too close to call" rather than a guess.

The rules are the BWF's: lines are in; a rally shot must land in the opponent's half of the
singles or doubles court; a serve must land in the service court diagonally opposite, beyond
the short service line and, in doubles, short of the doubles long service line.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from geometry.court_model import (
    COURT_LENGTH_M,
    DOUBLES_MARGIN_M,
    LONG_SERVICE_INSET_M,
    NET_CENTRE_HEIGHT_M,
    NET_POST_HEIGHT_M,
    NET_Y_M,
    SHORT_SERVICE_M,
    SINGLES_WIDTH_M,
)
from geometry.flight_fit import INLIER_SIGMAS, FlightFit, _Views, fit_flight
from geometry.rally_fit import Seen, at_terminal_velocity, fit_rally, terminal_velocity
from geometry.shuttle_physics import GRAVITY

CENTRE_X_M = SINGLES_WIDTH_M / 2  # the centre line is shared by both service courts
POST_X_M = (-DOUBLES_MARGIN_M, SINGLES_WIDTH_M + DOUBLES_MARGIN_M)  # posts stand on the doubles sidelines
CALL_CONFIDENCE = 0.95  # a landing is called in or out only when it is this sure
CALL_SAMPLES = 4000
PATH_STEP_S = 0.002  # resolution of the path the apex, net crossing and landing are read from
SHOWN_EVERY = 10  # the path a shot reports, for drawing: every 20 ms
MAX_FLIGHT_S = 3.0  # no shot stays up longer
STATE_STEP_S = 0.001
DIFFERENCE_STEP = 1e-3  # central-difference step on the flight's position (m) and velocity (m/s)
ONE_FLIGHT_RMS = 2.0  # pieces are one flight when fitting them together misses by at most this much more
ONE_FLIGHT_SHARE = 0.8  # and still explains this share of every piece's detections
# The court calibration's own error, which the flight fit does not see: the camera-tracking
# gate's budget (median floor error under 5 cm) until labelled real clips measure it.
CALIBRATION_SIGMA_M = 0.05


@dataclass(frozen=True)
class Estimate:
    value: float
    sigma: float  # one standard deviation


@dataclass(frozen=True)
class NetCrossing:
    x_m: float
    height_m: Estimate
    clearance_m: Estimate  # above the top of the net where it crossed


@dataclass(frozen=True)
class Landing:
    point: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    time_ms: float
    call: str  # "in" | "out" | "too close to call"
    p_in: float  # the chance it landed in, given the fit's uncertainty


@dataclass(frozen=True)
class ShotFacts:
    hit_ms: float
    hitter: str  # "near" | "far": the half it was hit from
    serve: bool
    contact: tuple[float, float, float]
    contact_sigma: tuple[float, float, float]
    speed_m_s: Estimate  # leaving the racket
    elevation_deg: Estimate  # above horizontal, leaving the racket
    apex_m: Estimate
    net: NetCrossing | None  # None when it never crossed the net
    flight_ms: float  # until it was played again, or until it landed
    landing: Landing | None  # None when it was played again before landing
    path: tuple[tuple[float, float, float, float], ...] = ()  # (time ms, x, y, z) from the hit to its end
    terminal_velocity: Estimate | None = None  # the shuttle's drag these facts were worked out with


def net_height_m(x: float) -> float:
    """Height of the top of the net across the court: 1.55 m at the posts, sagging to 1.524 m
    at the centre (taken as a parabola)."""
    half_span = (POST_X_M[1] - POST_X_M[0]) / 2
    return NET_CENTRE_HEIGHT_M + (NET_POST_HEIGHT_M - NET_CENTRE_HEIGHT_M) * ((x - CENTRE_X_M) / half_span) ** 2


def landing_area(court: str, hitter: str, serve_from_x: float | None = None) -> tuple[float, float, float, float]:
    """(x_low, x_high, y_low, y_high) where a shot from `hitter`'s half must land to be in, lines
    included. With `serve_from_x` (the server's contact), the diagonal service court."""
    if court not in ("singles", "doubles"):
        raise ValueError(f"court must be singles or doubles, not {court!r}")
    x_low, x_high = (0.0, SINGLES_WIDTH_M) if court == "singles" else POST_X_M
    near = hitter == "near"
    if serve_from_x is None:
        return (x_low, x_high, NET_Y_M, COURT_LENGTH_M) if near else (x_low, x_high, 0.0, NET_Y_M)
    # Diagonally opposite: the other side of the centre line, beyond the short service line.
    x_range = (x_low, CENTRE_X_M) if serve_from_x > CENTRE_X_M else (CENTRE_X_M, x_high)
    long_inset = LONG_SERVICE_INSET_M if court == "doubles" else 0.0
    if near:
        return (*x_range, NET_Y_M + SHORT_SERVICE_M, COURT_LENGTH_M - long_inset)
    return (*x_range, long_inset, NET_Y_M - SHORT_SERVICE_M)


_UNIT_SAMPLES = np.random.default_rng(0).standard_normal((CALL_SAMPLES, 2))


def call_landing(point, covariance, area) -> tuple[str, float]:
    """In, out or too close to call, and the chance it was in: a landing is called only when its
    uncertainty leaves it at least CALL_CONFIDENCE sure."""
    covariance = np.asarray(covariance, dtype=float)
    root = np.linalg.cholesky(covariance + np.eye(2) * 1e-12)
    samples = np.asarray(point, dtype=float) + _UNIT_SAMPLES @ root.T
    x_low, x_high, y_low, y_high = area
    p_in = float(np.mean((samples[:, 0] >= x_low) & (samples[:, 0] <= x_high) & (samples[:, 1] >= y_low) & (samples[:, 1] <= y_high)))
    if p_in >= CALL_CONFIDENCE:
        return "in", p_in
    if p_in <= 1.0 - CALL_CONFIDENCE:
        return "out", p_in
    return "too close to call", p_in


def _integrate(p: np.ndarray, v: np.ndarray, duration_s: float, step_s: float, terminal_velocity: float):
    """RK4 under gravity and drag for B flights at once, forwards or (duration < 0) backwards.
    Returns positions and velocities (n + 1, B, 3) at every step, the last at exactly duration."""
    k = GRAVITY / terminal_velocity**2

    def acceleration(vel: np.ndarray) -> np.ndarray:
        a = -k * np.linalg.norm(vel, axis=1)[:, None] * vel
        a[:, 2] -= GRAVITY
        return a

    count = max(1, int(np.ceil(abs(duration_s) / step_s - 1e-9)))
    h = duration_s / count
    ps, vs = [p], [v]
    for _ in range(count):
        a1 = acceleration(v)
        a2 = acceleration(v + 0.5 * h * a1)
        a3 = acceleration(v + 0.5 * h * a2)
        a4 = acceleration(v + h * a3)
        p = p + h * v + h * h / 6 * (a1 + a2 + a3)
        v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        ps.append(p)
        vs.append(v)
    return np.stack(ps), np.stack(vs)


def _facts(params: np.ndarray, fit: FlightFit, hit_ms: float, ends_ms: float | None) -> list[dict]:
    """For each row of (p0, v0) at fit.start_ms: the shot's facts as numbers (NaN where absent)."""
    p, v = params[:, :3], params[:, 3:]
    to_hit = (hit_ms - fit.start_ms) / 1000.0
    if abs(to_hit) > 1e-9:
        ps, vs = _integrate(p, v, to_hit, STATE_STEP_S, fit.terminal_velocity)
        p, v = ps[-1], vs[-1]
    duration = MAX_FLIGHT_S if ends_ms is None else max((ends_ms - hit_ms) / 1000.0, PATH_STEP_S)
    path, _ = _integrate(p, v, duration, PATH_STEP_S, fit.terminal_velocity)  # (n, B, 3)
    step = duration / (len(path) - 1)
    rows = []
    for b in range(len(params)):
        xyz = path[:, b]
        row = {
            "contact": p[b],
            "speed": float(np.linalg.norm(v[b])),
            "elevation": float(np.degrees(np.arctan2(v[b, 2], np.hypot(v[b, 0], v[b, 1])))),
            "landing": None,
        }
        end = len(xyz)
        if ends_ms is None:
            below = np.nonzero((xyz[1:, 2] <= 0.0) & (xyz[:-1, 2] > 0.0))[0]
            if len(below):
                i = int(below[0])
                f = xyz[i, 2] / (xyz[i, 2] - xyz[i + 1, 2])
                row["landing"] = (xyz[i, :2] + f * (xyz[i + 1, :2] - xyz[i, :2]), (i + f) * step)
                end = i + 2
        flown = xyz[:end]
        row["path"] = [(hit_ms + 1000.0 * k * step, *map(float, flown[k])) for k in range(0, len(flown), SHOWN_EVERY)]
        if row["landing"] is not None:
            (x, y), seconds = row["landing"]
            row["path"] = [point for point in row["path"] if point[0] < hit_ms + 1000.0 * seconds] + [(hit_ms + 1000.0 * seconds, float(x), float(y), 0.0)]
        row["apex"] = float(flown[:, 2].max())
        side = np.sign(flown[:, 1] - NET_Y_M)
        cross = np.nonzero(side[1:] != side[:-1])[0]
        row["net"] = None
        if len(cross):
            i = int(cross[0])
            f = (NET_Y_M - flown[i, 1]) / (flown[i + 1, 1] - flown[i, 1])
            at = flown[i] + f * (flown[i + 1] - flown[i])
            row["net"] = (float(at[0]), float(at[2]), float(at[2] - net_height_m(at[0])))
        rows.append(row)
    return rows


def flight_facts(
    fit: FlightFit, hit_ms: float, hitter: str, court: str = "singles", ends_ms: float | None = None, serve: bool = False
) -> ShotFacts:
    """A shot's facts from its fitted flight. `ends_ms` is when it was played again; None means
    it ended the rally, so it is followed to the floor and called."""
    x0 = np.concatenate([fit.p0, fit.v0])
    steps = np.eye(6) * DIFFERENCE_STEP
    batch = np.vstack([x0, x0 + steps, x0 - steps])
    rows = _facts(batch, fit, hit_ms, ends_ms)
    nominal = rows[0]

    def vector(row) -> np.ndarray | None:
        values = [*row["contact"], row["speed"], row["elevation"], row["apex"]]
        if nominal["net"] is not None:
            if row["net"] is None:
                return None
            values += [row["net"][1], row["net"][2]]
        if nominal["landing"] is not None:
            if row["landing"] is None:
                return None
            values += [*row["landing"][0]]
        return np.array(values, dtype=float)

    plus, minus = [vector(r) for r in rows[1:7]], [vector(r) for r in rows[7:]]
    centre = vector(nominal)
    jacobian = np.stack([
        (a - b) / (2 * DIFFERENCE_STEP) if a is not None and b is not None else np.zeros_like(centre) for a, b in zip(plus, minus)
    ], axis=1)
    covariance = jacobian @ np.asarray(fit.covariance) @ jacobian.T
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    def estimate(i: int) -> Estimate:
        return Estimate(float(centre[i]), float(sigma[i]))

    net = None
    if nominal["net"] is not None:
        net = NetCrossing(nominal["net"][0], estimate(6), estimate(7))
    landing = None
    flight_ms = (ends_ms - hit_ms) if ends_ms is not None else MAX_FLIGHT_S * 1000.0
    if nominal["landing"] is not None:
        i = len(centre) - 2
        point = tuple(float(c) for c in centre[i:])
        cov = covariance[i:, i:] + np.eye(2) * CALIBRATION_SIGMA_M**2
        area = landing_area(court, hitter, serve_from_x=float(centre[0]) if serve else None)
        verdict, p_in = call_landing(point, cov, area)
        flight_ms = float(nominal["landing"][1] * 1000.0)
        landing = Landing(point, tuple(tuple(float(c) for c in r) for r in cov), hit_ms + flight_ms, verdict, p_in)
    return ShotFacts(
        hit_ms, hitter, serve, tuple(float(c) for c in centre[:3]), tuple(float(s) for s in np.hypot(sigma[:3], CALIBRATION_SIGMA_M)),
        estimate(3), estimate(4), estimate(5), net, flight_ms, landing, tuple(tuple(round(float(v), 4) for v in point) for point in nominal["path"]),
    )


def _one_flight(pieces, times: np.ndarray, track, cameras) -> tuple[FlightFit, list[int], int]:
    """The pieces of one shot's flight (the shuttle left the view and came back, unhit) refitted as
    one. Pieces join in order, each only if one flight still explains nearly all the detections of
    every piece so far; the first that does not (a false track after the landing, an unseen hit)
    ends the flight. Each try starts from the pieces traced back to the first's start. Returns
    the fit, its detections and how many of the pieces it took."""
    first = pieces[0]
    fit, indices, used = first.fit, list(first.indices), 1
    for n in range(1, len(pieces)):
        if times[pieces[n].indices[0]] - first.fit.start_ms > MAX_FLIGHT_S * 1000.0:
            break  # no shot stays up this long
        members = pieces[: n + 1]
        joined = [i for piece in members for i in piece.indices]
        guesses = []
        for piece in members:
            back = (first.fit.start_ms - piece.fit.start_ms) / 1000.0
            ps, vs = _integrate(piece.fit.p0[None], piece.fit.v0[None], back, STATE_STEP_S, piece.fit.terminal_velocity)
            guess = np.concatenate([ps[-1, 0], vs[-1, 0]])
            if np.isfinite(guess).all():
                guesses.append(guess)
        trial = fit_flight(
            times[joined], np.array([track[i] for i in joined], dtype=float), [cameras[i] for i in joined],
            start_ms=first.fit.start_ms, terminal_velocity=first.fit.terminal_velocity, initial=guesses,
        )
        if trial is None or trial.rms_px > ONE_FLIGHT_RMS * max(piece.fit.rms_px for piece in members):
            break
        views = _Views([cameras[i] for i in joined])
        misses = np.hypot(*(views.project(trial.positions_at(times[joined])[None])[0] - np.array([track[i] for i in joined], dtype=float)).T)
        bound = INLIER_SIGMAS * 2 * max(piece.fit.rms_px for piece in members)
        at = 0
        explained = True
        for piece in members:
            share = float(np.mean(misses[at : at + len(piece.indices)] <= bound))
            at += len(piece.indices)
            explained = explained and share >= ONE_FLIGHT_SHARE
        if not explained:
            break
        fit, indices, used = trial, joined, n + 1
    return fit, indices, used


def rally_shots(
    analysis, track, times_ms, cameras, court: str = "singles", jitter_px: float = 2.0, drag: tuple[float, float] | None = None
) -> list[ShotFacts]:
    """Every shot of a rally (hits.find_hits's analysis of these detections): each hit with its
    flight (every piece seen until the next hit), the flights refitted together so neighbours
    share their contact point, then played again at the next hit or, for the last, followed to the
    floor. `drag` is the shuttle's terminal velocity and its standard deviation when known (from
    a whole session's rallies, say); otherwise it is measured from this rally's own flights."""
    times = np.asarray(times_ms, dtype=float)
    hits = sorted(analysis.hits, key=lambda h: h.time_ms)
    fitted = [k for k, piece in enumerate(analysis.flights) if piece.fit is not None]
    flights = []  # per shot: (hit index, piece numbers, fit, detections)
    for n, hit in enumerate(hits):
        first = next((k for k in fitted if hit.frame in analysis.flights[k].indices), None)
        if first is None:
            continue  # nothing seen of this shot's flight
        until = hits[n + 1].frame if n + 1 < len(hits) else None
        members = [k for k in fitted if k >= first and (until is None or analysis.flights[k].indices[0] < until)]
        fit, indices, used = _one_flight([analysis.flights[k] for k in members], times, track, cameras)
        flights.append((n, members[:used], fit, indices))
    # A hit hands over from one flight to the next when nothing lies between them.
    links = [
        (j, hits[flights[j + 1][0]].time_ms)
        for j in range(len(flights) - 1)
        if "flights" in hits[flights[j + 1][0]].evidence and flights[j + 1][1][0] - flights[j][1][-1] == 1
    ]
    seen = [Seen(times[indices], np.array([track[i] for i in indices], dtype=float), [cameras[i] for i in indices], fit) for _, _, fit, indices in flights]
    # This shuttle's drag, from the rally's own long flights; one standard deviation of it either
    # way is carried into every fact, so a landing far from its last sighting is not over-sure.
    drag, drag_sigma = drag if drag is not None else terminal_velocity(seen, jitter_px)
    middle = fit_rally(at_terminal_velocity(seen, drag, jitter_px), links, jitter_px)
    other = fit_rally(at_terminal_velocity([Seen(s.times_ms, s.pixels, s.cameras, f) for s, f in zip(seen, middle)], drag + drag_sigma, jitter_px), links, jitter_px)
    shots = []
    for (n, _, _, _), fit, shifted in zip(flights, middle, other):
        hit = hits[n]
        ends_ms = hits[n + 1].time_ms if n + 1 < len(hits) else None
        contact = fit.position_at(max(hit.time_ms, fit.start_ms))
        hitter = "near" if contact[1] < NET_Y_M else "far"
        serve = hit.evidence.endswith("serve")
        facts = flight_facts(fit, hit.time_ms, hitter, court, ends_ms, serve=serve)
        facts = _with_drag(facts, flight_facts(shifted, hit.time_ms, hitter, court, ends_ms, serve=serve), court)
        shots.append(replace(facts, terminal_velocity=Estimate(float(drag), float(drag_sigma))))
    return shots


def _with_drag(facts: ShotFacts, shifted: ShotFacts, court: str) -> ShotFacts:
    """A shot's facts with the change one standard deviation of drag makes added to their spread."""

    def widened(a: Estimate, b: Estimate) -> Estimate:
        return Estimate(a.value, float(np.hypot(a.sigma, b.value - a.value)))

    net = facts.net
    if net is not None and shifted.net is not None:
        net = NetCrossing(net.x_m, widened(net.height_m, shifted.net.height_m), widened(net.clearance_m, shifted.net.clearance_m))
    landing = facts.landing
    if landing is not None and shifted.landing is not None:
        change = np.asarray(shifted.landing.point) - np.asarray(landing.point)
        covariance = np.asarray(landing.covariance) + np.outer(change, change)
        area = landing_area(court, facts.hitter, serve_from_x=facts.contact[0] if facts.serve else None)
        verdict, p_in = call_landing(landing.point, covariance, area)
        landing = Landing(landing.point, tuple(tuple(float(c) for c in row) for row in covariance), landing.time_ms, verdict, p_in)
    contact_change = np.asarray(shifted.contact) - np.asarray(facts.contact)
    return replace(
        facts,
        contact_sigma=tuple(float(s) for s in np.hypot(facts.contact_sigma, contact_change)),
        speed_m_s=widened(facts.speed_m_s, shifted.speed_m_s),
        elevation_deg=widened(facts.elevation_deg, shifted.elevation_deg),
        apex_m=widened(facts.apex_m, shifted.apex_m),
        net=net,
        landing=landing,
    )
