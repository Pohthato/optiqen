# worker/geometry/rally_fit.py
"""Neighbouring flights of a rally fitted together.

The shuttle is where one flight ends when the next begins: a racket meets it at one point. So a
rally's flights are refitted at once, each to its own detections, with every hit adding that the
flight before it and the flight after it are at the same place then (to CONTACT_SIGMA_M). When
exactly "then" is, is fitted too, near the hit's measured time: a contact pinned a few ms late
would put the outgoing flight half a metre past it. A well-seen flight then pins the contact
point, and with it the depth, of a poorly seen neighbour: from behind the baseline, a far
player's short clear is metres of uncertain depth on its own, but it starts where the well-seen
serve before it ended.

A hit's contact is also within reach of its hitter (Reach), whose feet put them on the floor far
more surely than a far flight's own depth: from a low corner camera a far flight's first few
sightings, by the racket, carry its depth, and a couple of pixels there move it metres. Those
sightings are also the worst (the racket and the body around the shuttle, its blur at its
fastest), so they count for less, as measured on the real clips (EARLY_EXCESS). Beyond
the reach it costs as a hand-over gap does; within it, nothing, and a contact held at its edge
is said to be anywhere within it. A hitter the contact's line of sight does not pass within reach
of is no hitter of it (the other player, someone else tracked); one the flight cannot reach
without missing its own detections is let go, as a hand-over is.

A far flight's depth also hangs on the flight model more than its pixels let on: coming at a low
camera it barely moves in the image, and its depth rests on how fast it falls. On the real clips
the far players' flights met their later sightings from where the hitter stood only if their own
time ran a few per cent faster than the clip's, and stayed metres short of the hitter otherwise.
Why is not known (not drag, not a rolling shutter, not repeated frames). So a flight whose contact
stays well beyond its hitter's reach with the reach pulling it in (CLOCK_EVIDENCE_SIGMAS), the
flight model shown wrong for it, has its own time run at its own rate (FlightFit.clock),
CLOCK_SIGMA either way a priori: its depth is then as unsure as it is, and its hitter's feet set
it. The others keep the clip's clock: let free everywhere, it only loosened what the model got
right (on the shots benchmark, contacts and landings twice as far off, and a wrong call).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geometry.camera import Camera
from geometry.flight_fit import COVARIANCE_INFLATION, INLIER_SIGMAS, FlightFit, _Views, fit_flight, simulate_grid
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, GRAVITY

CONTACT_SIGMA_M = 0.05  # how closely two flights meet at a hit (the model shuttle is a point)
REACH_FLOOR_M = 0.05  # a reach's edge is never surer than this
SIGHT_TOP_M = 3.6  # no racket meets the shuttle higher (a tall player's jump smash)
# Detections just after a hit miss their flight more. On the real clips' near flights, 2.4 times
# as much in the first 70 ms as after 140 ms and 1.4 times in the next 70 ms: a detection's noise
# is 1 + EARLY_EXCESS exp(-t / EARLY_MS) times the rest's, t after the flight's start.
EARLY_EXCESS = 2.4
EARLY_MS = 60.0
# How much faster or slower a flight's own time may run than the clip's, one standard deviation:
# the clock that let each of the real clips' 22 far flights start where its hitter stood was 1.06
# in the middle, 7 % from the clip's (rms).
CLOCK_SIGMA = 0.07
CLOCK_EVIDENCE_SIGMAS = 2.0  # a contact this far beyond its hitter's reach, the reach pulling, shows its flight's model wrong
HANDOVER_SIGMA_MS = 15.0  # how far the meeting may be from the hit's measured time, a priori
WORSE_INLIERS = 0.7  # a flight keeping fewer of its own fit's inliers than this was forced to meet a stranger
WORSE_RMS = 2.0  # and so was one missing its detections by this many times more
ITERATIONS = 40
STEP = 1e-6  # relative forward-difference step for the Jacobian
VELOCITY_STEP_MS = 1.0
TERMINAL_PRIOR_SIGMA = 0.4  # m/s: shuttles differ (feather or nylon, speed grade, temperature)
TERMINAL_GRID = (5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5)
DRAG_FLIGHT_DETECTIONS = 20  # a flight this well seen, for this long, says something about drag
DRAG_FLIGHT_MS = 500.0
# One flight's own estimate scatters by about 7 % (median error 4.7 %, measured on the hard
# benchmark's flights); a clip's flights agreeing more closely than this are not trusted further.
FLIGHT_DRAG_SCATTER = 0.07


@dataclass
class RallyFit:
    flights: list[FlightFit]  # every flight, refitted together where it hands over
    links: list[tuple[int, float]]  # the hand-overs kept
    meetings: dict[int, tuple[float, float]]  # per kept hand-over (by its flight i): when the flights meet, ms, and its standard deviation
    reaches: list[Reach | None]  # per flight: its hitter's reach, if kept


@dataclass
class Reach:
    """Where a flight's hitter stood when they hit it: the contact, on the floor, is within
    `radius` (its standard deviation `sigma`) of `point` (covariance `covariance`)."""

    time_ms: float  # the hit
    point: np.ndarray  # (2,)
    covariance: np.ndarray  # (2, 2)
    radius: float
    sigma: float


@dataclass
class Seen:
    """One flight's detections and its own fit, the starting point."""

    times_ms: np.ndarray
    pixels: np.ndarray  # (N, 2)
    cameras: Sequence[Camera]
    fit: FlightFit


def terminal_velocity(flights: Sequence[Seen], jitter_px: float = 2.0) -> tuple[float, float, int]:
    """This shuttle's terminal velocity and its standard deviation. Each long, well-seen flight
    gives its own estimate (refitted across TERMINAL_GRID, the best by a parabola); the flights'
    median, as sure as their agreement and number allow, is weighed against the usual feather
    shuttle (6.7 +- 0.4 m/s), so a clip with little or discordant evidence stays near it. The
    third value is how many flights it rests on (0: the usual shuttle, assumed)."""
    grid = np.asarray(TERMINAL_GRID, dtype=float)
    estimates = []
    for seen in flights:
        times = np.asarray(seen.times_ms, dtype=float)
        if len(times) < DRAG_FLIGHT_DETECTIONS or times[-1] - times[0] < DRAG_FLIGHT_MS:
            continue
        views = _Views(seen.cameras)
        pixels = np.asarray(seen.pixels, dtype=float)
        guess = [np.concatenate([seen.fit.p0, seen.fit.v0])]
        curve = []
        for value in grid:
            fit = fit_flight(times, pixels, seen.cameras, start_ms=seen.fit.start_ms, terminal_velocity=float(value), jitter_px=jitter_px, initial=guess)
            if fit is None:
                break
            miss = np.hypot(*(views.project(fit.positions_at(times)[None])[0] - pixels).T)
            curve.append(float(np.sum(np.minimum(miss, 6 * jitter_px) ** 2)))
            guess = [np.concatenate([fit.p0, fit.v0])]
        if len(curve) < len(grid):
            continue
        k = int(np.argmin(curve))
        if not 0 < k < len(grid) - 1:
            continue  # at the edge of what a shuttle can be: this flight says nothing reliable
        a, b, c = curve[k - 1], curve[k], curve[k + 1]
        estimates.append(float(grid[k] + 0.5 * (a - c) / (a - 2 * b + c) * (grid[1] - grid[0])) if a - 2 * b + c > 0 else float(grid[k]))
    if not estimates:
        return FEATHER_TERMINAL_VELOCITY, TERMINAL_PRIOR_SIGMA, 0
    middle = float(np.median(estimates))
    scatter = max(1.4826 * float(np.median(np.abs(np.asarray(estimates) - middle))), FLIGHT_DRAG_SCATTER * middle)
    data_sigma = 1.25 * scatter / np.sqrt(len(estimates))  # a median is a little less sure than a mean
    weight = TERMINAL_PRIOR_SIGMA**2 / (TERMINAL_PRIOR_SIGMA**2 + data_sigma**2)
    value = FEATHER_TERMINAL_VELOCITY + weight * (middle - FEATHER_TERMINAL_VELOCITY)
    return float(value), float(np.sqrt(weight) * data_sigma), len(estimates)


def at_terminal_velocity(flights: Sequence[Seen], value: float, jitter_px: float = 2.0) -> list[Seen]:
    """The flights refitted with this terminal velocity (each warm-started from its own fit)."""
    out = []
    for seen in flights:
        fit = fit_flight(
            seen.times_ms, seen.pixels, seen.cameras, start_ms=seen.fit.start_ms, terminal_velocity=value,
            jitter_px=jitter_px, initial=[np.concatenate([seen.fit.p0, seen.fit.v0])],
        )
        out.append(Seen(seen.times_ms, seen.pixels, seen.cameras, fit if fit is not None else seen.fit))
    return out


def fit_rally(
    flights: Sequence[Seen], links: Sequence[tuple[int, float]], jitter_px: float = 2.0, terminal: float | None = None,
    reaches: Sequence[Reach | None] | None = None,
) -> RallyFit:
    """The flights refitted together. `links` holds (i, time_ms): flight i hands over to flight
    i + 1 at about that time; `reaches` each flight's hitter's reach (or None). A hand-over or a
    reach that makes a flight fit its own detections clearly worse than alone (a missed hit
    between them, a wrong one, the wrong hitter) is dropped, the flight's reach first (once its
    clock has been let free and that did not help), and the rest refitted; a flight with neither
    left keeps its own fit. A flight still well beyond its hitter's reach has its clock let free,
    and so has the flight handing over to it there, and they are refitted. `terminal` is the shuttle's terminal velocity (by default the first
    flight's)."""
    terminal = flights[0].fit.terminal_velocity if terminal is None and flights else terminal
    links = list(links)
    reaches = [None if reach is None or not _in_sight(seen, reach) else reach for seen, reach in zip(flights, reaches or [None] * len(flights))]
    clocked: set[int] = set()
    while True:
        evident = {i for i in clocked if reaches[i] is not None}  # a let-go reach is no evidence
        clocked = evident | {i for i, _ in links if i in clocked and i + 1 in evident}
        fits, meetings, beyond = _fit_together(flights, links, jitter_px, terminal, reaches, clocked)
        worse = {i for i, (seen, fit) in enumerate(zip(flights, fits)) if fit is None or _worse(seen, fit, jitter_px)}
        held_worse = {i for i in worse if reaches[i] is not None}
        if held_worse - clocked:  # its clock let free first: the flight model may be what is wrong
            clocked |= held_worse | {i for i, _ in links if i + 1 in held_worse}
            continue
        if held_worse:
            reaches = [None if i in worse else reach for i, reach in enumerate(reaches)]
            continue
        kept = [(i, t) for i, t in links if i not in worse and i + 1 not in worse]
        if len(kept) != len(links):
            links = kept
            continue
        short = {i for i, sigmas in beyond.items() if sigmas > CLOCK_EVIDENCE_SIGMAS}
        short |= {i for i, _ in links if i + 1 in short}  # the flight that brought the shuttle there ends at it
        if short <= clocked:
            break
        clocked |= short
    held = {k for i, _ in links for k in (i, i + 1)} | {i for i, reach in enumerate(reaches) if reach is not None}
    chosen = [fit if i in held and fit is not None else seen.fit for i, (seen, fit) in enumerate(zip(flights, fits))]
    return RallyFit(chosen, links, {i: meetings[i] for i, _ in links if i in meetings}, reaches)


def _noise(seen: Seen) -> np.ndarray:
    """Each detection's noise, as a multiple of the flight's later ones'."""
    after = np.maximum(np.asarray(seen.times_ms, dtype=float) - seen.fit.start_ms, 0.0)
    return 1.0 + EARLY_EXCESS * np.exp(-after / EARLY_MS)


def _worse(seen: Seen, fit: FlightFit, jitter_px: float) -> bool:
    """Whether a fit misses the flight's detections clearly more than the flight's own fit, each
    detection weighed by its own noise: a racket's worth of misses just after the hit is no sign
    of a wrong hand-over or hitter."""

    def misfit(of: FlightFit) -> tuple[float, int]:
        misses = np.hypot(*(_Views(seen.cameras).project(of.positions_at(seen.times_ms)[None])[0] - np.asarray(seen.pixels, dtype=float)).T) / _noise(seen)
        inliers = misses <= INLIER_SIGMAS * 2 * jitter_px
        return (float(np.sqrt(np.mean(misses[inliers] ** 2))) if inliers.any() else np.inf), int(inliers.sum())

    (rms, inliers), (own_rms, own_inliers) = misfit(fit), misfit(seen.fit)
    return inliers < WORSE_INLIERS * own_inliers or rms > WORSE_RMS * own_rms + 0.5


def _in_sight(seen: Seen, reach: Reach) -> bool:
    """Whether the flight's contact can be within the reach: whether the line of sight through
    its own contact passes, between the floor and SIGHT_TOP_M, within the reach (and three
    standard deviations) of where the hitter stood."""
    camera = seen.cameras[0]
    centre = np.asarray(camera.centre, dtype=float)
    way = seen.fit.position_at(max(reach.time_ms, seen.fit.start_ms)) - centre
    if abs(way[2]) < 1e-9:
        lo, hi = 0.0, 1e3
    else:
        lo, hi = sorted(((0.0 - centre[2]) / way[2], (SIGHT_TOP_M - centre[2]) / way[2]))
    lo, hi = max(lo, 0.0), min(max(hi, 0.0), 1e3)
    a, b = centre[:2] + lo * way[:2], centre[:2] + hi * way[:2]
    along = b - a
    share = float(np.clip((reach.point - a) @ along / max(float(along @ along), 1e-12), 0.0, 1.0))
    missed = float(np.linalg.norm(a + share * along - reach.point))
    sigma = float(np.sqrt(np.linalg.eigvalsh(reach.covariance)[-1] + reach.sigma**2 + REACH_FLOOR_M**2))
    return missed <= reach.radius + 3.0 * sigma


def _fit_together(
    flights: Sequence[Seen], links: Sequence[tuple[int, float]], jitter_px: float, terminal: float, reaches: Sequence[Reach | None],
    clocked: set[int] = frozenset(),
):
    """The joint fit: each flight's fit (None for one that lost its detections), the clocks of
    those in `clocked` fitted too; per link (by its flight i), the meeting time and its standard
    deviation; and per flight with a reach, how far beyond it its contact is, in standard
    deviations."""
    n, m = len(flights), len(links)
    if n == 0:
        return [], {}, {}
    x = np.array([np.concatenate([f.fit.p0, f.fit.v0, [f.fit.clock]]) for f in flights], dtype=float)  # per flight: p0, v0, clock
    measured = np.array([max(t, flights[i + 1].fit.start_ms) for i, t in links], dtype=float)
    tau = measured.copy()
    starts = [f.fit.start_ms for f in flights]
    views = [_Views(f.cameras) for f in flights]
    observed = [np.asarray(f.pixels, dtype=float).ravel() for f in flights]
    noise = [np.repeat(_noise(f), 2) for f in flights]
    ends = {i: k for k, (i, _) in enumerate(links)}  # flight i ends at link k
    begins = {i + 1: k for k, (i, _) in enumerate(links)}  # flight i + 1 begins at link k
    f_scale = 2 * jitter_px

    def evaluate(i: int, params: np.ndarray, tau: np.ndarray):
        """For B parameter rows of flight i: pixel residuals (B, 2N) in units of each detection's
        noise (the rest's, more just after the hit), and per link it takes part in
        its position and velocity (m/ms) at the meeting time; keyed None, its position at its
        hitter's hit, for a flight with a reach that begins at no link."""
        handovers = [k for k in (ends.get(i), begins.get(i)) if k is not None]
        extra = [t for k in handovers for t in (tau[k], tau[k] + VELOCITY_STEP_MS)]
        if reaches[i] is not None and i not in begins:  # one that begins at a hand-over is hit at the meeting
            extra.append(max(reaches[i].time_ms, starts[i]))
        sample_ms = np.concatenate([np.asarray(flights[i].times_ms, dtype=float), extra])
        relative = (sample_ms - starts[i]) / 1000.0
        order = np.argsort(relative)
        positions = np.empty((len(params), len(relative), 3))
        with np.errstate(all="ignore"):
            positions[:, order] = simulate_grid(params[:, :3], params[:, 3:6], np.maximum(relative[order], 0.0), terminal, gravity=GRAVITY * params[:, 6] ** 2)
        count = len(flights[i].times_ms)
        pixels = (views[i].project(positions[:, :count]).reshape(len(params), -1) - observed[i]) / noise[i]
        meet = {}
        for j, k in enumerate(handovers):
            at, later = positions[:, count + 2 * j], positions[:, count + 2 * j + 1]
            meet[k] = (at, (later - at) / VELOCITY_STEP_MS)
        if reaches[i] is not None and i not in begins:
            meet[None] = (positions[:, -1], np.zeros_like(positions[:, -1]))
        return np.where(np.isfinite(pixels), pixels, 1e6), meet

    def linearise(x: np.ndarray, tau: np.ndarray):
        blocks = []
        for i in range(n):
            steps = STEP * (1.0 + np.abs(x[i]))
            pixels, meet = evaluate(i, np.vstack([x[i], x[i] + np.diag(steps)]), tau)
            on = np.ones(7) if i in clocked else np.r_[np.ones(6), 0.0]  # the clip's clock: no say in the fit
            jac = ((pixels[1:] - pixels[0]) / steps[:, None]).T * on
            meet = {k: (at[0], ((at[1:] - at[0]) / steps[:, None]).T * on, velocity[0]) for k, (at, velocity) in meet.items()}
            blocks.append((pixels[0], jac, meet))
        return blocks

    def handover_terms(blocks, tau):
        """Per link: the gap (3,), its Jacobian on flight i, on flight i + 1, and on the meeting time."""
        terms = []
        for k, (i, _) in enumerate(links):
            a_at, a_jac, a_vel = blocks[i][2][k]
            b_at, b_jac, b_vel = blocks[i + 1][2][k]
            terms.append((i, a_at - b_at, a_jac, -b_jac, a_vel - b_vel))
        return terms

    def reach_terms(blocks, spread: bool = False):
        """Per flight with a reach: how far beyond it its contact is, in standard deviations, and
        that on the flight's parameters and on its meeting time (when it begins at one). With
        `spread`, for the uncertainty: a contact held at the edge is anywhere within the reach,
        so is placed along the way to it no surer than a point anywhere across it."""
        terms = []
        for i, reach in enumerate(reaches):
            if reach is None:
                continue
            k = begins.get(i)
            at, jac, velocity = blocks[i][2][k]
            apart = at[:2] - reach.point
            distance = max(float(np.linalg.norm(apart)), 1e-9)
            towards = apart / distance
            sigma = float(np.sqrt(towards @ reach.covariance @ towards + reach.sigma**2 + REACH_FLOOR_M**2))
            if spread:
                sigma = float(np.hypot(sigma, reach.radius / np.sqrt(3.0)))
            beyond = max(distance - reach.radius, 0.0) / sigma
            on = 1.0 if beyond > 0.0 else 0.0
            terms.append((i, k, beyond, on * (towards @ jac[:2]) / sigma, on * float(towards @ velocity[:2]) / sigma))
        return terms

    def cost(x: np.ndarray, blocks, tau) -> float:
        total = sum(float(x[i, 6] - 1.0) ** 2 for i in clocked) / CLOCK_SIGMA**2
        for residual, _, _ in blocks:
            miss2 = np.sum(residual.reshape(-1, 2) ** 2, axis=1)
            total += float(np.sum(2 * f_scale**2 * (np.sqrt(1 + miss2 / f_scale**2) - 1))) / jitter_px**2
        total += sum(float(gap @ gap) for _, gap, *_ in handover_terms(blocks, tau)) / CONTACT_SIGMA_M**2
        total += sum(beyond**2 for _, _, beyond, _, _ in reach_terms(blocks))
        return total + float(np.sum((tau - measured) ** 2)) / HANDOVER_SIGMA_MS**2

    size = 7 * n + m

    def normal_equations(x: np.ndarray, blocks, tau, pixels: bool = True):  # without the pixels: for the uncertainty
        hessian = np.zeros((size, size))
        gradient = np.zeros(size)
        if pixels:
            for i, (residual, jac, _) in enumerate(blocks):
                miss2 = np.repeat(np.sum(residual.reshape(-1, 2) ** 2, axis=1), 2)
                weight = 1.0 / np.sqrt(1.0 + miss2 / f_scale**2) / jitter_px**2
                s = slice(7 * i, 7 * i + 7)
                hessian[s, s] += jac.T @ (weight[:, None] * jac)
                gradient[s] += jac.T @ (weight * residual)
        for i in range(n):
            free = i in clocked
            hessian[7 * i + 6, 7 * i + 6] += 1.0 / CLOCK_SIGMA**2 if free else 1.0
            gradient[7 * i + 6] += (x[i, 6] - 1.0) / CLOCK_SIGMA**2 if free else 0.0
        for k, (i, gap, a_jac, b_jac, rate) in enumerate(handover_terms(blocks, tau)):
            row = np.zeros((3, size))
            row[:, 7 * i : 7 * i + 7] = a_jac
            row[:, 7 * i + 7 : 7 * i + 14] = b_jac
            row[:, 7 * n + k] = rate
            hessian += row.T @ row / CONTACT_SIGMA_M**2
            gradient += row.T @ gap / CONTACT_SIGMA_M**2
            hessian[7 * n + k, 7 * n + k] += 1.0 / HANDOVER_SIGMA_MS**2
            gradient[7 * n + k] += (tau[k] - measured[k]) / HANDOVER_SIGMA_MS**2
        for i, k, beyond, on_flight, on_time in reach_terms(blocks, spread=not pixels):
            row = np.zeros(size)
            row[7 * i : 7 * i + 7] = on_flight
            if k is not None:
                row[7 * n + k] = on_time
            hessian += np.outer(row, row)
            gradient += row * beyond
        return hessian, gradient

    blocks = linearise(x, tau)
    current = cost(x, blocks, tau)
    damping = 1e-3
    for _ in range(ITERATIONS):
        hessian, gradient = normal_equations(x, blocks, tau)
        system = hessian + damping * np.diag(np.diag(hessian) + 1e-9)
        try:
            step = -np.linalg.solve(system, gradient)
        except np.linalg.LinAlgError:
            break
        trial_x, trial_tau = x + step[: 7 * n].reshape(n, 7), tau + step[7 * n :]
        trial_tau = np.array([max(t, starts[i + 1]) for t, (i, _) in zip(trial_tau, links)])
        trial_blocks = linearise(trial_x, trial_tau)
        trial_cost = cost(trial_x, trial_blocks, trial_tau)
        if trial_cost < current:
            settled = current - trial_cost <= 1e-6 * (1 + current)
            x, tau, blocks, current, damping = trial_x, trial_tau, trial_blocks, trial_cost, damping / 3
            if settled:
                break
        else:
            damping *= 4
            if damping > 1e6:
                break

    # Uncertainty: the joint information, the pixel part scaled by the measured residual variance.
    misses = [np.hypot(*residual.reshape(-1, 2).T) for residual, _, _ in blocks]  # in units of each detection's noise
    inliers = [miss <= INLIER_SIGMAS * 2 * jitter_px for miss in misses]
    lost = {i for i, k in enumerate(inliers) if k.sum() < 3}  # flights the hand-overs pulled off their detections
    rows = sum(2 * int(k.sum()) for k in inliers)
    variance = max(sum(float(np.sum(miss[k] ** 2)) for miss, k in zip(misses, inliers)) / max(1, rows - 6 * n - len(clocked)), 1e-6)
    information, _ = normal_equations(x, blocks, tau, pixels=False)
    for i, (_, jac, _) in enumerate(blocks):
        keep = np.repeat(inliers[i], 2)
        information[7 * i : 7 * i + 7, 7 * i : 7 * i + 7] += jac[keep].T @ jac[keep] / variance
    covariance = COVARIANCE_INFLATION * np.linalg.pinv(information)
    fits = [
        None if i in lost else FlightFit(
            starts[i], x[i, :3].copy(), x[i, 3:6].copy(), terminal, covariance[7 * i : 7 * i + (7 if i in clocked else 6), 7 * i : 7 * i + (7 if i in clocked else 6)],
            float(np.sqrt(np.mean((misses[i] * noise[i][::2])[inliers[i]] ** 2))), int(inliers[i].sum()), len(misses[i]), float(x[i, 6]),
        )
        for i in range(n)
    ]
    meetings = {i: (float(tau[k]), float(np.sqrt(max(covariance[7 * n + k, 7 * n + k], 0.0)))) for k, (i, _) in enumerate(links)}
    return fits, meetings, {i: beyond for i, _, beyond, _, _ in reach_terms(blocks)}
