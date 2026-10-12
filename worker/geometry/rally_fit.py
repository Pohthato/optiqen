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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geometry.camera import Camera
from geometry.flight_fit import COVARIANCE_INFLATION, INLIER_SIGMAS, FlightFit, _Views, fit_flight, simulate_grid
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY

CONTACT_SIGMA_M = 0.05  # how closely two flights meet at a hit (the model shuttle is a point)
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
    flights: Sequence[Seen], links: Sequence[tuple[int, float]], jitter_px: float = 2.0, terminal: float | None = None
) -> RallyFit:
    """The flights refitted together. `links` holds (i, time_ms): flight i hands over to flight
    i + 1 at about that time. A hand-over that makes either flight fit its own detections clearly
    worse than alone (a missed hit between them, or a wrong one) is dropped and the rest refitted;
    a flight with no hand-over left keeps its own fit. `terminal` is the shuttle's terminal
    velocity (by default the first flight's)."""
    terminal = flights[0].fit.terminal_velocity if terminal is None and flights else terminal
    links = list(links)
    while True:
        fits, meetings = _fit_together(flights, links, jitter_px, terminal)
        worse = {
            i for i, (seen, fit) in enumerate(zip(flights, fits))
            if fit is None or fit.inliers < WORSE_INLIERS * seen.fit.inliers or fit.rms_px > WORSE_RMS * seen.fit.rms_px + 0.5
        }
        kept = [(i, t) for i, t in links if i not in worse and i + 1 not in worse]
        if len(kept) == len(links):
            break
        links = kept
    linked = {k for i, _ in links for k in (i, i + 1)}
    chosen = [fit if i in linked and fit is not None else seen.fit for i, (seen, fit) in enumerate(zip(flights, fits))]
    return RallyFit(chosen, links, {i: meetings[i] for i, _ in links if i in meetings})


def _fit_together(flights: Sequence[Seen], links: Sequence[tuple[int, float]], jitter_px: float, terminal: float):
    """The joint fit: each flight's fit (None for one that lost its detections) and, per link (by
    its flight i), the meeting time and its standard deviation."""
    n, m = len(flights), len(links)
    if n == 0:
        return [], {}
    x = np.array([np.concatenate([f.fit.p0, f.fit.v0]) for f in flights], dtype=float)
    measured = np.array([max(t, flights[i + 1].fit.start_ms) for i, t in links], dtype=float)
    tau = measured.copy()
    starts = [f.fit.start_ms for f in flights]
    views = [_Views(f.cameras) for f in flights]
    observed = [np.asarray(f.pixels, dtype=float).ravel() for f in flights]
    ends = {i: k for k, (i, _) in enumerate(links)}  # flight i ends at link k
    begins = {i + 1: k for k, (i, _) in enumerate(links)}  # flight i + 1 begins at link k
    f_scale = 2 * jitter_px

    def evaluate(i: int, params: np.ndarray, tau: np.ndarray):
        """For B parameter rows of flight i: pixel residuals (B, 2N), and per link it takes part in
        its position and velocity (m/ms) at the meeting time."""
        handovers = [k for k in (ends.get(i), begins.get(i)) if k is not None]
        extra = [t for k in handovers for t in (tau[k], tau[k] + VELOCITY_STEP_MS)]
        sample_ms = np.concatenate([np.asarray(flights[i].times_ms, dtype=float), extra])
        relative = (sample_ms - starts[i]) / 1000.0
        order = np.argsort(relative)
        positions = np.empty((len(params), len(relative), 3))
        with np.errstate(all="ignore"):
            positions[:, order] = simulate_grid(params[:, :3], params[:, 3:], np.maximum(relative[order], 0.0), terminal)
        count = len(flights[i].times_ms)
        pixels = views[i].project(positions[:, :count]).reshape(len(params), -1) - observed[i]
        meet = {}
        for j, k in enumerate(handovers):
            at, later = positions[:, count + 2 * j], positions[:, count + 2 * j + 1]
            meet[k] = (at, (later - at) / VELOCITY_STEP_MS)
        return np.where(np.isfinite(pixels), pixels, 1e6), meet

    def linearise(x: np.ndarray, tau: np.ndarray):
        blocks = []
        for i in range(n):
            steps = STEP * (1.0 + np.abs(x[i]))
            pixels, meet = evaluate(i, np.vstack([x[i], x[i] + np.diag(steps)]), tau)
            jac = ((pixels[1:] - pixels[0]) / steps[:, None]).T
            meet = {k: (at[0], ((at[1:] - at[0]) / steps[:, None]).T, velocity[0]) for k, (at, velocity) in meet.items()}
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

    def cost(blocks, tau) -> float:
        total = 0.0
        for residual, _, _ in blocks:
            miss2 = np.sum(residual.reshape(-1, 2) ** 2, axis=1)
            total += float(np.sum(2 * f_scale**2 * (np.sqrt(1 + miss2 / f_scale**2) - 1))) / jitter_px**2
        total += sum(float(gap @ gap) for _, gap, *_ in handover_terms(blocks, tau)) / CONTACT_SIGMA_M**2
        return total + float(np.sum((tau - measured) ** 2)) / HANDOVER_SIGMA_MS**2

    size = 6 * n + m

    def normal_equations(blocks, tau, pixels: bool = True):
        hessian = np.zeros((size, size))
        gradient = np.zeros(size)
        if pixels:
            for i, (residual, jac, _) in enumerate(blocks):
                miss2 = np.repeat(np.sum(residual.reshape(-1, 2) ** 2, axis=1), 2)
                weight = 1.0 / np.sqrt(1.0 + miss2 / f_scale**2) / jitter_px**2
                s = slice(6 * i, 6 * i + 6)
                hessian[s, s] += jac.T @ (weight[:, None] * jac)
                gradient[s] += jac.T @ (weight * residual)
        for k, (i, gap, a_jac, b_jac, rate) in enumerate(handover_terms(blocks, tau)):
            row = np.zeros((3, size))
            row[:, 6 * i : 6 * i + 6] = a_jac
            row[:, 6 * i + 6 : 6 * i + 12] = b_jac
            row[:, 6 * n + k] = rate
            hessian += row.T @ row / CONTACT_SIGMA_M**2
            gradient += row.T @ gap / CONTACT_SIGMA_M**2
            hessian[6 * n + k, 6 * n + k] += 1.0 / HANDOVER_SIGMA_MS**2
            gradient[6 * n + k] += (tau[k] - measured[k]) / HANDOVER_SIGMA_MS**2
        return hessian, gradient

    blocks = linearise(x, tau)
    current = cost(blocks, tau)
    damping = 1e-3
    for _ in range(ITERATIONS):
        hessian, gradient = normal_equations(blocks, tau)
        system = hessian + damping * np.diag(np.diag(hessian) + 1e-9)
        try:
            step = -np.linalg.solve(system, gradient)
        except np.linalg.LinAlgError:
            break
        trial_x, trial_tau = x + step[: 6 * n].reshape(n, 6), tau + step[6 * n :]
        trial_tau = np.array([max(t, starts[i + 1]) for t, (i, _) in zip(trial_tau, links)])
        trial_blocks = linearise(trial_x, trial_tau)
        trial_cost = cost(trial_blocks, trial_tau)
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
    misses = [np.hypot(*residual.reshape(-1, 2).T) for residual, _, _ in blocks]
    inliers = [miss <= INLIER_SIGMAS * 2 * jitter_px for miss in misses]
    lost = {i for i, k in enumerate(inliers) if k.sum() < 3}  # flights the hand-overs pulled off their detections
    rows = sum(2 * int(k.sum()) for k in inliers)
    variance = max(sum(float(np.sum(miss[k] ** 2)) for miss, k in zip(misses, inliers)) / max(1, rows - 6 * n), 1e-6)
    information, _ = normal_equations(blocks, tau, pixels=False)
    for i, (_, jac, _) in enumerate(blocks):
        keep = np.repeat(inliers[i], 2)
        information[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] += jac[keep].T @ jac[keep] / variance
    covariance = COVARIANCE_INFLATION * np.linalg.pinv(information)
    fits = [
        None if i in lost else FlightFit(
            starts[i], x[i, :3].copy(), x[i, 3:].copy(), terminal, covariance[6 * i : 6 * i + 6, 6 * i : 6 * i + 6],
            float(np.sqrt(np.mean(misses[i][inliers[i]] ** 2))), int(inliers[i].sum()), len(misses[i]),
        )
        for i in range(n)
    ]
    meetings = {i: (float(tau[k]), float(np.sqrt(max(covariance[6 * n + k, 6 * n + k], 0.0)))) for k, (i, _) in enumerate(links)}
    return fits, meetings
