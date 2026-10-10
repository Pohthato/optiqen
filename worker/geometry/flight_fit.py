# worker/geometry/flight_fit.py
"""One shuttle flight in 3D from its pixels in a video with a known camera for every frame.

The shuttle between two hits obeys dv/dt = -g e_z - (g / v_t^2) |v| v (geometry.shuttle_physics).
Its launch point and velocity (6 numbers) are fitted so that the simulated flight, seen through
each frame's camera, lands on the detections. One camera cannot see depth directly, but gravity
does: the same pixel path far away is a bigger, faster flight whose fall in pixels is slower, so
the fit is started from several depths along the first sighting's ray and the best one kept. The
loss is robust (a stray detection cannot drag the flight) and the fit's covariance gives an
uncertainty for anything derived from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from geometry.camera import Camera
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, GRAVITY, STEP_S

MIN_DETECTIONS = 5
SEED_HEIGHTS_M = (0.4, 1.2, 2.2, 3.2)  # where along the first ray the flight may start (by height)
SEED_DISTANCES_M = (4.0, 8.0, 13.0, 20.0)  # and by distance, for rays close to horizontal
COURT_BOX = ((-3.0, 8.2), (-4.0, 17.4), (0.0, 12.0))  # generous: where a flight can be at all
INLIER_SIGMAS = 3.0
FIT_STEP_S = 0.02  # RK4 step inside the fit: within 2 mm of a fine step even for smashes
PRUNE_AFTER = 15  # iterations before only the best few starts keep going
KEEP_STARTS = 5
SEED_SIGHTINGS = 3  # start seeds come from the first few sightings' rays (the first may be a stray)
# The fit's covariance, scaled so its 95 % region holds the truth 95 % of the time: on 192 synthetic
# flights (three views, 2 px jitter) it held 89.6 % unscaled and 96.4 % with variance x 1.5.
COVARIANCE_INFLATION = 1.5
WARM_OK_SIGMAS = 2.5  # a warm start whose median miss is within this many jitters needs no depth search


def simulate_batch(
    p0: np.ndarray, v0: np.ndarray, times_s: np.ndarray, terminal_velocity: float = FEATHER_TERMINAL_VELOCITY, step_s: float = STEP_S
) -> np.ndarray:
    """Positions (B, T, 3) of B launches at sorted, non-negative times (T,): RK4 in fixed steps,
    each launch stepped together, landing exactly on the requested times."""
    p = np.array(p0, dtype=np.float64).reshape(-1, 3)
    v = np.array(v0, dtype=np.float64).reshape(-1, 3)
    k = GRAVITY / terminal_velocity**2
    g = np.array([0.0, 0.0, -GRAVITY])

    def acceleration(vel: np.ndarray) -> np.ndarray:
        return g - k * np.sqrt(np.einsum("ij,ij->i", vel, vel))[:, None] * vel

    out = np.empty((len(p), len(times_s), 3))
    t = 0.0
    for index, target in enumerate(np.asarray(times_s, dtype=np.float64)):
        while t < target - 1e-12:
            h = min(step_s, target - t)
            a1 = acceleration(v)
            v2 = v + 0.5 * h * a1
            a2 = acceleration(v2)
            v3 = v + 0.5 * h * a2
            a3 = acceleration(v3)
            v4 = v + h * a3
            a4 = acceleration(v4)
            p = p + h / 6 * (v + 2 * v2 + 2 * v3 + v4)
            v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            t += h
        out[:, index] = p
    return out


def simulate_grid(
    p0: np.ndarray, v0: np.ndarray, times_s: np.ndarray, terminal_velocity: float = FEATHER_TERMINAL_VELOCITY, step_s: float = FIT_STEP_S
) -> np.ndarray:
    """Like simulate_batch, but faster: RK4 on a uniform grid, positions at the requested times
    by cubic Hermite interpolation from the grid's positions and velocities (within a millimetre
    of the exact path at 20 ms steps). Coordinates are stepped as (3, B) rows: fewer, larger
    array operations per step."""
    p = np.array(p0, dtype=np.float64).reshape(-1, 3).T.copy()
    v = np.array(v0, dtype=np.float64).reshape(-1, 3).T.copy()
    times_s = np.asarray(times_s, dtype=np.float64)
    k = GRAVITY / terminal_velocity**2
    steps = max(1, int(np.ceil(times_s.max() / step_s))) if len(times_s) else 1
    grid_p = np.empty((steps + 1, 3, p.shape[1]))
    grid_v = np.empty_like(grid_p)
    grid_p[0], grid_v[0] = p, v
    half, sixth = 0.5 * step_s, step_s / 6.0

    def acceleration(vel: np.ndarray) -> np.ndarray:
        a = vel * (-k * np.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]))
        a[2] -= GRAVITY
        return a

    for step in range(steps):
        a1 = acceleration(v)
        v2 = v + half * a1
        a2 = acceleration(v2)
        v3 = v + half * a2
        a3 = acceleration(v3)
        v4 = v + step_s * a3
        a4 = acceleration(v4)
        p = p + sixth * (v + 2 * (v2 + v3) + v4)
        v = v + sixth * (a1 + 2 * (a2 + a3) + a4)
        grid_p[step + 1], grid_v[step + 1] = p, v
    node = np.clip((times_s // step_s).astype(int), 0, steps - 1)
    u = ((times_s - node * step_s) / step_s)[:, None, None]
    p_a, p_b = grid_p[node], grid_p[node + 1]
    m_a, m_b = grid_v[node] * step_s, grid_v[node + 1] * step_s
    out = (2 * u**3 - 3 * u**2 + 1) * p_a + (u**3 - 2 * u**2 + u) * m_a + (-2 * u**3 + 3 * u**2) * p_b + (u**3 - u**2) * m_b
    return out.transpose(2, 0, 1)  # (B, T, 3)


class _Views:
    """The per-detection cameras as arrays, to project many trajectories at once."""

    def __init__(self, cameras: Sequence[Camera]):
        self.rotation = np.stack([camera.rotation for camera in cameras])
        self.tvec = np.stack([np.asarray(camera.tvec, dtype=np.float64).reshape(3) for camera in cameras])
        self.focal = np.array([camera.focal_px for camera in cameras])
        self.centre = np.array([[camera.cx, camera.cy] for camera in cameras])
        self.k1 = np.array([camera.k1 for camera in cameras])

    def project(self, points: np.ndarray) -> np.ndarray:
        """(B, N, 3) world points, point n through camera n -> (B, N, 2) pixels; far off when behind."""
        cam = np.einsum("nij,bnj->bni", self.rotation, points) + self.tvec
        depth = np.where(cam[..., 2] > 1e-3, cam[..., 2], 1e-3)
        xy = cam[..., :2] / depth[..., None]
        scale = 1.0 + self.k1[None, :, None] * np.sum(xy * xy, axis=-1, keepdims=True)
        pixels = self.focal[None, :, None] * xy * scale + self.centre[None]
        return np.where((cam[..., 2] > 1e-3)[..., None], pixels, 1e6)


@dataclass(frozen=True)
class FlightFit:
    start_ms: float
    p0: np.ndarray  # position at start_ms, court metres
    v0: np.ndarray  # velocity at start_ms, m/s
    terminal_velocity: float
    covariance: np.ndarray  # of (p0, v0), 6 x 6
    rms_px: float  # of the inliers
    inliers: int
    detections: int

    def positions_at(self, times_ms: Sequence[float]) -> np.ndarray:
        times = (np.asarray(times_ms, dtype=float) - self.start_ms) / 1000.0
        order = np.argsort(times)
        out = np.empty((len(times), 3))
        out[order] = simulate_batch(self.p0, self.v0, np.maximum(times[order], 0.0), self.terminal_velocity)[0]
        return out

    def position_at(self, time_ms: float) -> np.ndarray:
        return self.positions_at([time_ms])[0]

    def position_covariance(self, time_ms: float) -> np.ndarray:
        """3 x 3 covariance of the position at time_ms, propagated from the fit."""
        params = np.concatenate([self.p0, self.v0])
        steps = 1e-5 * (1.0 + np.abs(params))
        batch = np.repeat(params[None], 7, axis=0)
        batch[1:] += np.diag(steps)
        t = np.array([max(0.0, (time_ms - self.start_ms) / 1000.0)])
        positions = simulate_batch(batch[:, :3], batch[:, 3:], t, self.terminal_velocity)[:, 0]
        jacobian = ((positions[1:] - positions[0]) / steps[:, None]).T
        return jacobian @ self.covariance @ jacobian.T

    def within(self, time_ms: float, point: np.ndarray, confidence: float = 0.95) -> bool:
        """Whether `point` lies inside the fit's confidence region for the position at time_ms."""
        delta = np.asarray(point, dtype=float) - self.position_at(time_ms)
        distance2 = float(delta @ np.linalg.pinv(self.position_covariance(time_ms)) @ delta)
        return distance2 <= chi2.ppf(confidence, 3)


def _inside_box(point: np.ndarray) -> bool:
    return all(low <= value <= high for value, (low, high) in zip(point, COURT_BOX))


def _seeds(camera: Camera, pixel: np.ndarray) -> list[np.ndarray]:
    """Plausible start points along the ray of the first sighting."""
    seeds = []
    for height in SEED_HEIGHTS_M:
        xy = camera.pixel_to_plane(pixel[None], height)[0]
        if np.isfinite(xy).all():
            seeds.append(np.array([xy[0], xy[1], height]))
    near = camera.pixel_to_plane(pixel[None], camera.centre[2] - 1.0)[0]
    if np.isfinite(near).all():
        direction = np.array([near[0], near[1], camera.centre[2] - 1.0]) - camera.centre
    else:
        far = camera.pixel_to_plane(pixel[None], camera.centre[2] + 1.0)[0]
        direction = np.array([far[0], far[1], camera.centre[2] + 1.0]) - camera.centre
    direction /= np.linalg.norm(direction)
    seeds += [camera.centre + distance * direction for distance in SEED_DISTANCES_M]
    return [seed for seed in seeds if _inside_box(seed)]


def _robust_cost(residuals: np.ndarray, f_scale: float) -> np.ndarray:
    """Soft-L1 cost per problem of (S, 2N) residuals, on each detection's pixel miss."""
    miss2 = np.sum(residuals.reshape(len(residuals), -1, 2) ** 2, axis=2)
    return np.sum(2 * f_scale**2 * (np.sqrt(1 + miss2 / f_scale**2) - 1), axis=1)


def _jacobians(residuals_batch, x: np.ndarray) -> np.ndarray:
    """(S, n, m) forward-difference Jacobians of S problems, in one batched evaluation."""
    count, n = x.shape
    steps = 1e-6 * (1.0 + np.abs(x))
    batch = np.repeat(x[:, None, :], n + 1, axis=1)
    batch[:, 1:] += steps[:, :, None] * np.eye(n)[None]
    values = residuals_batch(batch.reshape(-1, n)).reshape(count, n + 1, -1)
    return (values[:, 1:] - values[:, :1]) / steps[:, :, None]


def _lm_batch(residuals_batch, x0: np.ndarray, f_scale: float, iterations: int = 40):
    """Levenberg-Marquardt on S problems at once (same data, different starts), with soft-L1
    weights on each detection's miss (re-weighted every step). After PRUNE_AFTER iterations only
    the KEEP_STARTS best starts continue. Returns (x, residuals, cost)."""
    x = np.array(x0, dtype=float)
    r = residuals_batch(x)
    cost = _robust_cost(r, f_scale)
    damping = np.full(len(x), 1e-2)
    settled = np.zeros(len(x), dtype=int)
    for iteration in range(iterations):
        if iteration == PRUNE_AFTER and len(x) > KEEP_STARTS:
            keep = np.argsort(cost)[:KEEP_STARTS]
            x, r, cost, damping, settled = x[keep], r[keep], cost[keep], damping[keep], settled[keep]
        jac = _jacobians(residuals_batch, x)
        miss2 = np.repeat(np.sum(r.reshape(len(r), -1, 2) ** 2, axis=2), 2, axis=1)
        weight = 1.0 / np.sqrt(1.0 + miss2 / f_scale**2)
        normal = np.einsum("snm,sm,skm->snk", jac, weight, jac)
        gradient = np.einsum("snm,sm,sm->sn", jac, weight, r)
        diagonal = np.einsum("snn->sn", normal)
        system = normal + (damping[:, None] * diagonal + 1e-9)[:, :, None] * np.eye(x.shape[1])[None]
        try:
            step = -np.linalg.solve(system, gradient[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        trial = x + step
        trial_r = residuals_batch(trial)
        trial_cost = _robust_cost(trial_r, f_scale)
        better = trial_cost < cost
        improvement = np.where(better, cost - trial_cost, 0.0)
        x[better], r[better], cost[better] = trial[better], trial_r[better], trial_cost[better]
        damping = np.where(better, damping / 3, damping * 4)
        stalled = (improvement <= 1e-5 * (1 + cost)) | (damping > 1e6)
        settled = np.where(stalled, settled + 1, 0)
        if np.all(settled >= 3):  # three steps in a row without real progress: done
            break
    return x, r, cost


def _best(result) -> tuple[np.ndarray, np.ndarray]:
    x, r, cost = result
    k = int(np.argmin(cost))
    return x[k], r[k]


def fit_flight(
    times_ms: Sequence[float],
    pixels: np.ndarray,
    cameras: Sequence[Camera],
    start_ms: float | None = None,
    terminal_velocity: float = FEATHER_TERMINAL_VELOCITY,
    jitter_px: float = 2.0,
    initial: Sequence[np.ndarray] = (),
) -> FlightFit | None:
    """The physical flight that best explains the detections (time, pixel, camera per detection).
    The flight is parameterised at start_ms (the hit, if known; else the first detection).
    `initial` holds starting guesses (position and velocity at that start, 6 numbers each), e.g.
    from neighbouring fits; when one fits well the depth search is skipped.
    None when there are too few detections or no start reproduces them."""
    times_ms = np.asarray(times_ms, dtype=float)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(times_ms) < MIN_DETECTIONS:
        return None
    order = np.argsort(times_ms)
    times_ms, pixels, cameras = times_ms[order], pixels[order], [cameras[i] for i in order]
    start = float(times_ms[0] if start_ms is None else min(start_ms, times_ms[0]))
    relative = (times_ms - start) / 1000.0
    views = _Views(cameras)
    observed = pixels.ravel()

    def residuals_batch(params: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):  # absurd trial launches overflow; they just score badly
            positions = simulate_grid(params[:, :3], params[:, 3:6], relative, terminal_velocity)
            values = views.project(positions).reshape(len(params), -1) - observed
        return np.where(np.isfinite(values), values, 1e6)

    f_scale = 2 * jitter_px
    duration = max(relative[-1], 1e-3)
    best_x, best_r = None, None
    if len(initial):
        best_x, best_r = _best(_lm_batch(residuals_batch, np.array(initial, dtype=float), f_scale))
    warm_ok = best_r is not None and float(np.median(np.hypot(*best_r.reshape(-1, 2).T))) <= WARM_OK_SIGMAS * jitter_px
    if not warm_ok:
        starts = []
        for i in range(min(SEED_SIGHTINGS, len(pixels))):
            for p_seed in _seeds(cameras[i], pixels[i]):
                last_ray = cameras[-1].pixel_to_plane(pixels[-1][None], p_seed[2])[0]
                towards = (np.array([last_ray[0], last_ray[1], p_seed[2]]) - p_seed) / duration if np.isfinite(last_ray).all() else np.zeros(3)
                starts.append(np.concatenate([p_seed, towards + np.array([0.0, 0.0, 0.5 * GRAVITY * duration])]))
        if starts:
            x, r = _best(_lm_batch(residuals_batch, np.array(starts), f_scale))
            if best_r is None or _robust_cost(r[None], f_scale)[0] < _robust_cost(best_r[None], f_scale)[0]:
                best_x, best_r = x, r
    if best_x is None or not np.isfinite(best_r).all():
        return None
    error = np.hypot(*best_r.reshape(-1, 2).T)
    inlier = error <= INLIER_SIGMAS * 2 * jitter_px
    if inlier.sum() < MIN_DETECTIONS:
        return None
    rows = np.repeat(inlier, 2)
    jacobian = _jacobians(residuals_batch, best_x[None])[0].T[rows]
    dof = max(1, rows.sum() - 6)
    variance = float(np.sum(best_r[rows] ** 2) / dof)
    covariance = COVARIANCE_INFLATION * variance * np.linalg.pinv(jacobian.T @ jacobian)
    return FlightFit(start, best_x[:3].copy(), best_x[3:].copy(), terminal_velocity, covariance, float(np.sqrt(np.mean(error[inlier] ** 2))), int(inlier.sum()), len(error))
