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
from scipy.optimize import least_squares
from scipy.stats import chi2

from geometry.camera import Camera
from geometry.shuttle_physics import FEATHER_TERMINAL_VELOCITY, GRAVITY, STEP_S

MIN_DETECTIONS = 5
SEED_HEIGHTS_M = (0.4, 1.2, 2.2, 3.2)  # where along the first ray the flight may start (by height)
SEED_DISTANCES_M = (4.0, 8.0, 13.0, 20.0)  # and by distance, for rays close to horizontal
COURT_BOX = ((-3.0, 8.2), (-4.0, 17.4), (0.0, 12.0))  # generous: where a flight can be at all
INLIER_SIGMAS = 3.0
FIT_STEP_S = 0.02  # RK4 step inside the fit: within 2 mm of a fine step even for smashes
FULL_FIT_SEEDS = 2  # seeds whose velocity-only fit is best go on to the full fit
SEED_SIGHTINGS = 3  # start seeds come from the first few sightings' rays (the first may be a stray)
# The fit's covariance, scaled so its 95 % region holds the truth 95 % of the time: on 192 synthetic
# flights (three views, 2 px jitter) it held 89.6 % unscaled and 96.4 % with variance x 1.5.
COVARIANCE_INFLATION = 1.5


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


def fit_flight(
    times_ms: Sequence[float],
    pixels: np.ndarray,
    cameras: Sequence[Camera],
    start_ms: float | None = None,
    terminal_velocity: float = FEATHER_TERMINAL_VELOCITY,
    jitter_px: float = 2.0,
) -> FlightFit | None:
    """The physical flight that best explains the detections (time, pixel, camera per detection).
    The flight is parameterised at start_ms (the hit, if known; else the first detection).
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
            positions = simulate_batch(params[:, :3], params[:, 3:6], relative, terminal_velocity, FIT_STEP_S)
            values = views.project(positions).reshape(len(params), -1) - observed
        return np.where(np.isfinite(values), values, 1e6)

    def fun(x: np.ndarray) -> np.ndarray:
        return residuals_batch(x[None])[0]

    def jac(x: np.ndarray) -> np.ndarray:
        steps = 1e-6 * (1.0 + np.abs(x))
        batch = np.repeat(x[None], len(x) + 1, axis=0)
        batch[1:] += np.diag(steps)
        values = residuals_batch(batch)
        return ((values[1:] - values[0]) / steps[:, None]).T

    duration = max(relative[-1], 1e-3)
    # Velocity first, with the start held at each seed; the best few then fit everything.
    staged = []
    seeds = [seed for i in range(min(SEED_SIGHTINGS, len(pixels))) for seed in _seeds(cameras[i], pixels[i])]
    for p_seed in seeds:
        last_ray = cameras[-1].pixel_to_plane(pixels[-1][None], p_seed[2])[0]
        towards = (np.array([last_ray[0], last_ray[1], p_seed[2]]) - p_seed) / duration if np.isfinite(last_ray).all() else np.zeros(3)
        v_seed = towards + np.array([0.0, 0.0, 0.5 * GRAVITY * duration])
        try:
            velocity = least_squares(
                lambda v, p=p_seed: fun(np.concatenate([p, v])), v_seed, jac=lambda v, p=p_seed: jac(np.concatenate([p, v]))[:, 3:],
                loss="soft_l1", f_scale=2 * jitter_px, max_nfev=12, ftol=1e-6, xtol=1e-6,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(velocity.cost):
            staged.append((velocity.cost, np.concatenate([p_seed, velocity.x])))
    best = None
    for _, start_params in sorted(staged, key=lambda item: item[0])[:FULL_FIT_SEEDS]:
        try:
            result = least_squares(fun, start_params, jac=jac, loss="soft_l1", f_scale=2 * jitter_px, max_nfev=60, ftol=1e-6, xtol=1e-6)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        return None
    error = np.hypot(*best.fun.reshape(-1, 2).T)
    inlier = error <= INLIER_SIGMAS * 2 * jitter_px
    if inlier.sum() < MIN_DETECTIONS:
        return None
    rows = np.repeat(inlier, 2)
    jacobian = jac(best.x)[rows]
    dof = max(1, rows.sum() - 6)
    variance = float(np.sum(best.fun[rows] ** 2) / dof)
    covariance = COVARIANCE_INFLATION * variance * np.linalg.pinv(jacobian.T @ jacobian)
    return FlightFit(start, best.x[:3].copy(), best.x[3:].copy(), terminal_velocity, covariance, float(np.sqrt(np.mean(error[inlier] ** 2))), int(inlier.sum()), len(error))
