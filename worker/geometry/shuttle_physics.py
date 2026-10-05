# worker/geometry/shuttle_physics.py
"""Shuttlecock flight: a projectile with quadratic air drag.

    dv/dt = -g e_z - (g / v_t^2) |v| v

v_t is the terminal velocity (about 6.7 m/s for a feather shuttle). Positions are court-frame
metres (z up). Integration is fixed-step RK4 on plain floats, fast enough to call inside an
optimiser (a 2 s flight is ~400 steps).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

GRAVITY = 9.81
FEATHER_TERMINAL_VELOCITY = 6.7
STEP_S = 0.005
LAUNCH_TOLERANCE_M = 0.01

Vector = tuple[float, float, float]


def _drag(terminal_velocity: float | None) -> float:
    return 0.0 if terminal_velocity is None else GRAVITY / terminal_velocity**2


def _acceleration(v: Vector, k: float) -> Vector:
    speed = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (-k * speed * v[0], -k * speed * v[1], -GRAVITY - k * speed * v[2])


def _rk4(p: Vector, v: Vector, h: float, k: float) -> tuple[Vector, Vector]:
    a1 = _acceleration(v, k)
    v2 = (v[0] + 0.5 * h * a1[0], v[1] + 0.5 * h * a1[1], v[2] + 0.5 * h * a1[2])
    a2 = _acceleration(v2, k)
    v3 = (v[0] + 0.5 * h * a2[0], v[1] + 0.5 * h * a2[1], v[2] + 0.5 * h * a2[2])
    a3 = _acceleration(v3, k)
    v4 = (v[0] + h * a3[0], v[1] + h * a3[1], v[2] + h * a3[2])
    a4 = _acceleration(v4, k)
    p_next = tuple(p[i] + h / 6 * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i]) for i in range(3))
    v_next = tuple(v[i] + h / 6 * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]) for i in range(3))
    return p_next, v_next  # type: ignore[return-value]


def _state(p0, v0) -> tuple[Vector, Vector]:
    p = tuple(float(c) for c in np.asarray(p0, dtype=np.float64).reshape(3))
    v = tuple(float(c) for c in np.asarray(v0, dtype=np.float64).reshape(3))
    return p, v  # type: ignore[return-value]


def simulate(p0, v0, times, terminal_velocity: float | None = FEATHER_TERMINAL_VELOCITY) -> np.ndarray:
    """Positions (N, 3) at non-negative, non-decreasing times after launch."""
    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(times_arr) and (times_arr[0] < 0 or np.any(np.diff(times_arr) < 0)):
        raise ValueError("times must be non-negative and non-decreasing")
    k = _drag(terminal_velocity)
    p, v = _state(p0, v0)
    t = 0.0
    out = np.empty((len(times_arr), 3))
    for index, target in enumerate(times_arr):
        while t < target - 1e-12:
            h = min(STEP_S, target - t)
            p, v = _rk4(p, v, h, k)
            t += h
        out[index] = p
    return out


def time_at_height(
    p0, v0, height: float, terminal_velocity: float | None = FEATHER_TERMINAL_VELOCITY, max_time: float = 6.0
) -> float | None:
    """First time the shuttle passes `height` while descending, or None within max_time."""
    k = _drag(terminal_velocity)
    p, v = _state(p0, v0)
    t = 0.0
    while t < max_time:
        p_next, v_next = _rk4(p, v, STEP_S, k)
        if p[2] > height >= p_next[2] and v_next[2] < 0:
            low, high = 0.0, STEP_S
            for _ in range(40):
                middle = (low + high) / 2
                if _rk4(p, v, middle, k)[0][2] > height:
                    low = middle
                else:
                    high = middle
            return t + high
        p, v, t = p_next, v_next, t + STEP_S
    return None


def solve_launch(p0, target, flight_time: float, terminal_velocity: float | None = FEATHER_TERMINAL_VELOCITY) -> np.ndarray:
    """Launch velocity that puts the shuttle at `target` exactly `flight_time` seconds after leaving p0."""
    if flight_time <= 0:
        raise ValueError("flight_time must be positive")
    start = np.asarray(p0, dtype=np.float64).reshape(3)
    goal = np.asarray(target, dtype=np.float64).reshape(3)
    guess = (goal - start) / flight_time + np.array([0.0, 0.0, 0.5 * GRAVITY * flight_time])
    result = least_squares(lambda v: simulate(start, v, [flight_time], terminal_velocity)[0] - goal, guess, method="lm")
    miss = float(np.max(np.abs(result.fun)))
    if miss > LAUNCH_TOLERANCE_M:
        raise ValueError(f"no launch reaches the target in {flight_time} s (miss {miss:.3f} m)")
    return result.x
