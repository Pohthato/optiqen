# worker/geometry/skeleton.py
"""A player's body in 3D, from their 2D joints in every frame and the camera of every frame.

One camera puts each joint somewhere along its line of sight. The body fixes where: bones keep
their length (the player's own, estimated over the clip), no foot goes below the floor and each
foot is on it unless clearly off it (which fixes how far along the line of sight the body is;
held to the floor one foot at a time, a few pixels of noise do not lift the body towards the
camera, as holding only the lower foot does), knees bend forward, the trunk is roughly upright
and the shoulders and hips roughly level. And a body moves smoothly: no joint faster than a
swinging wrist, the hips' middle speeding up, slowing down or turning no harder than a player
can, and a foot on the floor not sliding. That is what tells a jump from a body standing
further away: both look alike in one frame, but holding a jumping player to the floor would
slide both their feet a metre back and forth. And a planted foot, held still over its frames,
puts the body on the court more surely than any one frame.

The whole clip is fitted together: first each frame on its own (from the frame before), then
every frame at once. The constraints are firm; only what a detector or a learned model can get
wrong (a joint, the model's pose, whether a foot is down) is weighed robustly, so that one wild
value does not drag the body and a jump lifts it off the floor.

A learned 3D pose model can add what one camera lacks: how the body is posed in depth. Its
joints, relative to the middle of the hips and in the camera's axes, pull the fitted body's
own relative joints towards them (to PRIOR_SIGMA_M); the geometry still places the body on the
court, at its own scale and with its own bone lengths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu

from geometry.camera import Camera

# The joints fitted (COCO numbering): nose, shoulders, elbows, wrists, hips, knees, ankles.
JOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
SLOT = {joint: k for k, joint in enumerate(JOINTS)}
# Bones whose length a body keeps; a left and right pair share one length.
BONES = {
    "upper_arm": ((5, 7), (6, 8)), "forearm": ((7, 9), (8, 10)), "thigh": ((11, 13), (12, 14)), "shank": ((13, 15), (14, 16)),
    "shoulders": ((5, 6),), "hips": ((11, 12),), "flank": ((5, 11), (6, 12)), "diagonal": ((5, 12), (6, 11)), "neck": ((0, 5), (0, 6)),
}
PAIRS = [(name, SLOT[a], SLOT[b]) for name, pairs in BONES.items() for a, b in pairs]
# Starting lengths as fractions of stature (Drillis and Contini): limbs and widths as measured,
# the torso's sides and diagonals and the nose's distance to the shoulders from the same
# proportions (trunk 0.29 high, shoulders 0.13 and hips 0.09 either side, nose 0.05 ahead and
# 0.12 above the shoulders' middle), so that a body with these proportions fits exactly.
STARTING = {
    "upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shank": 0.246, "shoulders": 0.26, "hips": 0.18,
    "flank": float(np.hypot(0.29, 0.13 - 0.09)), "diagonal": float(np.hypot(0.29, 0.13 + 0.09)),
    "neck": float(np.sqrt(0.13**2 + 0.05**2 + 0.12**2)),
}
ANKLE_HEIGHT_M = 0.07
BONE_SIGMA_M = 0.02
FLOOR_SIGMA_M = 0.01  # no foot below the floor
DOWN_SIGMA_M = 0.02  # and each on it, unless the rest says otherwise (a step, a jump)
FLOOR_CAP = 4.5  # the most an ankle off the floor costs (squared standard deviations): further up, it is off it
MOVE_SIGMA_M_PER_S = 9.0  # how far a joint may move between frames, a priori (a wrist in a swing)
ACCEL_SIGMA_M_PER_S2 = 20.0  # how hard the hips' middle may speed up, slow down or turn (a lunge's stop)
LINK_GAP_S = 0.25  # frames further apart than this are not tied together
SLIDE_SIGMA_M_PER_S = 0.3  # how fast a foot on the floor may slide
CONTACT_M = 0.03  # an ankle this far above its floor height is still more likely down than up
KNEE_SIGMA_M = 0.02
LEAN_SIGMA_M = 0.25  # how far the shoulders' middle may be off above the hips' middle (about 30 degrees)
LEVEL_SIGMA_M = 0.10  # how far one shoulder (or hip) may be above the other
MIN_SCORE = 0.3
PRIOR_SIGMA_M = 0.08  # how closely a learned pose model's relative joints are trusted
ROBUST_SCALE = 2.0  # standard deviations beyond which a joint, the model's pose or a foot's slide counts less
MAX_STEPS = 40


@dataclass(frozen=True)
class Body:
    joints: np.ndarray  # (17, 3) court metres; NaN for the joints not fitted (eyes, ears)
    sigma: np.ndarray  # (17,) one standard deviation of each joint's position, metres
    rms_px: float  # how far the fitted joints miss the detected ones


class _Frame:
    """One frame's problem: its residuals and their analytic Jacobian over the joints' positions."""

    def __init__(self, camera: Camera, pixels, weights, lengths, previous, step_sigma, noise_px, prior=None):
        self.R = camera.rotation
        self.t = np.asarray(camera.tvec, dtype=float).reshape(3)
        self.f, self.cx, self.cy, self.k1 = camera.focal_px, camera.cx, camera.cy, camera.k1
        self.pixels, self.weights, self.lengths = pixels, weights, lengths
        self.previous, self.step_sigma, self.noise = previous, step_sigma, noise_px
        n = len(JOINTS)
        # A learned model's joints relative to the hips' middle, turned from the camera's axes into the court's.
        self.prior = None if prior is None else np.asarray(prior, dtype=float) @ self.R
        self.size = 2 * n + len(PAIRS) + 2 + 2 + 2 + 2 + 2 + (3 * n if previous is not None else 0) + (3 * n if prior is not None else 0)

    def __call__(self, x: np.ndarray):
        n = len(JOINTS)
        X = x.reshape(n, 3)
        r = np.zeros(self.size)
        J = np.zeros((self.size, 3 * n))
        row = 0
        # Reprojection, through the camera's one radial term.
        p = X @ self.R.T + self.t
        depth = np.where(p[:, 2] > 1e-3, p[:, 2], 1e-3)
        u, v = p[:, 0] / depth, p[:, 1] / depth
        s = 1.0 + self.k1 * (u * u + v * v)
        scale = self.weights / self.noise
        r[0 : 2 * n : 2] = (self.f * u * s + self.cx - self.pixels[:, 0]) * scale
        r[1 : 2 * n : 2] = (self.f * v * s + self.cy - self.pixels[:, 1]) * scale
        for k in range(n):
            du = np.array([1 / depth[k], 0.0, -u[k] / depth[k]])
            dv = np.array([0.0, 1 / depth[k], -v[k] / depth[k]])
            ds = self.k1 * (2 * u[k] * du + 2 * v[k] * dv)
            J[2 * k, 3 * k : 3 * k + 3] = self.f * (s[k] * du + u[k] * ds) @ self.R * scale[k]
            J[2 * k + 1, 3 * k : 3 * k + 3] = self.f * (s[k] * dv + v[k] * ds) @ self.R * scale[k]
        row = 2 * n
        # Bones.
        for name, a, b in PAIRS:
            span = X[a] - X[b]
            length = max(float(np.linalg.norm(span)), 1e-9)
            r[row] = (length - self.lengths[name]) / BONE_SIGMA_M
            J[row, 3 * a : 3 * a + 3] = span / length / BONE_SIGMA_M
            J[row, 3 * b : 3 * b + 3] = -span / length / BONE_SIGMA_M
            row += 1
        # The floor: no ankle below it, and each on it or clearly off it.
        ankles = (SLOT[15], SLOT[16])
        for a in ankles:
            gap = X[a, 2] - ANKLE_HEIGHT_M
            if gap < 0:
                r[row], J[row, 3 * a + 2] = gap / FLOOR_SIGMA_M, 1.0 / FLOOR_SIGMA_M
            row += 1
        for a in ankles:
            r[row], slope = _capped((X[a, 2] - ANKLE_HEIGHT_M) / DOWN_SIGMA_M)
            J[row, 3 * a + 2] = slope / DOWN_SIGMA_M
            row += 1
        # Knees bend forward (forward taken across the hips).
        across = X[SLOT[12]] - X[SLOT[11]]
        forward = np.array([-across[1], across[0], 0.0])
        forward /= max(np.linalg.norm(forward), 1e-9)
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
            h, k, a = SLOT[hip], SLOT[knee], SLOT[ankle]
            ahead = float((X[k] - (X[h] + X[a]) / 2) @ forward)
            if ahead < 0:
                r[row] = ahead / KNEE_SIGMA_M
                J[row, 3 * k : 3 * k + 3] = forward / KNEE_SIGMA_M
                J[row, 3 * h : 3 * h + 3] = J[row, 3 * a : 3 * a + 3] = -forward / (2 * KNEE_SIGMA_M)
            row += 1
        # The trunk roughly upright: the shoulders' middle above the hips' middle.
        shoulders, hips = (SLOT[5], SLOT[6]), (SLOT[11], SLOT[12])
        off = (X[shoulders[0]] + X[shoulders[1]] - X[hips[0]] - X[hips[1]]) / 2
        for axis in (0, 1):
            r[row] = off[axis] / LEAN_SIGMA_M
            for j, sign in ((shoulders[0], 1), (shoulders[1], 1), (hips[0], -1), (hips[1], -1)):
                J[row, 3 * j + axis] = sign * 0.5 / LEAN_SIGMA_M
            row += 1
        # Shoulders and hips roughly level.
        for a, b in (shoulders, hips):
            r[row] = (X[a, 2] - X[b, 2]) / LEVEL_SIGMA_M
            J[row, 3 * a + 2], J[row, 3 * b + 2] = 1.0 / LEVEL_SIGMA_M, -1.0 / LEVEL_SIGMA_M
            row += 1
        # Smoothly from the frame before.
        if self.previous is not None:
            r[row : row + 3 * n] = ((X - self.previous) / self.step_sigma).ravel()
            J[row : row + 3 * n, :] = np.eye(3 * n) / self.step_sigma
            row += 3 * n
        # Posed as the learned model sees it, about the hips' middle.
        if self.prior is not None:
            hips = (SLOT[11], SLOT[12])
            middle = (X[hips[0]] + X[hips[1]]) / 2
            r[row : row + 3 * n] = ((X - middle - self.prior) / PRIOR_SIGMA_M).ravel()
            block = np.eye(3 * n) / PRIOR_SIGMA_M
            for h in hips:
                for k in range(n):
                    block[3 * k : 3 * k + 3, 3 * h : 3 * h + 3] -= 0.5 * np.eye(3) / PRIOR_SIGMA_M
            J[row : row + 3 * n, :] = block
        robust = list(range(2 * n)) + (list(range(row, row + 3 * n)) if self.prior is not None else [])
        _soften(r, J, np.array(robust), ROBUST_SCALE)
        return r, J


def _capped(u: float) -> tuple[float, float]:
    """A residual whose square grows as u squared near 0 and levels off at FLOOR_CAP (on the
    floor, or off it: no pull once clearly up), and its slope."""
    k = np.exp(-FLOOR_CAP / 2) / (1 - np.exp(-FLOOR_CAP / 2))
    e = float(np.exp(-0.5 * u * u))
    cost = -2.0 * np.log((e + k) / (1 + k))
    if cost < 1e-12:
        return float(u / np.sqrt(1 + k)), float(1 / np.sqrt(1 + k))
    return float(np.sign(u) * np.sqrt(cost)), float(abs(u) * e / ((e + k) * np.sqrt(cost)))


def _soft(r: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft-L1 as residuals: one far beyond `scale` counts for about the square root of its size
    (a detector's wild joint, a jump off the floor), the rest as usual. The softened residuals
    and their slopes."""
    x = r / scale
    cost = 2.0 * (np.sqrt(1.0 + x * x) - 1.0)
    slope = np.ones_like(x)
    big = np.abs(x) > 1e-6
    slope[big] = np.abs(x[big]) / (np.sqrt(1.0 + x[big] ** 2) * np.sqrt(cost[big]))
    return scale * np.sign(x) * np.sqrt(cost), slope


def _soften(r: np.ndarray, J: np.ndarray, rows: np.ndarray, scale: float) -> None:
    """_soft on these rows of a residual vector and its dense Jacobian, in place."""
    r[rows], slope = _soft(r[rows], scale)
    J[rows] *= slope[:, None]


def _start(camera: Camera, pixels: np.ndarray, weights: np.ndarray, stature: float, facing: float) -> np.ndarray:
    """A standing body where the detected ankles meet the floor, facing `facing`."""
    from simulation.people import STANDING  # the standing template's proportions

    ankles = [pixels[SLOT[j]] for j in (15, 16) if weights[SLOT[j]] > 0]
    floor = camera.pixel_to_plane(np.mean(ankles, axis=0)[None] if ankles else np.array([[camera.cx, camera.cy]]), ANKLE_HEIGHT_M)[0]
    if not np.isfinite(floor).all():
        floor = np.array([2.6, 3.0])
    forward = np.array([np.cos(facing), np.sin(facing), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    template = STANDING[list(JOINTS)]
    return (np.array([floor[0], floor[1], 0.0]) + stature * (template[:, :1] * right + template[:, 1:2] * forward + template[:, 2:3] * np.array([0, 0, 1.0]))).ravel()


def _alone(problem: _Frame, x0: np.ndarray) -> np.ndarray:
    """One frame's body on its own (with the frame before, if any, in its problem)."""
    cache: dict = {}

    def residuals(x):
        cache["x"], (cache["r"], cache["J"]) = x.copy(), problem(x)
        return cache["r"]

    def jacobian(x):
        if "x" not in cache or not np.array_equal(cache["x"], x):
            residuals(x)
        return cache["J"]

    return least_squares(residuals, x0, jac=jacobian, max_nfev=100).x


def _links(times_s: np.ndarray) -> sparse.csr_matrix:
    """The residuals tying consecutive frames, linear in all the frames' joints: every joint moves
    no faster than MOVE_SIGMA allows, and the hips' middle speeds up, slows down or turns no
    harder than ACCEL_SIGMA allows."""
    n, size, count = len(JOINTS), 3 * len(JOINTS), len(times_s)
    rows, cols, values = [], [], []
    row = 0
    for i in range(1, count):
        dt = times_s[i] - times_s[i - 1]
        if dt > LINK_GAP_S:
            continue
        scale = 1.0 / (MOVE_SIGMA_M_PER_S * max(dt, 1e-3))
        for c in range(size):
            rows += [row, row]
            cols += [i * size + c, (i - 1) * size + c]
            values += [scale, -scale]
            row += 1
    hips = (SLOT[11], SLOT[12])
    for i in range(1, count - 1):
        before, after = times_s[i] - times_s[i - 1], times_s[i + 1] - times_s[i]
        if before > LINK_GAP_S or after > LINK_GAP_S:
            continue
        # The acceleration of the hips' middle m over three frames: 2 (v_after - v_before) / (before + after).
        span = before + after
        weights = {i + 1: 2 / (after * span), i: -2 / span * (1 / before + 1 / after), i - 1: 2 / (before * span)}
        for axis in range(3):
            for frame, weight in weights.items():
                for h in hips:
                    rows.append(row)
                    cols.append(frame * size + 3 * h + axis)
                    values.append(0.5 * weight / ACCEL_SIGMA_M_PER_S2)
            row += 1
    return sparse.csr_matrix((values, (rows, cols)), shape=(row, count * size))


def _contact(z: float) -> tuple[float, float]:
    """How surely an ankle at height z is on the floor (1 at its floor height or below, falling
    away above it), and its slope."""
    over = max(z - ANKLE_HEIGHT_M, 0.0)
    weight = float(np.exp(-0.5 * (over / CONTACT_M) ** 2))
    return weight, -over / CONTACT_M**2 * weight


def _slides(x: np.ndarray, times_s: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    """A foot on the floor does not slide: per ankle and pair of tied frames, its speed across the
    floor (to SLIDE_SIGMA), counted as much as the ankle is down in both frames. Softened (a
    foot can scrape) and with the Jacobian over all the frames' joints."""
    count, size = x.shape
    X = x.reshape(count, -1, 3)
    r, rows, cols, values = [], [], [], []
    for a in (SLOT[15], SLOT[16]):
        for j in range(1, count):
            i, dt = j - 1, times_s[j] - times_s[j - 1]
            if dt > LINK_GAP_S:
                continue
            (wi, si), (wj, sj) = _contact(X[i, a, 2]), _contact(X[j, a, 2])
            scale = 1.0 / (SLIDE_SIGMA_M_PER_S * max(dt, 1e-3))
            for axis in range(2):
                v = (X[j, a, axis] - X[i, a, axis]) * scale
                row = len(r)
                r.append(wi * wj * v)
                rows += [row] * 4
                cols += [j * size + 3 * a + axis, i * size + 3 * a + axis, j * size + 3 * a + 2, i * size + 3 * a + 2]
                values += [wi * wj * scale, -wi * wj * scale, v * wi * sj, v * wj * si]
    soft, slope = _soft(np.array(r, dtype=float), ROBUST_SCALE)
    J = sparse.csr_matrix((values, (rows, cols)), shape=(len(r), count * size))
    return soft, sparse.diags(slope) @ J


def _together(problems: Sequence[_Frame], x0: np.ndarray, links: sparse.csr_matrix, times_s: np.ndarray) -> tuple[np.ndarray, sparse.csc_matrix]:
    """Every frame's body at once (Levenberg-Marquardt on the sparse problem); the bodies and the
    problem's information matrix at them."""
    count, size = x0.shape

    def evaluate(x):
        parts = [problem(x[i]) for i, problem in enumerate(problems)]
        r_slides, J_slides = _slides(x, times_s)
        r = np.concatenate([part[0] for part in parts] + [links @ x.ravel(), r_slides])
        J = sparse.vstack([sparse.block_diag([part[1] for part in parts], format="csr"), links, J_slides], format="csr")
        return r, J

    x = x0.copy()
    r, J = evaluate(x)
    cost, damping = float(r @ r), 1e-3
    for _ in range(MAX_STEPS):
        information = (J.T @ J).tocsc()
        diagonal = information.diagonal() + 1e-9
        try:
            step = splu((information + sparse.diags(damping * diagonal)).tocsc()).solve(-(J.T @ r))
        except RuntimeError:
            damping *= 4
            continue
        candidate = x + step.reshape(count, size)
        r_new, J_new = evaluate(candidate)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            settled = cost - cost_new < 1e-7 * cost
            x, r, J, cost = candidate, r_new, J_new, cost_new
            damping = max(damping / 3, 1e-9)
            if settled:
                break
        else:
            damping *= 4
            if damping > 1e8:
                break
    return x, (J.T @ J).tocsc()


def _spreads(information: sparse.csc_matrix, count: int, size: int) -> np.ndarray:
    """Each frame's joints' standard deviations (metres, combined over x, y, z) from the
    information matrix: the diagonal of its inverse, a block of frames at a time."""
    lu = splu((information + sparse.identity(information.shape[0]) * 1e-9).tocsc())
    variances = np.zeros(count * size)
    block = max(1, 2000 // size)
    for first in range(0, count, block):
        last = min(first + block, count)
        unit = np.zeros((count * size, (last - first) * size))
        unit[first * size : last * size] = np.eye((last - first) * size)
        variances[first * size : last * size] = np.diag(lu.solve(unit)[first * size : last * size])
    return np.sqrt(np.maximum(variances.reshape(count, len(JOINTS), 3).sum(axis=2), 0.0))


def fit_bodies(
    frames: Sequence[tuple[np.ndarray, np.ndarray] | None], cameras: Sequence[Camera | None], times_ms: Sequence[float],
    facing: float, stature: float = 1.75, noise_px: float = 3.0, rounds: int = 2,
    priors: Sequence[np.ndarray | None] | None = None,
) -> list[Body | None]:
    """Each frame's body (None where the player or the camera is missing). `frames` holds the
    player's 17 COCO joints (pixels, scores) per frame; `facing` is the way they face the net
    (radians in the court plane) for the first frame. The bone lengths start from `stature` and
    are refitted from the clip's bodies `rounds` times. `priors`, per frame, holds a learned
    pose model's 13 joints (JOINTS order) relative to the middle of the hips, in metres along
    the camera's axes (x right, y down, z forward), or None."""
    lengths = {name: fraction * stature for name, fraction in STARTING.items()}
    used, inputs = [], []
    for index, (item, camera) in enumerate(zip(frames, cameras)):
        if item is None or camera is None:
            continue
        joints2d, scores = np.asarray(item[0], dtype=float), np.asarray(item[1], dtype=float)
        pixels = joints2d[list(JOINTS)]
        weights = np.where(scores[list(JOINTS)] >= MIN_SCORE, np.sqrt(np.clip(scores[list(JOINTS)], 0, 1)), 0.0)
        if (weights > 0).sum() >= 6:
            used.append(index)
            inputs.append((camera, pixels, weights, priors[index] if priors is not None else None))
    bodies: list[Body | None] = [None] * len(frames)
    if not used:
        return bodies
    times_s = np.array([times_ms[i] for i in used], dtype=float) / 1000.0

    # Each frame on its own first, from the frame before: where the clip's fit starts.
    x = np.zeros((len(used), 3 * len(JOINTS)))
    previous, previous_s = None, None
    for k, (camera, pixels, weights, prior) in enumerate(inputs):
        linked = previous is not None and times_s[k] - previous_s <= LINK_GAP_S
        step_sigma = MOVE_SIGMA_M_PER_S * max(times_s[k] - previous_s, 1e-3) if linked else 1.0
        problem = _Frame(camera, pixels, weights, lengths, previous if linked else None, step_sigma, noise_px, prior)
        start = previous.ravel() if linked else _start(camera, pixels, weights, stature, facing)
        x[k] = _alone(problem, start)
        previous, previous_s = x[k].reshape(-1, 3), times_s[k]

    links = _links(times_s)
    for round_ in range(rounds + 1):
        problems = [_Frame(camera, pixels, weights, lengths, None, 1.0, noise_px, prior) for camera, pixels, weights, prior in inputs]
        x, information = _together(problems, x, links, times_s)
        fitted = x.reshape(len(used), -1, 3)
        if round_ < rounds and len(used) >= 5:
            for name in BONES:
                lengths[name] = float(np.median([np.linalg.norm(j[a] - j[b]) for j in fitted for bone, a, b in PAIRS if bone == name]))

    spreads = _spreads(information, len(used), 3 * len(JOINTS))
    for k, index in enumerate(used):
        joints = fitted[k]
        camera, pixels, weights, _ = inputs[k]
        projected = camera.project(joints)
        misses = np.hypot(*(projected - pixels).T)[weights > 0]
        full, full_sigma = np.full((17, 3), np.nan), np.full(17, np.nan)
        full[list(JOINTS)], full_sigma[list(JOINTS)] = joints, spreads[k]
        bodies[index] = Body(full, full_sigma, float(np.sqrt(np.mean(misses**2))))
    return bodies
