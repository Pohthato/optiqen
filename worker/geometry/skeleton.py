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
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.optimize import least_squares

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
ANKLES, SHOULDERS, HIPS = [SLOT[15], SLOT[16]], [SLOT[5], SLOT[6]], [SLOT[11], SLOT[12]]
LEGS = [[SLOT[13], SLOT[11], SLOT[15]], [SLOT[14], SLOT[12], SLOT[16]]]  # knee, hip and ankle
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
FLOOR_CAP = 7.0  # the most an ankle off the floor costs (squared standard deviations): further up, it is off it
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
# What the fit's own uncertainty leaves out: its floor rule can be this wrong about a foot's
# height (one just off the floor is held a little down), which moves the whole body along the
# line of sight by that over the sine of the sight's angle to the floor (a few centimetres close
# up, over 10 cm for a far player seen from behind). Set on the tuning rallies so that 95 % of
# joints fall within twice their stated spread in every view, near and far.
FLOOR_MISS_M = 0.012


@dataclass(frozen=True)
class Body:
    joints: np.ndarray  # (17, 3) court metres; NaN for the joints not fitted (eyes, ears)
    sigma: np.ndarray  # (17,) one standard deviation of each joint's position, metres
    rms_px: float  # how far the fitted joints miss the detected ones
    # (39, 39) the fitted joints' covariance (JOINTS order, x y z each), metres squared: how their
    # errors go together (the whole body along the line of sight, say)
    covariance: np.ndarray | None = None


class _Frames:
    """Each frame's problem, for any number of frames at once: the residuals (frame after frame)
    and their analytic Jacobian over every frame's joints, sparse (a row touches a few joints of
    its own frame) and built from arrays, with no loop over frames or joints."""

    ENDS = np.array([(a, b) for _, a, b in PAIRS])

    def __init__(self, cameras: Sequence[Camera], pixels, weights, priors, lengths: dict, noise_px: float, previous=None, step_sigma=None):
        count, n = len(cameras), len(JOINTS)
        self.R = np.array([camera.rotation for camera in cameras])
        self.t = np.array([np.asarray(camera.tvec, dtype=float).reshape(3) for camera in cameras])
        self.f, self.cx, self.cy, self.k1 = (np.array([getattr(camera, name) for camera in cameras], dtype=float)[:, None] for name in ("focal_px", "cx", "cy", "k1"))
        self.pixels = np.asarray(pixels, dtype=float).reshape(count, n, 2)
        self.scale = np.asarray(weights, dtype=float).reshape(count, n) / noise_px
        self.lengths = np.array([lengths[name] for name, _, _ in PAIRS])
        # Each frame's frame before, if given, and how far a joint may have moved since.
        self.previous = None if previous is None else np.asarray(previous, dtype=float).reshape(count, n, 3)
        self.step_sigma = None if previous is None else np.asarray(step_sigma, dtype=float).reshape(count, 1, 1)
        # A learned model's joints relative to the hips' middle, turned from the camera's axes into the court's.
        posed = np.array([prior is not None for prior in priors])
        self.prior = np.zeros((count, n, 3))
        if posed.any():
            self.prior[posed] = np.array([prior for prior in priors if prior is not None], dtype=float) @ self.R[posed]
        # Which of a frame's residuals and Jacobian entries it has (the learned pose's only with a
        # pose), and the sparse Jacobian's structure: entries in the same place are summed.
        rows, cols, unposed = self._layout(previous is not None)
        self.has_row = posed[:, None] | (np.arange(unposed + 3 * n) < unposed)
        self.has_entry = posed[:, None] | (rows < unposed)
        starts = np.concatenate([[0], np.cumsum(self.has_row.sum(axis=1))[:-1]]).astype(np.int64)
        self.shape = (int(self.has_row.sum()), count * 3 * n)
        place = ((starts[:, None] + rows) * self.shape[1] + np.arange(count, dtype=np.int64)[:, None] * 3 * n + cols)[self.has_entry]
        unique, self.slot = np.unique(place, return_inverse=True)
        self.indices = unique % self.shape[1]
        self.indptr = np.searchsorted(unique // self.shape[1], np.arange(self.shape[0] + 1))

    @staticmethod
    def _layout(steps: bool) -> tuple[np.ndarray, np.ndarray, int]:
        """A frame's Jacobian entries' rows and columns, in the order __call__ lists them, and the
        row the learned pose's residuals start at."""
        n, xyz, two, joint = len(JOINTS), np.arange(3), np.arange(2), np.arange(len(JOINTS))
        parts = [np.broadcast_arrays(2 * joint[:, None, None] + two[:, None], 3 * joint[:, None, None] + xyz)]  # reprojection
        row = 2 * n
        parts.append(np.broadcast_arrays(row + np.arange(len(PAIRS))[:, None, None], 3 * _Frames.ENDS[:, :, None] + xyz))  # bones
        row += len(PAIRS)
        for _ in range(2):  # no ankle below the floor, and each on it
            parts.append((row + two, 3 * np.array(ANKLES) + 2))
            row += 2
        parts.append(np.broadcast_arrays(row + two[:, None, None], 3 * np.array(LEGS)[:, :, None] + xyz))  # knees
        row += 2
        parts.append(np.broadcast_arrays(row + two[:, None], 3 * np.array(SHOULDERS + HIPS) + two[:, None]))  # lean
        row += 2
        parts.append(np.broadcast_arrays(row + two[:, None], 3 * np.array([SHOULDERS, HIPS]) + 2))  # level
        row += 2
        if steps:
            parts.append((row + np.arange(3 * n), np.arange(3 * n)))
            row += 3 * n
        ends = np.stack([joint, np.full(n, HIPS[0]), np.full(n, HIPS[1])], axis=1)  # each joint and both hips
        parts.append(np.broadcast_arrays(row + 3 * joint[:, None, None] + xyz[:, None], 3 * ends[:, None, :] + xyz[:, None]))  # the learned pose
        return np.concatenate([r.ravel() for r, _ in parts]), np.concatenate([c.ravel() for _, c in parts]), row

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
        count, n = len(self.R), len(JOINTS)
        X = x.reshape(count, n, 3)
        # Reprojection, through the camera's one radial term.
        p = X @ self.R.transpose(0, 2, 1) + self.t[:, None]
        depth = np.where(p[..., 2] > 1e-3, p[..., 2], 1e-3)
        u, v = p[..., 0] / depth, p[..., 1] / depth
        s = 1.0 + self.k1 * (u * u + v * v)
        seen = np.stack([(self.f * u * s + self.cx - self.pixels[..., 0]) * self.scale, (self.f * v * s + self.cy - self.pixels[..., 1]) * self.scale], axis=2)
        zero = np.zeros_like(depth)
        du = np.stack([1 / depth, zero, -u / depth], axis=2)
        dv = np.stack([zero, 1 / depth, -v / depth], axis=2)
        ds = self.k1[..., None] * (2 * u[..., None] * du + 2 * v[..., None] * dv)
        turned = np.stack([s[..., None] * du + u[..., None] * ds, s[..., None] * dv + v[..., None] * ds], axis=2)
        seen_J = (self.f[..., None, None] * turned) @ self.R[:, None] * self.scale[..., None, None]
        # Bones.
        ends = X[:, self.ENDS]
        span = ends[:, :, 0] - ends[:, :, 1]
        length = np.maximum(np.linalg.norm(span, axis=2), 1e-9)
        bones = (length - self.lengths) / BONE_SIGMA_M
        along = span / length[..., None] / BONE_SIGMA_M
        bones_J = np.stack([along, -along], axis=2)
        # The floor: no ankle below it, and each on it or clearly off it.
        gap = X[:, ANKLES, 2] - ANKLE_HEIGHT_M
        below = gap < 0
        floor, floor_J = np.where(below, gap / FLOOR_SIGMA_M, 0.0), np.where(below, 1.0 / FLOOR_SIGMA_M, 0.0)
        down, slope = _capped(gap / DOWN_SIGMA_M)
        down_J = slope / DOWN_SIGMA_M
        # Knees bend forward (forward taken across the hips).
        across = X[:, HIPS[1]] - X[:, HIPS[0]]
        forward = np.stack([-across[:, 1], across[:, 0], np.zeros(count)], axis=1)
        forward /= np.maximum(np.linalg.norm(forward, axis=1), 1e-9)[:, None]
        legs = X[:, LEGS]
        ahead = np.sum((legs[:, :, 0] - (legs[:, :, 1] + legs[:, :, 2]) / 2) * forward[:, None], axis=2)
        back = ahead < 0
        knees = np.where(back, ahead / KNEE_SIGMA_M, 0.0)
        bend = np.where(back[..., None], forward[:, None], 0.0)
        knees_J = np.stack([bend / KNEE_SIGMA_M, -bend / (2 * KNEE_SIGMA_M), -bend / (2 * KNEE_SIGMA_M)], axis=2)
        # The trunk roughly upright: the shoulders' middle above the hips' middle.
        off = (X[:, SHOULDERS[0]] + X[:, SHOULDERS[1]] - X[:, HIPS[0]] - X[:, HIPS[1]]) / 2
        lean, lean_J = off[:, :2] / LEAN_SIGMA_M, np.broadcast_to(np.array([0.5, 0.5, -0.5, -0.5]) / LEAN_SIGMA_M, (count, 2, 4))
        # Shoulders and hips roughly level.
        level = (X[:, [SHOULDERS[0], HIPS[0]], 2] - X[:, [SHOULDERS[1], HIPS[1]], 2]) / LEVEL_SIGMA_M
        level_J = np.broadcast_to(np.array([1.0, -1.0]) / LEVEL_SIGMA_M, (count, 2, 2))
        terms = [(seen, seen_J), (bones, bones_J), (floor, floor_J), (down, down_J), (knees, knees_J), (lean, lean_J), (level, level_J)]
        # Smoothly from the frame before.
        if self.previous is not None:
            terms.append(((X - self.previous) / self.step_sigma, np.broadcast_to(1.0 / self.step_sigma, (count, n, 3))))
        # Posed as the learned model sees it, about the hips' middle.
        middle = (X[:, HIPS[0]] + X[:, HIPS[1]]) / 2
        pose = (X - middle[:, None] - self.prior) / PRIOR_SIGMA_M
        terms.append((pose, np.array([1.0, -0.5, -0.5]) / PRIOR_SIGMA_M))  # the joint itself and each hip
        # Robust: a detected joint, and the model's pose.
        for k in (0, -1):
            value, J = terms[k]
            value, slope = _soft(value, ROBUST_SCALE)
            terms[k] = value, J * slope[..., None]
        r = np.concatenate([value.reshape(count, -1) for value, _ in terms], axis=1)[self.has_row]
        values = np.concatenate([np.reshape(J, (count, -1)) for _, J in terms], axis=1)[self.has_entry]
        data = np.bincount(self.slot, weights=values, minlength=len(self.indices))
        return r, sparse.csr_matrix((data, self.indices, self.indptr), shape=self.shape)


def _capped(u):
    """A residual whose square grows as u squared near 0 and levels off at FLOOR_CAP (on the
    floor, or off it: no pull once clearly up), and its slope; u a number or an array."""
    k = np.exp(-FLOOR_CAP / 2) / (1 - np.exp(-FLOOR_CAP / 2))
    e = np.exp(-0.5 * u * u)
    cost = -2.0 * np.log((e + k) / (1 + k))
    small = cost < 1e-12
    root = np.sqrt(np.where(small, 1.0, cost))
    return np.where(small, u / np.sqrt(1 + k), np.sign(u) * root), np.where(small, 1 / np.sqrt(1 + k), np.abs(u) * e / ((e + k) * root))


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


def _alone(problem: _Frames, x0: np.ndarray) -> np.ndarray:
    """One frame's body on its own (with the frame before, if any, in its problem)."""
    cache: dict = {}

    def residuals(x):
        cache["x"], (cache["r"], cache["J"]) = x.copy(), problem(x)
        return cache["r"]

    def jacobian(x):
        if "x" not in cache or not np.array_equal(cache["x"], x):
            residuals(x)
        return cache["J"].toarray()

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


def _contact(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How surely an ankle at height z is on the floor (1 at its floor height or below, falling
    away above it), and its slope."""
    over = np.maximum(z - ANKLE_HEIGHT_M, 0.0)
    weight = np.exp(-0.5 * (over / CONTACT_M) ** 2)
    return weight, -over / CONTACT_M**2 * weight


def _slides(x: np.ndarray, times_s: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    """A foot on the floor does not slide: per ankle and pair of tied frames, its speed across the
    floor (to SLIDE_SIGMA), counted as much as the ankle is down in both frames. Softened (a
    foot can scrape) and with the Jacobian over all the frames' joints."""
    count, size = x.shape
    X = x.reshape(count, -1, 3)
    j = np.flatnonzero(np.diff(times_s) <= LINK_GAP_S) + 1
    i = j - 1
    scale = (1.0 / (SLIDE_SIGMA_M_PER_S * np.maximum(times_s[j] - times_s[i], 1e-3)))[:, None]
    # Per ankle, pair of tied frames and axis across the floor.
    w, s = _contact(X[:, ANKLES, 2].T)
    wi, si, wj, sj = (a[:, :, None] for a in (w[:, i], s[:, i], w[:, j], s[:, j]))
    v = (X[j][:, ANKLES, :2] - X[i][:, ANKLES, :2]).transpose(1, 0, 2) * scale
    a, axis = 3 * np.array(ANKLES)[:, None, None], np.arange(2)
    cols = np.stack(np.broadcast_arrays(j[:, None] * size + a + axis, i[:, None] * size + a + axis, j[:, None] * size + a + 2, i[:, None] * size + a + 2), axis=3)
    values = np.stack(np.broadcast_arrays(wi * wj * scale, -wi * wj * scale, v * wi * sj, v * wj * si), axis=3)
    soft, slope = _soft((wi * wj * v).ravel(), ROBUST_SCALE)
    J = sparse.csr_matrix((values.ravel(), (np.repeat(np.arange(soft.size), 4), cols.ravel())), shape=(soft.size, count * size))
    return soft, sparse.diags(slope) @ J


def _together(problem: _Frames, x0: np.ndarray, links: sparse.csr_matrix, times_s: np.ndarray) -> tuple[np.ndarray, sparse.csc_matrix]:
    """Every frame's body at once (Levenberg-Marquardt on the sparse problem); the bodies and the
    problem's information matrix at them."""
    count, size = x0.shape

    def evaluate(x):
        r_frames, J_frames = problem(x.ravel())
        r_slides, J_slides = _slides(x, times_s)
        return np.concatenate([r_frames, links @ x.ravel(), r_slides]), sparse.vstack([J_frames, links, J_slides], format="csr")

    x = x0.copy()
    r, J = evaluate(x)
    information, gradient = J.T @ J, J.T @ r
    cost, damping = float(r @ r), 1e-3
    for _ in range(MAX_STEPS):
        # The damped normal equations are banded (a frame is tied only to its neighbours).
        try:
            factor = cholesky_banded(_band(information + sparse.diags(damping * (information.diagonal() + 1e-9))), lower=True)
        except np.linalg.LinAlgError:
            damping *= 4
            continue
        candidate = x + cho_solve_banded((factor, True), -gradient).reshape(count, size)
        r_new, J_new = evaluate(candidate)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            settled = cost - cost_new < 1e-7 * cost
            x, r, cost = candidate, r_new, cost_new
            information, gradient = J_new.T @ J_new, J_new.T @ r_new
            damping = max(damping / 3, 1e-9)
            if settled:
                break
        else:
            damping *= 4
            if damping > 1e8:
                break
    return x, information


def _band(matrix: sparse.spmatrix, width: int = 0) -> np.ndarray:
    """A symmetric matrix's lower band, as a banded Cholesky takes it: band[d, j] = matrix[j + d, j],
    at least `width` below the diagonal."""
    coo = sparse.tril(matrix, format="coo")
    gap, n = coo.row - coo.col, matrix.shape[0]
    rows = max(int(gap.max(initial=0)), width) + 1
    return np.bincount(gap * n + coo.col, weights=coo.data, minlength=rows * n).reshape(rows, n)


def _covariances(information: sparse.csc_matrix, count: int, size: int) -> np.ndarray:
    """Each frame's joints' covariance (size by size, metres squared): that frame's block of the
    information matrix's inverse. The matrix is banded (a frame is tied only to its neighbours),
    so its inverse's band comes from a banded Cholesky factor by Takahashi's recursion, in time
    that grows with the clip's length rather than its square:
    Z[i, j] = (delta_ij / L[i, i] - sum over k > i of L[k, i] Z[k, j]) / L[i, i]."""
    n = information.shape[0]
    factor = cholesky_banded(_band(information + sparse.identity(n) * 1e-9, size - 1), lower=True)  # factor[d, j] = L[j + d, j]
    b = len(factor) - 1
    blocks = np.zeros((count, size, size))
    window = np.zeros((b + 1, b + 1))  # Z[j : j + b + 1, j : j + b + 1] for the last row j done (nought past the end)
    for i in range(n - 1, -1, -1):
        span = min(b, n - 1 - i)
        below = factor[1 : span + 1, i]
        row = np.empty(b + 1)  # Z[i, i + d]
        row[1:] = -(below @ window[:span, :b]) / factor[0, i]  # right off the diagonal: only rows below i are used
        row[0] = (1.0 / factor[0, i] - below @ row[1 : span + 1]) / factor[0, i]  # the diagonal uses this row's own
        shifted = np.empty_like(window)
        shifted[1:, 1:], shifted[0], shifted[1:, 0] = window[:b, :b], row, row[1:]
        window = shifted
        if i % size == 0:
            blocks[i // size] = window[:size, :size]
    return blocks


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
        problem = _Frames([camera], [pixels], [weights], [prior], lengths, noise_px, [previous] if linked else None, [step_sigma])
        start = previous.ravel() if linked else _start(camera, pixels, weights, stature, facing)
        x[k] = _alone(problem, start)
        previous, previous_s = x[k].reshape(-1, 3), times_s[k]

    links = _links(times_s)
    for round_ in range(rounds + 1):
        x, information = _together(_Frames(*zip(*inputs), lengths, noise_px), x, links, times_s)
        fitted = x.reshape(len(used), -1, 3)
        if round_ < rounds and len(used) >= 5:
            for name in BONES:
                lengths[name] = float(np.median([np.linalg.norm(j[a] - j[b]) for j in fitted for bone, a, b in PAIRS if bone == name]))

    covariances = _covariances(information, len(used), 3 * len(JOINTS))
    for k, index in enumerate(used):
        joints = fitted[k]
        camera, pixels, weights, _ = inputs[k]
        centre = -camera.rotation.T @ np.asarray(camera.tvec, dtype=float).reshape(3)
        sight = (joints[SLOT[11]] + joints[SLOT[12]]) / 2 - centre
        sight /= max(float(np.linalg.norm(sight)), 1e-9)
        along = FLOOR_MISS_M / max(abs(float(sight[2])), 0.05)
        covariances[k] += along**2 * np.kron(np.ones((len(JOINTS), len(JOINTS))), np.outer(sight, sight))
        projected = camera.project(joints)
        misses = np.hypot(*(projected - pixels).T)[weights > 0]
        variances = np.maximum(np.diag(covariances[k]), 0.0).reshape(len(JOINTS), 3).sum(axis=1)
        full, full_sigma = np.full((17, 3), np.nan), np.full(17, np.nan)
        full[list(JOINTS)], full_sigma[list(JOINTS)] = joints, np.sqrt(variances)
        bodies[index] = Body(full, full_sigma, float(np.sqrt(np.mean(misses**2))), covariances[k])
    return bodies
