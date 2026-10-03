# worker/geometry/calibrate.py
"""Single-image camera calibration against the BWF singles court model.

Given pixel observations of named court keypoints, recover the pinhole camera
(focal length, pose, one radial term). Floor keypoints give a homography, whose
decomposition seeds a robust nonlinear refinement over all keypoints. Net-post
and net-tape points are off the floor plane and pin the focal length when
visible. The caller learns how far to trust the result from `tier`, the
in-sample `rms_px`, and the leave-one-out floor error.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from geometry.camera import Camera
from geometry.court_model import KEYPOINTS

TYPICAL_FOCAL_OVER_LONG_SIDE = 0.72  # 26 mm-equivalent phone main lens
OUTLIER_PX = 8.0
MIN_KEYPOINTS = 4
MIN_K1_POINTS = 8
VALIDATED_MAX_RMS_PX = 3.0
VALIDATED_MAX_LOO_CM = 15.0
APPROX_MAX_RMS_PX = 8.0
APPROX_MAX_LOO_CM = 40.0
VALIDATED_MIN_REDUNDANCY = 2
_PRIOR_FOCAL_PX_PER_LOG = 6.0
_PRIOR_K1_PX_PER_UNIT = 30.0


@dataclass(frozen=True)
class CameraSolution:
    camera: Camera
    tier: str  # "validated" | "approximate" | "unavailable"
    rms_px: float
    floor_rms_cm: float
    loo_floor_cm: float | None
    redundancy: int
    inliers: tuple[str, ...]
    outliers: tuple[str, ...]


def _non_collinear(points_xy: np.ndarray) -> bool:
    centred = points_xy - points_xy.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return bool(singular[0] > 0 and singular[-1] / singular[0] > 0.02)


def _focal_from_homography(homography: np.ndarray, cx: float, cy: float) -> float | None:
    g = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ homography
    h1, h2 = g[:, 0], g[:, 1]
    denominator = h1[2] * h2[2]
    if abs(denominator) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / denominator
        if f2 > 0:
            return float(np.sqrt(f2))
    denominator = h2[2] ** 2 - h1[2] ** 2
    if abs(denominator) > 1e-12:
        f2 = ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / denominator
        if f2 > 0:
            return float(np.sqrt(f2))
    return None


def _pose_from_homography(
    homography: np.ndarray, focal: float, cx: float, cy: float, floor_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    m = np.linalg.inv(k) @ homography
    lam = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    # The homography fixes the pose only up to sign. Pick the sign that puts the observed
    # points in front of the camera; the court origin itself may be behind it.
    depths = np.column_stack([floor_xy, np.ones(len(floor_xy))]) @ m[2]
    if np.median(depths) < 0:
        lam = -lam
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    u, _, vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return cv2.Rodrigues(rotation)[0].ravel(), t


def _unpack(params: np.ndarray, cx: float, cy: float, fit_k1: bool) -> Camera:
    return Camera(
        float(np.exp(params[6])), cx, cy, params[:3].copy(), params[3:6].copy(), float(params[7]) if fit_k1 else 0.0
    )


def _pack(camera: Camera, fit_k1: bool) -> np.ndarray:
    base = [*np.asarray(camera.rvec).ravel(), *np.asarray(camera.tvec).ravel(), np.log(camera.focal_px)]
    return np.array(base + ([camera.k1] if fit_k1 else []), dtype=np.float64)


def _refine(
    start: np.ndarray,
    world: np.ndarray,
    pixels: np.ndarray,
    cx: float,
    cy: float,
    focal_typical: float,
    fit_k1: bool,
) -> tuple[np.ndarray, float]:
    def residuals(params: np.ndarray) -> np.ndarray:
        projected = _unpack(params, cx, cy, fit_k1).project(world)
        flat = (projected - pixels).ravel()
        flat[~np.isfinite(flat)] = 1e3
        terms = [flat, [_PRIOR_FOCAL_PX_PER_LOG * (params[6] - np.log(focal_typical))]]
        if fit_k1:
            terms.append([_PRIOR_K1_PX_PER_UNIT * params[7]])
        return np.concatenate(terms)

    result = least_squares(residuals, start, loss="soft_l1", f_scale=2.0, method="trf", max_nfev=200)
    return result.x, float(result.cost)


def _fit(
    names: list[str],
    observations: dict[str, tuple[float, float]],
    cx: float,
    cy: float,
    focal_typical: float,
) -> Camera | None:
    world = np.array([KEYPOINTS[name] for name in names])
    pixels = np.array([observations[name] for name in names], dtype=np.float64)
    floor = world[:, 2] == 0.0
    if floor.sum() < MIN_KEYPOINTS or not _non_collinear(world[floor, :2]):
        return None
    method = cv2.RANSAC if floor.sum() >= 5 else 0
    homography, _ = cv2.findHomography(world[floor, :2].astype(np.float64), pixels[floor], method, 5.0)
    if homography is None:
        return None
    seeds = [focal_typical * scale for scale in (0.7, 1.0, 1.4)]
    from_homography = _focal_from_homography(homography, cx, cy)
    if from_homography is not None and 0.25 * focal_typical < from_homography < 4.0 * focal_typical:
        seeds.append(from_homography)
    fit_k1 = len(names) >= MIN_K1_POINTS
    best: tuple[np.ndarray, float] | None = None
    for focal in seeds:
        rvec, tvec = _pose_from_homography(homography, focal, cx, cy, world[floor, :2])
        start = _pack(Camera(focal, cx, cy, rvec, tvec), fit_k1)
        params, cost = _refine(start, world, pixels, cx, cy, focal_typical, fit_k1)
        if best is None or cost < best[1]:
            best = (params, cost)
    assert best is not None
    return _unpack(best[0], cx, cy, fit_k1)


def _refit_warm(
    names: list[str],
    observations: dict[str, tuple[float, float]],
    camera: Camera,
    cx: float,
    cy: float,
    focal_typical: float,
) -> Camera | None:
    world = np.array([KEYPOINTS[name] for name in names])
    floor = world[:, 2] == 0.0
    if floor.sum() < MIN_KEYPOINTS or not _non_collinear(world[floor, :2]):
        return None
    pixels = np.array([observations[name] for name in names], dtype=np.float64)
    fit_k1 = len(names) >= MIN_K1_POINTS
    params, _ = _refine(_pack(camera, fit_k1), world, pixels, cx, cy, focal_typical, fit_k1)
    return _unpack(params, cx, cy, fit_k1)


def _floor_error_cm(camera: Camera, names: list[str], observations: dict[str, tuple[float, float]]) -> np.ndarray:
    pixels = np.array([observations[name] for name in names], dtype=np.float64)
    truth = np.array([KEYPOINTS[name][:2] for name in names])
    return np.linalg.norm(camera.pixel_to_plane(pixels, 0.0) - truth, axis=1) * 100.0


def _residual_px(camera: Camera, names: list[str], observations: dict[str, tuple[float, float]]) -> np.ndarray:
    projected = camera.project(np.array([KEYPOINTS[name] for name in names]))
    observed = np.array([observations[name] for name in names], dtype=np.float64)
    distance = np.linalg.norm(projected - observed, axis=1)
    distance[~np.isfinite(distance)] = 1e3
    return distance


def _leave_one_out_cm(
    camera: Camera,
    inliers: list[str],
    floor_names: list[str],
    observations: dict[str, tuple[float, float]],
    cx: float,
    cy: float,
    focal_typical: float,
) -> float | None:
    if len(floor_names) < MIN_KEYPOINTS + 1:
        return None
    errors: list[float] = []
    for held_out in floor_names:
        rest = [name for name in inliers if name != held_out]
        refit = _refit_warm(rest, observations, camera, cx, cy, focal_typical)
        if refit is None:
            continue
        errors.append(float(_floor_error_cm(refit, [held_out], observations)[0]))
    if not errors:
        return None
    return float(np.sqrt(np.mean(np.square(errors))))


def _physically_plausible(camera: Camera, names: list[str]) -> bool:
    """A real camera is above the floor and has every inlier in front of it. Mirrored
    labels (left/right or near/far swapped) fit a camera reflected below the floor."""
    if camera.centre[2] <= 0:
        return False
    world = np.array([KEYPOINTS[name] for name in names])
    depth = world @ camera.rotation[2] + float(np.asarray(camera.tvec, dtype=np.float64).reshape(3)[2])
    return bool(np.all(depth > 0))


def _tier(rms_px: float, loo_floor_cm: float | None, redundancy: int) -> str:
    validated = (
        redundancy >= VALIDATED_MIN_REDUNDANCY
        and loo_floor_cm is not None
        and rms_px <= VALIDATED_MAX_RMS_PX
        and loo_floor_cm <= VALIDATED_MAX_LOO_CM
    )
    if validated:
        return "validated"
    if rms_px <= APPROX_MAX_RMS_PX and (loo_floor_cm is None or loo_floor_cm <= APPROX_MAX_LOO_CM):
        return "approximate"
    return "unavailable"


def solve_camera(
    observations: dict[str, tuple[float, float]],
    image_size: tuple[int, int],
    focal_prior_px: float | None = None,
) -> CameraSolution | None:
    """Return the best camera for the observed keypoints, or None when under-determined."""
    width, height = image_size
    cx, cy = width / 2, height / 2
    focal_typical = focal_prior_px or TYPICAL_FOCAL_OVER_LONG_SIDE * max(width, height)
    names = [name for name, pixel in observations.items() if name in KEYPOINTS and np.isfinite(pixel).all()]
    if len(names) < MIN_KEYPOINTS:
        return None
    camera = _fit(names, observations, cx, cy, focal_typical)
    if camera is None:
        return None
    residual = _residual_px(camera, names, observations)
    inliers = [name for name, value in zip(names, residual) if value <= OUTLIER_PX]
    outliers = [name for name, value in zip(names, residual) if value > OUTLIER_PX]
    if outliers and len(inliers) >= MIN_KEYPOINTS:
        refit = _fit(inliers, observations, cx, cy, focal_typical)
        if refit is not None:
            camera = refit
        else:
            inliers, outliers = names, []
    else:
        inliers, outliers = names, []
    floor_names = [name for name in inliers if KEYPOINTS[name][2] == 0.0]
    floor_rms_cm = float(np.sqrt(np.mean(_floor_error_cm(camera, floor_names, observations) ** 2)))
    rms_px = float(np.sqrt(np.mean(np.square(_residual_px(camera, inliers, observations)))))
    loo = _leave_one_out_cm(camera, inliers, floor_names, observations, cx, cy, focal_typical)
    redundancy = len(inliers) - MIN_KEYPOINTS
    return CameraSolution(
        camera=camera,
        tier=_tier(rms_px, loo, redundancy) if _physically_plausible(camera, inliers) else "unavailable",
        rms_px=rms_px,
        floor_rms_cm=floor_rms_cm,
        loo_floor_cm=loo,
        redundancy=redundancy,
        inliers=tuple(inliers),
        outliers=tuple(outliers),
    )
