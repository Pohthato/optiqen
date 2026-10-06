# worker/geometry/tracking.py
"""A camera for every frame of a phone video, by locking the court model onto the painted lines.

Each frame, the floor lines and the net tape are projected through the current camera estimate.
Along each projected line we search for the real line (geometry.lines) and then refine the
camera so the model lies on what was found, with a robust loss so players, shoes and other
bright clutter cannot drag it. The net tape is 1.5 m above the floor, so it pins the focal
length that floor lines alone leave ambiguous for a phone straight behind the court.

The focal length and lens distortion are refined once, when the court is first found, and
then held: a phone's main lens does not zoom mid-rally. When a frame cannot be fitted with
confidence (a hand over the lens, a person walking past the phone) it is reported as lost,
with no camera, and the court is searched for again from the last good camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares

from geometry.camera import Camera
from geometry.court_model import court_lines, net_tape_lines
from geometry.lines import find_line_offsets, line_response, sample_lines

SAMPLE_SPACING_M = 0.25
REFERENCE_WIDTH_PX = 1280.0  # search radii below are for a frame this wide and scale with it
TRACK_RADII_PX = (10.0, 5.0, 3.0)
ACQUIRE_RADII_PX = (16.0, 8.0, 4.0, 3.0)
ACQUIRE_SHIFT_FRACTION = 0.06  # how far (of the frame width) the court may have moved while lost
MIN_RADIUS_PX = 2.5
TANGENT_STEP_M = 0.05
INLIER_PX = 2.0
MIN_INLIERS = 24
MIN_INLIER_SHARE = 0.6
MIN_COVERAGE = 0.35  # of the model that should be in view; a wrong lock explains little of it
MIN_LINE_INLIERS = 3
MIN_LINES = 4
MAX_RMS_PX = 1.5
MAX_STEP_DEG = 2.0  # a steadily held phone does not turn further than this between frames
MAX_STEP_M = 0.25
_PRIOR_FOCAL_PX_PER_LOG = 6.0
_PRIOR_K1_PX_PER_UNIT = 30.0


@dataclass(frozen=True)
class LineFit:
    camera: Camera
    visible: int  # model samples that project into the frame
    matched: int  # model samples with a line found near them
    inliers: int  # matched samples within INLIER_PX of the fitted model
    rms_px: float  # RMS distance of the inliers from their lines
    lines: int  # model lines with at least MIN_LINE_INLIERS inliers
    crossing: bool  # those lines include both along-court and across-court directions

    @property
    def confident(self) -> bool:
        return (
            self.inliers >= MIN_INLIERS
            and self.inliers >= MIN_INLIER_SHARE * self.matched
            and self.inliers >= MIN_COVERAGE * self.visible
            and self.rms_px <= MAX_RMS_PX
            and self.lines >= MIN_LINES
            and self.crossing
        )


@dataclass(frozen=True)
class TrackedFrame:
    state: str  # "anchored" | "tracked" | "reanchored" | "lost"
    camera: Camera | None  # None when lost: no camera is better than a wrong one
    inliers: int
    rms_px: float


@lru_cache(maxsize=1)
def _model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sample_lines(_lines(), SAMPLE_SPACING_M)


@lru_cache(maxsize=1)
def _lines() -> list[tuple[str, np.ndarray, np.ndarray]]:
    return court_lines() + net_tape_lines()


@lru_cache(maxsize=1)
def _floor_directions() -> tuple[np.ndarray, np.ndarray]:
    """Per model line: is it a floor line running across the court, or along it."""
    on_floor = np.array([start[2] == 0.0 and end[2] == 0.0 for _, start, end in _lines()])
    across = np.array([abs(end[0] - start[0]) > abs(end[1] - start[1]) for _, start, end in _lines()])
    return on_floor & across, on_floor & ~across


def _radii(radii: tuple[float, ...], width: int) -> list[float]:
    return [max(MIN_RADIUS_PX, radius * width / REFERENCE_WIDTH_PX) for radius in radii]


def _project(camera: Camera, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels and unit image normals of the model samples, and which samples are in the frame."""
    points, directions, _ = _model()
    pixels = camera.project(points)
    tangent = camera.project(points + TANGENT_STEP_M * directions) - pixels
    length = np.linalg.norm(tangent, axis=1)
    width, height = size
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / length[:, None]
        visible = (
            np.isfinite(pixels).all(axis=1)
            & np.isfinite(normals).all(axis=1)
            & (length > 1e-3)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] <= height - 1)
        )
    return pixels, normals, visible


def _refine(camera: Camera, world: np.ndarray, targets: np.ndarray, normals: np.ndarray, fit_intrinsics: bool) -> Camera:
    """Camera that puts each world sample on the line found for it (distance along the normal)."""
    focal_start = np.log(camera.focal_px)

    def unpack(params: np.ndarray) -> Camera:
        if fit_intrinsics:
            return Camera(float(np.exp(params[6])), camera.cx, camera.cy, params[:3].copy(), params[3:6].copy(), float(params[7]))
        return Camera(camera.focal_px, camera.cx, camera.cy, params[:3].copy(), params[3:6].copy(), camera.k1)

    def residuals(params: np.ndarray) -> np.ndarray:
        distance = ((unpack(params).project(world) - targets) * normals).sum(axis=1)
        distance[~np.isfinite(distance)] = 50.0
        if not fit_intrinsics:
            return distance
        priors = [_PRIOR_FOCAL_PX_PER_LOG * (params[6] - focal_start), _PRIOR_K1_PX_PER_UNIT * params[7]]
        return np.concatenate([distance, priors])

    start = [*np.asarray(camera.rvec, dtype=np.float64).ravel(), *np.asarray(camera.tvec, dtype=np.float64).ravel()]
    if fit_intrinsics:
        start += [focal_start, camera.k1]
    result = least_squares(residuals, np.array(start), loss="soft_l1", f_scale=1.0, method="trf", max_nfev=60)
    return unpack(result.x)


def fit_frame(
    response: np.ndarray,
    camera: Camera,
    radii_px: tuple[float, ...] = TRACK_RADII_PX,
    fit_intrinsics: bool = False,
) -> LineFit | None:
    """Lock the court model onto the lines in `response` (geometry.lines.line_response), starting
    from `camera` and searching coarse to fine. None when too few lines are found to try."""
    points, _, line_ids = _model()
    height, width = response.shape
    current = camera
    for radius in _radii(radii_px, width):
        pixels, normals, visible = _project(current, (width, height))
        candidates = np.flatnonzero(visible)
        offsets, _ = find_line_offsets(response, pixels[candidates], normals[candidates], radius)
        found = np.isfinite(offsets)
        if found.sum() < MIN_INLIERS:
            return None
        matched = candidates[found]
        targets = pixels[matched] + offsets[found, None] * normals[matched]
        current = _refine(current, points[matched], targets, normals[matched], fit_intrinsics)
    distance = ((current.project(points[matched]) - targets) * normals[matched]).sum(axis=1)
    _, _, visible = _project(current, (width, height))
    inlier = np.abs(distance) <= INLIER_PX
    strong = np.bincount(line_ids[matched][inlier], minlength=len(_lines())) >= MIN_LINE_INLIERS
    across, along = _floor_directions()
    return LineFit(
        camera=current,
        visible=int(visible.sum()),
        matched=int(len(matched)),
        inliers=int(inlier.sum()),
        rms_px=float(np.sqrt(np.mean(distance[inlier] ** 2))) if inlier.any() else float("inf"),
        lines=int(strong.sum()),
        crossing=bool((strong & across).any() and (strong & along).any()),
    )


def _turn_to_shift(camera: Camera, shift: np.ndarray) -> Camera:
    """The camera turned about its own centre so the image moves by `shift` pixels."""
    delta = cv2.Rodrigues(np.array([-shift[1], shift[0], 0.0]).reshape(3, 1) / camera.focal_px)[0]
    rotation = delta @ camera.rotation
    tvec = delta @ np.asarray(camera.tvec, dtype=np.float64).reshape(3)
    return Camera(camera.focal_px, camera.cx, camera.cy, cv2.Rodrigues(rotation)[0].ravel(), tvec, camera.k1)


def _best_shift(response: np.ndarray, camera: Camera, max_shift_px: float) -> np.ndarray:
    """The image shift that puts the most line response under the projected model, searched
    coarse to fine on a blurred response so a near miss still scores."""
    height, width = response.shape
    pixels, _, visible = _project(camera, (width, height))
    pixels = pixels[visible]
    best = np.zeros(2)
    if len(pixels) == 0:
        return best
    step = max(1.0, max_shift_px / 10)
    span = max_shift_px
    while True:
        blurred = cv2.GaussianBlur(response, (0, 0), max(1.0, step))
        grid = np.arange(-span, span + 1e-9, step)
        shifts = best + np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
        where = pixels[None, :, :] + shifts[:, None, :]
        scores = cv2.remap(
            blurred,
            where[..., 0].astype(np.float32),
            where[..., 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).mean(axis=1)
        best = shifts[int(np.argmax(scores))]
        if step <= 1.0:
            return best
        span, step = step, max(1.0, step / 4)


def acquire(response: np.ndarray, camera: Camera, fit_intrinsics: bool = False) -> LineFit | None:
    """Find the court when the camera is only roughly known: after the first calibration, or
    after the court was lost. A whole-court shift search first, then the coarse-to-fine fit."""
    width = response.shape[1]
    shift = _best_shift(response, camera, ACQUIRE_SHIFT_FRACTION * width)
    return fit_frame(response, _turn_to_shift(camera, shift), ACQUIRE_RADII_PX, fit_intrinsics)


def _small_step(previous: Camera, current: Camera) -> bool:
    turn = cv2.Rodrigues(current.rotation @ previous.rotation.T)[0]
    return np.degrees(np.linalg.norm(turn)) <= MAX_STEP_DEG and np.linalg.norm(current.centre - previous.centre) <= MAX_STEP_M


def track_video(frames: Iterable[np.ndarray], anchor: Camera, fit_intrinsics: bool = True) -> list[TrackedFrame]:
    """A camera per frame, starting from `anchor` (e.g. solve_camera on a few tapped court points)."""
    track: list[TrackedFrame] = []
    last_good = anchor
    following = False
    anchored = False
    for frame in frames:
        response = line_response(frame)
        fit, state = None, "lost"
        if following:
            fit = fit_frame(response, last_good)
            if fit is not None and fit.confident and _small_step(last_good, fit.camera):
                state = "tracked"
        if state == "lost":
            fit = acquire(response, last_good, fit_intrinsics=fit_intrinsics and not anchored)
            if fit is not None and fit.confident:
                state = "reanchored" if anchored else "anchored"
        if state == "lost":
            following = False
            track.append(TrackedFrame("lost", None, fit.inliers if fit else 0, fit.rms_px if fit else float("inf")))
            continue
        assert fit is not None
        following, anchored, last_good = True, True, fit.camera
        track.append(TrackedFrame(state, fit.camera, fit.inliers, fit.rms_px))
    return track
