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

A court slid by one line spacing (the doubles sideline on the singles sideline, the baseline on
the long service line) puts many model lines on real ones and passes every per-fit check, so
whenever the court is searched for, the slid alternatives are fitted too and the one that
explains clearly more of the court wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator

import cv2
import numpy as np
from scipy.optimize import least_squares

from geometry.calibrate import PRIOR_FOCAL_PX_PER_LOG, PRIOR_K1_PX_PER_UNIT, solve_camera
from geometry.camera import Camera
from geometry.court_model import COLUMNS, KEYPOINTS, LINE_WIDTH_M, NET_TAPE_WIDTH_M, ROWS, court_lines, net_tape_lines
from geometry.lines import find_line_offsets, line_response, sample_lines

SAMPLE_SPACING_M = 0.25
REFERENCE_LONG_SIDE_PX = 1280.0  # search radii below are for a frame this long and scale with it
TRACK_RADII_PX = (10.0, 5.0, 3.0)
ACQUIRE_RADII_PX = (16.0, 8.0, 4.0, 3.0)
ACQUIRE_SHIFT_FRACTION = 0.06  # how far (of the long side) the court may have moved while lost
SHIFT_CANDIDATES = 3
MIN_RADIUS_PX = 2.5
TANGENT_STEP_M = 0.05
INLIER_PX = 2.0  # up to a 1920 px long side; larger frames scale it
MIN_INLIERS = 24
MIN_INLIER_SHARE = 0.6
MIN_COVERAGE = 0.35  # of the model that should be in view; a wrong lock explains little of it
MIN_LINE_INLIERS = 3
MIN_LINES = 4
MAX_STEP_DEG = 2.0  # a steadily held phone does not turn further than this between frames
MAX_STEP_M = 0.25
ALIAS_GAIN = 1.1  # a slid court must explain this many times the inliers to win
ALIAS_ROUNDS = 3
ANCHOR_TAP_FRACTION = 0.0075  # of the long side: how far the lines may put the labelled points (median)


@dataclass(frozen=True)
class LineFit:
    camera: Camera
    visible: int  # model samples that project into the frame
    matched: int  # model samples with a line found near them
    inliers: int  # matched samples within INLIER_PX of the fitted model
    rms_px: float  # RMS distance of the inliers from their lines (reported, not a check)
    lines: int  # model lines with at least MIN_LINE_INLIERS inliers
    crossing: bool  # those lines include both along-court and across-court directions

    @property
    def confident(self) -> bool:
        return (
            self.inliers >= MIN_INLIERS
            and self.inliers >= MIN_INLIER_SHARE * self.matched
            and self.inliers >= MIN_COVERAGE * self.visible
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
def _widths() -> np.ndarray:
    """Painted width of the line under each model sample, in metres."""
    points, _, _ = _model()
    return np.where(points[:, 2] > 0, NET_TAPE_WIDTH_M, LINE_WIDTH_M)


@lru_cache(maxsize=1)
def _floor_directions() -> tuple[np.ndarray, np.ndarray]:
    """Per model line: is it a floor line running across the court, or along it."""
    on_floor = np.array([start[2] == 0.0 and end[2] == 0.0 for _, start, end in _lines()])
    across = np.array([abs(end[0] - start[0]) > abs(end[1] - start[1]) for _, start, end in _lines()])
    return on_floor & across, on_floor & ~across


class AnchorError(ValueError):
    """The labelled court points and the lines in the frame do not give a trustworthy camera."""


def _radii(radii: tuple[float, ...], size: tuple[int, int]) -> list[float]:
    return [max(MIN_RADIUS_PX, radius * max(size) / REFERENCE_LONG_SIDE_PX) for radius in radii]


def _inlier_px(size: tuple[int, int]) -> float:
    return INLIER_PX * max(1.0, max(size) / 1920.0)


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
        priors = [PRIOR_FOCAL_PX_PER_LOG * (params[6] - focal_start), PRIOR_K1_PX_PER_UNIT * params[7]]
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
    for radius in _radii(radii_px, (width, height)):
        pixels, normals, visible = _project(current, (width, height))
        candidates = np.flatnonzero(visible)
        # The search covers the position error plus half the line: near the phone a line is wide.
        depth = points[candidates] @ current.rotation[2] + float(np.asarray(current.tvec, dtype=np.float64).reshape(3)[2])
        search = radius + 0.5 * _widths()[candidates] * current.focal_px / depth + 1.0
        offsets = find_line_offsets(response, pixels[candidates], normals[candidates], search)
        found = np.isfinite(offsets)
        if found.sum() < MIN_INLIERS:
            return None
        matched = candidates[found]
        targets = pixels[matched] + offsets[found, None] * normals[matched]
        current = _refine(current, points[matched], targets, normals[matched], fit_intrinsics)
    distance = ((current.project(points[matched]) - targets) * normals[matched]).sum(axis=1)
    _, _, visible = _project(current, (width, height))
    inlier = np.abs(distance) <= _inlier_px((width, height))
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


def _shift_scores(response: np.ndarray, pixels: np.ndarray, shifts: np.ndarray, blur_px: float) -> np.ndarray:
    """Mean line response under the projected model for each image shift (blurred so a near miss still scores)."""
    blurred = cv2.GaussianBlur(response, (0, 0), max(1.0, blur_px))
    where = pixels[None, :, :] + shifts[:, None, :]
    return cv2.remap(
        blurred,
        where[..., 0].astype(np.float32),
        where[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).mean(axis=1)


def _shift_candidates(response: np.ndarray, camera: Camera, max_shift_px: float) -> list[np.ndarray]:
    """The best few image shifts of the projected model onto the line response. Banners and
    spectators respond too, so the top score can be wrong; each candidate is checked by a fit."""
    height, width = response.shape
    pixels, _, visible = _project(camera, (width, height))
    pixels = pixels[visible]
    if len(pixels) == 0:
        return []
    step = max(1.0, max_shift_px / 10)
    grid = np.arange(-max_shift_px, max_shift_px + 1e-9, step)
    shifts = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    scores = _shift_scores(response, pixels, shifts, step).reshape(len(grid), len(grid)).astype(np.float32)
    peaks = (scores >= cv2.dilate(scores, np.ones((3, 3), np.uint8))).ravel()
    order = [index for index in np.argsort(-scores.ravel()) if peaks[index]][:SHIFT_CANDIDATES]
    candidates = []
    for index in order:
        best, span, fine = shifts[index], step, max(1.0, step / 4)
        while True:
            local = np.arange(-span, span + 1e-9, fine)
            nearby = best + np.stack(np.meshgrid(local, local), axis=-1).reshape(-1, 2)
            best = nearby[int(np.argmax(_shift_scores(response, pixels, nearby, fine)))]
            if fine <= 1.0:
                break
            span, fine = fine, max(1.0, fine / 4)
        candidates.append(best)
    return candidates


def _moved(camera: Camera, dx: float, dy: float) -> Camera:
    """The camera moved over the floor by (dx, dy) metres, facing the same way."""
    centre = camera.centre + np.array([dx, dy, 0.0])
    return Camera(camera.focal_px, camera.cx, camera.cy, camera.rvec, -camera.rotation @ centre, camera.k1)


@lru_cache(maxsize=1)
def _alias_slides() -> list[tuple[float, float]]:
    """One line spacing in each direction: across, doubles to singles sideline and sideline to
    centre line; along, baseline to long service line and the long gaps between service lines."""
    across = {COLUMNS["sl"] - COLUMNS["dl"], COLUMNS["c"] - COLUMNS["sl"]}
    along = {ROWS["long0"] - ROWS["back0"], ROWS["short0"] - ROWS["long0"], ROWS["short1"] - ROWS["short0"]}
    return [(sign * step, 0.0) for step in across for sign in (1, -1)] + [(0.0, sign * step) for step in along for sign in (1, -1)]


def _resolve_aliases(response: np.ndarray, fit: LineFit, fit_intrinsics: bool) -> LineFit:
    """Refit from the court slid by each line spacing; move while a slide explains clearly more."""
    for _ in range(ALIAS_ROUNDS):
        rivals = (fit_frame(response, _moved(fit.camera, dx, dy), ACQUIRE_RADII_PX, fit_intrinsics) for dx, dy in _alias_slides())
        best = max((rival for rival in rivals if rival is not None and rival.confident), key=lambda rival: rival.inliers, default=None)
        if best is None or best.inliers < ALIAS_GAIN * fit.inliers:
            return fit
        fit = best
    return fit


def acquire(response: np.ndarray, camera: Camera, fit_intrinsics: bool = False) -> LineFit | None:
    """Find the court when the camera is only roughly known: after the first calibration, or
    after the court was lost. Coarse-to-fine fits from `camera` and from the best whole-court
    shifts of the model onto the lines; a partial lock can look confident, so the fit that
    explains the most of the court wins, and then the slid courts are tried against it."""
    max_shift = ACQUIRE_SHIFT_FRACTION * max(response.shape)
    starts = [camera] + [_turn_to_shift(camera, shift) for shift in _shift_candidates(response, camera, max_shift)]
    fits = [fit for fit in (fit_frame(response, start, ACQUIRE_RADII_PX, fit_intrinsics) for start in starts) if fit is not None]
    best = max(fits, key=lambda fit: (fit.confident, fit.inliers), default=None)
    if best is not None and best.confident:
        best = _resolve_aliases(response, best, fit_intrinsics)
    return best


def anchor_on_points(frame: np.ndarray, points: dict[str, tuple[float, float]]) -> LineFit:
    """The camera for a frame from court points placed on it (e.g. in the labeller): a first
    camera from the points, then the lines refine it and fix the focal length. The lines must
    put the points back where they were placed; a point with the wrong name (a doubles corner
    as a singles corner, the long service line as the baseline) cannot pass."""
    height, width = frame.shape[:2]
    solution = solve_camera(points, (width, height))
    if solution is None or solution.tier == "unavailable":
        raise AnchorError("the labelled court points do not give a usable camera: place at least 4, and check their names and positions")
    fit = acquire(line_response(frame), solution.camera, fit_intrinsics=True)
    if fit is None or not fit.confident:
        raise AnchorError("the court lines could not be found near the labelled points")
    names = [name for name in points if name in KEYPOINTS]
    placed = np.array([points[name] for name in names], dtype=np.float64)
    reprojected = fit.camera.project(np.array([KEYPOINTS[name] for name in names]))
    miss = float(np.median(np.linalg.norm(reprojected - placed, axis=1)))
    if not miss <= ANCHOR_TAP_FRACTION * max(width, height):
        raise AnchorError(
            f"the court lines put the labelled points {miss:.0f} px (median) from where they were placed: "
            "check each point's name and the frame it was placed on"
        )
    return fit


def _small_step(previous: Camera, current: Camera) -> bool:
    turn = cv2.Rodrigues(current.rotation @ previous.rotation.T)[0]
    return np.degrees(np.linalg.norm(turn)) <= MAX_STEP_DEG and np.linalg.norm(current.centre - previous.centre) <= MAX_STEP_M


def iter_track(frames: Iterable[np.ndarray], anchor: Camera, fit_intrinsics: bool = True) -> Iterator[TrackedFrame]:
    """A camera per frame, yielded as each frame is read, starting from `anchor` (e.g. from
    anchor_on_points). The first frame is searched for from the anchor the same way as after a
    loss, so tracking can start before the anchor's frame as long as the phone was within that
    search of the anchor's pose (otherwise those frames are lost until it is).

    States: "anchored" (first found), "tracked" (fitted from the previous frame and within a
    steady hand's step of it), "reanchored" (found again by the full search, after a loss or a
    jump bigger than a steady hand makes), "lost" (no camera)."""
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
            yield TrackedFrame("lost", None, fit.inliers if fit else 0, fit.rms_px if fit else float("inf"))
            continue
        assert fit is not None
        following, anchored, last_good = True, True, fit.camera
        yield TrackedFrame(state, fit.camera, fit.inliers, fit.rms_px)


def model_polylines(camera: Camera, size: tuple[int, int]) -> list[np.ndarray]:
    """The court model as image polylines (one per model line, visible samples only), for drawing."""
    _, _, line_ids = _model()
    pixels, _, visible = _project(camera, size)
    return [pixels[(line_ids == line) & visible] for line in range(len(_lines())) if ((line_ids == line) & visible).sum() >= 2]
