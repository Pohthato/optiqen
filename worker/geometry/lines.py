# worker/geometry/lines.py
"""Painted court lines in a frame: a line-likeness image, and where each predicted line really sits."""
from __future__ import annotations

import cv2
import numpy as np

END_CLEARANCE_M = 0.12  # skip line ends, where two lines cross and the profile is ambiguous
RIVAL_RATIO = 0.5  # a peak at least this fraction of the strongest one competes on distance
MIN_STRENGTH = 0.04
SEARCH_STEP_PX = 0.5


def line_response(frame: np.ndarray, max_line_px: int | None = None) -> np.ndarray:
    """Float32 image in [0, 1]: how much brighter each pixel is than its surroundings at the
    scale of a painted line (white top-hat). Floors, walls and lighting gradients give ~0."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    size = max_line_px or max(7, gray.shape[0] // 30)
    size += 1 - size % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return tophat.astype(np.float32) / 255.0


def sample_lines(
    lines: list[tuple[str, np.ndarray, np.ndarray]], spacing_m: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points along each (name, start, end) segment, ends excluded; the segment's unit
    direction at each point; and the segment's index in `lines`."""
    points, directions, line_ids = [], [], []
    for line_id, (_, start, end) in enumerate(lines):
        length = float(np.linalg.norm(end - start))
        along = (end - start) / length
        count = max(1, int((length - 2 * END_CLEARANCE_M) // spacing_m))
        span = length - 2 * END_CLEARANCE_M
        for distance in END_CLEARANCE_M + span * (np.arange(count) + 0.5) / count:
            points.append(start + along * distance)
            directions.append(along)
            line_ids.append(line_id)
    return np.array(points), np.array(directions), np.array(line_ids, dtype=int)


def find_line_offsets(
    response: np.ndarray,
    pixels: np.ndarray,
    normals: np.ndarray,
    radius_px: float | np.ndarray,
    min_strength: float = MIN_STRENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """For each predicted line pixel, the signed distance along its unit normal to the centre
    of the line actually seen there, and that line's strength (NaN and 0 when none is found).

    The profile along the normal is searched within radius_px (one for all points, or one each). Of the peaks at least half as
    strong as the best one, the nearest wins, so a neighbouring parallel line a few pixels away
    does not steal the match. The centre is the centroid of the peak above half its height, which
    stays accurate for lines many pixels wide. Bright areas wider than the search are rejected.
    """
    pixels = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    normals = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    count = len(pixels)
    offsets = np.full(count, np.nan)
    strengths = np.zeros(count)
    if count == 0:
        return offsets, strengths
    height, width = response.shape
    radii = np.broadcast_to(np.asarray(radius_px, dtype=np.float64), (count,))
    steps = np.arange(-radii.max(), radii.max() + 1e-9, SEARCH_STEP_PX)
    window = np.abs(steps)[None, :] <= radii[:, None] + 1e-9
    where = pixels[:, None, :] + steps[None, :, None] * normals[:, None, :]
    profiles = cv2.remap(
        response,
        where[..., 0].astype(np.float32),
        where[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float64)
    profiles[~window] = 0.0
    inside = (where[..., 0] >= 0) & (where[..., 0] <= width - 1) & (where[..., 1] >= 0) & (where[..., 1] <= height - 1)
    usable = (inside | ~window).all(axis=1)

    left = np.pad(profiles[:, :-1], ((0, 0), (1, 0)), constant_values=-np.inf)
    right = np.pad(profiles[:, 1:], ((0, 0), (0, 1)), constant_values=-np.inf)
    peaks = (profiles >= left) & (profiles > right) & (profiles >= min_strength) & window
    strongest = np.where(peaks, profiles, 0.0).max(axis=1)
    rivals = peaks & (profiles >= RIVAL_RATIO * strongest[:, None])
    distance = np.where(rivals, np.abs(steps)[None, :], np.inf)
    choice = distance.argmin(axis=1)
    found = usable & np.isfinite(distance.min(axis=1))

    rows = np.arange(count)
    peak = profiles[rows, choice]
    half = 0.5 * peak
    above = profiles > half[:, None]
    run = np.cumsum(~above, axis=1)
    in_run = above & (run == run[rows, choice][:, None])
    first = window.argmax(axis=1)
    last = window.shape[1] - 1 - window[:, ::-1].argmax(axis=1)
    found &= ~in_run[rows, first] & ~in_run[rows, last]
    weights = np.where(in_run, profiles - half[:, None], 0.0)
    total = weights.sum(axis=1)
    found &= total > 0
    offsets[found] = (weights[found] * steps[None, :]).sum(axis=1) / total[found]
    strengths[found] = peak[found]
    return offsets, strengths
