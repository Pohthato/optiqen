# worker/geometry/quality.py
"""How much of the game a given camera placement can support."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, NET_Y_M, SINGLES_WIDTH_M

TIER_A_MIN_ELEVATION_DEG = 12.0
TIER_A_MIN_VISIBLE = 0.85
TIER_B_MIN_ELEVATION_DEG = 5.0
TIER_B_MIN_VISIBLE = 0.40


@dataclass(frozen=True)
class GeometryQuality:
    tier: str  # "A" | "B" | "C"
    elevation_deg: float
    visible_fraction: float
    distance_m: float
    reasons: tuple[str, ...]


def court_visibility(camera: Camera, image_size: tuple[int, int]) -> float:
    """Fraction of a regular grid over the singles court that lands inside the frame."""
    width, height = image_size
    xs = np.linspace(0.0, SINGLES_WIDTH_M, 11)
    ys = np.linspace(0.0, COURT_LENGTH_M, 21)
    grid = np.array([[x, y, 0.0] for y in ys for x in xs])
    pixels = camera.project(grid)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return float(inside.mean())


def assess_geometry(camera: Camera, image_size: tuple[int, int]) -> GeometryQuality:
    court_centre = np.array([SINGLES_WIDTH_M / 2, NET_Y_M, 0.0])
    offset = camera.centre - court_centre
    elevation = math.degrees(math.atan2(float(offset[2]), float(np.linalg.norm(offset[:2]))))
    distance = float(np.linalg.norm(offset))
    visible = court_visibility(camera, image_size)
    reasons: list[str] = []
    if elevation < TIER_B_MIN_ELEVATION_DEG:
        reasons.append("Camera is almost at floor level; court depth cannot be measured reliably. Raise the phone.")
    elif elevation < TIER_A_MIN_ELEVATION_DEG:
        reasons.append("Camera is low; depth estimates carry wide uncertainty. Raise the phone for full 3D analysis.")
    if visible < TIER_B_MIN_VISIBLE:
        reasons.append("Less than 40% of the court is in frame. Step back or widen the view.")
    elif visible < TIER_A_MIN_VISIBLE:
        reasons.append("Part of the court is out of frame. Step back until both baselines are visible for full analysis.")
    if elevation >= TIER_A_MIN_ELEVATION_DEG and visible >= TIER_A_MIN_VISIBLE:
        tier = "A"
    elif elevation >= TIER_B_MIN_ELEVATION_DEG and visible >= TIER_B_MIN_VISIBLE:
        tier = "B"
    else:
        tier = "C"
    return GeometryQuality(tier, elevation, visible, distance, tuple(reasons))
