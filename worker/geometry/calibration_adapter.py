# worker/geometry/calibration_adapter.py
"""Bridge between the UI corner taps and the camera solver."""
from __future__ import annotations

import math
from typing import Any

from geometry.calibrate import solve_camera
from geometry.court_model import CORNER_LABEL_TO_KEYPOINT
from geometry.quality import assess_geometry

UNSOLVED_REASON = (
    "The tapped corners do not match a real court view (check that left/right and near/far are not swapped); "
    "court geometry is unavailable for this clip."
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def corners_to_observations(corners: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, tuple[float, float]]:
    """UI corners are percent-of-frame; return pixel observations keyed by court keypoint."""
    width, height = image_size
    observed: dict[str, tuple[float, float]] = {}
    for corner in corners:
        name = CORNER_LABEL_TO_KEYPOINT.get(corner.get("label"))
        x, y = corner.get("x"), corner.get("y")
        if name is None or not (_is_number(x) and _is_number(y)):
            continue
        observed[name] = (float(x) * width / 100, float(y) * height / 100)
    return observed


def summarise_calibration(
    corners: list[dict[str, Any]], image_size: tuple[int, int], court_type: str = "singles"
) -> dict[str, Any] | None:
    """Camera facts to attach to the accepted calibration; None when the corners cannot be solved."""
    if court_type != "singles":
        # v1 models the singles court only; doubles corners would be solved against the wrong lines.
        return None
    solution = solve_camera(corners_to_observations(corners, image_size), image_size)
    if solution is None:
        return None
    quality = assess_geometry(solution.camera, image_size)
    # Capture advice from a camera the solver rejected would be invented, not measured.
    trusted = solution.tier != "unavailable"
    return {
        "cameraTier": solution.tier,
        "rmsPx": round(solution.rms_px, 3),
        "focalPx": round(solution.camera.focal_px, 1),
        "k1": round(solution.camera.k1, 4),
        "looFloorCm": None if solution.loo_floor_cm is None else round(solution.loo_floor_cm, 2),
        "geometryTier": quality.tier if trusted else "C",
        "elevationDeg": round(quality.elevation_deg, 1),
        "visibleFraction": round(quality.visible_fraction, 3),
        "reasons": list(quality.reasons) if trusted else [UNSOLVED_REASON],
    }
