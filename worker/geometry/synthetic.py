# worker/geometry/synthetic.py
"""Synthetic cameras and keypoint observations for tests and evaluation."""
from __future__ import annotations

import numpy as np

from geometry.camera import Camera
from geometry.court_model import COLUMNS, KEYPOINTS, NET_Y_M

LANDSCAPE = (1920, 1080)
PORTRAIT = (1080, 1920)
CENTRE_X = COLUMNS["c"]
FOCAL_PX = 1300.0


def standard_scenes() -> dict[str, tuple[Camera, tuple[int, int]]]:
    """Camera placements a phone user plausibly holds, tripod or hand."""
    aim = (CENTRE_X, NET_Y_M, 0.0)
    return {
        "behind_elevated": (Camera.look_at((CENTRE_X, -3.5, 3.0), (CENTRE_X, 7.0, 0.0), FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "behind_low": (Camera.look_at((CENTRE_X + 0.4, -2.5, 1.5), (CENTRE_X, 7.0, 0.0), FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "side_elevated": (Camera.look_at((-4.5, NET_Y_M, 3.5), aim, FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "corner_elevated": (Camera.look_at((-3.0, -3.0, 3.5), aim, FOCAL_PX, LANDSCAPE), LANDSCAPE),
        "behind_portrait": (Camera.look_at((CENTRE_X, -4.0, 3.2), (CENTRE_X, 6.0, 0.0), FOCAL_PX, PORTRAIT), PORTRAIT),
    }


def observe(
    camera: Camera,
    image_size: tuple[int, int],
    names: list[str] | None = None,
    noise_px: float = 0.0,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Pixel observations of court keypoints that land inside the image."""
    rng = np.random.default_rng(seed)
    selected = names if names is not None else list(KEYPOINTS)
    pixels = camera.project(np.array([KEYPOINTS[name] for name in selected]))
    width, height = image_size
    observed: dict[str, tuple[float, float]] = {}
    for name, pixel in zip(selected, pixels):
        if not np.isfinite(pixel).all():
            continue
        noisy = pixel + rng.normal(0.0, noise_px, size=2) if noise_px > 0 else pixel
        if 0 <= noisy[0] < width and 0 <= noisy[1] < height:
            observed[name] = (float(noisy[0]), float(noisy[1]))
    return observed
