# worker/simulation/camera_path.py
"""Camera motion for synthetic clips: a tripod, or a steadily held landscape phone."""
from __future__ import annotations

import cv2
import numpy as np

from geometry.camera import Camera

SHAKE_FREQUENCIES_HZ = (0.35, 0.9, 1.7)


def tripod(camera: Camera, frames: int) -> list[Camera]:
    return [camera] * frames


def _shake(frames: int, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth signal in [-1, 1]: mean of sinusoids with random phase and jittered frequency."""
    t = np.arange(frames) / fps
    signal = np.zeros(frames)
    for frequency in SHAKE_FREQUENCIES_HZ:
        signal += np.sin(2 * np.pi * frequency * rng.uniform(0.8, 1.2) * t + rng.uniform(0, 2 * np.pi))
    return signal / len(SHAKE_FREQUENCIES_HZ)


def stable_handheld(
    camera: Camera,
    frames: int,
    fps: float,
    seed: int = 0,
    rotation_deg: float = 0.3,
    translation_m: float = 0.015,
) -> list[Camera]:
    """A steadily held phone: small smooth rotation (camera axes) and translation (world axes)."""
    rng = np.random.default_rng(seed)
    rotation = np.stack([_shake(frames, fps, rng) for _ in range(3)], axis=1) * np.radians(rotation_deg)
    translation = np.stack([_shake(frames, fps, rng) for _ in range(3)], axis=1) * translation_m
    base_rotation, base_centre = camera.rotation, camera.centre
    path: list[Camera] = []
    for delta_rotation, delta_centre in zip(rotation, translation):
        rotation_matrix = cv2.Rodrigues(delta_rotation.reshape(3, 1))[0] @ base_rotation
        centre = base_centre + delta_centre
        path.append(
            Camera(
                camera.focal_px,
                camera.cx,
                camera.cy,
                cv2.Rodrigues(rotation_matrix)[0].ravel(),
                -rotation_matrix @ centre,
                camera.k1,
            )
        )
    return path
