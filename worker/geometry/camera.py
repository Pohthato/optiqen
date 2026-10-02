# worker/geometry/camera.py
"""Pinhole camera with one radial-distortion term.

World frame is the court frame from court_model (z up). `rvec`/`tvec` map world
points into the camera frame (OpenCV convention: x right, y down, z forward).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    focal_px: float
    cx: float
    cy: float
    rvec: np.ndarray
    tvec: np.ndarray
    k1: float = 0.0

    @property
    def rotation(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64).reshape(3, 1))[0]

    @property
    def centre(self) -> np.ndarray:
        return -self.rotation.T @ np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @classmethod
    def look_at(
        cls,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        focal_px: float,
        image_size: tuple[int, int],
        k1: float = 0.0,
    ) -> "Camera":
        position_v = np.asarray(position, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - position_v
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        rvec = cv2.Rodrigues(rotation)[0].ravel()
        tvec = -rotation @ position_v
        return cls(focal_px, image_size[0] / 2, image_size[1] / 2, rvec, tvec, k1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2); NaN where the point is behind the camera."""
        world = np.atleast_2d(np.asarray(points, dtype=np.float64))
        cam = world @ self.rotation.T + np.asarray(self.tvec, dtype=np.float64).reshape(3)
        out = np.full((len(world), 2), np.nan)
        front = cam[:, 2] > 1e-6
        x = cam[front, 0] / cam[front, 2]
        y = cam[front, 1] / cam[front, 2]
        scale = 1.0 + self.k1 * (x * x + y * y)
        out[front, 0] = self.focal_px * x * scale + self.cx
        out[front, 1] = self.focal_px * y * scale + self.cy
        return out

    def pixel_to_plane(self, pixels: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Pixels (N, 2) -> world (x, y) on the horizontal plane at height z; NaN if the ray misses it."""
        px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        xd = (px[:, 0] - self.cx) / self.focal_px
        yd = (px[:, 1] - self.cy) / self.focal_px
        x, y = xd.copy(), yd.copy()
        for _ in range(12):
            scale = 1.0 + self.k1 * (x * x + y * y)
            x, y = xd / scale, yd / scale
        rays = np.stack([x, y, np.ones_like(x)], axis=1) @ self.rotation
        centre = self.centre
        out = np.full((len(px), 2), np.nan)
        dz = rays[:, 2]
        valid = np.abs(dz) > 1e-9
        s = np.full(len(px), np.nan)
        s[valid] = (z - centre[2]) / dz[valid]
        hit = valid & (s > 0)
        out[hit] = centre[:2] + s[hit, None] * rays[hit, :2]
        return out
