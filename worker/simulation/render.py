# worker/simulation/render.py
"""Draw one synthetic frame of a singles court as a phone would see it."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from geometry.camera import Camera
from geometry.court_model import COLUMNS, COURT_LENGTH_M, NET_CENTRE_HEIGHT_M, NET_POST_HEIGHT_M, NET_Y_M, court_lines

FLOOR_BGR = (64, 112, 44)
LINE_BGR = (232, 232, 232)
WALL_TOP_BGR = np.array([70.0, 70.0, 75.0])
WALL_BOTTOM_BGR = np.array([120.0, 118.0, 115.0])
NET_BGR = (40, 40, 40)
TAPE_BGR = (240, 240, 240)
POST_BGR = (30, 30, 160)
PLAYER_BGR = (60, 45, 35)
SHUTTLE_BGR = (250, 250, 250)
LINE_WIDTH_M = 0.04
LINE_SEGMENT_M = 0.25
NET_DEPTH_M = 0.76
TAPE_DEPTH_M = 0.075
NET_OPACITY = 0.35
PLAYER_RADIUS_M = 0.25
PLAYER_HEIGHT_M = 1.75
SHUTTLE_SIZE_M = 0.065
_SHIFT = 4
_SCALE = 1 << _SHIFT
_MAX_PIXEL = 1e5


def _to_fixed(pixels: np.ndarray) -> np.ndarray | None:
    """Fixed-point int32 pixels for OpenCV's sub-pixel drawing; None if anything is behind or absurdly far."""
    if not np.isfinite(pixels).all() or np.abs(pixels).max() > _MAX_PIXEL:
        return None
    return np.round(pixels * _SCALE).astype(np.int32)


def _fill(image: np.ndarray, camera: Camera, corners, color, antialias: bool = True) -> None:
    fixed = _to_fixed(camera.project(np.asarray(corners, dtype=np.float64)))
    if fixed is not None:
        cv2.fillConvexPoly(image, fixed, color, lineType=cv2.LINE_AA if antialias else cv2.LINE_8, shift=_SHIFT)


def _depth(camera: Camera, point) -> float:
    return float((camera.rotation @ np.asarray(point, dtype=np.float64) + np.asarray(camera.tvec, dtype=np.float64).reshape(3))[2])


def _background(height: int, width: int) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, height)[:, None]
    rows = (WALL_TOP_BGR[None, :] * (1 - ramp) + WALL_BOTTOM_BGR[None, :] * ramp).astype(np.uint8)
    return np.repeat(rows[:, None, :], width, axis=1)


def _draw_floor(image: np.ndarray, camera: Camera) -> None:
    # 1 m tiles with a faint checker texture; no anti-aliasing so tiles meet without seams.
    for x0 in np.arange(-3.0, 8.0, 1.0):
        for y0 in np.arange(-6.0, COURT_LENGTH_M + 6.0, 1.0):
            shade = 4 * ((int(x0) + int(y0)) % 3 - 1)
            color = tuple(int(c + shade) for c in FLOOR_BGR)
            corners = [[x0, y0, 0.0], [x0 + 1, y0, 0.0], [x0 + 1, y0 + 1, 0.0], [x0, y0 + 1, 0.0]]
            _fill(image, camera, corners, color, antialias=False)


def _draw_lines(image: np.ndarray, camera: Camera) -> None:
    half = LINE_WIDTH_M / 2
    for _, start, end in court_lines():
        length = float(np.linalg.norm(end - start))
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0], 0.0]) * half
        steps = max(1, int(np.ceil(length / LINE_SEGMENT_M)))
        for i in range(steps):
            a = start + (end - start) * (i / steps)
            b = start + (end - start) * ((i + 1) / steps)
            _fill(image, camera, [a - normal, b - normal, b + normal, a + normal], LINE_BGR)


def _draw_net(image: np.ndarray, camera: Camera) -> None:
    left, centre, right = COLUMNS["dl"], COLUMNS["c"], COLUMNS["dr"]
    spans = [((left, NET_POST_HEIGHT_M), (centre, NET_CENTRE_HEIGHT_M)), ((centre, NET_CENTRE_HEIGHT_M), (right, NET_POST_HEIGHT_M))]
    mesh = image.copy()
    for (xa, za), (xb, zb) in spans:
        _fill(mesh, camera, [[xa, NET_Y_M, za - TAPE_DEPTH_M], [xb, NET_Y_M, zb - TAPE_DEPTH_M], [xb, NET_Y_M, zb - NET_DEPTH_M], [xa, NET_Y_M, za - NET_DEPTH_M]], NET_BGR)
    cv2.addWeighted(mesh, NET_OPACITY, image, 1 - NET_OPACITY, 0, dst=image)
    for (xa, za), (xb, zb) in spans:
        _fill(image, camera, [[xa, NET_Y_M, za], [xb, NET_Y_M, zb], [xb, NET_Y_M, zb - TAPE_DEPTH_M], [xa, NET_Y_M, za - TAPE_DEPTH_M]], TAPE_BGR)
    for x in (left, right):
        ends = _to_fixed(camera.project(np.array([[x, NET_Y_M, 0.0], [x, NET_Y_M, NET_POST_HEIGHT_M]])))
        if ends is not None:
            thickness = max(1, int(round(0.04 * camera.focal_px / max(_depth(camera, (x, NET_Y_M, 0.8)), 0.1))))
            cv2.line(image, tuple(int(v) for v in ends[0]), tuple(int(v) for v in ends[1]), POST_BGR, thickness, cv2.LINE_AA, _SHIFT)


def _draw_player(image: np.ndarray, camera: Camera, xy: tuple[float, float]) -> None:
    angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    ring = np.column_stack([xy[0] + PLAYER_RADIUS_M * np.cos(angles), xy[1] + PLAYER_RADIUS_M * np.sin(angles)])
    points = np.vstack([np.column_stack([ring, np.zeros(16)]), np.column_stack([ring, np.full(16, PLAYER_HEIGHT_M)])])
    fixed = _to_fixed(camera.project(points))
    if fixed is not None:
        cv2.fillConvexPoly(image, cv2.convexHull(fixed), PLAYER_BGR, lineType=cv2.LINE_AA, shift=_SHIFT)


def _draw_shuttle(image: np.ndarray, camera: Camera, position, previous) -> None:
    centre = _to_fixed(camera.project(np.asarray([position], dtype=np.float64)))
    depth = _depth(camera, position)
    if centre is None or depth <= 0:
        return
    radius = max(1.0, SHUTTLE_SIZE_M * camera.focal_px / depth / 2)
    if previous is not None:
        start = _to_fixed(camera.project(np.asarray([previous], dtype=np.float64)))
        if start is not None:
            thickness = max(2, int(round(2 * radius)))
            cv2.line(image, tuple(int(v) for v in start[0]), tuple(int(v) for v in centre[0]), SHUTTLE_BGR, thickness, cv2.LINE_AA, _SHIFT)
    cv2.circle(image, tuple(int(v) for v in centre[0]), int(round(radius * _SCALE)), SHUTTLE_BGR, -1, cv2.LINE_AA, _SHIFT)


def render_frame(
    camera: Camera,
    image_size: tuple[int, int],
    shuttle=None,
    shuttle_previous=None,
    players=(),
    noise_sigma: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """BGR frame: walls, floor, lines, then net/players/shuttle far-to-near, then sensor noise."""
    width, height = image_size
    image = _background(height, width).copy()
    _draw_floor(image, camera)
    _draw_lines(image, camera)
    drawables: list[tuple[float, Callable[[], None]]] = [
        (_depth(camera, (COLUMNS["c"], NET_Y_M, 1.0)), lambda: _draw_net(image, camera)),
    ]
    for xy in players:
        drawables.append((_depth(camera, (xy[0], xy[1], PLAYER_HEIGHT_M / 2)), lambda xy=xy: _draw_player(image, camera, xy)))
    if shuttle is not None:
        drawables.append((_depth(camera, shuttle), lambda: _draw_shuttle(image, camera, shuttle, shuttle_previous)))
    for _, draw in sorted(drawables, key=lambda item: -item[0]):
        draw()
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noisy = image.astype(np.int16) + np.round(rng.normal(0.0, noise_sigma, image.shape)).astype(np.int16)
        image = np.clip(noisy, 0, 255).astype(np.uint8)
    return image
