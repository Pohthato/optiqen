# worker/simulation/people.py
"""People on and around a court, as a pose detector would report them: the 17 COCO joints of a
standing body (proportions of stature, after Drillis and Contini), placed in court metres,
projected through the camera, with detector noise, joints out of view and missed people."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, NET_Y_M, SINGLES_WIDTH_M

# Joint offsets as fractions of stature: (to the body's right, forward, up), COCO order.
STANDING = np.array([
    (0.00, 0.05, 0.94),  # nose
    (-0.03, 0.04, 0.95), (0.03, 0.04, 0.95),  # eyes
    (-0.06, 0.00, 0.94), (0.06, 0.00, 0.94),  # ears
    (-0.13, 0.00, 0.82), (0.13, 0.00, 0.82),  # shoulders
    (-0.17, 0.00, 0.63), (0.17, 0.00, 0.63),  # elbows
    (-0.18, 0.02, 0.49), (0.18, 0.02, 0.49),  # wrists
    (-0.09, 0.00, 0.53), (0.09, 0.00, 0.53),  # hips
    (-0.07, 0.01, 0.29), (0.07, 0.01, 0.29),  # knees
    (-0.06, 0.00, 0.04), (0.06, 0.00, 0.04),  # ankles
])


@dataclass(frozen=True)
class Detected:
    """One person as a detector reports them: joint pixels, joint confidences and a box."""

    joints: np.ndarray  # (17, 2)
    scores: np.ndarray  # (17,)
    box: np.ndarray  # (4,) x0, y0, x1, y1


def skeleton(position: tuple[float, float], facing: float, stature: float = 1.75, raised: float = 0.0) -> np.ndarray:
    """A standing body's 17 joints in court metres. `facing` is the direction the body faces
    (radians in the court plane, 0 along +x); `raised` lifts it (a chair, a jump)."""
    forward = np.array([np.cos(facing), np.sin(facing), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    base = np.array([position[0], position[1], raised])
    return base + stature * (STANDING[:, :1] * right + STANDING[:, 1:2] * forward + STANDING[:, 2:3] * up)


def detect(joints: np.ndarray, camera: Camera, size: tuple[int, int], rng: np.random.Generator, noise_px: float = 3.0) -> Detected | None:
    """What a detector reports for a body: joints in view with noise and high confidence, joints
    out of view with none; None when too little of the body is in view."""
    pixels = camera.project(joints) + rng.normal(0.0, noise_px, (len(joints), 2))
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < size[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < size[1])
    if inside.sum() < 5:
        return None
    scores = np.where(inside, rng.uniform(0.6, 0.95, len(joints)), 0.0)
    seen = pixels[inside]
    box = np.array([seen[:, 0].min(), seen[:, 1].min(), seen[:, 0].max(), seen[:, 1].max()])
    return Detected(np.where(inside[:, None], pixels, 0.0), scores, box)


def bystanders(rng: np.random.Generator) -> list[tuple[tuple[float, float], float, float, float]]:
    """People who are not the players: (position, facing, stature, raised). The umpire on a high
    chair at the net, line judges behind the baselines, spectators beyond the court, and a
    player on the next court."""
    people = [((-2.4, NET_Y_M), 0.0, 1.7, 1.2)]  # umpire's chair, facing across the court
    people += [((SINGLES_WIDTH_M + 2.5, y), np.pi, 1.7, 0.5) for y in (2.0, COURT_LENGTH_M - 2.0)]  # line judges, seated
    people += [((rng.uniform(-3, 8), rng.uniform(16.5, 19.0)), -np.pi / 2, rng.uniform(1.5, 1.9), 0.0) for _ in range(5)]
    people += [((SINGLES_WIDTH_M + 4.5, rng.uniform(2.0, 11.0)), np.pi / 2, 1.8, 0.0)]  # the next court's player
    return people
