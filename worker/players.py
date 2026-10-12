# worker/players.py
"""Who is who: the two players among everyone a pose detector finds.

Every detected person's ankles are put on the floor through the frame's camera. A player is a
person standing in one half of the court, with room around the lines for a player chasing a
shot (CHASE_SIDE_M beside the doubles sidelines, CHASE_BACK_M behind the baseline). The umpire
on the chair, line judges, spectators and the next court's players stand elsewhere. From one
frame to the next a player can only have run so far, so identity carries by position.
Singles: one player per half.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, NET_Y_M, SINGLES_WIDTH_M

ANKLE_HEIGHT_M = 0.08  # an ankle joint's height above the floor when the foot is down
MIN_JOINT_SCORE = 0.3
CHASE_SIDE_M = 1.0  # room beside the doubles sideline a player may be in
CHASE_BACK_M = 2.5  # and behind the baseline
MAX_SPEED_M_S = 9.0  # no player runs faster
GATE_M = 0.6  # and the floor position wobbles this much from frame to frame on its own
NOISE_PX = 4.0  # a detector's joint error, which far away moves the floor point by much more
HEAD_HEIGHT_M = (0.8, 2.6)  # a nose above the feet, crouched in a lunge to jumping: a standing person
ANKLES = (15, 16)
NOSE, SHOULDERS = 0, (5, 6)
HALVES = ("near", "far")


@dataclass(frozen=True)
class Person:
    """One person as a pose detector reports them."""

    joints: np.ndarray  # (17, 2) pixels, COCO order
    scores: np.ndarray  # (17,) confidences
    box: np.ndarray | None = None  # x0, y0, x1, y1


@dataclass(frozen=True)
class PlayerFrame:
    detection: int  # index of the person in that frame's detections
    floor: tuple[float, float]  # where the player stands, court metres
    joints: np.ndarray
    scores: np.ndarray


def floor_position(person: Person, camera: Camera) -> tuple[float, float] | None:
    """Where a person stands: their ankles put on the floor through the camera (or, with no
    ankle seen, the bottom of their box at floor level)."""
    ankles = [person.joints[j] for j in ANKLES if person.scores[j] >= MIN_JOINT_SCORE]
    if ankles:
        pixel, height = np.mean(ankles, axis=0), ANKLE_HEIGHT_M
    elif person.box is not None:
        pixel, height = np.array([(person.box[0] + person.box[2]) / 2, person.box[3]]), 0.0
    else:
        return None
    point = camera.pixel_to_plane(np.asarray(pixel, dtype=float)[None], height)[0]
    return (float(point[0]), float(point[1])) if np.isfinite(point).all() else None


def floor_spread(person: Person, camera: Camera) -> float:
    """How far NOISE_PX of detector error moves a person's floor point: a few centimetres close
    to the camera, most of a metre at the far baseline seen from behind."""
    ankles = [person.joints[j] for j in ANKLES if person.scores[j] >= MIN_JOINT_SCORE]
    if not ankles:
        return GATE_M
    pixel = np.mean(ankles, axis=0)
    points = camera.pixel_to_plane(np.array([pixel - (0, NOISE_PX), pixel + (0, NOISE_PX)], dtype=float), ANKLE_HEIGHT_M)
    return float(np.hypot(*(points[1] - points[0]))) / 2 if np.isfinite(points).all() else GATE_M


def _ray(camera: Camera, pixel: np.ndarray) -> np.ndarray:
    """The world direction of a pixel's line of sight (undistorted as Camera.pixel_to_plane does)."""
    xd, yd = (pixel[0] - camera.cx) / camera.focal_px, (pixel[1] - camera.cy) / camera.focal_px
    x, y = xd, yd
    for _ in range(12):
        scale = 1.0 + camera.k1 * (x * x + y * y)
        x, y = xd / scale, yd / scale
    return np.array([x, y, 1.0]) @ camera.rotation


def stands_there(person: Person, camera: Camera, floor: tuple[float, float]) -> bool:
    """Whether the person really stands at that floor point: their head (or shoulders) must be
    at a human height straight above it. A person seated high (the umpire) is not where their
    feet put them, and their head says so."""
    if person.scores[NOSE] >= MIN_JOINT_SCORE:
        pixel, (low, high) = person.joints[NOSE], HEAD_HEIGHT_M
    else:
        shoulders = [person.joints[j] for j in SHOULDERS if person.scores[j] >= MIN_JOINT_SCORE]
        if not shoulders:
            return True  # nothing to check against
        pixel, (low, high) = np.mean(shoulders, axis=0), (0.85 * HEAD_HEIGHT_M[0], 0.85 * HEAD_HEIGHT_M[1])
    ray, centre = _ray(camera, np.asarray(pixel, dtype=float)), camera.centre
    across = ray[:2]
    if float(across @ across) < 1e-12:
        return True  # looking straight down: the floor point says it all
    along = float((np.asarray(floor) - centre[:2]) @ across / (across @ across))
    return along > 0 and low <= centre[2] + along * ray[2] <= high


def half_of(floor: tuple[float, float]) -> str | None:
    """The half a person standing there plays in, or None when nobody playing stands there."""
    x, y = floor
    if not -DOUBLES_MARGIN_M - CHASE_SIDE_M <= x <= SINGLES_WIDTH_M + DOUBLES_MARGIN_M + CHASE_SIDE_M:
        return None
    if -CHASE_BACK_M <= y < NET_Y_M:
        return "near"
    if NET_Y_M < y <= COURT_LENGTH_M + CHASE_BACK_M:
        return "far"
    return None


def track_players(
    frames: Sequence[Sequence[Person]], cameras: Sequence[Camera | None], times_ms: Sequence[float]
) -> dict[str, list[PlayerFrame | None]]:
    """Per frame, each half's player: the person in that half who continues the player's track
    (within reach of a run from where they last were), or, with no track yet or after losing it,
    the one standing most squarely in the half. None where the player is not found."""
    tracks: dict[str, list[PlayerFrame | None]] = {half: [] for half in HALVES}
    last: dict[str, tuple[float, np.ndarray, float] | None] = {half: None for half in HALVES}
    for people, camera, time_ms in zip(frames, cameras, times_ms):
        floors = [floor_position(person, camera) if camera is not None else None for person in people]
        floors = [floor if floor is not None and stands_there(person, camera, floor) else None for person, floor in zip(people, floors)]
        for half in HALVES:
            candidates = [k for k, floor in enumerate(floors) if floor is not None and half_of(floor) == half]
            chosen = None
            if candidates and last[half] is None:
                chosen = min(candidates, key=lambda k: _offcourt(floors[k]))  # the first sighting: the most plausible
            elif candidates:
                # Within reach of a run from where the player was last seen; the reach grows while
                # they are unseen, so a lost player is found again rather than swapped for another.
                then, where, spread = last[half]
                reach = GATE_M + MAX_SPEED_M_S * (time_ms - then) / 1000.0 + 3 * spread
                distance = {k: float(np.hypot(*(np.asarray(floors[k]) - where))) for k in candidates}
                near = [k for k in candidates if distance[k] <= reach + 3 * floor_spread(people[k], camera)]
                chosen = min(near, key=distance.get) if near else None
            if chosen is None:
                tracks[half].append(None)
                continue
            person = people[chosen]
            last[half] = (float(time_ms), np.asarray(floors[chosen]), floor_spread(person, camera))
            tracks[half].append(PlayerFrame(chosen, floors[chosen], person.joints, person.scores))
    return tracks


def _offcourt(floor: tuple[float, float]) -> float:
    """How far outside the singles court a floor point is (0 inside)."""
    x, y = floor
    return float(np.hypot(max(0.0, -x, x - SINGLES_WIDTH_M), max(0.0, -y, y - COURT_LENGTH_M)))
