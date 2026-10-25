# worker/players.py
"""Who is who: the two players among everyone a pose detector finds.

Every detected person's ankles are put on the floor through the frame's camera. A player is a
person standing in one half of the court, with room around the lines for a player chasing a
shot (CHASE_SIDE_M beside the doubles sidelines, CHASE_BACK_M behind the baseline). The umpire
on the chair, line judges, spectators and the next court's players stand elsewhere. From one
frame to the next a player can only have run so far, so identity carries by position.
Officials, coaches and spectators on chairs beside and behind the court sit where a player
could be, and from a low camera their feet put them anywhere along their line of sight. Their
heads give them away: low, and still at one place in the picture for a second. A player only
crouches or lunges for a moment, and waits standing. Singles: one player per half.
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
SEATED_HEAD_M = 1.3  # a nose lower than this is a seated person's, or a player's crouched or lunging
SEATED_S = 1.0  # a low head at one place this long, over the clip, is someone sitting there
STILL_PX = 2 * NOISE_PX  # how far a seated person's head moves in the picture
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


def head(person: Person, camera: Camera, floor: tuple[float, float]) -> tuple[np.ndarray, float] | None:
    """The person's head (nose, or else shoulders) if they stand at that floor point: the point on
    its line of sight straight above the floor point, in court metres, and its height as a nose's
    (shoulders are 0.85 of that). None with no head seen, or looking straight down."""
    if person.scores[NOSE] >= MIN_JOINT_SCORE:
        pixel, scale = person.joints[NOSE], 1.0
    else:
        shoulders = [person.joints[j] for j in SHOULDERS if person.scores[j] >= MIN_JOINT_SCORE]
        if not shoulders:
            return None
        pixel, scale = np.mean(shoulders, axis=0), 0.85
    ray, centre = _ray(camera, np.asarray(pixel, dtype=float)), camera.centre
    across = ray[:2]
    if float(across @ across) < 1e-12:
        return None
    point = centre + float((np.asarray(floor) - centre[:2]) @ across / (across @ across)) * ray
    return point, float(point[2]) / scale


def stands_there(person: Person, camera: Camera, floor: tuple[float, float]) -> bool:
    """Whether the person really stands at that floor point: their head (or shoulders) must be
    at a human height straight above it. A person seated high (the umpire) is not where their
    feet put them, and their head says so."""
    found = head(person, camera, floor)
    if found is None:
        return True  # nothing to check against
    point, height = found
    in_front = float((point - camera.centre) @ camera.rotation[2]) > 0
    return in_front and HEAD_HEIGHT_M[0] <= height <= HEAD_HEIGHT_M[1]


def sitting(heads: Sequence[Sequence[tuple[np.ndarray, float] | None]], cameras: Sequence[Camera | None], times_ms: Sequence[float]) -> list[list[bool]]:
    """Per frame, which people are sitting: their head is low, and over the clip a low head is
    at that place in the picture (within STILL_PX, each frame's put through this frame's
    camera) for SEATED_S in all. `heads` is head() of each person per frame (None without one)."""
    low = [(index, found[0]) for index, row in enumerate(heads) for found in row if found is not None and found[1] < SEATED_HEAD_M]
    flags = [[False] * len(row) for row in heads]
    if not low or len(times_ms) < 2:
        return flags
    frame_s = (times_ms[-1] - times_ms[0]) / (len(times_ms) - 1) / 1000.0
    which, points = np.array([index for index, _ in low]), np.array([point for _, point in low])
    for row, camera, out in zip(heads, cameras, flags):
        if camera is None or not any(found is not None and found[1] < SEATED_HEAD_M for found in row):
            continue
        pixels = camera.project(points)
        for k, found in enumerate(row):
            if found is not None and found[1] < SEATED_HEAD_M:
                here = camera.project(found[0])[0]
                same = np.hypot(*(pixels - here).T) <= STILL_PX
                out[k] = len(np.unique(which[same])) * frame_s >= SEATED_S
    return flags


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
    """Per frame, each half's player: the person standing in that half who continues the player's
    track (within reach of a run from where they last were), or, with no track yet or after
    losing it, the one standing most squarely in the half. None where the player is not found."""
    tracks: dict[str, list[PlayerFrame | None]] = {half: [] for half in HALVES}
    last: dict[str, tuple[float, np.ndarray, float] | None] = {half: None for half in HALVES}
    seen = [[floor_position(person, camera) if camera is not None else None for person in people] for people, camera in zip(frames, cameras)]
    heads = [[head(person, camera, floor) if floor is not None else None for person, floor in zip(people, row)] for people, camera, row in zip(frames, cameras, seen)]
    for people, camera, time_ms, row, sits in zip(frames, cameras, times_ms, seen, sitting(heads, cameras, times_ms)):
        floors = [floor if floor is not None and not sat and stands_there(person, camera, floor) else None for person, floor, sat in zip(people, row, sits)]
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
