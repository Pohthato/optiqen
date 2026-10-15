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


# Bone lengths as fractions of stature (Drillis and Contini), the ones a moving body keeps.
SEGMENTS = {
    "thigh": 0.245, "shank": 0.246, "upper_arm": 0.186, "forearm": 0.146,
    "hip_half": 0.09, "shoulder_half": 0.13, "torso": 0.29, "head": 0.12, "ankle": 0.04,
}
RACKET_M = 0.62  # from the hand to where the racket meets the shuttle
STROKE_S = 0.18  # how long either side of a hit the body is in its stroke
STEP_S = 0.25  # one step: each foot is planted this long, then swings this long to its next place
LIFT_M = 0.10  # how high a swinging foot is lifted, mid-step
BASE_FROM_NET_M = 2.8  # where a player heads between shots: on the centre line, this far from the net
RECOVER_M_S = 2.5  # how fast they head back there
RUN_M_S = 3.5  # and how fast, on average, to the next shot: further than that, they reach for it
REACT_S = 0.15  # after the opponent's hit, before they set off for the next shot (unless it is fast)
SPLIT_HOP_M = 0.05  # a split step: a hop this high, landing near the opponent's hit,
SPLIT_DIP_M = 0.07  # then the hips this much lower as the legs take the landing
SPLIT_HOP_S, SPLIT_DIP_S = 0.2, 0.3


def _bend(root: np.ndarray, target: np.ndarray, upper: float, lower: float, hint: np.ndarray) -> np.ndarray:
    """The middle joint of a two-bone limb from root to target, bent towards `hint`."""
    span = target - root
    distance = float(np.clip(np.linalg.norm(span), abs(upper - lower) + 1e-6, upper + lower - 1e-6))
    axis = span / max(np.linalg.norm(span), 1e-9)
    along = (upper**2 - lower**2 + distance**2) / (2 * distance)
    out = float(np.sqrt(max(upper**2 - along**2, 0.0)))
    side = hint - (hint @ axis) * axis
    side = side / max(np.linalg.norm(side), 1e-9)
    return root + along * axis + out * side


def _reach(root: np.ndarray, target: np.ndarray, length: float) -> np.ndarray:
    """The target if a limb of this length can touch it, else the nearest point it can."""
    span = target - root
    distance = float(np.linalg.norm(span))
    return target if distance <= length * 0.999 else root + span * (length * 0.999 / distance)


def body(
    stature: float, pelvis: np.ndarray, facing: float, feet: tuple[np.ndarray, np.ndarray],
    hands: tuple[np.ndarray, np.ndarray], lean: float = 0.0, turn: float = 0.0,
) -> np.ndarray:
    """The 17 COCO joints of a body with these bone lengths: pelvis centre, the direction it faces,
    where the (left, right) feet are on the floor and where the hands are, how far the trunk
    leans forward and how far the shoulders (with the head) are turned from the hips,
    anticlockwise seen from above (radians). Knees bend forward, elbows down and back; limbs
    that cannot reach reach as far as they can."""
    seg = {name: fraction * stature for name, fraction in SEGMENTS.items()}
    forward = np.array([np.cos(facing), np.sin(facing), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    trunk = np.cos(lean) * up + np.sin(lean) * forward
    hips = [pelvis - seg["hip_half"] * right, pelvis + seg["hip_half"] * right]
    neck = pelvis + seg["torso"] * trunk
    right, forward = np.cos(turn) * right + np.sin(turn) * forward, np.cos(turn) * forward - np.sin(turn) * right  # the shoulders' own
    shoulders = [neck - seg["shoulder_half"] * right, neck + seg["shoulder_half"] * right]
    leg = seg["thigh"] + seg["shank"]
    ankles = [_reach(hip, np.asarray(foot) + seg["ankle"] * up, leg) for hip, foot in zip(hips, feet)]
    knees = [_bend(hip, ankle, seg["thigh"], seg["shank"], forward) for hip, ankle in zip(hips, ankles)]
    arm = seg["upper_arm"] + seg["forearm"]
    wrists = [_reach(shoulder, np.asarray(hand), arm) for shoulder, hand in zip(shoulders, hands)]
    elbows = [_bend(shoulder, wrist, seg["upper_arm"], seg["forearm"], -up - 0.5 * forward) for shoulder, wrist in zip(shoulders, wrists)]
    nose = neck + seg["head"] * trunk + 0.05 * stature * forward
    eyes = [nose - 0.03 * stature * right + 0.01 * stature * up - 0.01 * stature * forward, nose + 0.03 * stature * right + 0.01 * stature * up - 0.01 * stature * forward]
    ears = [nose - 0.06 * stature * right - 0.05 * stature * forward, nose + 0.06 * stature * right - 0.05 * stature * forward]
    return np.array([nose, *eyes, *ears, *shoulders, *elbows, *wrists, *hips, *knees, *ankles])


def _sign(side: str) -> float:
    return 1.0 if side == "near" else -1.0


def _stance(contact, side: str, stature: float = 1.75) -> np.ndarray:
    """Where the body's centre is when the racket meets the shuttle: the racket shoulder an arm
    (60 to 100 percent stretched, nearly straight for an overhead) and a racket from the contact,
    behind it and to its left (a right-hander); the shoulder lower in a lunge, higher in a jump."""
    x, y, z = contact.position
    rng = _style(contact, side)
    stretch = rng.uniform(0.85, 1.0) if z > 1.8 else rng.uniform(0.6, 0.95)
    reach = RACKET_M + stretch * (SEGMENTS["upper_arm"] + SEGMENTS["forearm"]) * stature
    shoulder = 0.79 * stature + (0.3 if z > 2.7 else 0.0) - (0.13 * stature if z < 1.0 else 0.0)
    across = float(np.sqrt(max(reach**2 - (z - shoulder) ** 2, 0.04)))
    away = np.array([-0.3, -(0.95 if z < 1.0 else (0.3 if z > 1.8 else 0.55))]) * _sign(side)
    shoulder_at = np.array([x, y]) + across * away / np.linalg.norm(away)
    return shoulder_at - np.array([_sign(side), 0.0]) * SEGMENTS["shoulder_half"] * stature


def _waypoints(rally, side: str) -> list[tuple[float, np.ndarray]]:
    """(time, body centre) the player passes through: at each of their hits, their stance (or as
    near it as RUN_M_S allows); at each of the opponent's, as far back towards base as they got
    since their own hit (at RECOVER_M_S), where they wait REACT_S before setting off for the
    next, unless it comes fast."""
    from geometry.court_model import NET_Y_M, SINGLES_WIDTH_M

    base = np.array([SINGLES_WIDTH_M / 2, NET_Y_M - _sign(side) * BASE_FROM_NET_M])
    contacts = sorted(rally.contacts, key=lambda c: c.time)
    first = contacts[0]
    points = [(first.time - 1.0, _stance(first, side) if first.hitter == side else base)]
    for k, contact in enumerate(contacts):
        then, there = points[-1]
        if contact.hitter == side:
            gap = _stance(contact, side) - there
            room = max(contact.time - then, 0.0) * RUN_M_S
            points.append((contact.time, there + gap * min(1.0, room / max(float(np.linalg.norm(gap)), 1e-9))))
            continue
        gap = base - there
        room = max(contact.time - then - 0.1, 0.0) * RECOVER_M_S
        reached = there + gap * min(1.0, room / max(float(np.linalg.norm(gap)), 1e-9))
        points.append((contact.time, reached))
        if k + 1 == len(contacts) or contacts[k + 1].time - contact.time > REACT_S + 0.3:
            points.append((contact.time + REACT_S, reached))
    kept = [points[0]]
    for time, where in points[1:]:
        if time > kept[-1][0] + 1e-6:
            kept.append((time, where))
    return kept


def _centre(rally, side: str, time: float) -> np.ndarray:
    """The body's centre on the floor: between waypoints, a smooth start and stop (minimum jerk)."""
    points = _waypoints(rally, side)
    times = [t for t, _ in points]
    if time <= times[0]:
        return points[0][1].copy()
    if time >= times[-1]:
        return points[-1][1].copy()
    k = int(np.searchsorted(times, time))
    (t0, p0), (t1, p1) = points[k - 1], points[k]
    u = (time - t0) / (t1 - t0)
    return p0 + (p1 - p0) * (10 * u**3 - 15 * u**4 + 6 * u**5)


def _foot(rally, side: str, time: float, which: int, stature: float) -> np.ndarray:
    """Where one foot is (the ankle's floor point, lifted while it swings): planted where it last
    landed, or swinging to where it lands next. The feet take turns, each planted for STEP_S and
    swinging for STEP_S; a foot lands beside the body and, when the body moves, ahead of it by
    half the way the body goes while the foot is down."""
    facing = np.pi / 2 if side == "near" else -np.pi / 2
    right = np.array([np.sin(facing), -np.cos(facing)])
    across = (-1.0 if which == 0 else 1.0) * 0.16 * stature * right

    def landing(at: float) -> np.ndarray:
        centre = _centre(rally, side, at)
        velocity = (_centre(rally, side, at + 0.05) - centre) / 0.05
        return np.array([*(centre + across + 0.5 * velocity * STEP_S), 0.0])

    period = 2 * STEP_S
    landed = (which * STEP_S) + np.floor((time - which * STEP_S) / period) * period
    if time < landed + STEP_S:
        return landing(landed)
    u = (time - landed - STEP_S) / STEP_S
    start, end = landing(landed), landing(landed + period)
    foot = start + (end - start) * (0.5 - 0.5 * np.cos(np.pi * u))
    foot[2] = LIFT_M * min(float(np.linalg.norm(end - start)) / 0.3, 1.0) * np.sin(np.pi * u)
    return foot


def _style(contact, side: str) -> np.random.Generator:
    """The same random choices every time for one player at one hit (how they play it)."""
    return np.random.default_rng(int(round(contact.time * 1000)) * 2 + (0 if side == "near" else 1))


def split_landing(contact, side: str) -> float | None:
    """When the player's split step lands around this hit of the opponent's (seconds), or None for
    the one in seven they skip: from 80 ms before the hit to 120 ms after it."""
    rng = _style(contact, side)
    landing = contact.time + rng.uniform(-0.08, 0.12)
    return None if rng.random() < 1 / 7 else float(landing)


def playing(rally, side: str, time: float, stature: float = 1.75) -> np.ndarray:
    """A player's body at a time in a simulated rally: stepping to the place they hit from (a foot
    down stays where it is), back towards base after their hit, and a split step as the opponent
    hits. At each of their hits the racket reaches the shuttle with the shoulders turned
    side-on, lunging for a low one (the leading foot stepping out, the hips low, the front knee
    bent) and jumping for a high one."""
    centre = _centre(rally, side, time)
    facing = np.pi / 2 if side == "near" else -np.pi / 2
    forward = np.array([np.cos(facing), np.sin(facing), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    hip_height = 0.50 * stature
    # Stepping: the feet take turns, the pelvis lower the faster the player goes.
    pace = min(float(np.linalg.norm(_centre(rally, side, time + 0.05) - centre)) / 0.05 / 3.0, 1.0)
    feet = [_foot(rally, side, time, 0, stature), _foot(rally, side, time, 1, stature)]
    pelvis = np.array([*centre, hip_height - 0.05 * pace])
    # The split step: a little hop to land near the opponent's hit, then the hips take the landing.
    for contact in (c for c in rally.contacts if c.hitter != side):
        landing = split_landing(contact, side)
        if landing is None:
            continue
        hop, after = time - (landing - SPLIT_HOP_S), time - landing
        if 0.0 <= hop <= SPLIT_HOP_S:
            lift = SPLIT_HOP_M * np.sin(np.pi * hop / SPLIT_HOP_S)
            pelvis = pelvis + lift * up
            feet = [foot + lift * up for foot in feet]
        if 0.0 <= after <= SPLIT_DIP_S:
            pelvis = pelvis - SPLIT_DIP_M * np.sin(np.pi * after / SPLIT_DIP_S) * up
    shoulder = pelvis + 0.29 * stature * up
    hands = [shoulder - 0.25 * right - 0.35 * up + 0.15 * forward, shoulder + 0.25 * right - 0.30 * up + 0.20 * forward]
    lean, turn = 0.08, 0.0
    for contact in (c for c in rally.contacts if c.hitter == side):
        weight = float(np.exp(-(((time - contact.time) / STROKE_S) ** 2)))
        if weight < 0.02:
            continue
        rng = _style(contact, side)
        point = np.asarray(contact.position, dtype=float)
        # Side-on for the stroke: the racket shoulder back, most for an overhead.
        side_on = rng.uniform(15, 45) if point[2] > 1.8 else (rng.uniform(5, 35) if point[2] >= 1.0 else rng.uniform(-10, 20))
        turn -= np.radians(side_on) * weight
        if point[2] < 1.0:  # a lunge: the right foot steps out towards it (lifted on the way out and back), the hips drop
            toward = np.array([*(point[:2] - _stance(contact, side, stature)), 0.0])  # from where they lunge, so the foot stays put
            toward = toward / max(np.linalg.norm(toward), 1e-9)
            step = np.array([*(point[:2] - 0.35 * toward[:2]), 0.0])
            out = min(1.0, 1.6 * weight)  # out there and planted through the stroke's middle
            feet[1] = (1 - out) * feet[1] + out * step
            feet[1][2] += 4 * LIFT_M * out * (1 - out)
            pelvis = pelvis - weight * rng.uniform(0.10, 0.17) * stature * up
            lean += 0.4 * weight
        elif point[2] > 2.7:  # a jump smash: off the floor
            lift = rng.uniform(0.2, 0.45) * weight
            pelvis = pelvis + lift * up
            feet = [foot + lift * up for foot in feet]
        # The racket arm and racket in line from the shoulder to the shuttle.
        racket_shoulder = pelvis + 0.29 * stature * up + 0.13 * stature * right
        line = racket_shoulder - point
        hand = point + RACKET_M * line / max(np.linalg.norm(line), 1e-9)
        hands[1] = (1 - weight) * hands[1] + weight * hand
    return body(stature, pelvis, facing, (feet[0], feet[1]), (hands[0], hands[1]), lean, turn)


def model_pose(camera, joints13: np.ndarray, rng: np.random.Generator, joint_sigma: float = 0.06, angle_deg: float = 8.0, scale_sigma: float = 0.07) -> np.ndarray:
    """What a learned 3D pose model might say of a body (13 joints, geometry.skeleton.JOINTS
    order): its joints about the middle of the hips in the camera's axes, turned a little,
    scaled a little and each a little off."""
    import cv2

    middle = (joints13[7] + joints13[8]) / 2
    relative = (joints13 - middle) @ camera.rotation.T
    axis = rng.normal(size=3)
    turn = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.radians(rng.normal(0, angle_deg)))[0]
    return (1 + rng.normal(0, scale_sigma)) * relative @ turn.T + rng.normal(0, joint_sigma, relative.shape)

