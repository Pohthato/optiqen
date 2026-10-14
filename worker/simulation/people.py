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
    hands: tuple[np.ndarray, np.ndarray], lean: float = 0.0,
) -> np.ndarray:
    """The 17 COCO joints of a body with these bone lengths: pelvis centre, the direction it faces,
    where the (left, right) feet are on the floor and where the hands are, and how far the
    trunk leans forward (radians). Knees bend forward, elbows down and back; limbs that cannot
    reach reach as far as they can."""
    seg = {name: fraction * stature for name, fraction in SEGMENTS.items()}
    forward = np.array([np.cos(facing), np.sin(facing), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    trunk = np.cos(lean) * up + np.sin(lean) * forward
    hips = [pelvis - seg["hip_half"] * right, pelvis + seg["hip_half"] * right]
    neck = pelvis + seg["torso"] * trunk
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


def _foot(rally, side: str, time: float, which: int, stature: float) -> np.ndarray:
    """Where one foot is (the ankle's floor point, lifted while it swings): planted where it last
    landed, or swinging to where it lands next. The feet take turns, each planted for STEP_S and
    swinging for STEP_S; a foot lands beside the body and, when the body moves, ahead of it by
    half the way the body goes while the foot is down."""
    from simulation.rally import player_positions  # the court plan of the rally's players

    facing = np.pi / 2 if side == "near" else -np.pi / 2
    right = np.array([np.sin(facing), -np.cos(facing)])
    across = (-1.0 if which == 0 else 1.0) * 0.16 * stature * right

    def landing(at: float) -> np.ndarray:
        centre = np.asarray(player_positions(rally, at)[side], dtype=float)
        velocity = (np.asarray(player_positions(rally, at + 0.05)[side], dtype=float) - centre) / 0.05
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


def playing(rally, side: str, time: float, stature: float = 1.75) -> np.ndarray:
    """A player's body at a time in a simulated rally: stepping between the places they hit from
    (a foot down stays where it is), in a stance, and at each of their hits reaching the racket
    to the shuttle, lunging for a low one (the leading foot stepping out and back) and jumping
    for a high one."""
    from simulation.rally import player_positions  # the court plan of the rally's players

    centre = np.asarray(player_positions(rally, time)[side], dtype=float)
    facing = np.pi / 2 if side == "near" else -np.pi / 2
    forward = np.array([np.cos(facing), np.sin(facing), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    hip_height = 0.50 * stature
    # Stepping: the feet take turns, the pelvis lower the faster the player goes.
    moving = float(np.linalg.norm(np.asarray(player_positions(rally, time + 0.05)[side]) - centre)) / 0.05
    pace = min(moving / 3.0, 1.0)
    feet = [_foot(rally, side, time, 0, stature), _foot(rally, side, time, 1, stature)]
    pelvis = np.array([*centre, hip_height - 0.05 * pace])
    shoulder = pelvis + 0.29 * stature * np.array([0, 0, 1.0])
    hands = [shoulder - 0.25 * right - 0.35 * np.array([0, 0, 1.0]) + 0.15 * forward,
             shoulder + 0.25 * right - 0.30 * np.array([0, 0, 1.0]) + 0.20 * forward]
    lean = 0.08
    for contact in (c for c in rally.contacts if c.hitter == side):
        weight = float(np.exp(-(((time - contact.time) / STROKE_S) ** 2)))
        if weight < 0.02:
            continue
        point = np.asarray(contact.position, dtype=float)
        to_body = np.array([*(centre - point[:2]), 0.0])
        to_body = to_body / max(np.linalg.norm(to_body), 1e-9)
        hand = point + RACKET_M * (to_body if point[2] < 1.6 else np.array([0.0, 0.0, -1.0]) * 0.7 + to_body * 0.3)
        hands[1] = (1 - weight) * hands[1] + weight * hand
        if point[2] < 1.0:  # a lunge: the right foot steps out towards it (lifted on the way out and back), the hips drop
            step = np.array([*(point[:2] - 0.25 * to_body[:2]), 0.0])
            feet[1] = (1 - weight) * feet[1] + weight * step
            feet[1][2] += 4 * LIFT_M * weight * (1 - weight)
            pelvis = pelvis - weight * np.array([0.0, 0.0, 0.12 * stature])
            lean += 0.35 * weight
        elif point[2] > 2.7:  # a jump smash: off the floor
            lift = 0.3 * weight
            pelvis = pelvis + np.array([0.0, 0.0, lift])
            feet = [foot + np.array([0.0, 0.0, lift]) for foot in feet]
    return body(stature, pelvis, facing, (feet[0], feet[1]), (hands[0], hands[1]), lean)


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

