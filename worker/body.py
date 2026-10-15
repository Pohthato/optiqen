# worker/body.py
"""What a player's body did at each of their hits, from their fitted bodies (geometry.skeleton)
and the shot's contact point (shots), in court metres, each fact with its uncertainty.

At the moment the racket met the shuttle:
- Reach: how high the contact was against the player's own standing reach with the racket (from
  their bone lengths: legs straight, arm straight up, racket in line). Below it, they let the
  shuttle drop; above it, they jumped for it.
- Where: the contact in front of (or behind) the middle of the hips along the way the hips
  face, and to the side, towards the racket arm.
- The racket arm's elbow angle (180 degrees straight); how far the trunk is turned side-on (the
  racket shoulder back from the hip line); the hips' height and the front knee's angle (how
  deep a lunge); how far the lower foot is off the floor (a jump); and the centre of mass in
  front of (or behind) the middle of the feet, from segment masses (de Leva, 1996).

Before the shot, at the opponent's hit that it answers:
- How far the player was from base (on the centre line, BASE_FROM_NET_M from the net).
- The split step: a hop and its landing, the hips rising and then dropping lowest around the
  opponent's hit as the legs load to push off. How far they dropped from the hop's top, and
  when they were lowest against the opponent's hit; no split step when the hips' lowest point
  is not inside that time, or the drop is under SPLIT_MIN_M or not clear of its uncertainty.

The body at a moment is interpolated between the frames either side. Every fact is worked out on
bodies drawn from the fit's uncertainty (each frame's joints' covariance: their errors go
together, the whole body along the line of sight) and the contact point's, so its spread is
what they allow; a moment between frames adds the joints' movement over half a frame.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter

from geometry.court_model import NET_Y_M, SINGLES_WIDTH_M
from geometry.skeleton import ANKLE_HEIGHT_M, JOINTS, Body
from shots import Estimate

RACKET_REACH_M = 0.6  # from the wrist to where a racket meets the shuttle, the arm and racket in line
BONES_OFF = 0.01  # how far the fitted bone lengths may be off together (they run up to 1 % long)
BASE_FROM_NET_M = 2.8
DRAWS = 400
# Segment masses (fractions of the body's) and where each segment's centre of mass lies from its
# first joint to its second (de Leva 1996, men's values; head at the nose, hands at the wrists).
SEGMENT_MASS = (
    (0.0694, (0,), None, 0.0),  # head
    (0.4346, (5, 6), (11, 12), 0.4486),  # trunk
    (0.0271, (5,), (7,), 0.5772), (0.0271, (6,), (8,), 0.5772),  # upper arms
    (0.0162, (7,), (9,), 0.4574), (0.0162, (8,), (10,), 0.4574),  # forearms
    (0.0061, (9,), None, 0.0), (0.0061, (10,), None, 0.0),  # hands
    (0.1416, (11,), (13,), 0.4095), (0.1416, (12,), (14,), 0.4095),  # thighs
    (0.0433, (13,), (15,), 0.4459), (0.0433, (14,), (16,), 0.4459),  # shanks
    (0.0137, (15,), None, 0.0), (0.0137, (16,), None, 0.0),  # feet
)
HOP_S = -0.35  # from this long before the opponent's hit, a split step's hop is looked for
SPLIT_S = (-0.10, 0.35)  # and around it, its lowest point
SMOOTH_S = 0.15  # the hips' height is smoothed over this long (a local quadratic)
SPLIT_MIN_M = 0.06  # a split step drops the hips at least this much from the hop's top
SPLIT_SEEN = 2.5  # and by this many standard deviations
# The fit holds a foot only a few centimetres off the floor a little down, so a split step's hop
# comes out up to this much low: added to the drop's uncertainty.
HOP_FLATTENED_M = 0.02


@dataclass(frozen=True)
class Readiness:
    """The player at the opponent's hit, before their own shot."""

    from_base_m: Estimate
    split_drop_m: Estimate  # the hips' drop from their highest to their lowest around the opponent's hit
    split_lowest_ms: Estimate | None  # when they were lowest, from the opponent's hit; None: no split step


@dataclass(frozen=True)
class BodyFacts:
    hit_ms: float
    hitter: str
    arm: str  # "left" | "right": the racket arm, the one whose wrist is nearer the contact
    above_reach_m: Estimate  # contact height minus the standing reach with the racket
    ahead_m: Estimate  # contact in front of the hips' middle, along the way the hips face
    aside_m: Estimate  # and to the racket arm's side
    elbow_deg: Estimate
    side_on_deg: Estimate  # the shoulder line turned from the hip line, the racket shoulder back
    hips_height_m: Estimate
    front_knee_deg: Estimate  # the knee of the foot further forward (180 straight)
    off_floor_m: Estimate  # the lower foot's height off the floor
    weight_ahead_m: Estimate  # centre of mass in front of the feet's middle
    ready: Readiness | None = None  # None for a serve, or with the body unknown at the opponent's hit


def standing_reach(bodies: Sequence[Body | None]) -> Estimate:
    """The player's standing reach with the racket: ankle height, shank, thigh, the trunk from the
    hips' middle to the shoulders' middle, upper arm and forearm (each the median over the clip,
    both sides together), and the racket."""
    joints = np.array([b.joints for b in bodies if b is not None])
    if len(joints) == 0:
        raise ValueError("no bodies")

    def median(values: np.ndarray) -> tuple[float, float]:
        """A median and its standard error."""
        values = values[np.isfinite(values)]
        return float(np.median(values)), float(1.2533 * np.std(values) / np.sqrt(max(len(values), 1)))

    def bone(*pairs) -> tuple[float, float]:
        return median(np.concatenate([np.linalg.norm(joints[:, a] - joints[:, b], axis=1) for a, b in pairs]))

    trunk = np.linalg.norm((joints[:, 5] + joints[:, 6] - joints[:, 11] - joints[:, 12]) / 2, axis=1)
    parts = [bone((13, 15), (14, 16)), bone((11, 13), (12, 14)), median(trunk), bone((5, 7), (6, 8)), bone((7, 9), (8, 10))]
    body = sum(p[0] for p in parts)
    return Estimate(ANKLE_HEIGHT_M + body + RACKET_REACH_M, float(np.sqrt(sum(p[1] ** 2 for p in parts) + (BONES_OFF * body) ** 2)))


def body_at(bodies: Sequence[Body | None], times_ms: Sequence[float], time_ms: float) -> Body | None:
    """The body at a moment between frames, interpolated between the fitted frames either side
    (no more than a frame and a half apart), its uncertainty widened by the joints' movement over
    half that gap. None without both."""
    times = np.asarray(times_ms, dtype=float)
    after = int(np.searchsorted(times, time_ms))
    if after < len(times) and abs(times[after] - time_ms) < 1e-6 and bodies[after] is not None:
        return bodies[after]
    before = after - 1
    if before < 0 or after >= len(times) or bodies[before] is None or bodies[after] is None:
        return None
    gap = times[after] - times[before]
    if len(times) > 1 and gap > 1.5 * float(np.median(np.diff(times))):
        return None
    f = (time_ms - times[before]) / gap
    a, b = bodies[before], bodies[after]
    joints = (1 - f) * a.joints + f * b.joints
    between = (np.linalg.norm(b.joints - a.joints, axis=1) / 2) ** 2 / 3  # per axis
    sigma = np.sqrt(((1 - f) * a.sigma + f * b.sigma) ** 2 + 3 * between)
    covariance = None
    if a.covariance is not None and b.covariance is not None:
        covariance = (1 - f) * a.covariance + f * b.covariance + np.diag(np.repeat(np.nan_to_num(between[list(JOINTS)]), 3))
    return Body(joints, sigma, (1 - f) * a.rms_px + f * b.rms_px, covariance)


def _facing(joints: np.ndarray) -> np.ndarray:
    """The way the hips face, across the floor: square to the line from the left hip to the right."""
    across = joints[..., 12, :2] - joints[..., 11, :2]
    forward = np.stack([-across[..., 1], across[..., 0]], axis=-1)
    return forward / np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-9)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The angle at b between the lines to a and to c, degrees."""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def centre_of_mass(joints: np.ndarray) -> np.ndarray:
    """The body's centre of mass from its segments' (joints (..., 17, 3))."""
    total = np.zeros(joints.shape[:-2] + (3,))
    for mass, first, second, along in SEGMENT_MASS:
        start = np.mean([joints[..., j, :] for j in first], axis=0)
        point = start if second is None else start + along * (np.mean([joints[..., j, :] for j in second], axis=0) - start)
        total = total + mass * point
    return total / sum(segment[0] for segment in SEGMENT_MASS)


def _facts(joints: np.ndarray, contact: np.ndarray, reach: np.ndarray, arm: str) -> dict[str, np.ndarray]:
    """Every fact at contact for bodies (..., 17, 3), contacts (..., 3) and reaches (...)."""
    hips = (joints[..., 11, :] + joints[..., 12, :]) / 2
    forward = _facing(joints)
    to_right = np.stack([forward[..., 1], -forward[..., 0]], axis=-1)
    arm_side = 1.0 if arm == "right" else -1.0
    offset = contact[..., :2] - hips[..., :2]
    shoulder, elbow, wrist = (6, 8, 10) if arm == "right" else (5, 7, 9)
    shoulder_line = joints[..., 6, :2] - joints[..., 5, :2]
    hip_line = joints[..., 12, :2] - joints[..., 11, :2]
    # The shoulder line's turn from the hip line, anticlockwise seen from above; the racket shoulder
    # goes back clockwise for a right-hander, anticlockwise for a left-hander.
    turn = np.degrees(np.arctan2(hip_line[..., 0] * shoulder_line[..., 1] - hip_line[..., 1] * shoulder_line[..., 0], np.sum(hip_line * shoulder_line, axis=-1)))
    left_ahead = np.sum((joints[..., 15, :2] - hips[..., :2]) * forward, axis=-1) >= np.sum((joints[..., 16, :2] - hips[..., :2]) * forward, axis=-1)
    knee = np.where(left_ahead, _angle(joints[..., 11, :], joints[..., 13, :], joints[..., 15, :]), _angle(joints[..., 12, :], joints[..., 14, :], joints[..., 16, :]))
    feet = (joints[..., 15, :2] + joints[..., 16, :2]) / 2
    return {
        "above_reach_m": contact[..., 2] - reach,
        "ahead_m": np.sum(offset * forward, axis=-1),
        "aside_m": arm_side * np.sum(offset * to_right, axis=-1),
        "elbow_deg": _angle(joints[..., shoulder, :], joints[..., elbow, :], joints[..., wrist, :]),
        "side_on_deg": -arm_side * turn,
        "hips_height_m": hips[..., 2],
        "front_knee_deg": knee,
        "off_floor_m": np.minimum(joints[..., 15, 2], joints[..., 16, 2]) - ANKLE_HEIGHT_M,
        "weight_ahead_m": np.sum((centre_of_mass(joints)[..., :2] - feet) * forward, axis=-1),
    }


def _drawn(body: Body, rng: np.random.Generator, draws: int = DRAWS) -> np.ndarray:
    """Bodies (draws, 17, 3) drawn about this one: from its covariance when it has one (the joints'
    errors together), else each joint off by its standard deviation, shared over x, y and z."""
    joints = np.where(np.isfinite(body.joints), body.joints, 0.0)
    spread = np.where(np.isfinite(body.sigma), body.sigma, 0.0)[None, :, None] / np.sqrt(3)
    drawn = joints[None] + rng.normal(size=(draws,) + joints.shape) * spread
    if body.covariance is not None:
        values, vectors = np.linalg.eigh(body.covariance)
        root = vectors * np.sqrt(np.maximum(values, 0.0))
        drawn[:, list(JOINTS)] = joints[list(JOINTS)][None] + (rng.normal(size=(draws, len(values))) @ root.T).reshape(draws, len(JOINTS), 3)
    return drawn


def hit_body(
    bodies: Sequence[Body | None], times_ms: Sequence[float], hit_ms: float, hitter: str,
    contact: Sequence[float], contact_sigma: Sequence[float], reach: Estimate, rng: np.random.Generator | None = None,
) -> BodyFacts | None:
    """The body facts at one hit, or None when the hitter's body is not known at that moment."""
    found = body_at(bodies, times_ms, hit_ms)
    if found is None or not np.isfinite(found.joints[list(JOINTS)]).all():
        return None
    joints = found.joints
    contact = np.asarray(contact, dtype=float)
    arm = "right" if np.linalg.norm(joints[10] - contact) <= np.linalg.norm(joints[9] - contact) else "left"
    rng = rng or np.random.default_rng(0)
    contacts = contact[None] + rng.normal(size=(DRAWS, 3)) * np.asarray(contact_sigma, dtype=float)[None]
    reaches = reach.value + rng.normal(size=DRAWS) * reach.sigma
    best = _facts(np.where(np.isfinite(joints), joints, 0.0), contact, np.array(reach.value), arm)
    spread = _facts(_drawn(found, rng), contacts, reaches, arm)
    values = {}
    for name, value in best.items():
        draws = spread[name]
        if name.endswith("_deg"):
            draws = value + (draws - value + 180.0) % 360.0 - 180.0  # no wrap-around for angles near +-180
        values[name] = Estimate(float(value), float(np.std(draws)))
    return BodyFacts(hit_ms=float(hit_ms), hitter=hitter, arm=arm, **values)


def _base(side: str) -> np.ndarray:
    return np.array([SINGLES_WIDTH_M / 2, NET_Y_M - (1.0 if side == "near" else -1.0) * BASE_FROM_NET_M])


def _split(heights: np.ndarray, times_s: np.ndarray, hit_s: float, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows of hips' heights over frames: the drop from the highest point before the lowest
    one, when the smoothed height was lowest (refined between frames by a parabola) and whether
    that lowest point is inside the split step's time rather than at its edge."""
    smooth = savgol_filter(heights, window, 2, axis=-1) if window >= 3 else heights
    inside = np.flatnonzero((times_s >= hit_s + SPLIT_S[0] - 1e-9) & (times_s <= hit_s + SPLIT_S[1] + 1e-9))
    rows = np.arange(smooth.shape[0])
    k = inside[np.argmin(smooth[:, inside], axis=-1)]
    interior = (k > inside[0]) & (k < inside[-1])
    before = np.where(np.arange(smooth.shape[1])[None] <= k[:, None], smooth, -np.inf)
    a, b, c = smooth[rows, np.maximum(k - 1, 0)], smooth[rows, k], smooth[rows, np.minimum(k + 1, smooth.shape[1] - 1)]
    curve = a - 2 * b + c
    shift = np.where(interior & (curve > 1e-12), 0.5 * (a - c) / np.where(curve > 1e-12, curve, 1.0), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    step = float(np.median(np.diff(times_s)))
    return before.max(axis=-1) - (b - 0.25 * (a - c) * shift), times_s[k] + shift * step, interior


def _height_spread(body: Body) -> float:
    """One standard deviation of the hips' middle's height."""
    if body.covariance is None:
        return float((body.sigma[11] + body.sigma[12]) / 2 / np.sqrt(3))
    a, b = 3 * JOINTS.index(11) + 2, 3 * JOINTS.index(12) + 2
    c = body.covariance
    return float(np.sqrt(max(0.25 * (c[a, a] + c[b, b] + 2 * c[a, b]), 0.0)))


def readiness(
    bodies: Sequence[Body | None], times_ms: Sequence[float], opponent_hit_ms: float, side: str, rng: np.random.Generator | None = None,
) -> Readiness | None:
    """The player at the opponent's hit: how far from base, and their split step. None when the
    body is not known through the moments the split step needs."""
    at = body_at(bodies, times_ms, opponent_hit_ms)
    times = np.asarray(times_ms, dtype=float) / 1000.0
    hit = opponent_hit_ms / 1000.0
    frames = np.flatnonzero((times >= hit + HOP_S - 1e-9) & (times <= hit + SPLIT_S[1] + 1e-9))
    if at is None or len(frames) < 7 or any(bodies[k] is None for k in frames):
        return None
    t = times[frames]
    if np.max(np.diff(t)) > 1.5 * np.median(np.diff(t)) or t[0] > hit + HOP_S + 0.05 or t[-1] < hit + SPLIT_S[1] - 0.05:
        return None
    rng = rng or np.random.default_rng(0)
    # How far from base: the hips' middle across the floor.
    hips = (at.joints[11, :2] + at.joints[12, :2]) / 2
    drawn = _drawn(at, rng)
    from_base = np.linalg.norm((drawn[:, 11, :2] + drawn[:, 12, :2]) / 2 - _base(side), axis=1)
    # The split step, from the hips' height over the frames around the hit.
    heights = np.array([(bodies[k].joints[11, 2] + bodies[k].joints[12, 2]) / 2 for k in frames])
    spreads = np.array([_height_spread(bodies[k]) for k in frames])
    window = int(round(SMOOTH_S / np.median(np.diff(t)))) // 2 * 2 + 1
    drop, lowest, interior = _split(heights[None], t, hit, window)
    drops, lows, _ = _split(heights[None] + rng.normal(size=(DRAWS, len(t))) * spreads[None], t, hit, window)
    split = Estimate(float(drop[0]), float(np.hypot(np.std(drops), HOP_FLATTENED_M)))
    seen = bool(interior[0]) and split.value >= SPLIT_MIN_M and split.value > SPLIT_SEEN * split.sigma
    timed = Estimate(float((lowest[0] - hit) * 1000.0), float(np.std(lows) * 1000.0)) if seen else None
    return Readiness(Estimate(float(np.linalg.norm(hips - _base(side))), float(np.std(from_base))), split, timed)


def rally_body_facts(shots: Sequence, bodies: dict[str, Sequence[Body | None]], times_ms: Sequence[float]) -> list[BodyFacts | None]:
    """Each shot's body facts (shots.ShotFacts or anything with hit_ms, hitter, contact and
    contact_sigma), with the player's readiness at the opponent's hit before it; None where the
    hitter's body is not known."""
    reach = {half: standing_reach(seen) if any(b is not None for b in seen) else None for half, seen in bodies.items()}
    facts: list[BodyFacts | None] = []
    for k, shot in enumerate(shots):
        half = shot.hitter
        if reach.get(half) is None:
            facts.append(None)
            continue
        fact = hit_body(bodies[half], times_ms, shot.hit_ms, half, shot.contact, shot.contact_sigma, reach[half])
        if fact is not None and k > 0 and shots[k - 1].hitter != half:
            fact = replace(fact, ready=readiness(bodies[half], times_ms, shots[k - 1].hit_ms, half))
        facts.append(fact)
    return facts
