# worker/hits.py
"""Racket hits from the shuttle's path and the sound track together.

The shuttle's image path turns sharply at a hit. For every gap between two detections we ask
how much better two smooth curves, one either side, explain the nearby path than a single curve
does (the gain, relative to the shuttle's speed), and where those two paths meet (the contact,
between frames). Where the shuttle only appears (a serve from a hand the detector did not see)
and a moving flight follows, the path marks a start instead.

Sound leads where there is sound: a hit is heard 0-80 ms after it happens (travel to the phone,
plus any audio/video offset in the file), so each sound picks the best split just before it.
A sound with no turn or start in that window is not a hit: a neighbouring court's hits are as
sharp as the rally's. Turns away from any sound are hits only when they are strong (quiet net
shots), or, in a video without sound, whenever they are clear turns. Hits from a start are timed
by the sound less the delay measured on this clip's own turn hits on the same half of the court.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from geometry.camera import Camera
from geometry.court_model import NET_Y_M

SIDE_POINTS = 5  # detections fitted on each side of a split
TURN_GAIN = 0.05  # misfit of one curve, relative to the shuttle's speed, that marks a turn
HEARD_TURN_GAIN = 0.07  # a weaker turn is enough when a sound says a hit happened just then
HEARD_TURN_JUMP = 0.8  # but the path must still change direction or speed sharply there
CLEAR_TURN_GAIN = 0.11  # with a velocity jump of CLEAR_TURN_JUMP, a hit even without a sound
CLEAR_TURN_JUMP = 1.0  # velocity change at the meeting point, relative to the faster path
QUIET_HIT_GAIN = 0.2  # in a video with sound, a hit nobody heard must be clearer still
QUIET_HIT_JUMP = 1.2
GAP_TIMING_FRAMES = 2  # missing frames at a split this many or more: time the hit by its sound
MAX_WINDOW_GAP_FRAMES = 3  # missing frames allowed inside a fitting window
TRACK_GAP_FRAMES = 4  # this many frames without the shuttle end a track
HEARD_AFTER_HIT_MS = (-10.0, 80.0)  # a hit is heard this long after it happens
TRACK_AFTER_SOUND_MS = (-90.0, 150.0)  # a track starts this long after the hit is heard
SAME_HIT_MS = 100.0
SAME_TURN_MS = 150.0  # splits this close belong to one turn
ENTRY_BORDER_FRACTION = 0.08  # a track starting this near the frame's edge is the shuttle coming back into view
DEFAULT_DELAY_MS = 25.0  # sound travel plus A/V offset when this clip gives no measurement
ONSET_GAP_MS = 40.0  # onsets for hit finding: a neighbouring court's sound must not hide a hit 40-120 ms later
START_DETECTIONS = 6  # a start must open a flight: this many detections a smooth curve fits
START_MAX_RMS_PX = 4.0
START_MIN_MOVE_PX = 15.0
CONTACT_HEIGHT_M = 1.5


@dataclass(frozen=True)
class FlightBreak:
    kind: str  # "turn" | "start"
    time_ms: float
    strength: float  # turns: the gain at the chosen split; 0 for starts
    pixel: tuple[float, float]
    frame: int
    jump: float = 0.0  # turns: velocity change where the two paths meet, relative to the faster

    @property
    def clear(self) -> bool:
        return self.kind == "turn" and self.strength >= CLEAR_TURN_GAIN and self.jump >= CLEAR_TURN_JUMP


@dataclass(frozen=True)
class Hit:
    time_ms: float  # when the racket met the shuttle
    heard_ms: float | None  # when it was heard, if a sound went with it
    evidence: str  # "sound and turn" | "sound and start" | "turn"
    pixel: tuple[float, float]
    frame: int


@dataclass(frozen=True)
class _Split:
    gain: float
    cost: float
    jump: float  # change of velocity where the paths meet, relative to the faster of the two
    gap: int  # frames the detector missed right at the split
    time_ms: float  # where the two fitted paths meet
    pixel: tuple[float, float]
    frame: int  # the last detection before the split


def _fit(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    degree = min(2, len(t) - 1)
    return np.polyfit(t, p[:, 0], degree), np.polyfit(t, p[:, 1], degree)


def _cost(t: np.ndarray, p: np.ndarray, curve: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.sum((np.polyval(curve[0], t) - p[:, 0]) ** 2 + (np.polyval(curve[1], t) - p[:, 1]) ** 2))


def _splits(track: Sequence[tuple[float, float] | None], times_ms: Sequence[float]) -> list[_Split]:
    seen = [index for index, point in enumerate(track) if point is not None]
    times = np.asarray(times_ms, dtype=float)[seen] / 1000.0  # seconds keep the polynomial fits well scaled
    points = np.array([track[index] for index in seen], dtype=float).reshape(-1, 2)
    side = SIDE_POINTS
    splits = []
    for split in range(side, len(seen) - side + 1):
        lo, hi = split - side, split + side
        if seen[hi - 1] - seen[lo] > hi - lo - 1 + MAX_WINDOW_GAP_FRAMES:
            continue
        centre = times[split - 1]
        t, p = times[lo:hi] - centre, points[lo:hi]
        before, after = _fit(t[:side], p[:side]), _fit(t[side:], p[side:])
        cost = _cost(t[:side], p[:side], before) + _cost(t[side:], p[side:], after)
        speed2 = float(np.median(np.sum(np.diff(p, axis=0) ** 2, axis=1)))
        gain = (_cost(t, p, _fit(t, p)) - cost) / len(t) / (speed2 + 4.0)
        # The contact: where the two paths come closest, between the frames either side of the split.
        half_frame = 0.5 * float(np.median(np.diff(t)))
        taus = np.linspace(t[side - 1] - half_frame, t[side] + half_frame, 41)
        gap = np.hypot(np.polyval(before[0], taus) - np.polyval(after[0], taus), np.polyval(before[1], taus) - np.polyval(after[1], taus))
        best = taus[int(np.argmin(gap))]
        pixel = ((np.polyval(before[0], best) + np.polyval(after[0], best)) / 2, (np.polyval(before[1], best) + np.polyval(after[1], best)) / 2)
        v_in = np.array([np.polyval(np.polyder(before[0]), best), np.polyval(np.polyder(before[1]), best)])
        v_out = np.array([np.polyval(np.polyder(after[0]), best), np.polyval(np.polyder(after[1]), best)])
        jump = float(np.linalg.norm(v_out - v_in) / max(np.linalg.norm(v_in), np.linalg.norm(v_out), 30.0))
        splits.append(_Split(gain, cost, jump, seen[split] - seen[split - 1] - 1, (centre + best) * 1000.0, (float(pixel[0]), float(pixel[1])), seen[split - 1]))
    return splits


def _turns(splits: list[_Split], threshold: float) -> list[FlightBreak]:
    """Runs of splits where one curve cannot explain the path; each run's turn is its best split."""
    turns, run = [], []
    for split in splits + [None]:
        if split is not None and split.gain >= threshold and (not run or split.frame - run[-1].frame <= 2):
            run.append(split)
            continue
        if run:
            best = min(run, key=lambda s: s.cost)
            turns.append(FlightBreak("turn", best.time_ms, best.gain, best.pixel, best.frame, best.jump))
        run = [split] if split is not None and split.gain >= threshold else []
    return turns


def _starts(track: Sequence[tuple[float, float] | None], times_ms: Sequence[float]) -> list[FlightBreak]:
    """Where a flight begins after a long enough gap: the shuttle appears and the next detections
    follow one smooth, moving path (a lone false detection, or a shuttle lying still, is not one)."""
    seen = [index for index, point in enumerate(track) if point is not None]
    breaks = []
    for k, index in enumerate(seen):
        gap_before = index - seen[k - 1] - 1 if k > 0 else index
        following = seen[k : k + START_DETECTIONS]
        if gap_before < TRACK_GAP_FRAMES or len(following) < START_DETECTIONS:
            continue
        if following[-1] - index > START_DETECTIONS + MAX_WINDOW_GAP_FRAMES:
            continue
        t = np.asarray(times_ms, dtype=float)[following] / 1000.0
        p = np.array([track[i] for i in following], dtype=float)
        rms = np.sqrt(_cost(t - t[0], p, _fit(t - t[0], p)) / len(t))
        if rms <= START_MAX_RMS_PX and np.linalg.norm(p[-1] - p[0]) >= START_MIN_MOVE_PX:
            breaks.append(FlightBreak("start", float(times_ms[index]), 0.0, tuple(map(float, track[index])), index))
    return breaks


def _at_border(pixel: tuple[float, float], frame_size: tuple[int, int]) -> bool:
    margin = ENTRY_BORDER_FRACTION * min(frame_size)
    return pixel[0] < margin or pixel[1] < margin or pixel[0] > frame_size[0] - margin or pixel[1] > frame_size[1] - margin


def court_side(camera: Camera, pixel: tuple[float, float]) -> str | None:
    """The half of the court a hit at this pixel was made in: where the pixel's ray is at a
    typical racket height. None when the ray never comes down to that height."""
    point = camera.pixel_to_plane(np.array([pixel], dtype=float), CONTACT_HEIGHT_M)[0]
    if not np.isfinite(point).all():
        return None
    return "near" if point[1] < NET_Y_M else "far"


def find_hits(
    track: Sequence[tuple[float, float] | None],
    times_ms: Sequence[float],
    onsets: Sequence | None,
    side: Callable[[tuple[float, float], int], str | None] | None = None,
    frame_size: tuple[int, int] | None = None,
) -> list[Hit]:
    """Hits from a per-frame shuttle track (pixel or None per frame, with each frame's timestamp)
    and sound onsets (None when the video has no sound; detect them with min_gap_ms=ONSET_GAP_MS).
    `side(pixel, frame)` names the hitter's half ("near"/"far") so sound delays are measured per
    half: from a phone behind one baseline, hits on the far half are heard ~30 ms later. With
    `frame_size`, a track starting at the frame's edge is the shuttle coming back into view."""
    splits = _splits(track, times_ms)
    turns = _turns(splits, TURN_GAIN)
    if onsets is None:
        return [Hit(t.time_ms, None, "turn", t.pixel, t.frame) for t in turns if t.clear]
    hits: list[Hit] = []
    unexplained = []
    # Each sound picks the best-fitting split just before it. Two sounds cannot claim splits of
    # one turn: a neighbouring court's sound just before a real hit would take the turn's run-up.
    claims: list[tuple[_Split, object]] = []
    for onset in onsets:
        low, high = onset.time_ms - HEARD_AFTER_HIT_MS[1], onset.time_ms - HEARD_AFTER_HIT_MS[0]
        nearby = [s for s in splits if low <= s.time_ms <= high and s.gain >= HEARD_TURN_GAIN and s.jump >= HEARD_TURN_JUMP]
        if not nearby:
            unexplained.append(onset)
            continue
        best = min(nearby, key=lambda split: split.cost)
        rivals = [claim for claim in claims if abs(claim[0].time_ms - best.time_ms) < SAME_TURN_MS]
        if any(rival[0].cost <= best.cost for rival in rivals):
            unexplained.append(onset)
            continue
        for rival in rivals:
            claims.remove(rival)
            unexplained.append(rival[1])
        claims.append((best, onset))
    # Sound delays (travel to the phone plus any audio/video offset) from turns timed cleanly.
    def half(pixel: tuple[float, float], frame: int) -> str | None:
        return side(pixel, frame) if side else None

    delays: dict[str | None, list[float]] = {}
    for split, onset in claims:
        if split.gap < GAP_TIMING_FRAMES:
            delays.setdefault(half(split.pixel, split.frame), []).append(onset.time_ms - split.time_ms)
    every = [delay for values in delays.values() for delay in values]

    def delay_for(pixel: tuple[float, float], frame: int) -> float:
        measured = delays.get(half(pixel, frame)) or every
        return float(np.median(measured)) if measured else DEFAULT_DELAY_MS

    for split, onset in claims:
        # Where the detector missed frames right at the hit, the paths are extrapolated across the
        # gap; the sound, less this clip's measured delay, is the better clock.
        time_ms = split.time_ms if split.gap < GAP_TIMING_FRAMES else onset.time_ms - delay_for(split.pixel, split.frame)
        hits.append(Hit(time_ms, onset.time_ms, "sound and turn", split.pixel, split.frame))
    starts = [b for b in _starts(track, times_ms) if not (frame_size and _at_border(b.pixel, frame_size))]
    # Each start is claimed by one sound: the one whose timing fits this clip's measured delay.
    for start in starts:
        expected = start.time_ms + delay_for(start.pixel, start.frame)
        heard = [o for o in unexplained if o.time_ms + TRACK_AFTER_SOUND_MS[0] <= start.time_ms <= o.time_ms + TRACK_AFTER_SOUND_MS[1]]
        if not heard:
            continue
        onset = min(heard, key=lambda o: abs(o.time_ms - expected))
        unexplained.remove(onset)
        hits.append(Hit(onset.time_ms - delay_for(start.pixel, start.frame), onset.time_ms, "sound and start", start.pixel, start.frame))
    for turn in turns:
        quiet_hit = turn.strength >= QUIET_HIT_GAIN and turn.jump >= QUIET_HIT_JUMP
        if quiet_hit and all(abs(turn.time_ms - hit.time_ms) > SAME_HIT_MS for hit in hits):
            hits.append(Hit(turn.time_ms, None, "turn", turn.pixel, turn.frame))
    return sorted(hits, key=lambda h: h.time_ms)
