# worker/hits.py
"""Racket hits as the boundaries between physically possible flights.

A shuttle between two hits can only fly one way: under gravity and drag (geometry.flight_fit).
So the detections of a rally are cut into pieces at every place a hit might be (each sound and
each break in the track), and each piece is fitted as one flight in 3D through the per-frame
cameras. Then neighbouring pieces are merged, weakest boundary first, while one flight explains
the pair nearly as well as two: what a split explains is measured in units of the detector's
noise, and a boundary a sound backs needs less of it. What survives are hits. A neighbouring
court's sound mid-flight merges away; a far-court smash that barely bends on screen is an
impossible single flight, so it stays.

A hit is timed where its two fitted flights are at the same place in the image (their drag counts,
their poorly seen depth does not), or, with sound, when it was heard less the sound's travel from
the fitted contact point to the phone and the phone's audio/video offset, which the clip's hits
agree on. A flight that appears after a pause, from below the service law's height, is a serve.
A flight whose incoming flight went unseen is a hit only if a sound times it, traced back to then
it was where a racket can be, and the flight before the gap does not explain it: a shuttle coming
back into view mid-flight is not a hit, whatever is heard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from geometry.camera import Camera
from geometry.flight_fit import FlightFit, _Views, fit_flight

SPEED_OF_SOUND_M_S = 343.0
ONSET_GAP_MS = 40.0  # onsets for hit finding: a neighbouring court's sound must not hide a hit 40-120 ms later
HEARD_AFTER_HIT_MS = 90.0  # a hit is heard up to this long after it happens (travel plus A/V offset)
HEARD_BEFORE_FLIGHT_MS = (-120.0, 150.0)  # a serve's flight is first seen this long after the sound
LINK_GAP_FRAMES = 3  # missed frames allowed inside a tracklet
LINK_MIN_PX = 25.0  # a detection this far from the tracklet's prediction starts a new tracklet
START_SPEED_RAD_S = 8.0  # how fast a tracklet's second detection may move away from its first (a 40 m/s serve 5 m away)
STILL_PX = 10.0  # a tracklet that moves less than this is a still object, not the shuttle
HARD_GAP_MS = 500.0  # longer without the shuttle: no flight bridges it
MIN_PIECE = 6  # detections a piece needs to be fitted as a flight
SOUND_SPLIT_GAIN = 40.0  # noise units a sound-backed boundary must explain to be a hit
FLIGHT_SPLIT_GAIN = 150.0  # and a boundary with no sound
OUTLIER_SIGMAS = 6.0  # a detection's misfit is capped here when costing a fit
MAX_CONTACT_HEIGHT_M = 3.5
CONTACT_BOX = ((-1.5, 6.7), (-2.0, 15.4))  # where a racket can meet the shuttle (x, y)
MIN_OFFSET_HITS = 2  # sound-backed hits that must agree to measure the clip's audio/video offset
OFFSET_AGREEMENT_MS = 10.0  # how closely they must agree
SLIDE = 4  # a surviving boundary is tried this many detections either way, again while it keeps moving
MAX_SLIDE = 12  # in all
IMAGE_TURN_GAIN = 25.0  # noise units two image curves must explain over one to propose a hit there
IMAGE_SIDE = 5  # detections each side of a split when proposing image bends
IMAGE_WINDOW_MS = 160.0  # path either side of a hit used to time it in the image
IMAGE_SPLIT_SEARCH = 3  # detections either way the image split may move from the flights' boundary
IMAGE_MEET_PX = 5.0  # in detector noise: image curves that come no closer than this do not meet
SOUND_AGREEMENT_MS = 20.0  # a flights time further than this from the sound's says the flights are wrong
SOUND_SEARCH_MS = 120.0  # a boundary's sound, less its travel, lies this close to the sightings either side
SOUND_RESPLIT_COST = 25.0  # noise units the flights may lose when the boundary moves to the sound's time
SOUND_NEAR_FRAMES = 2.5  # a sound this close to the flights' time times the hit even if they fit worse there
SERVE_MAX_HEIGHT_M = 1.6  # the service law's 1.15 m at contact, plus what a serve rises before it is seen
MAX_FLIGHT_MS = 2500.0  # no shot stays up longer: a flight out of view this long has landed
MIN_HIT_INTERVAL_MS = 150.0  # no two hits come closer, even in the fastest exchanges
SERVE_QUIET_MS = 1500.0  # without sound, a serve needs this long with no shuttle before it


@dataclass
class Piece:
    """Detections fitted as one flight; `start_ms` is where the flight may begin (a boundary)."""

    indices: list[int]
    start_ms: float
    fit: FlightFit | None = None
    cost: float = np.inf  # capped squared misfit, in noise units


@dataclass(frozen=True)
class Hit:
    time_ms: float  # when the racket met the shuttle
    heard_ms: float | None  # when it was heard, if a sound went with it
    met_ms: float  # when the two fitted flights meet (or a serve's flight is first seen)
    evidence: str  # "[sound and ]flights" | "[sound and ]serve" | "sound and outgoing flight" (the incoming one unseen)
    position: tuple[float, float, float] | None  # the contact point in court metres
    frame: int  # the first detection after the hit
    gain: float  # what the split explains, in noise units (inf for serves)


@dataclass
class RallyAnalysis:
    hits: list[Hit]
    flights: list[Piece]
    noise_px: float
    audio_offset_ms: float | None = None  # how late the audio runs behind the video, once sound travel is taken off
    rejected_sounds: list[float] = field(default_factory=list)


def link_tracklets(track: Sequence[tuple[float, float] | None], times_ms: Sequence[float], focal_px: float = 1000.0) -> list[list[int]]:
    """Runs of detections that continue one another (each within reach of where the previous
    two said it would be; a second detection within reach of the fastest a shuttle starts off).
    Still objects and lone detections are dropped."""
    tracklets: list[list[int]] = []
    current: list[int] = []
    for index, point in enumerate(track):
        if point is None:
            continue
        if current and index - current[-1] - 1 <= LINK_GAP_FRAMES:
            last = np.asarray(track[current[-1]])
            elapsed = times_ms[index] - times_ms[current[-1]]
            start_reach = max(3 * LINK_MIN_PX, focal_px * START_SPEED_RAD_S * elapsed / 1000.0)
            if len(current) >= 2:
                rate = (last - np.asarray(track[current[-2]])) / (times_ms[current[-1]] - times_ms[current[-2]])
                predicted, reach = last + rate * elapsed, max(LINK_MIN_PX, 0.5 * float(np.linalg.norm(rate)) * elapsed)
            else:
                predicted, reach = last, start_reach
            if np.linalg.norm(np.asarray(point) - predicted) <= reach:
                current.append(index)
                continue
            if len(current) == 2 and np.linalg.norm(np.asarray(point) - last) <= start_reach:
                current = current[1:] + [index]  # the first of the two was not this flight's
                continue
        if current:
            tracklets.append(current)
        current = [index]
    if current:
        tracklets.append(current)
    return [t for t in tracklets if len(t) >= 2 and np.ptp(np.array([track[i] for i in t]), axis=0).max() >= STILL_PX]


class _Fitter:
    def __init__(self, track, times_ms, cameras, jitter_px):
        self.track, self.times, self.cameras, self.jitter = track, np.asarray(times_ms, dtype=float), cameras, jitter_px
        self.cache: dict[tuple[int, int, float], Piece] = {}

    def fit(self, indices: list[int], start_ms: float, initial: Sequence[np.ndarray] = ()) -> Piece:
        key = (indices[0], indices[-1], start_ms)
        if key in self.cache:
            return self.cache[key]
        piece = Piece(list(indices), start_ms)
        if len(indices) >= MIN_PIECE:
            piece.fit = fit_flight(
                self.times[indices], np.array([self.track[i] for i in indices]), [self.cameras[i] for i in indices],
                start_ms=start_ms, jitter_px=self.jitter, initial=initial,
            )
            if piece.fit is not None:
                piece.cost = float(np.sum(np.minimum(self.misfits(piece), OUTLIER_SIGMAS * self.jitter) ** 2) / self.jitter**2)
        self.cache[key] = piece
        return piece

    def misfits(self, piece: Piece) -> np.ndarray:
        idx = piece.indices
        pixels = _Views([self.cameras[i] for i in idx]).project(piece.fit.positions_at(self.times[idx])[None])[0]
        return np.hypot(*(pixels - np.array([self.track[i] for i in idx])).T)

    def merged(self, a: Piece, b: Piece) -> Piece:
        initial = [np.concatenate([piece.fit.p0, piece.fit.v0]) for piece in (a, b) if piece.fit is not None]
        return self.fit(a.indices + b.indices, a.start_ms, initial)


def _meet(a: FlightFit, b: FlightFit, low_ms: float, high_ms: float) -> tuple[float, np.ndarray]:
    """When and where flight a (running on) and flight b come closest, between low and high."""
    times = np.linspace(low_ms, high_ms, 41)
    gap = np.linalg.norm(a.positions_at(times) - b.positions_at(np.maximum(times, b.start_ms)), axis=1)
    best = int(np.argmin(gap))
    return float(times[best]), (a.position_at(times[best]) + b.position_at(max(times[best], b.start_ms))) / 2


def _projected_meet(a: FlightFit, b: FlightFit, camera: Camera, low_ms: float, high_ms: float) -> float:
    """When flight a (running on) and flight b are at the same place in the image, between low and
    high: the fitted flights carry the drag a smash loses speed to, and in the image their depth,
    the poorly seen part of a 3D fit, does not count."""
    times = np.linspace(low_ms, high_ms, 41)
    views = _Views([camera] * len(times))
    gap = np.hypot(*(views.project(a.positions_at(times)[None])[0] - views.project(b.positions_at(np.maximum(times, b.start_ms))[None])[0]).T)
    return float(times[int(np.argmin(gap))])


def _image_meet(track, times: np.ndarray, before: list[int], after: list[int], jitter_px: float) -> float | None:
    """When the shuttle's image path before a hit and after it cross: of the detections within
    IMAGE_WINDOW_MS of the boundary, the cut (up to IMAGE_SPLIT_SEARCH detections either way)
    where a quadratic in time either side fits best, then the time between the frames either side
    of it where the two curves come closest. None when they never come close: one side is not
    the shuttle."""
    centre_ms = 0.5 * (times[before[-1]] + times[after[0]])
    joint = [i for i in before + after if abs(times[i] - centre_ms) <= IMAGE_WINDOW_MS]
    middle = sum(1 for i in joint if i in set(before))
    t_all = (np.asarray(times[joint], dtype=float) - centre_ms) / 1000.0
    p_all = np.array([track[i] for i in joint], dtype=float)

    def fitted(sel: slice):
        curves = [np.polyfit(t_all[sel], p_all[sel, axis], 2) for axis in (0, 1)]
        return curves, float(sum(np.sum((np.polyval(curves[axis], t_all[sel]) - p_all[sel, axis]) ** 2) for axis in (0, 1)))

    best = None
    for cut in range(max(3, middle - IMAGE_SPLIT_SEARCH), min(len(joint) - 3, middle + IMAGE_SPLIT_SEARCH) + 1):
        (c_left, e_left), (c_right, e_right) = fitted(slice(0, cut)), fitted(slice(cut, None))
        if best is None or e_left + e_right < best[0]:  # the same detections for every cut: totals compare
            best = (e_left + e_right, cut, c_left, c_right)
    if best is None:
        return None
    _, cut, c_left, c_right = best
    taus = np.linspace(t_all[cut - 1], t_all[cut], 41)
    gap = np.hypot(*(np.polyval(c_left[axis], taus) - np.polyval(c_right[axis], taus) for axis in (0, 1)))
    if gap.min() > IMAGE_MEET_PX * jitter_px:
        return None
    return float(centre_ms + 1000.0 * taus[int(np.argmin(gap))])


def _image_turns(track, times: np.ndarray, tracklet: list[int], jitter_px: float) -> list[int]:
    """Detections (in the tracklet) before which the image path bends: two quadratics in time,
    either side, explain IMAGE_TURN_GAIN noise units more than one. Local peaks only."""
    if len(tracklet) < 2 * IMAGE_SIDE:
        return []
    t = (times[tracklet] - times[tracklet[0]]) / 1000.0
    p = np.array([track[i] for i in tracklet], dtype=float)

    def misfit(sel) -> float:
        return float(sum(np.sum((np.polyval(np.polyfit(t[sel], p[sel, axis], 2), t[sel]) - p[sel, axis]) ** 2) for axis in (0, 1)))

    gains = np.zeros(len(tracklet))
    for cut in range(IMAGE_SIDE, len(tracklet) - IMAGE_SIDE + 1):
        left, right = np.arange(cut - IMAGE_SIDE, cut), np.arange(cut, cut + IMAGE_SIDE)
        gains[cut] = (misfit(np.concatenate([left, right])) - misfit(left) - misfit(right)) / jitter_px**2
    return [tracklet[cut] for cut in range(1, len(tracklet)) if gains[cut] >= IMAGE_TURN_GAIN and gains[cut] >= gains[cut - 1] and gains[cut] >= gains[min(cut + 1, len(gains) - 1)]]


def _plausible_contact(point: np.ndarray, spread: np.ndarray = np.zeros(3)) -> bool:
    """Whether a racket can meet the shuttle at `point`, give or take `spread` per axis."""
    (x_low, x_high), (y_low, y_high) = CONTACT_BOX
    x, y, z = point
    sx, sy, sz = spread
    return x_low - sx <= x <= x_high + sx and y_low - sy <= y <= y_high + sy and -sz <= z <= MAX_CONTACT_HEIGHT_M + sz


def _spread(fit: FlightFit, time_ms: float) -> np.ndarray:
    """Two standard deviations of the fitted position, per axis."""
    return 2.0 * np.sqrt(np.maximum(np.diag(fit.position_covariance(time_ms)), 0.0))


def find_hits(
    track: Sequence[tuple[float, float] | None],
    times_ms: Sequence[float],
    cameras: Sequence[Camera],
    onsets: Sequence | None = None,
    jitter_px: float = 2.0,
    trace: list | None = None,
) -> RallyAnalysis:
    """Hits and the flights between them, from per-frame shuttle detections (pixel or None),
    their timestamps, the camera of every frame, and sound onsets (None for a silent video;
    detect them with min_gap_ms=ONSET_GAP_MS)."""
    times = np.asarray(times_ms, dtype=float)
    fitter = _Fitter(track, times, cameras, jitter_px)
    tracklets = link_tracklets(track, times, cameras[0].focal_px if len(cameras) else 1000.0)
    kept = sorted(i for tracklet in tracklets for i in tracklet)
    if not kept:
        return RallyAnalysis([], [], jitter_px)
    tracklet_starts = {tracklet[0] for tracklet in tracklets}
    sounds = sorted(onset.time_ms for onset in onsets) if onsets is not None else []

    # Where a hit might be, as positions in `kept` (a boundary before kept[k]), and the sound behind it.
    hard = {k for k in range(1, len(kept)) if times[kept[k]] - times[kept[k - 1]] > HARD_GAP_MS}
    soft: dict[int, float | None] = {k: None for k in range(1, len(kept)) if k not in hard and kept[k] in tracklet_starts}
    position = {index: k for k, index in enumerate(kept)}
    for tracklet in tracklets:
        for index in _image_turns(track, times, tracklet, jitter_px):
            if position[index] not in hard:
                soft.setdefault(position[index], None)
    for heard in sounds:
        window = [k for k in range(1, len(kept)) if k not in hard and heard - HEARD_AFTER_HIT_MS <= times[kept[k - 1]] <= heard]
        if window:
            k = min(window, key=lambda k: abs(times[kept[k]] - (heard - 20.0)))
            if soft.get(k) is None:
                soft[k] = heard
    edges = [0] + sorted(hard | set(soft)) + [len(kept)]
    pieces = [
        fitter.fit([kept[i] for i in range(lo, hi)], float(times[kept[lo - 1]] if lo in soft else times[kept[lo]]))
        for lo, hi in zip(edges, edges[1:])
    ]
    sound_of = [soft.get(edge) for edge in edges[1:-1]]  # the sound at the boundary after pieces[i]
    if trace is not None:
        trace.append(("tracklets", [(times[t[0]], times[t[-1]], len(t)) for t in tracklets]))
        trace.append(("pieces", [(times[p.indices[0]], times[p.indices[-1]], len(p.indices), p.fit is not None, round(p.cost, 1)) for p in pieces], list(sound_of)))
    joinable = [edge not in hard for edge in edges[1:-1]]

    # Merge, weakest boundary first, while one flight explains a pair nearly as well as two.
    def own(piece: Piece) -> float:
        # Too short for a flight of its own: with a flight's six numbers free it fits exactly.
        return piece.cost if piece.fit is not None else 0.0

    def excess(i: int) -> float:
        a, b = pieces[i], pieces[i + 1]
        if not joinable[i]:
            return np.inf
        joined = fitter.merged(a, b)
        if joined.fit is None:
            # Scraps join while together they are still too few to fit; enough detections that no
            # flight explains hold a hit, and a flight does not take what it cannot explain.
            return -np.inf if a.fit is None and b.fit is None and len(joined.indices) < MIN_PIECE else np.inf
        return joined.cost - own(a) - own(b) - (SOUND_SPLIT_GAIN if sound_of[i] is not None else FLIGHT_SPLIT_GAIN)

    while len(pieces) > 1:
        scores = [excess(i) for i in range(len(pieces) - 1)]
        i = int(np.argmin(scores))
        if scores[i] >= 0:
            break
        if trace is not None:
            trace.append(("merge", times[pieces[i].indices[-1]], sound_of[i], round(float(scores[i]), 1)))
        joined = fitter.merged(pieces[i], pieces[i + 1])
        if joined.fit is None:  # both halves unfittable: keep them together without a flight
            joined = Piece(pieces[i].indices + pieces[i + 1].indices, pieces[i].start_ms)
        pieces[i : i + 2] = [joined]
        sound_of.pop(i)  # a sound whose boundary merged away was not a hit here
        joinable.pop(i)

    def resplit(a: Piece, b: Piece, cut: int) -> tuple[Piece, Piece]:
        """Flights a and b refitted with the boundary before the cut-th of their detections."""
        joint = a.indices + b.indices
        left = fitter.fit(joint[:cut], a.start_ms, [np.concatenate([a.fit.p0, a.fit.v0])])
        start = float(times[joint[cut - 1]])
        guesses = [np.concatenate([b.fit.p0, b.fit.v0])]
        if start >= b.start_ms:
            later = b.fit.positions_at([start, start + 1.0])
            guesses.append(np.concatenate([later[0], (later[1] - later[0]) * 1000.0]))
        return left, fitter.fit(joint[cut:], start, guesses)

    # Slide each surviving boundary to where its two flights fit best; drop it if it no longer pays.
    i = 0
    while i < len(pieces) - 1:
        a, b = pieces[i], pieces[i + 1]
        if joinable[i] and a.fit is not None and b.fit is not None:
            joint = a.indices + b.indices
            best = (a.cost + b.cost, a, b)
            moved = 0
            while True:
                centre = len(best[1].indices)
                for cut in range(max(MIN_PIECE, centre - SLIDE), min(len(joint) - MIN_PIECE, centre + SLIDE) + 1):
                    if cut != centre:
                        left, right = resplit(best[1], best[2], cut)
                        if left.fit is not None and right.fit is not None and left.cost + right.cost < best[0]:
                            best = (left.cost + right.cost, left, right)
                shift = abs(len(best[1].indices) - centre)
                moved += shift
                if shift < SLIDE or moved >= MAX_SLIDE:
                    break
            pieces[i], pieces[i + 1] = best[1], best[2]
            if excess(i) < 0:
                joined = fitter.merged(pieces[i], pieces[i + 1])
                pieces[i : i + 2] = [joined]
                sound_of.pop(i)
                joinable.pop(i)
                continue
        i += 1

    fitted = [p for p in pieces if p.fit is not None]
    noise = 1.2 * float(np.median(np.concatenate([fitter.misfits(p) for p in fitted]))) if fitted else jitter_px

    frame_ms = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    def travel_ms(where: np.ndarray, frame: int) -> float:
        return float(np.linalg.norm(np.asarray(where) - cameras[frame].centre)) / SPEED_OF_SOUND_M_S * 1000.0

    def first_seen(piece: Piece) -> int:
        """The piece's first detection its flight explains: a false one before it is not the shuttle."""
        explained = fitter.misfits(piece) <= OUTLIER_SIGMAS * noise
        return piece.indices[int(np.argmax(explained))] if explained.any() else piece.indices[0]

    # Hits: surviving boundaries between two flights (or a flight and a scrap too short for one),
    # timed by where the flights meet in the image, and plausible serves.
    def meeting(i: int):
        a, b = pieces[i], pieces[i + 1]
        low, high = float(times[a.indices[-1]]), float(times[b.indices[0]])
        if a.fit is not None and b.fit is not None:
            return _projected_meet(a.fit, b.fit, cameras[b.indices[0]], low, high), _meet(a.fit, b.fit, low, high)[1]
        # One side is too short for its own flight: the contact is where the other one ends.
        where = a.fit.position_at(low) if a.fit is not None else b.fit.position_at(high)
        return _image_meet(track, times, a.indices, b.indices, jitter_px), where

    boundaries = []  # [piece index, met, where, sound]
    for i in range(len(pieces) - 1):
        if joinable[i] and (pieces[i].fit is not None or pieces[i + 1].fit is not None):
            met, where = meeting(i)
            seen = pieces[i] if pieces[i].fit is not None else pieces[i + 1]
            if _plausible_contact(where, _spread(seen.fit, met if met is not None else float(times[seen.indices[0]]))):
                boundaries.append([i, met, where, None])
    def continues(i: int) -> bool:
        """Whether the flight before piece i, carried on out of view, also explains piece i."""
        before, after = pieces[i - 1] if i > 0 else None, pieces[i]
        if before is None or before.fit is None or times[after.indices[0]] - times[before.indices[-1]] > MAX_FLIGHT_MS:
            return False
        joined = fitter.merged(before, after)
        return joined.fit is not None and joined.cost - before.cost - after.cost < SOUND_SPLIT_GAIN

    serves = []  # [piece index, first detection, origin, sound, kind]
    for i, piece in enumerate(pieces):
        if piece.fit is None or not (i == 0 or not joinable[i - 1]):
            continue  # after a scrap, the boundary before it already decided
        frame = first_seen(piece)
        first_ms = float(times[frame])
        origin = piece.fit.position_at(first_ms)
        if not _plausible_contact(origin):
            continue
        heard = [s for s in sounds if HEARD_BEFORE_FLIGHT_MS[0] <= first_ms - s <= HEARD_BEFORE_FLIGHT_MS[1]]
        if onsets is not None and not heard:
            continue
        # A serve comes after a pause, from below the service law's height (half a frame back from
        # where it is first seen). Anything else is a hit whose incoming flight went unseen: with
        # a sound to time it, a hit; with none, a shuttle coming back into view.
        earlier = [times[j] for j in kept if times[j] < first_ms]
        rise = (piece.fit.position_at(first_ms + 1.0)[2] - origin[2]) * 1000.0
        height = origin[2] - rise * 0.5 * frame_ms / 1000.0
        # Along the line of sight depth, and with it height, is uncertain: low enough within two
        # standard deviations is low enough.
        if (not earlier or first_ms - max(earlier) >= SERVE_QUIET_MS) and height - _spread(piece.fit, first_ms)[2] <= SERVE_MAX_HEIGHT_M:
            kind = "serve"
        elif heard and not continues(i):
            kind = "outgoing flight"
        else:
            continue
        serves.append([i, frame, origin, min(heard, key=lambda s: abs(first_ms - s)) if heard else None, kind])

    # Each sound to at most one boundary: the one whose flights met just before it was heard
    # (allowing for the sound's travel), nearest first. A boundary may sit a frame or two off, so
    # the search is wide; the check below keeps only sounds the flights agree with.
    taken = {serve[3] for serve in serves if serve[3] is not None}
    options = []
    for n, (i, met, where, _) in enumerate(boundaries):
        low, high = float(times[pieces[i].indices[-1]]), float(times[pieces[i + 1].indices[0]])
        reference = met if met is not None else 0.5 * (low + high)
        travel = travel_ms(where, pieces[i + 1].indices[0])
        options += [(abs(heard - travel - reference), n, heard) for heard in sounds if low - SOUND_SEARCH_MS <= heard - travel <= high + SOUND_SEARCH_MS]
    for _, n, heard in sorted(options):
        if boundaries[n][3] is None and heard not in taken:
            boundaries[n][3] = heard
            taken.add(heard)

    # A hit is heard after its sound travels from the contact point to the phone, plus whatever the
    # phone's audio runs behind its video: one offset for the clip, taken from the boundaries whose
    # flights time them and on which enough of them agree. With it, the sound times a hit to a few ms.
    offset = _consensus([heard - travel_ms(where, pieces[i + 1].indices[0]) - met for i, met, where, heard in boundaries if heard is not None and met is not None])
    hits = []
    for entry in boundaries:
        i, met, where, heard = entry
        a, b = pieces[i], pieces[i + 1]
        met, where = meeting(i)  # the boundary before may have moved this flight's start
        frame = b.indices[0]
        when = met
        if heard is not None and (offset is not None or met is None):
            by_sound = heard - travel_ms(where, frame) - (offset or 0.0)
            if a.fit is None or b.fit is None:
                if met is None or abs(met - by_sound) > SOUND_AGREEMENT_MS:
                    # A scrap's image curve is weaker evidence; a flight seen leaving bounds the hit.
                    when = min(by_sound, float(times[frame])) if b.fit is not None else by_sound
            elif abs(met - by_sound) > SOUND_AGREEMENT_MS:
                # The flights and the sound disagree: move the boundary to the sound's time and
                # keep the sound only if the flights still fit nearly as well.
                joint = a.indices + b.indices
                cut = int(np.searchsorted(times[joint], by_sound))
                moved = resplit(a, b, cut) if MIN_PIECE <= cut <= len(joint) - MIN_PIECE else (Piece([], 0.0), Piece([], 0.0))
                fits = moved[0].fit is not None and moved[1].fit is not None and moved[0].cost + moved[1].cost - a.cost - b.cost <= SOUND_RESPLIT_COST
                if trace is not None:
                    trace.append(("sound check", met, by_sound, heard, cut, len(a.indices), moved[0].cost + moved[1].cost - a.cost - b.cost))
                if fits:
                    pieces[i], pieces[i + 1] = moved
                    a, b = moved
                    met, where = meeting(i)
                    frame = b.indices[0]
                if fits or abs(met - by_sound) <= SOUND_NEAR_FRAMES * frame_ms:
                    when = by_sound
                else:
                    heard = None  # someone else's sound
        if when is None:
            when = float(times[frame]) - 0.5 * frame_ms  # between the sightings either side
        gain = fitter.merged(a, b).cost - own(a) - own(b)
        evidence = ("sound and " if heard is not None else "") + "flights"
        hits.append(Hit(when, heard, when if met is None else met, evidence, tuple(float(v) for v in where), frame, gain))
    for i, frame, origin, heard, kind in serves:
        first_ms = float(times[frame])
        when = first_ms - 0.5 * frame_ms
        if heard is not None:
            when = min(heard - travel_ms(origin, frame) - (offset or 0.0), first_ms)
            # Traced back to when the sound says it was hit, the flight must be where a racket can
            # be; a shuttle coming back into view mid-flight, next to someone else's sound, is not.
            back = fitter.fit([j for j in pieces[i].indices if j >= frame], when)
            if back.fit is not None and _plausible_contact(back.fit.position_at(when)):
                origin = back.fit.position_at(when)
            elif kind == "outgoing flight":
                continue
        hits.append(Hit(when, heard, first_ms, ("sound and " if heard is not None else "") + kind, tuple(float(v) for v in origin), frame, np.inf))
    # Two hits closer than any exchange allows are one: keep the better-explained.
    hits.sort(key=lambda h: h.time_ms)
    k = 0
    while k < len(hits) - 1:
        if hits[k + 1].time_ms - hits[k].time_ms < MIN_HIT_INTERVAL_MS:
            hits.pop(k if hits[k].gain < hits[k + 1].gain else k + 1)
        else:
            k += 1
    used = {hit.heard_ms for hit in hits}
    return RallyAnalysis(sorted(hits, key=lambda h: h.time_ms), pieces, noise, offset, [s for s in sounds if s not in used])


def _consensus(values: Sequence[float]) -> float | None:
    """The value at least MIN_OFFSET_HITS of `values` agree on (within OFFSET_AGREEMENT_MS of one
    another), the median of the largest such group; None when no group is large enough."""
    values = np.sort(np.asarray(values, dtype=float))
    best = None
    for lo in range(len(values)):
        hi = int(np.searchsorted(values, values[lo] + OFFSET_AGREEMENT_MS, side="right"))
        group = values[lo:hi]
        if len(group) >= MIN_OFFSET_HITS and (best is None or len(group) > len(best) or (len(group) == len(best) and np.ptp(group) < np.ptp(best))):
            best = group
    return None if best is None else float(np.median(best))
