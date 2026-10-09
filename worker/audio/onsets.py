# worker/audio/onsets.py
"""Racket-hit onsets in a sound track. The twin of detectOnsets in tools/labeller/labeller-core.js:
the labeller suggests hits from the same onsets the analysis uses, and the labeller's test suite
runs both on the same audio.

Per 2.5 ms frame we take the energy in the 2-9 kHz band where the racket "thwack" lives (two
second-order high-passes and two low-passes), so voices, footsteps and broadband hiss count for
little. An onset is a frame ~3x louder than both the running noise floor around it and the 20 ms
just before it: the sharp attack is what tells a racket hit from a shout or a reverb tail, so
sustained sounds trigger at most once while a hit inside them is still found. Strength is how
much louder (log10) the peak is, so quieter hits from a neighbouring court rank lower.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

FRAME_S = 0.0025
RISE = 0.5  # log10 of the energy ratio over the local noise floor: ~3x louder
MIN_GAP_MS = 120
ATTACK_MS = 20  # a hit must also jump this much above the 20 ms just before it
BLOCK_S = 0.5  # the noise floor is the median of +/- 2 such blocks around each frame
FLOOR_MIN = -8.0  # log10 energy: digital silence must not make faint noise look like hits
PEAK_WINDOW_S = 0.03
BAND_LOW_HZ = 2000.0
BAND_HIGH_HZ = 9000.0


@dataclass(frozen=True)
class Onset:
    time_ms: float  # when the hit is heard in the track (it happened earlier: sound travels)
    strength: float


def _round(value: float) -> int:
    """JavaScript's Math.round (halves round up), so frame sizes match the labeller exactly."""
    return math.floor(value + 0.5)


def _biquad(kind: str, frequency: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Second-order filter coefficients (RBJ audio-EQ cookbook, Q = 1/sqrt(2))."""
    w0 = 2 * math.pi * frequency / sample_rate
    cos = math.cos(w0)
    alpha = math.sin(w0) / (2 * math.sqrt(0.5))
    b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2] if kind == "highpass" else [(1 - cos) / 2, 1 - cos, (1 - cos) / 2]
    a0 = 1 + alpha
    return np.array(b) / a0, np.array([1.0, -2 * cos / a0, (1 - alpha) / a0])


def _band_levels(samples: np.ndarray, sample_rate: int, frame: int, count: int) -> np.ndarray:
    """log10 of the mean 2-9 kHz energy in each frame."""
    top = min(BAND_HIGH_HZ, 0.45 * sample_rate)
    filtered = np.asarray(samples, dtype=np.float64)
    for kind, frequency in (("highpass", BAND_LOW_HZ), ("highpass", BAND_LOW_HZ), ("lowpass", top), ("lowpass", top)):
        b, a = _biquad(kind, frequency, sample_rate)
        filtered = lfilter(b, a, filtered)
    energy = np.square(filtered[: count * frame]).reshape(count, frame).mean(axis=1)
    return np.log10(energy + 1e-12)


def _floors(level: np.ndarray, block: int) -> np.ndarray:
    """A running noise floor per frame: it follows crowd noise and quiet passages."""
    count = len(level)
    blocks = math.ceil(count / block)
    block_floor = [float(np.median(level[b * block : min(count, (b + 1) * block)])) for b in range(blocks)]
    floor_at = [max(FLOOR_MIN, float(np.median(block_floor[max(0, b - 2) : min(blocks, b + 3)]))) for b in range(blocks)]
    return np.repeat(floor_at, block)[:count]


def detect_onsets(samples: np.ndarray, sample_rate: int, rise: float = RISE, min_gap_ms: float = MIN_GAP_MS) -> list[Onset]:
    frame = max(2, _round(sample_rate * FRAME_S))
    count = len(samples) // frame
    if count < 8:
        return []
    level = _band_levels(samples, sample_rate, frame, count)
    floor = _floors(level, max(1, _round(BLOCK_S * sample_rate / frame)))
    gap = max(1, _round(min_gap_ms / 1000 * sample_rate / frame))
    attack = max(1, _round(ATTACK_MS / 1000 * sample_rate / frame))
    peak_window = max(1, _round(PEAK_WINDOW_S * sample_rate / frame))

    index = np.arange(count)
    first = np.maximum(0, index - attack)
    sums = np.concatenate([[0.0], np.cumsum(level)])
    recent = np.where(index > 0, (sums[index] - sums[first]) / np.maximum(index - first, 1), floor)
    triggered = np.flatnonzero((level - floor >= rise) & (level - recent >= rise))

    onsets: list[Onset] = []
    next_allowed = 0
    for f in triggered:
        if f < next_allowed:
            continue
        # The hit can start late in the previous frame if that frame is already clearly louder.
        start = f - 1 if f > 0 and level[f - 1] - floor[f - 1] > rise * 0.3 else f
        peak = float(level[f : min(count, f + peak_window)].max())
        time_ms = math.floor(start * frame / sample_rate * 10000 + 0.5) / 10
        strength = math.floor((peak - floor[f]) * 100 + 0.5) / 100
        onsets.append(Onset(time_ms, strength))
        next_allowed = f + gap
    return onsets
