# worker/simulation/audio.py
"""Audio for synthetic clips: background noise plus a racket 'thwack' at every contact."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48_000
HIT_DURATION_S = 0.03
HIT_DECAY_S = 0.006
DISTRACTOR_SPACING_S = 0.15
DISTRACTOR_LEVEL = 0.25


def _hit_sound(rng: np.random.Generator, sample_rate: int, amplitude: float) -> np.ndarray:
    count = int(HIT_DURATION_S * sample_rate)
    t = np.arange(count) / sample_rate
    burst = rng.normal(size=count) * np.exp(-t / HIT_DECAY_S)
    sos = butter(4, [2000, 9000], btype="bandpass", fs=sample_rate, output="sos")
    sound = sosfilt(sos, burst)
    return (amplitude * sound / np.max(np.abs(sound))).astype(np.float32)


def render_audio(
    contact_times: list[float],
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 0,
    distractors: int = 0,
    offset_s: float = 0.0,
    hit_amplitude: float = 0.5,
    noise_amplitude: float = 0.01,
    distractor_times: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Mono float32 track and the times of the distractor hits (a neighbouring court). With
    `distractor_times` they are placed exactly there, however close to the rally's hits;
    otherwise `distractors` of them are placed at random, away from the rally's hits."""
    rng = np.random.default_rng(seed)
    samples = (noise_amplitude * rng.normal(size=int(round(duration_s * sample_rate)))).astype(np.float32)

    def place(time: float, amplitude: float) -> None:
        start = int(round(time * sample_rate))
        sound = _hit_sound(rng, sample_rate, amplitude)
        if 0 <= start < len(samples):
            end = min(len(samples), start + len(sound))
            samples[start:end] += sound[: end - start]

    for time in contact_times:
        place(time + offset_s, hit_amplitude)
    heard = [time + offset_s for time in contact_times]
    if distractor_times is not None:
        for time in distractor_times:
            place(time, hit_amplitude * DISTRACTOR_LEVEL)
        return np.clip(samples, -1.0, 1.0), list(distractor_times)
    distractor_times = []
    attempts = 0
    while len(distractor_times) < distractors and attempts < 10_000:
        attempts += 1
        candidate = float(rng.uniform(0.0, duration_s - HIT_DURATION_S))
        if all(abs(candidate - other) >= DISTRACTOR_SPACING_S for other in heard + distractor_times):
            distractor_times.append(candidate)
    for time in sorted(distractor_times):
        place(time, hit_amplitude * DISTRACTOR_LEVEL)
    return np.clip(samples, -1.0, 1.0), sorted(distractor_times)
