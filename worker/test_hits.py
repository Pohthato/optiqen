# worker/test_hits.py
import unittest

import numpy as np

from audio.onsets import detect_onsets
from evaluation.hits_benchmark import SIZE
from geometry.camera import Camera
from hits import ONSET_GAP_MS, _consensus, find_hits, link_tracklets
from simulation.audio import SAMPLE_RATE
from simulation.clip import make_clip
from simulation.rally import canned_rally

FPS = 30.0
CAMERA = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 1300.0 * 1280 / 1920, SIZE)
RALLY = canned_rally()
TRUTH_MS = [contact.time * 1000 for contact in RALLY.contacts]


def clip_of(**options):
    return make_clip(CAMERA, SIZE, RALLY, fps=FPS, handheld=True, seed=3, tail_s=0.5, render=False, **options)


def times(count: int) -> list[float]:
    return [index * 1000.0 / FPS for index in range(count)]


def seen(clip, seed: int = 3, jitter_px: float = 1.0) -> list:
    """The shuttle's pixel in every frame it is in view, with detector jitter."""
    rng = np.random.default_rng(seed)
    track = []
    for camera, position in zip(clip.cameras, clip.shuttle):
        pixel = None if position is None else camera.project(np.array([position]))[0]
        inside = pixel is not None and 0 <= pixel[0] < SIZE[0] and 0 <= pixel[1] < SIZE[1]
        track.append(tuple(pixel + rng.normal(0, jitter_px, 2)) if inside else None)
    return track


def errors(hits, truth_ms=TRUTH_MS) -> list[float]:
    return [min(abs(hit.time_ms - t) for hit in hits) for t in truth_ms]


def near(hits, time_ms: float, within_ms: float = 100.0) -> list:
    return [hit for hit in hits if abs(hit.time_ms - time_ms) <= within_ms]


class FlightTests(unittest.TestCase):
    def test_without_sound_hits_are_where_the_flights_meet(self):
        clip = clip_of()
        result = find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, None)
        misses = errors(result.hits)
        self.assertGreaterEqual(sum(e <= 33 for e in misses), len(TRUTH_MS) - 1)
        self.assertLessEqual(len(result.hits), len(TRUTH_MS) + 1)
        # Between two seen flights the timing is sharp; a serve, with only its outgoing flight
        # and nothing heard, can only be placed between the sightings either side.
        between = [hit for hit in result.hits if hit.evidence == "flights"]
        self.assertLess(float(np.median([min(abs(hit.time_ms - t) for t in TRUTH_MS) for hit in between])), 10.0)
        self.assertEqual(result.hits[0].evidence, "serve")

    def test_hit_times_follow_the_frame_timestamps(self):
        # Phone video often has uneven frame timing; a flight's clock is the timestamps.
        rng = np.random.default_rng(1)
        stamps = list(np.cumsum(rng.uniform(28.0, 39.0, 260)))
        track = []
        for stamp in stamps:
            position = RALLY.shuttle_at(stamp / 1000.0)
            track.append(None if position is None else tuple(CAMERA.project(position[None])[0] + rng.normal(0, 1.0, 2)))
        result = find_hits(track, stamps, [CAMERA] * len(stamps), None)
        inside = [t for t in TRUTH_MS[1:] if t < stamps[-1] - 300]
        self.assertLess(float(np.median(errors(result.hits, inside))), 10.0)

    def test_a_still_object_after_a_sound_is_not_a_serve(self):
        clip = clip_of()
        track = [None] * 15 + [(400.0 + 0.5 * (i % 2), 500.0) for i in range(12)] + [None] * 20
        onsets = [o for o in detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS) if o.time_ms < 1500]
        self.assertEqual(find_hits(track, times(len(track)), clip.cameras[: len(track)], onsets).hits, [])

    def test_a_shuttle_first_seen_high_in_flight_is_not_a_serve(self):
        # The clip starts with the clear already past its peak: nothing before it, but no racket
        # can be up there, and with no sound nothing says a hit was made.
        clip = clip_of()
        clear = RALLY.flights[1]
        late = (clear.start_time + 0.6 * (clear.end_time - clear.start_time)) * 1000
        track = [point if t >= late else None for point, t in zip(seen(clip), times(len(clip.cameras)))]
        result = find_hits(track, times(len(track)), clip.cameras, None)
        self.assertEqual(near(result.hits, late, 300.0), [])


class SoundTests(unittest.TestCase):
    def test_with_sound_hits_are_timed_to_a_few_ms(self):
        clip = clip_of()
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        result = find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, onsets)
        self.assertTrue(all(e <= 33 for e in errors(result.hits)))
        self.assertLess(float(np.median(errors(result.hits))), 5.0)
        self.assertEqual(result.hits[0].evidence, "sound and serve")

    def test_the_phones_audio_offset_is_measured_and_taken_off(self):
        clip = clip_of(audio_offset_s=0.06)
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        result = find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, onsets)
        self.assertAlmostEqual(result.audio_offset_ms, 60.0, delta=8.0)
        self.assertLess(float(np.median(errors(result.hits))), 5.0)

    def test_a_neighbouring_courts_sound_in_mid_flight_is_not_a_hit(self):
        middles = [(f.start_time + f.end_time) / 2 for f in RALLY.flights]
        clip = clip_of(distractor_times=middles)
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        result = find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, onsets)
        self.assertTrue(all(not near(result.hits, m * 1000) for m in middles))
        self.assertGreaterEqual(sum(e <= 33 for e in errors(result.hits)), len(TRUTH_MS) - 1)
        self.assertTrue(result.rejected_sounds)

    def test_a_sound_80_ms_before_every_hit_does_not_hide_the_hits(self):
        before = [heard - 0.08 for heard in clip_of().contact_audio_times]
        clip = clip_of(distractor_times=before)
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        result = find_hits(seen(clip), times(len(clip.cameras)), clip.cameras, onsets)
        self.assertGreaterEqual(sum(e <= 33 for e in errors(result.hits)), len(TRUTH_MS) - 1)
        self.assertFalse(any(hit.heard_ms is not None and any(abs(hit.heard_ms - d * 1000) < 5 for d in before) for hit in result.hits))

    def test_a_flight_seen_again_after_leaving_view_is_not_a_hit_whatever_is_heard(self):
        # The clear's peak is out of view for 0.7 s; a sound from the next court comes just
        # before it is seen again. The same flight explains both sides of the gap.
        clear = RALLY.flights[1]
        peak = (clear.start_time + clear.end_time) / 2
        gap = (peak - 0.35, peak + 0.35)
        clip = clip_of(distractor_times=[gap[1] - 0.06])
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        track = [None if gap[0] * 1000 <= t <= gap[1] * 1000 else point for point, t in zip(seen(clip), times(len(clip.cameras)))]
        result = find_hits(track, times(len(track)), clip.cameras, onsets)
        self.assertEqual(near(result.hits, gap[1] * 1000, 200.0), [])


class PartTests(unittest.TestCase):
    def test_the_offset_is_what_enough_hits_agree_on(self):
        self.assertAlmostEqual(_consensus([31.0, 29.0, 30.0, 140.0, -80.0]), 30.0)
        self.assertIsNone(_consensus([10.0, 60.0, 120.0]))
        self.assertIsNone(_consensus([]))

    def test_a_fast_start_links_and_a_still_object_does_not(self):
        start = [(100.0, 600.0), (260.0, 520.0)] + [(260.0 + 150.0 * i, 520.0 - 70.0 * i + 4.0 * i * i) for i in range(1, 8)]
        still = [(900.0 + 0.5 * (i % 2), 300.0) for i in range(10)]
        tracklets = link_tracklets(start + [None] * 10 + still, times(len(start) + 10 + len(still)), focal_px=870.0)
        self.assertEqual(tracklets, [list(range(len(start)))])


if __name__ == "__main__":
    unittest.main()
