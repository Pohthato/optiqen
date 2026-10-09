# worker/test_hits.py
import unittest

import numpy as np

from audio.onsets import Onset, detect_onsets
from evaluation.hits_benchmark import SIZE, Scenario, simulate
from geometry.camera import Camera
from hits import ONSET_GAP_MS, court_side, find_hits
from simulation.audio import SAMPLE_RATE
from simulation.clip import make_clip
from simulation.rally import canned_rally

FPS = 30.0
CAMERA = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 1300.0 * 1280 / 1920, SIZE)


def times(count: int, fps: float = FPS) -> list[float]:
    return [index * 1000.0 / fps for index in range(count)]


def truth_track(clip, seed: int, jitter_px: float = 1.0) -> list:
    rng = np.random.default_rng(seed)
    track = []
    for camera, position in zip(clip.cameras, clip.shuttle):
        pixel = None if position is None else camera.project(np.array([position]))[0]
        inside = pixel is not None and 0 <= pixel[0] < SIZE[0] and 0 <= pixel[1] < SIZE[1]
        track.append(tuple(pixel + rng.normal(0, jitter_px, 2)) if inside else None)
    return track


def found(hits, truth_ms, tolerance=33.0):
    return sum(any(abs(hit.time_ms - t) <= tolerance for hit in hits) for t in truth_ms)


def flight(start_frame: int, count: int, origin=(400.0, 500.0), velocity=(9.0, -12.0)) -> list:
    """A smooth, moving image path: `count` detections from `start_frame`."""
    return [None] * start_frame + [(origin[0] + velocity[0] * i, origin[1] + velocity[1] * i + 0.4 * i * i) for i in range(count)]


class TurnTests(unittest.TestCase):
    def test_without_sound_turns_are_timed_where_the_two_paths_meet(self):
        sim = simulate(Scenario(seed=5, jitter_px=1.0, dropout=0.05, false_per_s=0.0, neighbour_per_s=0.0, sound=False))
        hits = find_hits(sim.track, sim.times_ms, None)
        truth = [contact.time * 1000 for contact in sim.clip.rally.contacts[1:]]
        errors = [min(abs(hit.time_ms - t) for t in truth) for hit in hits]
        self.assertTrue(hits)
        self.assertLess(float(np.median(errors)), 10.0)

    def test_a_smooth_flight_has_no_hit(self):
        track = flight(0, 40)
        self.assertEqual(find_hits(track, times(len(track)), None), [])

    def test_hit_times_follow_the_frame_timestamps(self):
        # Phone video often has uneven frame timing; the path's time axis is the timestamps.
        rng = np.random.default_rng(1)
        stamps = np.cumsum(rng.uniform(28.0, 39.0, 40))
        turn_at = stamps[20]
        track = [(500.0 + 0.6 * abs(t - turn_at), 300.0 + (t - turn_at) * 0.5) for t in stamps]  # the shuttle reverses
        hits = find_hits(track, list(stamps), [Onset(float(turn_at) + 30.0, 3.0)])
        self.assertEqual(len(hits), 1)
        self.assertLess(abs(hits[0].time_ms - turn_at), 10.0)


class StartTests(unittest.TestCase):
    def test_a_serve_is_a_sound_followed_by_a_moving_flight(self):
        track = flight(15, 20)
        hits = find_hits(track, times(len(track)), [Onset(470.0, 3.0)])
        self.assertEqual([hit.evidence for hit in hits], ["sound and start"])

    def test_a_lone_false_detection_after_a_sound_is_not_a_serve(self):
        track = [None] * 15 + [(400.0, 500.0)] + [None] * 20
        self.assertEqual(find_hits(track, times(len(track)), [Onset(470.0, 3.0)]), [])

    def test_a_still_object_after_a_sound_is_not_a_serve(self):
        track = [None] * 15 + [(400.0 + 0.5 * (i % 2), 500.0) for i in range(12)]
        self.assertEqual(find_hits(track, times(len(track)), [Onset(470.0, 3.0)]), [])

    def test_the_shuttle_coming_back_into_view_is_not_a_hit(self):
        track = flight(15, 20, origin=(640.0, 5.0), velocity=(1.0, 12.0))
        self.assertEqual(find_hits(track, times(len(track)), [Onset(470.0, 3.0)], frame_size=SIZE), [])
        self.assertEqual(len(find_hits(track, times(len(track)), [Onset(470.0, 3.0)])), 1)


class NeighbouringCourtTests(unittest.TestCase):
    def test_a_sound_80_ms_before_every_hit_does_not_hide_the_hits(self):
        clip = make_clip(CAMERA, SIZE, canned_rally(), fps=FPS, handheld=True, seed=3, tail_s=0.3, render=False)
        before = [heard - 0.08 for heard in clip.contact_audio_times]
        clip = make_clip(CAMERA, SIZE, canned_rally(), fps=FPS, handheld=True, seed=3, tail_s=0.3, render=False, distractor_times=before)
        track = truth_track(clip, seed=3)
        onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS)
        hits = find_hits(track, times(len(track)), onsets, side=lambda pixel, frame: court_side(clip.cameras[frame], pixel), frame_size=SIZE)
        truth = [contact.time * 1000 for contact in clip.rally.contacts]
        self.assertGreaterEqual(found(hits, truth), len(truth) - 1)
        self.assertFalse(any(hit.heard_ms is not None and any(abs(hit.heard_ms - d * 1000) < 5 for d in before) for hit in hits))

    def test_a_sound_with_nothing_in_the_flight_is_not_a_hit(self):
        track = flight(0, 40)
        self.assertEqual(find_hits(track, times(len(track)), [Onset(600.0, 3.0)]), [])


class CourtSideTests(unittest.TestCase):
    def test_the_half_of_the_court_a_hit_was_made_in(self):
        near = CAMERA.project(np.array([[2.59, 2.0, 1.5]]))[0]
        far = CAMERA.project(np.array([[2.59, 12.0, 1.5]]))[0]
        self.assertEqual(court_side(CAMERA, tuple(near)), "near")
        self.assertEqual(court_side(CAMERA, tuple(far)), "far")

    def test_a_ray_that_never_comes_down_to_racket_height_has_no_side(self):
        sky = CAMERA.project(np.array([[2.59, 60.0, 30.0]]))[0]
        self.assertIsNone(court_side(CAMERA, tuple(sky)))


if __name__ == "__main__":
    unittest.main()
