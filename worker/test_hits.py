# worker/test_hits.py
import unittest

import numpy as np

from audio.onsets import Onset, detect_onsets
from geometry.camera import Camera
from hits import court_side, find_hits, flight_breaks
from simulation.audio import SAMPLE_RATE
from simulation.clip import make_clip
from simulation.rally import canned_rally

SIZE = (1280, 720)
FPS = 30.0
CAMERA = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 1300.0 * 1280 / 1920, SIZE)
# The thresholds were tuned on seeds 0-9; these rallies were not looked at.
SEEDS = (20, 21, 22, 23, 24)


def clip_and_track(seed, distractors=3, noise_px=1.0, dropout=0.1):
    """A synthetic rally and what a good detector gives: the true pixel with noise, some frames missed."""
    clip = make_clip(CAMERA, SIZE, canned_rally(), fps=FPS, handheld=True, seed=seed, tail_s=0.3, distractors=distractors, render=False)
    rng = np.random.default_rng(seed)
    track = []
    for camera, position in zip(clip.cameras, clip.shuttle):
        if position is None or rng.random() < dropout:
            track.append(None)
            continue
        pixel = camera.project(np.array([position]))[0]
        inside = 0 <= pixel[0] < SIZE[0] and 0 <= pixel[1] < SIZE[1]
        track.append(tuple(pixel + rng.normal(0, noise_px, 2)) if inside else None)
    return clip, track


def f1_at(found_ms, truth_ms, tolerance_ms=33.0):
    unused = list(found_ms)
    matched = 0
    for truth in truth_ms:
        best = min(unused, key=lambda t: abs(t - truth), default=None)
        if best is not None and abs(best - truth) <= tolerance_ms:
            matched += 1
            unused.remove(best)
    return 2 * matched / (len(found_ms) + len(truth_ms)) if found_ms or truth_ms else 1.0


class FlightBreakTests(unittest.TestCase):
    def test_clear_turns_are_timed_where_the_two_paths_meet(self):
        clip, track = clip_and_track(20)
        turns = [b for b in flight_breaks(track, FPS) if b.kind == "turn" and b.clear]
        truth = [contact.time * 1000 for contact in clip.rally.contacts[1:]]  # the serve has no incoming path
        errors = [min(abs(turn.time_ms - t) for t in truth) for turn in turns]
        self.assertGreaterEqual(len(turns), len(truth) - 2)
        self.assertLess(float(np.median(errors)), 10.0)

    def test_where_the_shuttle_appears_and_disappears(self):
        track = [None] * 5 + [(100.0 + 5 * i, 200.0) for i in range(10)] + [None] * 5
        breaks = flight_breaks(track, FPS)
        self.assertEqual([(b.kind, b.frame) for b in breaks], [("start", 5), ("end", 14)])

    def test_a_smooth_flight_has_no_clear_turn(self):
        t = np.arange(30) / FPS
        track = [(100 + 300 * x, 500 - 400 * x + 600 * x * x) for x in t]
        self.assertFalse([b for b in flight_breaks(track, FPS) if b.kind == "turn" and b.clear])


class CourtSideTests(unittest.TestCase):
    def test_the_half_of_the_court_a_hit_was_made_in(self):
        near = CAMERA.project(np.array([[2.59, 2.0, 1.5]]))[0]
        far = CAMERA.project(np.array([[2.59, 12.0, 1.5]]))[0]
        self.assertEqual(court_side(CAMERA, tuple(near)), "near")
        self.assertEqual(court_side(CAMERA, tuple(far)), "far")

    def test_a_ray_that_never_comes_down_to_racket_height_has_no_side(self):
        sky = CAMERA.project(np.array([[2.59, 60.0, 30.0]]))[0]
        self.assertIsNone(court_side(CAMERA, tuple(sky)))


class FindHitsTests(unittest.TestCase):
    def test_hits_are_found_within_33_ms_and_neighbouring_court_hits_are_not(self):
        scores = []
        for seed in SEEDS:
            clip, track = clip_and_track(seed)
            onsets = detect_onsets(clip.audio, SAMPLE_RATE)
            hits = find_hits(track, onsets, FPS, side=lambda b: court_side(clip.cameras[b.frame], b.pixel), frame_size=SIZE)
            scores.append(f1_at([hit.time_ms for hit in hits], [contact.time * 1000 for contact in clip.rally.contacts]))
            for distractor in clip.distractor_times:
                self.assertFalse(any(hit.heard_ms is not None and abs(hit.heard_ms - distractor * 1000) < 5 for hit in hits), (seed, distractor))
        self.assertGreaterEqual(float(np.mean(scores)), 0.95, scores)

    def test_the_serve_is_found_from_its_sound_and_the_shuttle_appearing(self):
        clip, track = clip_and_track(21)
        hits = find_hits(track, detect_onsets(clip.audio, SAMPLE_RATE), FPS, frame_size=SIZE)
        serve = clip.rally.contacts[0].time * 1000
        self.assertTrue(any(hit.evidence == "sound and start" and abs(hit.time_ms - serve) <= 33 for hit in hits), hits[:2])

    def test_without_sound_clear_turns_are_still_hits(self):
        scores = []
        for seed in SEEDS:
            clip, track = clip_and_track(seed, distractors=0)
            hits = find_hits(track, None, FPS)
            self.assertTrue(all(hit.heard_ms is None for hit in hits))
            scores.append(f1_at([hit.time_ms for hit in hits], [contact.time * 1000 for contact in clip.rally.contacts[1:]]))
        self.assertGreaterEqual(float(np.mean(scores)), 0.75, scores)

    def test_the_shuttle_coming_back_into_view_is_not_a_hit(self):
        track = [None] * 10 + [(640.0 + i, 5.0 + 12 * i) for i in range(12)]
        self.assertEqual(find_hits(track, [Onset(300.0, 3.0)], FPS, frame_size=SIZE), [])
        self.assertEqual(len(find_hits(track, [Onset(300.0, 3.0)], FPS)), 1)

    def test_a_sound_with_no_turn_or_start_near_it_is_not_a_hit(self):
        t = np.arange(40) / FPS
        track = [(100 + 300 * x, 500 - 400 * x + 600 * x * x) for x in t]  # one smooth flight
        hits = find_hits(track, [Onset(600.0, 3.0)], FPS)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
