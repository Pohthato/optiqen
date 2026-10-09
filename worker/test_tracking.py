# worker/test_tracking.py
import unittest

import cv2
import numpy as np

from evaluation.track import floor_error_cm
from geometry.calibrate import solve_camera
from geometry.camera import Camera
from geometry.court_model import COLUMNS, ROWS
from geometry.lines import line_response
from geometry.synthetic import PORTRAIT, observe
from geometry.tracking import AnchorError, acquire, anchor_on_points, fit_frame, iter_track
from simulation.clip import make_clip
from simulation.rally import Shot, build_rally
from simulation.render import render_frame

SIZE = (960, 540)
TRUTH = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 650.0, SIZE)
CORNER = Camera.look_at((-2.5, -2.5, 3.5), (2.59, 6.7, 0.0), 650.0, SIZE)
SIDE = Camera.look_at((-4.5, 6.7, 3.5), (2.59, 6.7, 0.0), 650.0, SIZE)
VIEWS = {"behind": TRUTH, "corner": CORNER, "side": SIDE}
CORNER_NAMES = ["back0_sl", "back0_sr", "back1_sr", "back1_sl"]
PLAYERS = [(1.6, 2.2), (3.4, 11.0)]


def error_cm(truth: Camera, estimate: Camera, size=SIZE) -> float:
    return float(np.median(floor_error_cm(truth, estimate, size)))


def turned(camera: Camera, rotation_deg=(0.0, 0.0, 0.0), move_m=(0.0, 0.0, 0.0), focal_px=None) -> Camera:
    """The camera rotated about its own axes (degrees) and moved in the world (metres)."""
    delta = cv2.Rodrigues(np.radians(np.asarray(rotation_deg, dtype=np.float64)).reshape(3, 1))[0]
    rotation = delta @ camera.rotation
    centre = camera.centre + np.asarray(move_m, dtype=np.float64)
    return Camera(focal_px or camera.focal_px, camera.cx, camera.cy, cv2.Rodrigues(rotation)[0].ravel(), -rotation @ centre, camera.k1)


def covered_lens(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(18, 4, (SIZE[1], SIZE[0], 3)).clip(0, 255).astype(np.uint8)


class FitFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response = line_response(render_frame(TRUTH, SIZE, players=PLAYERS, seed=1))

    def test_snaps_a_nudged_camera_back_onto_the_lines(self):
        fit = fit_frame(self.response, turned(TRUTH, (0.3, -0.4, 0.2), (0.04, -0.03, 0.02)))
        self.assertIsNotNone(fit)
        self.assertTrue(fit.confident)
        self.assertLess(fit.rms_px, 1.0)
        self.assertLess(error_cm(TRUTH, fit.camera), 2.0)

    def test_keeps_the_focal_length_unless_asked_to_refine_it(self):
        fit = fit_frame(self.response, turned(TRUTH, (0.2, 0.0, 0.0), focal_px=670.0))
        self.assertEqual(fit.camera.focal_px, 670.0)
        self.assertEqual(fit.camera.k1, 0.0)

    def test_refines_the_focal_length_from_a_four_corner_start(self):
        # Four corners seen from straight behind cannot fix the focal length; the net tape can.
        start = solve_camera(observe(TRUTH, SIZE, CORNER_NAMES, noise_px=1.0, seed=2), SIZE).camera
        fit = acquire(self.response, start, fit_intrinsics=True)
        self.assertTrue(fit.confident)
        self.assertLess(abs(fit.camera.focal_px - 650.0) / 650.0, 0.03)
        self.assertLess(error_cm(TRUTH, fit.camera), 2.0)

    def test_works_from_the_corner_and_the_side(self):
        for truth in (CORNER, SIDE):
            response = line_response(render_frame(truth, SIZE, players=PLAYERS, seed=5))
            fit = fit_frame(response, turned(truth, (0.3, 0.2, -0.2), (0.03, 0.03, 0.0)))
            self.assertTrue(fit.confident)
            self.assertLess(error_cm(truth, fit.camera), 2.0)

    def test_works_on_a_portrait_frame(self):
        size = (PORTRAIT[0] // 2, PORTRAIT[1] // 2)
        truth = Camera.look_at((2.59, -4.0, 3.2), (2.59, 6.0, 0.0), 650.0, size)
        response = line_response(render_frame(truth, size, players=PLAYERS, seed=11))
        fit = fit_frame(response, turned(truth, (0.3, -0.4, 0.2), (0.04, -0.03, 0.02)))
        self.assertTrue(fit.confident)
        self.assertLess(error_cm(truth, fit.camera, size), 2.0)

    def test_a_covered_lens_is_not_a_confident_fit(self):
        fit = fit_frame(line_response(covered_lens()), TRUTH)
        self.assertTrue(fit is None or not fit.confident)

    def test_another_view_of_the_court_is_not_a_confident_fit(self):
        response = line_response(render_frame(SIDE, SIZE, players=PLAYERS, seed=6))
        fit = fit_frame(response, TRUTH)
        self.assertTrue(fit is None or not fit.confident)


class NearLinesTests(unittest.TestCase):
    def test_lines_close_to_the_phone_are_used_even_when_wider_than_the_search(self):
        close = Camera.look_at((2.59, -1.0, 1.3), (2.59, 4.0, 0.0), 650.0, SIZE)
        response = line_response(render_frame(close, SIZE, seed=8))
        fit = fit_frame(response, turned(close, (0.2, 0.2, 0.0)))
        self.assertTrue(fit.confident)
        self.assertGreater(fit.matched, 0.7 * fit.visible)


def cluttered(frame: np.ndarray, seed: int) -> np.ndarray:
    """Dense white strokes just above the far court, like banners, lettering and spectators in a real hall."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    height, width = out.shape[:2]
    for _ in range(1500):
        x, y = rng.uniform(0, width), rng.uniform(0.2 * height, 0.45 * height)
        angle, length = rng.uniform(0, np.pi), rng.uniform(5, 40)
        end = (int(x + length * np.cos(angle)), int(y + length * np.sin(angle)))
        cv2.line(out, (int(x), int(y)), end, (235, 235, 235), int(rng.integers(1, 4)), cv2.LINE_AA)
    return out


class AcquireTests(unittest.TestCase):
    def test_a_close_start_is_kept_when_the_hall_is_cluttered(self):
        response = line_response(cluttered(render_frame(TRUTH, SIZE, players=PLAYERS, seed=9), seed=9))
        fit = acquire(response, turned(TRUTH, (0.3, -0.4, 0.2), (0.04, -0.03, 0.02)))
        self.assertTrue(fit.confident)
        self.assertLess(error_cm(TRUTH, fit.camera), 2.0)

    def test_finds_the_court_after_the_phone_turned(self):
        response = line_response(render_frame(TRUTH, SIZE, players=PLAYERS, seed=7))
        fit = acquire(response, turned(TRUTH, (1.0, 2.0, 0.3), (0.05, 0.0, -0.03)))
        self.assertTrue(fit.confident)
        self.assertLess(error_cm(TRUTH, fit.camera), 2.0)

    def test_a_court_slid_by_one_line_spacing_is_not_accepted(self):
        # Slid by a line spacing, many model lines sit on real ones (doubles on singles sideline,
        # baseline on long service line, service line on service line) and the fit looks confident.
        slides = [
            (COLUMNS["sl"] - COLUMNS["dl"], 0.0),
            (COLUMNS["dl"] - COLUMNS["sl"], 0.0),
            (COLUMNS["sl"] - COLUMNS["c"], 0.0),
            (0.0, ROWS["long0"] - ROWS["back0"]),
            (0.0, ROWS["back0"] - ROWS["long0"]),
            (0.0, ROWS["long0"] - ROWS["short0"]),
            (0.0, ROWS["short1"] - ROWS["short0"]),
        ]
        # Never confident and wrong; the short slides a moved phone can cause are put right.
        short = {(round(dx, 2), round(dy, 2)) for dx, dy in slides[:2] + slides[3:5]}
        for view, truth in VIEWS.items():
            response = line_response(render_frame(truth, SIZE, players=PLAYERS, seed=3))
            for dx, dy in slides:
                slide = (round(dx, 2), round(dy, 2))
                with self.subTest(view=view, slide=slide):
                    fit = acquire(response, turned(truth, move_m=(-dx, -dy, 0.0)))
                    if slide in short or (fit is not None and fit.confident):
                        self.assertTrue(fit is not None and fit.confident)
                        self.assertLess(error_cm(truth, fit.camera), 2.0)


class AnchorOnPointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = render_frame(TRUTH, SIZE, players=PLAYERS, seed=12)

    def test_labelled_points_and_the_lines_give_the_camera(self):
        taps = observe(TRUTH, SIZE, CORNER_NAMES + ["short0_sl", "short0_sr"], noise_px=1.0, seed=13)
        fit = anchor_on_points(self.frame, taps)
        self.assertTrue(fit.confident)
        self.assertLess(abs(fit.camera.focal_px - 650.0) / 650.0, 0.03)
        self.assertLess(error_cm(TRUTH, fit.camera), 2.0)

    def test_doubles_corners_named_as_singles_corners_are_refused(self):
        seen = observe(TRUTH, SIZE, ["back0_dl", "back0_dr", "back1_dr", "back1_dl"], noise_px=1.0, seed=14)
        taps = {name.replace("_dl", "_sl").replace("_dr", "_sr"): pixel for name, pixel in seen.items()}
        with self.assertRaisesRegex(AnchorError, "name"):
            anchor_on_points(self.frame, taps)

    def test_the_long_service_line_named_as_the_baseline_is_refused(self):
        seen = observe(TRUTH, SIZE, ["long0_sl", "long0_sr", "back1_sr", "back1_sl"], noise_px=1.0, seed=15)
        taps = {name.replace("long0", "back0"): pixel for name, pixel in seen.items()}
        with self.assertRaisesRegex(AnchorError, "name"):
            anchor_on_points(self.frame, taps)

    def test_too_few_points_are_refused(self):
        with self.assertRaisesRegex(AnchorError, "court points"):
            anchor_on_points(self.frame, observe(TRUTH, SIZE, CORNER_NAMES[:3]))

    def test_a_frame_without_the_court_is_refused(self):
        with self.assertRaisesRegex(AnchorError, "lines"):
            anchor_on_points(covered_lens(), observe(TRUTH, SIZE, CORNER_NAMES))


class TrackTests(unittest.TestCase):
    def test_follows_a_slow_pan_without_losing_the_court(self):
        cameras = [turned(TRUTH, (0.03 * i, 0.05 * i, 0.0), (0.002 * i, 0.0, 0.0)) for i in range(60)]
        frames = [render_frame(camera, SIZE, players=PLAYERS, seed=i) for i, camera in enumerate(cameras)]
        track = list(iter_track(frames, turned(cameras[0], (0.2, -0.3, 0.0), (0.03, 0.0, 0.0))))
        self.assertEqual(len(track), len(frames))
        self.assertEqual(track[0].state, "anchored")
        self.assertTrue(all(step.state == "tracked" for step in track[1:]))
        errors = np.concatenate([floor_error_cm(camera, step.camera, SIZE) for camera, step in zip(cameras, track)])
        self.assertLess(float(np.median(errors)), 3.0)
        self.assertLess(float(np.percentile(errors, 95)), 10.0)

    def test_a_sudden_turn_must_pass_a_full_search_before_it_is_trusted(self):
        cameras = [TRUTH] * 4 + [turned(TRUTH, (0.0, 3.0, 0.0))] * 3
        frames = [render_frame(camera, SIZE, players=PLAYERS, seed=20 + i) for i, camera in enumerate(cameras)]
        track = list(iter_track(frames, turned(TRUTH, (0.2, -0.3, 0.0))))
        self.assertEqual([step.state for step in track], ["anchored", "tracked", "tracked", "tracked", "reanchored", "tracked", "tracked"])
        self.assertLess(max(error_cm(camera, step.camera) for camera, step in zip(cameras, track)), 2.0)

    def test_finds_the_court_again_when_the_phone_moved_while_covered(self):
        before, after = SIDE, turned(SIDE, (0.0, 0.5, 0.0), (0.0, 0.5, 0.0))
        frames = [render_frame(before, SIZE, players=PLAYERS, seed=30 + i) for i in range(3)]
        frames += [covered_lens(i) for i in range(3)]
        frames += [render_frame(after, SIZE, players=PLAYERS, seed=40 + i) for i in range(3)]
        track = list(iter_track(frames, turned(before, (0.2, -0.3, 0.0))))
        self.assertEqual([step.state for step in track], ["anchored", "tracked", "tracked", "lost", "lost", "lost", "reanchored", "tracked", "tracked"])
        self.assertLess(max(error_cm(after, step.camera) for step in track[6:]), 2.0)

    def test_recovers_on_the_first_clear_frame_after_two_seconds_covered(self):
        rally = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.0), 1.9, 2.2), Shot("clear", (4.2, 0.8), 2.6, None)])
        clip = make_clip(TRUTH, SIZE, rally, fps=30.0, handheld=True, seed=3, tail_s=1.0, occlusions=[(1.0, 3.0)])
        anchor = solve_camera(observe(clip.cameras[0], SIZE, CORNER_NAMES, noise_px=1.0, seed=4), SIZE).camera
        track = list(iter_track(clip.frames, anchor))
        covered = [step for step, occluded in zip(track, clip.occluded) if occluded]
        self.assertTrue(covered)
        self.assertTrue(all(step.state == "lost" and step.camera is None for step in covered))
        first_after = clip.occluded.index(True) + len(covered)
        self.assertEqual(track[first_after].state, "reanchored")
        errors = [error_cm(camera, step.camera) for camera, step in zip(clip.cameras, track) if step.camera is not None]
        self.assertLess(float(np.median(errors)), 5.0)
        self.assertLess(max(errors), 20.0)


if __name__ == "__main__":
    unittest.main()
