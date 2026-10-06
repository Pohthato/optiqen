# worker/test_tracking.py
import unittest

import cv2
import numpy as np

from geometry.calibrate import solve_camera
from geometry.camera import Camera
from geometry.court_model import COLUMNS, COURT_LENGTH_M
from geometry.lines import line_response
from geometry.synthetic import observe
from geometry.tracking import acquire, fit_frame, track_video
from simulation.clip import make_clip
from simulation.rally import Shot, build_rally
from simulation.render import render_frame

SIZE = (960, 540)
TRUTH = Camera.look_at((2.59, -5.0, 3.0), (2.59, 7.0, 0.0), 650.0, SIZE)
CORNER = Camera.look_at((-2.5, -2.5, 3.5), (2.59, 6.7, 0.0), 650.0, SIZE)
SIDE = Camera.look_at((-4.5, 6.7, 3.5), (2.59, 6.7, 0.0), 650.0, SIZE)
CORNER_NAMES = ["back0_sl", "back0_sr", "back1_sr", "back1_sl"]
PLAYERS = [(1.6, 2.2), (3.4, 11.0)]
GRID = np.array(
    [[x, y, 0.0] for x in np.linspace(COLUMNS["dl"], COLUMNS["dr"], 7) for y in np.linspace(0.0, COURT_LENGTH_M, 14)]
)


def floor_error_cm(truth: Camera, estimate: Camera) -> np.ndarray:
    """Where each floor point really is versus where the estimate puts the pixel it appears at."""
    pixels = truth.project(GRID)
    width, height = SIZE
    seen = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    estimated = estimate.pixel_to_plane(pixels[seen], 0.0)
    return np.linalg.norm(estimated - GRID[seen, :2], axis=1) * 100.0


def turned(camera: Camera, rotation_deg=(0.0, 0.0, 0.0), move_m=(0.0, 0.0, 0.0), focal_px=None) -> Camera:
    """The camera rotated about its own axes (degrees) and moved in the world (metres)."""
    delta = cv2.Rodrigues(np.radians(np.asarray(rotation_deg, dtype=np.float64)).reshape(3, 1))[0]
    rotation = delta @ camera.rotation
    centre = camera.centre + np.asarray(move_m, dtype=np.float64)
    return Camera(focal_px or camera.focal_px, camera.cx, camera.cy, cv2.Rodrigues(rotation)[0].ravel(), -rotation @ centre, camera.k1)


class FitFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response = line_response(render_frame(TRUTH, SIZE, players=PLAYERS, seed=1))

    def test_snaps_a_nudged_camera_back_onto_the_lines(self):
        fit = fit_frame(self.response, turned(TRUTH, (0.3, -0.4, 0.2), (0.04, -0.03, 0.02)))
        self.assertIsNotNone(fit)
        self.assertTrue(fit.confident)
        self.assertLess(fit.rms_px, 1.0)
        self.assertLess(float(np.median(floor_error_cm(TRUTH, fit.camera))), 2.0)

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
        self.assertLess(float(np.median(floor_error_cm(TRUTH, fit.camera))), 2.0)

    def test_works_from_the_corner_and_the_side(self):
        for truth in (CORNER, SIDE):
            response = line_response(render_frame(truth, SIZE, players=PLAYERS, seed=5))
            fit = fit_frame(response, turned(truth, (0.3, 0.2, -0.2), (0.03, 0.03, 0.0)))
            self.assertTrue(fit.confident)
            self.assertLess(float(np.median(floor_error_cm(truth, fit.camera))), 2.0)

    def test_a_covered_lens_is_not_a_confident_fit(self):
        dark = np.random.default_rng(0).normal(18, 4, (SIZE[1], SIZE[0], 3)).clip(0, 255).astype(np.uint8)
        fit = fit_frame(line_response(dark), TRUTH)
        self.assertTrue(fit is None or not fit.confident)

    def test_another_view_of_the_court_is_not_a_confident_fit(self):
        response = line_response(render_frame(SIDE, SIZE, players=PLAYERS, seed=6))
        fit = fit_frame(response, TRUTH)
        self.assertTrue(fit is None or not fit.confident)


class AcquireTests(unittest.TestCase):
    def test_finds_the_court_after_the_phone_turned(self):
        response = line_response(render_frame(TRUTH, SIZE, players=PLAYERS, seed=7))
        fit = acquire(response, turned(TRUTH, (1.0, 2.0, 0.3), (0.05, 0.0, -0.03)))
        self.assertTrue(fit.confident)
        self.assertLess(float(np.median(floor_error_cm(TRUTH, fit.camera))), 2.0)


class TrackVideoTests(unittest.TestCase):
    def test_follows_a_slow_pan_without_losing_the_court(self):
        cameras = [turned(TRUTH, (0.03 * i, 0.05 * i, 0.0), (0.002 * i, 0.0, 0.0)) for i in range(60)]
        frames = [render_frame(camera, SIZE, players=PLAYERS, seed=i) for i, camera in enumerate(cameras)]
        track = track_video(frames, turned(cameras[0], (0.2, -0.3, 0.0), (0.03, 0.0, 0.0)))
        self.assertEqual(len(track), len(frames))
        self.assertEqual(track[0].state, "anchored")
        self.assertTrue(all(step.state == "tracked" for step in track[1:]))
        errors = np.concatenate([floor_error_cm(camera, step.camera) for camera, step in zip(cameras, track)])
        self.assertLess(float(np.median(errors)), 3.0)
        self.assertLess(float(np.percentile(errors, 95)), 10.0)

    def test_recovers_after_the_lens_is_covered_for_two_seconds(self):
        rally = build_rally((3.3, 3.6, 1.0), [Shot("serve", (1.5, 12.0), 1.9, 2.2), Shot("clear", (4.2, 0.8), 2.6, None)])
        clip = make_clip(TRUTH, SIZE, rally, fps=30.0, handheld=True, seed=3, tail_s=1.0, occlusions=[(1.0, 3.0)])
        anchor = solve_camera(observe(clip.cameras[0], SIZE, CORNER_NAMES, noise_px=1.0, seed=4), SIZE).camera
        track = track_video(clip.frames, anchor)
        covered = [step for step, occluded in zip(track, clip.occluded) if occluded]
        self.assertTrue(covered)
        self.assertTrue(all(step.state == "lost" and step.camera is None for step in covered))
        first_after = clip.occluded.index(True) + len(covered)
        recovered = [i for i in range(first_after, len(track)) if track[i].camera is not None]
        self.assertTrue(recovered)
        self.assertLessEqual(recovered[0] - first_after, 1)
        self.assertEqual(track[recovered[0]].state, "reanchored")
        errors = [
            float(np.median(floor_error_cm(camera, step.camera)))
            for camera, step in zip(clip.cameras, track)
            if step.camera is not None
        ]
        self.assertLess(float(np.median(errors)), 5.0)
        self.assertLess(max(errors), 20.0)


if __name__ == "__main__":
    unittest.main()
