# worker/test_calibrate.py
import unittest

import numpy as np

from geometry.calibrate import solve_camera
from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, KEYPOINTS, NET_Y_M, SINGLES_WIDTH_M
from geometry.synthetic import LANDSCAPE, observe, standard_scenes

GRID = np.array(
    [[x, y, 0.0] for x in np.linspace(0, SINGLES_WIDTH_M, 6) for y in np.linspace(0, COURT_LENGTH_M, 12)]
)


def grid_error_cm(true_camera: Camera, solved_camera: Camera) -> np.ndarray:
    estimated = solved_camera.pixel_to_plane(true_camera.project(GRID), 0.0)
    return np.linalg.norm(estimated - GRID[:, :2], axis=1) * 100.0


class CalibrationSolverTests(unittest.TestCase):
    def test_recovers_pose_and_focal_without_noise(self):
        for name, (camera, size) in standard_scenes().items():
            solution = solve_camera(observe(camera, size), size)
            self.assertIsNotNone(solution, name)
            self.assertLess(grid_error_cm(camera, solution.camera).max(), 0.5, name)
            self.assertLess(abs(solution.camera.focal_px / camera.focal_px - 1), 0.005, name)

    def test_recovers_focal_and_floor_with_pixel_noise(self):
        for name, (camera, size) in standard_scenes().items():
            solution = solve_camera(observe(camera, size, noise_px=1.0, seed=3), size)
            self.assertIsNotNone(solution, name)
            errors = grid_error_cm(camera, solution.camera)
            self.assertLess(np.median(errors), 3.0, name)
            self.assertLess(errors.max(), 15.0, name)
            self.assertLess(abs(solution.camera.focal_px / camera.focal_px - 1), 0.03, name)
            self.assertIn(solution.tier, ("validated", "approximate"), name)

    def test_four_floor_keypoints_give_an_approximate_solution(self):
        camera, size = standard_scenes()["behind_elevated"]
        names = ["long0_sl", "long0_sr", "long1_sl", "long1_sr"]
        observed = observe(camera, size, names=names)
        self.assertEqual(len(observed), 4)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        self.assertEqual(solution.tier, "approximate")
        self.assertEqual(solution.redundancy, 0)
        self.assertIsNone(solution.loo_floor_cm)
        self.assertLess(grid_error_cm(camera, solution.camera).max(), 5.0)

    def test_partial_court_far_half_still_solves(self):
        camera, size = standard_scenes()["behind_elevated"]
        far = {k: v for k, v in observe(camera, size, noise_px=1.0, seed=5).items() if KEYPOINTS[k][1] >= NET_Y_M}
        solution = solve_camera(far, size)
        self.assertIsNotNone(solution)
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)
        self.assertLess(abs(solution.camera.focal_px / camera.focal_px - 1), 0.03)

    def test_rejects_outlier_taps(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size, noise_px=1.0, seed=5)
        bad = ["long0_sr", "short0_dr"]
        for name in bad:
            self.assertIn(name, observed)
            observed[name] = (observed[name][0] + 70.0, observed[name][1] - 50.0)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        self.assertEqual(set(solution.outliers), set(bad))
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)

    def test_recovers_lens_distortion(self):
        camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0, LANDSCAPE, k1=-0.08)
        solution = solve_camera(observe(camera, LANDSCAPE, noise_px=0.7, seed=2), LANDSCAPE)
        self.assertIsNotNone(solution)
        self.assertLess(abs(solution.camera.k1 + 0.08), 0.03)
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)

    def test_wrong_focal_prior_still_converges(self):
        camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1000.0, LANDSCAPE)
        solution = solve_camera(observe(camera, LANDSCAPE, noise_px=1.0, seed=2), LANDSCAPE)
        self.assertIsNotNone(solution)
        self.assertLess(abs(solution.camera.focal_px / 1000.0 - 1), 0.05)

    def test_gross_noise_is_never_validated(self):
        camera, size = standard_scenes()["behind_elevated"]
        solution = solve_camera(observe(camera, size, noise_px=12.0, seed=1), size)
        self.assertTrue(solution is None or solution.tier != "validated")

    def test_fewer_than_four_keypoints_returns_none(self):
        camera, size = standard_scenes()["behind_elevated"]
        three = observe(camera, size, names=["long0_sl", "long0_sr", "long1_sl"])
        self.assertIsNone(solve_camera(three, size))

    def test_collinear_floor_points_return_none(self):
        camera, size = standard_scenes()["behind_elevated"]
        column = {k: v for k, v in observe(camera, size).items() if k.endswith("_sl")}
        self.assertGreaterEqual(len(column), 4)
        self.assertIsNone(solve_camera(column, size))

    def test_off_floor_points_alone_return_none(self):
        camera, size = standard_scenes()["behind_elevated"]
        off_floor = {k: v for k, v in observe(camera, size).items() if KEYPOINTS[k][2] > 0}
        self.assertIsNone(solve_camera(off_floor, size))

    def test_unknown_names_are_ignored(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size, noise_px=0.5, seed=1)
        observed["not_a_keypoint"] = (10.0, 10.0)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        self.assertNotIn("not_a_keypoint", solution.inliers + solution.outliers)

    def test_non_finite_pixels_are_ignored(self):
        camera, size = standard_scenes()["behind_elevated"]
        observed = observe(camera, size, noise_px=0.5, seed=1)
        names = [name for name in observed if KEYPOINTS[name][2] == 0.0][:2]
        observed[names[0]] = (float("nan"), 100.0)
        observed[names[1]] = (float("inf"), 100.0)
        solution = solve_camera(observed, size)
        self.assertIsNotNone(solution)
        for name in names:
            self.assertNotIn(name, solution.inliers + solution.outliers)
        self.assertLess(np.median(grid_error_cm(camera, solution.camera)), 3.0)

    def test_identical_pixels_do_not_crash_or_validate(self):
        same = {name: (500.0, 500.0) for name in ("back0_sl", "back0_sr", "back1_sl", "back1_sr")}
        solution = solve_camera(same, LANDSCAPE)
        self.assertTrue(solution is None or solution.tier == "unavailable")


if __name__ == "__main__":
    unittest.main()
