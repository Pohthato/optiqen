# worker/test_skeleton.py
import unittest

import numpy as np

import geometry.skeleton as sk
from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from simulation.camera_path import stable_handheld
from simulation.people import SEGMENTS, model_pose, playing
from simulation.rally import random_rally

IDX = list(sk.JOINTS)


def clip(view, side, seed, first, count, noise_px=3.0, prior=True):
    """A stretch of a random rally's player seen from a held phone: the detections, cameras,
    times, a model's poses and the true joints."""
    rng = np.random.default_rng(seed)
    rally = random_rally(rng)
    position, target = VIEWS[view]
    base = Camera.look_at(tuple(np.asarray(position) + rng.uniform(-0.5, 0.5, 3) * (1, 1, 0.4)), target, FOCAL_PX, SIZE)
    cameras = stable_handheld(base, first + count, 30.0, seed=seed)[first:]
    truth = [playing(rally, side, (first + i) / 30.0) for i in range(count)]
    frames = [(c.project(t) + rng.normal(0, noise_px, (17, 2)), np.where(rng.random(17) < 0.05, 0.0, 0.9)) for c, t in zip(cameras, truth)]
    model_rng = np.random.default_rng(seed + 100)
    priors = [model_pose(c, t[IDX], model_rng) for c, t in zip(cameras, truth)] if prior else None
    times = [(first + i) * 1000 / 30 for i in range(count)]
    return frames, cameras, times, priors, np.array([t[IDX] for t in truth])


class PartsTests(unittest.TestCase):
    def test_a_frame_s_jacobian_matches_its_differences(self):
        frames, cameras, _, priors, truth = clip("corner", "near", 2, 30, 2)
        rng = np.random.default_rng(0)
        lengths = {name: fraction * 1.75 for name, fraction in sk.STARTING.items()}
        pixels, scores = frames[1][0][IDX], frames[1][1][IDX]
        for prior, previous in ((None, None), (priors[1], truth[0])):
            problem = sk._Frame(cameras[1], pixels, np.sqrt(scores), lengths, previous, 0.3, 3.0, prior)
            x = (truth[1] + rng.normal(0, 0.05, truth[1].shape)).ravel()
            r, J = problem(x)
            numeric = np.array([(problem(x + e)[0] - problem(x - e)[0]) / 2e-7 for e in np.eye(x.size) * 1e-7]).T
            np.testing.assert_allclose(J, numeric, atol=1e-4 * max(1.0, np.abs(J).max()))

    def test_the_floor_pull_levels_off_for_a_lifted_foot(self):
        near, half_near = sk._capped(0.5)[0], sk._capped(0.25)[0]
        far_up, slope = sk._capped(20.0)
        self.assertAlmostEqual(near / half_near, 2.0, delta=0.02)  # a plain residual near the floor
        self.assertAlmostEqual(far_up**2, sk.FLOOR_CAP, places=6)
        self.assertLess(slope, 1e-12)

    def test_the_links_measure_each_joint_s_speed_and_the_hips_acceleration(self):
        times = np.array([0.0, 0.03, 0.07, 0.1, 0.6, 0.633])
        x = np.zeros((len(times), len(sk.JOINTS), 3))
        x[:, :, 0] = (2.0 * times + 2.5 * times**2)[:, None]  # 5 m/s^2 along x
        r = sk._links(times) @ x.ravel()
        pairs = 4  # 0-1, 1-2, 2-3 and 4-5; 3-4 is too far apart
        speeds = r[: pairs * 3 * len(sk.JOINTS)].reshape(pairs, len(sk.JOINTS), 3) * sk.MOVE_SIGMA_M_PER_S
        np.testing.assert_allclose(speeds[0, :, 0], 2.0 + 2.5 * 0.03, rtol=1e-9)
        accelerations = r[pairs * 3 * len(sk.JOINTS) :].reshape(-1, 3) * sk.ACCEL_SIGMA_M_PER_S2
        np.testing.assert_allclose(accelerations, [[5.0, 0, 0], [5.0, 0, 0]], atol=1e-9)

    def test_each_frame_s_covariance_is_its_block_of_the_inverse(self):
        from scipy import sparse

        count, size = 10, 3 * len(sk.JOINTS)
        n = count * size
        J = sparse.random(3 * n, n, density=0.01, random_state=1, format="csr")
        neighbours = np.abs(np.subtract.outer(np.arange(n) // size, np.arange(n) // size)) <= 2  # tied to two frames either side
        information = (J.T @ J).toarray() * neighbours + np.eye(n) * 5.0
        inverse = np.linalg.inv(information)
        blocks = sk._covariances(sparse.csc_matrix(information), count, size)
        for f in range(count):
            np.testing.assert_allclose(blocks[f], inverse[f * size : (f + 1) * size, f * size : (f + 1) * size], atol=1e-12)

    def test_a_foot_s_slide_jacobian_matches_its_differences(self):
        rng = np.random.default_rng(1)
        times = np.array([0.0, 0.033, 0.066, 0.1])
        x = rng.normal(0, 0.05, (len(times), 3 * len(sk.JOINTS)))
        x.reshape(len(times), -1, 3)[:, (sk.SLOT[15], sk.SLOT[16]), 2] = sk.ANKLE_HEIGHT_M + rng.uniform(-0.02, 0.06, (len(times), 2))
        r, J = sk._slides(x, times)
        flat = x.ravel()
        numeric = np.array([(sk._slides((flat + e).reshape(x.shape), times)[0] - sk._slides((flat - e).reshape(x.shape), times)[0]) / 2e-7 for e in np.eye(flat.size) * 1e-7]).T
        np.testing.assert_allclose(J.toarray(), numeric, atol=1e-5 * np.abs(J.toarray()).max())


class SimulatedPlayerTests(unittest.TestCase):
    def test_a_foot_on_the_floor_stays_put_and_bones_keep_their_length(self):
        rally = random_rally(np.random.default_rng(4))
        times = np.arange(0, rally.end_time, 1 / 60)
        for side in ("near", "far"):
            bodies = np.array([playing(rally, side, t) for t in times])
            for ankle in (15, 16):
                down = bodies[:, ankle, 2] < SEGMENTS["ankle"] * 1.75 + 0.005
                moving = np.linalg.norm(np.diff(bodies[:, ankle, :2], axis=0), axis=1) * 60
                self.assertLess(np.percentile(moving[down[1:] & down[:-1]], 99), 0.1)
                self.assertGreater(bodies[:, ankle, 2].max(), 0.15)  # it steps
            shank = np.linalg.norm(bodies[:, 15] - bodies[:, 13], axis=1)
            np.testing.assert_allclose(shank, SEGMENTS["shank"] * 1.75, atol=1e-9)


class FitTests(unittest.TestCase):
    def test_perfect_joints_and_pose_give_the_body(self):
        # Not exactly: a stepping foot a few centimetres off the floor, as it lifts and lands, is
        # held a little to the floor, which moves the body by a few centimetres.
        frames, cameras, times, priors, truth = clip("side", "near", 3, 0, 12, noise_px=0.0)
        priors = [(t - (t[7] + t[8]) / 2) @ c.rotation.T for t, c in zip(truth, cameras)]
        bodies = sk.fit_bodies(frames, cameras, times, facing=np.pi / 2, priors=priors)
        fitted = np.array([b.joints[IDX] for b in bodies])
        self.assertLess(np.median(np.linalg.norm(fitted - truth, axis=2)), 0.04)

    def test_a_jump_is_a_jump_not_a_step_back(self):
        # Behind the near baseline, the far player jumps for a smash around frames 44-58. Held to
        # the floor, the jump would look like the player moving half a metre away and back.
        frames, cameras, times, priors, truth = clip("behind", "far", 5, 30, 40)
        bodies = sk.fit_bodies(frames, cameras, times, facing=-np.pi / 2, priors=priors)
        fitted = np.array([b.joints[IDX] for b in bodies])
        hips_true, hips_fit = (truth[:, 7] + truth[:, 8]) / 2, (fitted[:, 7] + fitted[:, 8]) / 2
        top = int(np.argmax(hips_true[:, 2]))
        self.assertGreater(hips_true[top, 2] - hips_true[0, 2], 0.2)  # it is a jump
        self.assertLess(abs(hips_fit[top, 2] - hips_true[top, 2]), 0.08)
        self.assertLess(np.max(np.linalg.norm(hips_fit - hips_true, axis=1)), 0.3)
        self.assertLess(np.median(np.linalg.norm(fitted - truth, axis=2)), 0.2)  # the far player from behind: placed along the line of sight to about 20 cm
        sigma = np.array([b.sigma[IDX] for b in bodies])
        self.assertTrue(np.isfinite(sigma).all() and (sigma > 0).all())


if __name__ == "__main__":
    unittest.main()
