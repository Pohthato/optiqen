# worker/test_skeleton.py
import dataclasses
import unittest

import numpy as np
from scipy import sparse

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


class Frame:
    """One frame's problem as first written, joint by joint, with a dense Jacobian: what
    skeleton._Frames, all frames at once, must reproduce."""

    def __init__(self, camera, pixels, weights, lengths, previous, step_sigma, noise_px, prior=None):
        self.R = camera.rotation
        self.t = np.asarray(camera.tvec, dtype=float).reshape(3)
        self.f, self.cx, self.cy, self.k1 = camera.focal_px, camera.cx, camera.cy, camera.k1
        self.pixels, self.weights, self.lengths = pixels, weights, lengths
        self.previous, self.step_sigma, self.noise = previous, step_sigma, noise_px
        n = len(sk.JOINTS)
        self.prior = None if prior is None else np.asarray(prior, dtype=float) @ self.R
        self.size = 2 * n + len(sk.PAIRS) + 2 + 2 + 2 + 2 + 2 + (3 * n if previous is not None else 0) + (3 * n if prior is not None else 0)

    def __call__(self, x):
        SLOT, n = sk.SLOT, len(sk.JOINTS)
        X = x.reshape(n, 3)
        r = np.zeros(self.size)
        J = np.zeros((self.size, 3 * n))
        p = X @ self.R.T + self.t
        depth = np.where(p[:, 2] > 1e-3, p[:, 2], 1e-3)
        u, v = p[:, 0] / depth, p[:, 1] / depth
        s = 1.0 + self.k1 * (u * u + v * v)
        scale = self.weights / self.noise
        r[0 : 2 * n : 2] = (self.f * u * s + self.cx - self.pixels[:, 0]) * scale
        r[1 : 2 * n : 2] = (self.f * v * s + self.cy - self.pixels[:, 1]) * scale
        for k in range(n):
            du = np.array([1 / depth[k], 0.0, -u[k] / depth[k]])
            dv = np.array([0.0, 1 / depth[k], -v[k] / depth[k]])
            ds = self.k1 * (2 * u[k] * du + 2 * v[k] * dv)
            J[2 * k, 3 * k : 3 * k + 3] = self.f * (s[k] * du + u[k] * ds) @ self.R * scale[k]
            J[2 * k + 1, 3 * k : 3 * k + 3] = self.f * (s[k] * dv + v[k] * ds) @ self.R * scale[k]
        row = 2 * n
        for name, a, b in sk.PAIRS:
            span = X[a] - X[b]
            length = max(float(np.linalg.norm(span)), 1e-9)
            r[row] = (length - self.lengths[name]) / sk.BONE_SIGMA_M
            J[row, 3 * a : 3 * a + 3] = span / length / sk.BONE_SIGMA_M
            J[row, 3 * b : 3 * b + 3] = -span / length / sk.BONE_SIGMA_M
            row += 1
        ankles = (SLOT[15], SLOT[16])
        for a in ankles:
            gap = X[a, 2] - sk.ANKLE_HEIGHT_M
            if gap < 0:
                r[row], J[row, 3 * a + 2] = gap / sk.FLOOR_SIGMA_M, 1.0 / sk.FLOOR_SIGMA_M
            row += 1
        for a in ankles:
            r[row], slope = sk._capped((X[a, 2] - sk.ANKLE_HEIGHT_M) / sk.DOWN_SIGMA_M)
            J[row, 3 * a + 2] = slope / sk.DOWN_SIGMA_M
            row += 1
        across = X[SLOT[12]] - X[SLOT[11]]
        forward = np.array([-across[1], across[0], 0.0])
        forward /= max(np.linalg.norm(forward), 1e-9)
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
            h, k, a = SLOT[hip], SLOT[knee], SLOT[ankle]
            ahead = float((X[k] - (X[h] + X[a]) / 2) @ forward)
            if ahead < 0:
                r[row] = ahead / sk.KNEE_SIGMA_M
                J[row, 3 * k : 3 * k + 3] = forward / sk.KNEE_SIGMA_M
                J[row, 3 * h : 3 * h + 3] = J[row, 3 * a : 3 * a + 3] = -forward / (2 * sk.KNEE_SIGMA_M)
            row += 1
        shoulders, hips = (SLOT[5], SLOT[6]), (SLOT[11], SLOT[12])
        off = (X[shoulders[0]] + X[shoulders[1]] - X[hips[0]] - X[hips[1]]) / 2
        for axis in (0, 1):
            r[row] = off[axis] / sk.LEAN_SIGMA_M
            for j, sign in ((shoulders[0], 1), (shoulders[1], 1), (hips[0], -1), (hips[1], -1)):
                J[row, 3 * j + axis] = sign * 0.5 / sk.LEAN_SIGMA_M
            row += 1
        for a, b in (shoulders, hips):
            r[row] = (X[a, 2] - X[b, 2]) / sk.LEVEL_SIGMA_M
            J[row, 3 * a + 2], J[row, 3 * b + 2] = 1.0 / sk.LEVEL_SIGMA_M, -1.0 / sk.LEVEL_SIGMA_M
            row += 1
        if self.previous is not None:
            r[row : row + 3 * n] = ((X - self.previous) / self.step_sigma).ravel()
            J[row : row + 3 * n, :] = np.eye(3 * n) / self.step_sigma
            row += 3 * n
        if self.prior is not None:
            middle = (X[hips[0]] + X[hips[1]]) / 2
            r[row : row + 3 * n] = ((X - middle - self.prior) / sk.PRIOR_SIGMA_M).ravel()
            block = np.eye(3 * n) / sk.PRIOR_SIGMA_M
            for h in hips:
                for k in range(n):
                    block[3 * k : 3 * k + 3, 3 * h : 3 * h + 3] -= 0.5 * np.eye(3) / sk.PRIOR_SIGMA_M
            J[row : row + 3 * n, :] = block
        robust = np.array(list(range(2 * n)) + (list(range(row, row + 3 * n)) if self.prior is not None else []))
        r[robust], slope = sk._soft(r[robust], sk.ROBUST_SCALE)
        J[robust] *= slope[:, None]
        return r, J


class PartsTests(unittest.TestCase):
    def test_a_frame_s_jacobian_matches_its_differences(self):
        frames, cameras, _, priors, truth = clip("corner", "near", 2, 30, 2)
        rng = np.random.default_rng(0)
        lengths = {name: fraction * 1.75 for name, fraction in sk.STARTING.items()}
        pixels, scores = frames[1][0][IDX], frames[1][1][IDX]
        for prior, previous in ((None, None), (priors[1], truth[0])):
            problem = sk._Frames([cameras[1]], [pixels], [np.sqrt(scores)], [prior], lengths, 3.0, None if previous is None else [previous], [0.3])
            x = (truth[1] + rng.normal(0, 0.05, truth[1].shape)).ravel()
            J = problem(x)[1].toarray()
            numeric = np.array([(problem(x + e)[0] - problem(x - e)[0]) / 2e-7 for e in np.eye(x.size) * 1e-7]).T
            np.testing.assert_allclose(J, numeric, atol=1e-4 * max(1.0, np.abs(J).max()))

    def test_every_frame_at_once_is_each_frame_on_its_own(self):
        # A lens with radial distortion, missed joints, frames without the model's pose, ankles
        # below the floor and a knee bent back: every rule's every branch.
        frames, cameras, _, priors, truth = clip("corner", "near", 2, 30, 9)
        cameras = [dataclasses.replace(c, k1=-0.04) for c in cameras]
        priors = [None if k % 3 == 1 else p for k, p in enumerate(priors)]
        lengths = {name: fraction * 1.75 for name, fraction in sk.STARTING.items()}
        pixels = [f[0][IDX] for f in frames]
        weights = [np.where(f[1][IDX] >= sk.MIN_SCORE, np.sqrt(f[1][IDX]), 0.0) for f in frames]
        x = truth + np.random.default_rng(1).normal(0, 0.05, truth.shape)
        x[0, [sk.SLOT[15], sk.SLOT[16]], 2] = sk.ANKLE_HEIGHT_M - 0.03
        across = x[0, sk.SLOT[12]] - x[0, sk.SLOT[11]]
        x[0, sk.SLOT[13], :2] -= 0.3 * np.array([-across[1], across[0]]) / np.hypot(*across[:2])
        self.assertTrue(any(min(w) == 0 for w in weights))
        floor = 2 * len(IDX) + len(sk.PAIRS)
        for previous, steps in ((None, None), (truth + 0.02, np.full(len(truth), 0.3))):
            problem = sk._Frames(cameras, pixels, weights, priors, lengths, 3.0, previous, steps)
            r, J = problem(x.ravel())
            parts = [Frame(cameras[k], pixels[k], weights[k], lengths, None if previous is None else previous[k], 0.3, 3.0, priors[k])(x[k].ravel()) for k in range(len(x))]
            np.testing.assert_allclose(r, np.concatenate([part[0] for part in parts]), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(J.toarray(), sparse.block_diag([part[1] for part in parts]).toarray(), rtol=1e-9, atol=1e-9)
            self.assertTrue((r[floor : floor + 2] < 0).all() and r[floor + 4] < 0)  # both ankles below the floor, a knee bent back

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
