# worker/test_flight_fit.py
import unittest

import numpy as np

from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from geometry.flight_fit import fit_flight, simulate_batch
from geometry.shuttle_physics import simulate
from simulation.camera_path import stable_handheld
from simulation.rally import random_rally


def observed_flights(seed: int, view: str, fps: float, jitter_px: float = 2.0, dropout: float = 0.1, outliers: float = 0.0):
    """Every flight of a random rally as a steadily held phone sees it: per-frame cameras, the
    detections inside the frame (jittered, some missed, some wild), and the true 3D flight."""
    rng = np.random.default_rng(seed)
    rally = random_rally(rng)
    position, target = VIEWS[view]
    base = Camera.look_at(tuple(np.asarray(position) + rng.uniform(-0.5, 0.5, 3) * (1, 1, 0.4)), target, FOCAL_PX, SIZE)
    count = int(np.ceil((rally.end_time + 0.2) * fps))
    cameras = stable_handheld(base, count, fps, seed=seed)
    flights = []
    for flight in rally.flights:
        frames = [i for i in range(count) if flight.start_time <= i / fps <= flight.end_time]
        times_ms, pixels, cams = [], [], []
        for i in frames:
            if rng.random() < dropout:
                continue
            truth = simulate(flight.p0, flight.v0, [i / fps - flight.start_time], rally.terminal_velocity)[0]
            pixel = cameras[i].project(np.array([truth]))[0]
            if not (0 <= pixel[0] < SIZE[0] and 0 <= pixel[1] < SIZE[1]):
                continue
            if rng.random() < outliers:
                pixel = rng.uniform((0, 0), SIZE)
            else:
                pixel = pixel + rng.normal(0, jitter_px, 2)
            times_ms.append(i * 1000.0 / fps)
            pixels.append(pixel)
            cams.append(cameras[i])
        flights.append((flight, rally.terminal_velocity, np.array(times_ms), np.array(pixels).reshape(-1, 2), cams))
    return flights


def end_error_m(flight, terminal_velocity, fit) -> float:
    """Where the fit puts the shuttle when the flight ended (a receive point, or the landing)."""
    truth = simulate(flight.p0, flight.v0, [flight.end_time - flight.start_time], terminal_velocity)[0]
    return float(np.linalg.norm(fit.position_at(flight.end_time * 1000.0) - truth))


class SimulateBatchTests(unittest.TestCase):
    def test_matches_the_reference_simulator_for_many_launches_at_once(self):
        rng = np.random.default_rng(0)
        p0 = rng.uniform((0, 1, 0.5), (5, 12, 2.8), (5, 3))
        v0 = rng.uniform((-5, -25, -10), (5, 25, 15), (5, 3))
        times = np.array([0.0, 0.013, 0.2, 0.71, 1.4])
        batch = simulate_batch(p0, v0, times)
        for k in range(5):
            np.testing.assert_allclose(batch[k], simulate(p0[k], v0[k], times), atol=1e-6)


class FitFlightTests(unittest.TestCase):
    def test_a_clean_flight_is_recovered_exactly(self):
        flight, vt, times, pixels, cams = observed_flights(1, "behind", 30.0, jitter_px=0.0, dropout=0.0)[1]
        fit = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt)
        self.assertLess(end_error_m(flight, vt, fit), 0.02)
        self.assertLess(float(np.linalg.norm(fit.v0 - flight.v0)), 0.1)

    def test_noisy_flights_from_every_view_and_frame_rate(self):
        errors, speed_errors = [], []
        for seed in range(4):
            for view in VIEWS:
                for fps in (30.0, 60.0):
                    for flight, vt, times, pixels, cams in observed_flights(100 + seed, view, fps):
                        if len(times) < 8:
                            continue
                        fit = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt)
                        self.assertIsNotNone(fit)
                        errors.append(end_error_m(flight, vt, fit))
                        speed_errors.append(abs(np.linalg.norm(fit.v0) - np.linalg.norm(flight.v0)) / np.linalg.norm(flight.v0))
        self.assertLess(float(np.median(errors)), 0.30, np.percentile(errors, [50, 90]))
        self.assertLess(float(np.median(speed_errors)), 0.10)

    def test_wild_detections_do_not_drag_the_fit(self):
        errors = []
        for seed in range(4):
            for flight, vt, times, pixels, cams in observed_flights(200 + seed, "corner", 30.0, outliers=0.1):
                if len(times) >= 8:
                    fit = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt)
                    errors.append(end_error_m(flight, vt, fit))
        self.assertLess(float(np.median(errors)), 0.30)

    def test_the_stated_uncertainty_is_honest(self):
        inside, total = 0, 0
        for seed in range(6):
            for flight, vt, times, pixels, cams in observed_flights(300 + seed, "behind", 30.0):
                if len(times) < 8:
                    continue
                fit = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt)
                truth = simulate(flight.p0, flight.v0, [flight.end_time - flight.start_time], vt)[0]
                total += 1
                inside += fit.within(flight.end_time * 1000.0, truth, confidence=0.95)
        self.assertGreaterEqual(inside / total, 0.8, (inside, total))

    def test_a_good_starting_guess_is_refined_without_a_depth_search(self):
        flight, vt, times, pixels, cams = observed_flights(1, "corner", 30.0)[2]
        guess = np.concatenate([np.asarray(flight.p0) + 0.3, np.asarray(flight.v0) * 1.05])
        warm = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt, initial=[guess])
        cold = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt)
        self.assertLess(end_error_m(flight, vt, warm), max(0.3, 1.5 * end_error_m(flight, vt, cold)))

    def test_a_bad_starting_guess_falls_back_to_the_depth_search(self):
        flight, vt, times, pixels, cams = observed_flights(1, "corner", 30.0)[2]
        nonsense = np.array([0.0, 0.0, 9.0, 40.0, 40.0, 40.0])
        fit = fit_flight(times, pixels, cams, start_ms=flight.start_time * 1000, terminal_velocity=vt, initial=[nonsense])
        self.assertLess(end_error_m(flight, vt, fit), 0.5)

    def test_too_few_detections_give_no_flight(self):
        flight, vt, times, pixels, cams = observed_flights(1, "behind", 30.0)[1]
        self.assertIsNone(fit_flight(times[:4], pixels[:4], cams[:4], start_ms=flight.start_time * 1000))


if __name__ == "__main__":
    unittest.main()
