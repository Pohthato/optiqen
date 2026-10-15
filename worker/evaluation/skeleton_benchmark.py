# worker/evaluation/skeleton_benchmark.py
"""Players' bodies in 3D against the truth on synthetic rallies.

Per scenario: a random rally's near or far player (stepping, lunging, reaching, jumping for
smashes), seen from one of the three views at 30 or 60 fps through a steadily held phone. The
detector's joints carry 3 px of noise, 5 % of them are missed, and (unless --no-model) a
learned 3D pose model's output is emulated: the true pose about the hips' middle, turned 8
degrees, scaled 7 % and each joint 6 cm off, independently every frame. The bodies are fitted
(geometry.skeleton.fit_bodies) and compared with the truth: joint error, how far the body is
put along the camera's line of sight (a bias there is the classic failure of one camera),
the error while airborne, jitter (the fitted joints' frame-to-frame acceleration beyond the
truth's) and whether each joint's stated uncertainty covers its error.

    python -m evaluation.skeleton_benchmark --seeds 3 --first-seed 100 --workers 6 --out skeleton.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from geometry.skeleton import JOINTS, fit_bodies
from simulation.camera_path import stable_handheld
from simulation.people import model_pose, playing
from simulation.rally import random_rally

SECONDS = 5.0
AIRBORNE_M = 0.12  # both ankles this high: off the floor
# A joint's stated sigma combines its x, y and z spreads. If the spreads are right, its error is
# within 2 sigma at least as often as a chi-square with one degree of freedom is within 4 (all
# the spread along one axis, as along a line of sight) and at most as often as one with three is
# within 12 (spread evenly).
COVERED = (float(chi2.cdf(4.0, 1)), float(chi2.cdf(12.0, 3)))


def score(job: tuple[str, str, float, int, bool]) -> dict:
    view, side, fps, seed, model = job
    rng = np.random.default_rng(seed)
    rally = random_rally(rng)
    position, target = VIEWS[view]
    base = Camera.look_at(tuple(np.asarray(position) + rng.uniform(-0.5, 0.5, 3) * (1, 1, 0.4)), target, FOCAL_PX, SIZE)
    count = int(min(SECONDS, rally.end_time) * fps)
    cameras = stable_handheld(base, count, fps, seed=seed)
    idx = list(JOINTS)
    truth = np.array([playing(rally, side, i / fps) for i in range(count)])
    frames = [(c.project(t) + rng.normal(0, 3.0, (17, 2)), np.where(rng.random(17) < 0.05, 0.0, 0.9)) for c, t in zip(cameras, truth)]
    model_rng = np.random.default_rng(seed + 100)
    priors = [model_pose(c, t[idx], model_rng) for c, t in zip(cameras, truth)] if model else None
    times = [i * 1000.0 / fps for i in range(count)]
    bodies = fit_bodies(frames, cameras, times, facing=np.pi / 2 if side == "near" else -np.pi / 2, priors=priors)
    fitted = np.array([b.joints[idx] for b in bodies])
    sigma = np.array([b.sigma[idx] for b in bodies])
    true = truth[:, idx]
    error = np.linalg.norm(fitted - true, axis=2)
    centre = np.array([-c.rotation.T @ np.asarray(c.tvec).reshape(3) for c in cameras])
    hips_fit, hips_true = (fitted[:, 7] + fitted[:, 8]) / 2, (true[:, 7] + true[:, 8]) / 2
    ray = hips_true - centre
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    along = np.sum((hips_fit - hips_true) * ray, axis=1)
    airborne = np.minimum(true[:, 11, 2], true[:, 12, 2]) > AIRBORNE_M
    dt = 1.0 / fps
    shake = np.linalg.norm(np.diff(fitted, 2, axis=0) - np.diff(true, 2, axis=0), axis=2) / dt**2
    return {
        "view": view, "side": side, "fps": fps, "seed": seed, "model": model,
        "error": error.ravel().tolist(), "along": along.tolist(), "airborne": np.repeat(airborne, len(idx)).tolist(),
        "covered": (error <= 2 * sigma).ravel().tolist(), "ratio": (error / sigma).ravel().tolist(), "jitter": float(np.median(shake)),
    }


def summarise(results: list[dict]) -> dict:
    out: dict = {}
    for key in sorted({(r["view"], r["side"]) for r in results}):
        group = [r for r in results if (r["view"], r["side"]) == key]
        error = np.concatenate([r["error"] for r in group])
        airborne = np.concatenate([r["airborne"] for r in group]).astype(bool)
        along = np.concatenate([r["along"] for r in group])
        covered = np.concatenate([r["covered"] for r in group]).astype(bool)
        out[f"{key[0]}, {key[1]}"] = {
            "jointErrorCm": {"median": round(float(np.median(error)) * 100, 1), "p90": round(float(np.percentile(error, 90)) * 100, 1)},
            "alongSightCm": {"median": round(float(np.median(along)) * 100, 1), "p90Abs": round(float(np.percentile(np.abs(along), 90)) * 100, 1)},
            "airborneJointErrorCm": round(float(np.median(error[airborne])) * 100, 1) if airborne.any() else None,
            "airborneShare": round(float(airborne.mean()), 3),
            "within2Sigma": round(float(covered.mean()), 3),
            "errorInSigmasP95": round(float(np.percentile(np.concatenate([r["ratio"] for r in group]), 95)), 2),
            "jitterMS2": round(float(np.median([r["jitter"] for r in group])), 1),
            "frames": int(len(error) // len(JOINTS)),
        }
    error = np.concatenate([r["error"] for r in results])
    out["all"] = {"jointErrorCm": {"median": round(float(np.median(error)) * 100, 1), "p90": round(float(np.percentile(error, 90)) * 100, 1)},
                  "within2Sigma": round(float(np.mean(np.concatenate([r["covered"] for r in results]))), 3),
                  "errorInSigmasP95": round(float(np.percentile(np.concatenate([r["ratio"] for r in results]), 95)), 2),
                  "within2SigmaIfCalibrated": [round(c, 3) for c in COVERED]}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark players' 3D bodies on synthetic rallies.")
    parser.add_argument("--seeds", type=int, default=3, help="rallies per view, half and frame rate")
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--fps", type=float, nargs="+", default=[30.0, 60.0])
    parser.add_argument("--no-model", action="store_true", help="geometry alone, without a learned 3D pose model")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    jobs = [(view, side, fps, args.first_seed + k, not args.no_model) for view in VIEWS for side in ("near", "far") for fps in args.fps for k in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(score, jobs))
    else:
        results = [score(job) for job in jobs]
    summary = summarise(results)
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "jobs": [{k: v for k, v in r.items() if k in ("view", "side", "fps", "seed", "jitter")} for r in results]}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
