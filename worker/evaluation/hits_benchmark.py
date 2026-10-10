# worker/evaluation/hits_benchmark.py
"""Hit finding on synthetic rallies made hard on purpose.

    python -m evaluation.hits_benchmark --scenarios 60 --out hits-report.json

Each scenario is a random singles rally (simulation.rally.random_rally) filmed by a steadily held
phone from behind, the corner or the side (its position jittered), at 30 or 60 fps, followed by
two seconds with no play. The shuttle detector is simulated with position jitter, missed frames
and false detections (still blobs that last a few frames, during play and between rallies), and a
neighbouring court's hits are heard at uniformly random times, as close to the rally's as chance
puts them. A hit counts when found within 33 ms.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audio.onsets import detect_onsets
from geometry.camera import Camera
from hits import ONSET_GAP_MS, Hit, find_hits
from simulation.audio import SAMPLE_RATE
from simulation.clip import SyntheticClip, make_clip
from simulation.rally import random_rally

SIZE = (1280, 720)
FOCAL_PX = 1300.0 * SIZE[0] / 1920
VIEWS = {
    "behind": ((2.59, -5.0, 3.0), (2.59, 7.0, 0.0)),
    "corner": ((-2.5, -2.5, 3.0), (2.59, 6.7, 0.0)),
    "side": ((-4.5, 6.7, 3.0), (2.59, 6.7, 0.0)),
}
TOLERANCE_MS = 33.0
OBSERVABLE_MS = 250.0  # a hit is observable when the shuttle is detected this close before and after it
OBSERVABLE_DETECTIONS = 3
IDLE_AFTER_S = 2.0


@dataclass(frozen=True)
class Scenario:
    seed: int
    view: str = "behind"
    fps: float = 30.0
    jitter_px: float = 2.0
    dropout: float = 0.1
    false_per_s: float = 0.5  # false detections (each a still blob lasting 1-6 frames)
    neighbour_per_s: float = 0.4  # a neighbouring court's hits, heard at random times
    sound: bool = True


@dataclass
class Simulated:
    clip: SyntheticClip
    track: list[tuple[float, float] | None]
    times_ms: list[float]
    onsets: list | None
    false_frames: int


def simulate(scenario: Scenario) -> Simulated:
    rng = np.random.default_rng(scenario.seed)
    position, target = VIEWS[scenario.view]
    position = tuple(np.asarray(position) + rng.uniform(-0.5, 0.5, 3) * (1, 1, 0.4))
    camera = Camera.look_at(position, target, FOCAL_PX, SIZE)
    rally = random_rally(rng)
    duration = rally.end_time + IDLE_AFTER_S
    neighbours = sorted(rng.uniform(0.0, duration - 0.05, rng.poisson(scenario.neighbour_per_s * duration)).tolist())
    clip = make_clip(
        camera, SIZE, rally, fps=scenario.fps, handheld=True, seed=scenario.seed, tail_s=IDLE_AFTER_S,
        distractor_times=neighbours, render=False,
    )
    track: list[tuple[float, float] | None] = []
    blob: tuple[np.ndarray, int] | None = None
    false_frames = 0
    for camera_at, shuttle in zip(clip.cameras, clip.shuttle):
        seen = None
        if shuttle is not None and rng.random() >= scenario.dropout:
            pixel = camera_at.project(np.array([shuttle]))[0]
            if 0 <= pixel[0] < SIZE[0] and 0 <= pixel[1] < SIZE[1]:
                seen = pixel + rng.normal(0.0, scenario.jitter_px, 2)
        if blob is None and rng.random() < scenario.false_per_s / scenario.fps:
            blob = (np.array([rng.uniform(0.1, 0.9) * SIZE[0], rng.uniform(0.35, 0.95) * SIZE[1]]), int(rng.integers(1, 7)))
        if blob is not None:
            # A still white object (a shoe, a line end) wins over the shuttle some of the time.
            if seen is None or rng.random() < 0.3:
                seen = blob[0] + rng.normal(0.0, scenario.jitter_px, 2)
                false_frames += 1
            blob = (blob[0], blob[1] - 1) if blob[1] > 1 else None
        track.append(None if seen is None else (float(seen[0]), float(seen[1])))
    times_ms = [index * 1000.0 / scenario.fps for index in range(len(track))]
    onsets = detect_onsets(clip.audio, SAMPLE_RATE, min_gap_ms=ONSET_GAP_MS) if scenario.sound else None
    return Simulated(clip, track, times_ms, onsets, false_frames)


def observable(clip: SyntheticClip, track: list, times_ms: list[float]) -> list[bool]:
    """Per true hit: whether the shuttle (the real one, not a false detection) was detected at
    least OBSERVABLE_DETECTIONS times within OBSERVABLE_MS after it, and before it too unless it is
    the serve. A hit the camera never saw is beyond any method that only has this video."""
    real = [
        t for t, point, shuttle in zip(times_ms, track, clip.shuttle)
        if point is not None and shuttle is not None
    ]
    flags = []
    for index, contact in enumerate(clip.rally.contacts):
        at = contact.time * 1000.0
        after = sum(1 for t in real if at < t <= at + OBSERVABLE_MS)
        before = sum(1 for t in real if at - OBSERVABLE_MS <= t < at)
        flags.append(after >= OBSERVABLE_DETECTIONS and (index == 0 or before >= OBSERVABLE_DETECTIONS))
    return flags


def score(clip: SyntheticClip, hits: list[Hit], seen: list[bool] | None = None) -> dict[str, Any]:
    """Hits matched to true contacts one to one within the tolerance (nearest first). With `seen`
    (per true hit, from observable()), recall is also given over the hits the camera saw."""
    truth = [(contact.time * 1000.0, contact.kind) for contact in clip.rally.contacts]
    seen = seen if seen is not None else [True] * len(truth)
    pairs = sorted(
        ((abs(hit.time_ms - time), h, t) for h, hit in enumerate(hits) for t, (time, _) in enumerate(truth) if abs(hit.time_ms - time) <= TOLERANCE_MS),
    )
    used_hits, used_truth, errors = set(), set(), []
    for error, h, t in pairs:
        if h not in used_hits and t not in used_truth:
            used_hits.add(h)
            used_truth.add(t)
            errors.append(error)
    neighbour_hits = sum(
        1 for h, hit in enumerate(hits)
        if h not in used_hits and hit.heard_ms is not None and any(abs(hit.heard_ms - d * 1000.0) < 5.0 for d in clip.distractor_times)
    )
    return {
        "truth": len(truth),
        "found": len(used_truth),
        "observable": sum(seen),
        "foundObservable": sum(1 for t in used_truth if seen[t]),
        "claimed": len(hits),
        "neighbourSounds": len(clip.distractor_times),
        "neighbourHits": neighbour_hits,
        "errorsMs": errors,
        "missedKinds": [kind for t, (_, kind) in enumerate(truth) if t not in used_truth and seen[t]],
        "unseenKinds": [kind for t, (_, kind) in enumerate(truth) if not seen[t]],
    }


def _default_finder(sim: Simulated, scenario: Scenario) -> list[Hit]:
    return find_hits(sim.track, sim.times_ms, sim.clip.cameras, sim.onsets).hits


def _scored(scenario: Scenario, finder=_default_finder) -> dict[str, Any]:
    sim = simulate(scenario)
    return {**asdict(scenario), **score(sim.clip, finder(sim, scenario), observable(sim.clip, sim.track, sim.times_ms))}


def run(scenarios: list[Scenario], finder: Callable[[Simulated, Scenario], list[Hit]] | None = None, workers: int = 1) -> dict[str, Any]:
    """Scores per scenario and pooled; `finder` defaults to hits.find_hits. With workers > 1
    the scenarios run in parallel processes (the default finder only)."""
    if workers > 1 and finder is None:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_scored, scenarios))
    else:
        rows = [_scored(scenario, finder or _default_finder) for scenario in scenarios]
    return {"scenarios": rows, "summary": summarise(rows)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def pooled(subset: list[dict[str, Any]]) -> dict[str, Any]:
        found = sum(r["found"] for r in subset)
        truth = sum(r["truth"] for r in subset)
        claimed = sum(r["claimed"] for r in subset)
        errors = [e for r in subset for e in r["errorsMs"]]
        sounds = sum(r["neighbourSounds"] for r in subset)
        return {
            "scenarios": len(subset),
            "f1": round(2 * found / (claimed + truth), 3) if claimed + truth else None,
            "recall": round(found / truth, 3) if truth else None,
            "recallOfObservable": round(sum(r["foundObservable"] for r in subset) / max(1, sum(r["observable"] for r in subset)), 3),
            "observableShare": round(sum(r["observable"] for r in subset) / truth, 3) if truth else None,
            "precision": round(found / claimed, 3) if claimed else None,
            "medianErrorMs": round(float(np.median(errors)), 1) if errors else None,
            "p90ErrorMs": round(float(np.percentile(errors, 90)), 1) if errors else None,
            "neighbourSoundsTakenAsHits": f"{sum(r['neighbourHits'] for r in subset)} of {sounds}",
        }

    missed: dict[str, int] = {}
    unseen: dict[str, int] = {}
    for row in rows:
        for kind in row["missedKinds"]:
            missed[kind] = missed.get(kind, 0) + 1
        for kind in row["unseenKinds"]:
            unseen[kind] = unseen.get(kind, 0) + 1
    by = {
        "all": pooled(rows),
        **{f"view={view}": pooled([r for r in rows if r["view"] == view]) for view in sorted({r["view"] for r in rows})},
        **{f"fps={fps:g}": pooled([r for r in rows if r["fps"] == fps]) for fps in sorted({r["fps"] for r in rows})},
        **{f"sound={sound}": pooled([r for r in rows if r["sound"] == sound]) for sound in sorted({r["sound"] for r in rows})},
        # The gate holds on every view and frame rate, with sound and without.
        **{
            f"{name}={value:g},sound={sound}" if isinstance(value, float) else f"{name}={value},sound={sound}": pooled(subset)
            for name in ("view", "fps")
            for value in sorted({r[name] for r in rows})
            for sound in (True, False)
            if (subset := [r for r in rows if r[name] == value and r["sound"] == sound])
        },
    }
    return {"tolerance": f"±{TOLERANCE_MS:g} ms", "pooled": by, "missedObservableByKind": missed, "unobservableByKind": unseen}


def standard_scenarios(count: int, first_seed: int = 1000) -> list[Scenario]:
    """A fixed spread: every view at 30 and 60 fps, each with sound and without (one in three
    silent), 1-3 px jitter, 5-20 % dropout."""
    scenarios = []
    for index in range(count):
        rng = np.random.default_rng(first_seed + index)
        scenarios.append(
            Scenario(
                seed=first_seed + index,
                view=list(VIEWS)[index % len(VIEWS)],
                fps=(30.0, 60.0)[(index // len(VIEWS)) % 2],
                jitter_px=float(rng.uniform(1.0, 3.0)),
                dropout=float(rng.uniform(0.05, 0.2)),
                sound=(index // (2 * len(VIEWS))) % 3 != 2,
            )
        )
    return scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark hit finding on hard synthetic rallies.")
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=1000, help="another seed gives rallies the method was not tuned on")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    args = parser.parse_args(argv)
    report = run(standard_scenarios(args.scenarios, args.first_seed), workers=args.workers)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
