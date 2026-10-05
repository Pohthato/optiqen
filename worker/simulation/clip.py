# worker/simulation/clip.py
"""Assemble a synthetic clip (frames, cameras, shuttle truth, audio) and write it to disk.

    python -m simulation.clip --out ../synthetic/clip-001 [--fps 60] [--tripod] [--seed 0]

Writes video.mp4 (OpenCV mp4v; readable by the worker, not by browsers), audio.wav and
truth.json. Frame i is time i / fps; every per-frame truth list is indexed the same way.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import wavfile

from geometry.camera import Camera
from simulation.audio import SAMPLE_RATE, render_audio
from simulation.camera_path import stable_handheld, tripod
from simulation.rally import Rally, canned_rally, player_positions
from simulation.render import render_frame

SPEED_OF_SOUND_M_S = 343.0


@dataclass
class SyntheticClip:
    fps: float
    image_size: tuple[int, int]
    cameras: list[Camera]
    frames: list[np.ndarray]
    shuttle: list[np.ndarray | None]
    rally: Rally
    audio: np.ndarray
    sample_rate: int
    audio_offset_s: float
    distractor_times: list[float]
    contact_audio_times: list[float]


def make_clip(
    base_camera: Camera,
    image_size: tuple[int, int],
    rally: Rally | None = None,
    fps: float = 60.0,
    handheld: bool = True,
    seed: int = 0,
    tail_s: float = 0.5,
    audio_offset_s: float = 0.0,
    distractors: int = 0,
    noise_sigma: float = 2.0,
    sound_delay: bool = True,
) -> SyntheticClip:
    if image_size[0] % 2 or image_size[1] % 2:
        raise ValueError(f"frame size {image_size} must be even: the mp4v writer silently crops odd sizes")
    rally = rally or canned_rally()
    duration = rally.end_time + tail_s
    count = math.ceil(duration * fps)
    cameras = stable_handheld(base_camera, count, fps, seed=seed) if handheld else tripod(base_camera, count)
    exposure = 0.5 / fps
    frames: list[np.ndarray] = []
    shuttle: list[np.ndarray | None] = []
    for index in range(count):
        time = index / fps
        position = rally.shuttle_at(time)
        previous = rally.shuttle_at(max(0.0, time - exposure)) if position is not None else None
        players = list(player_positions(rally, time).values())
        frames.append(render_frame(cameras[index], image_size, position, previous, players, noise_sigma, seed + index))
        shuttle.append(position)
    # A hit is heard once its sound has travelled to the phone (~15-45 ms on a court).
    heard = []
    for contact in rally.contacts:
        delay = 0.0
        if sound_delay:
            camera = cameras[min(count - 1, round(contact.time * fps))]
            delay = float(np.linalg.norm(np.asarray(contact.position) - camera.centre)) / SPEED_OF_SOUND_M_S
        heard.append(contact.time + delay)
    audio, distractor_times = render_audio(heard, duration, seed=seed, distractors=distractors, offset_s=audio_offset_s)
    contact_audio_times = [time + audio_offset_s for time in heard]
    return SyntheticClip(
        fps, image_size, cameras, frames, shuttle, rally, audio, SAMPLE_RATE, audio_offset_s, distractor_times, contact_audio_times
    )


def _camera_entry(camera: Camera) -> dict[str, Any]:
    return {
        "focalPx": float(camera.focal_px),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "rvec": [float(v) for v in np.asarray(camera.rvec).ravel()],
        "tvec": [float(v) for v in np.asarray(camera.tvec).ravel()],
        "k1": float(camera.k1),
    }


def camera_from_truth(entry: dict[str, Any]) -> Camera:
    return Camera(entry["focalPx"], entry["cx"], entry["cy"], np.array(entry["rvec"]), np.array(entry["tvec"]), entry["k1"])


def truth_dict(clip: SyntheticClip) -> dict[str, Any]:
    return {
        "fps": clip.fps,
        "imageSize": list(clip.image_size),
        "exposureS": 0.5 / clip.fps,
        "conventions": {
            "imageSize": "[width, height] in pixels",
            "time": "frame i is at i / fps seconds; every per-frame list is indexed by frame",
            "shuttle": "court metres (x across, y along, z up) at the frame time: the LEADING end of the motion-blur streak, which covers [t - exposureS, t]; null outside flight",
            "cameras": "rvec/tvec map world to camera (OpenCV: x right, y down, z forward); k1 is one radial term on normalised coordinates",
            "audioTimeMs": "when the hit is heard in audio.wav: contact time + sound travel to the camera + audio offset",
            "audioOffset": "positive offsetMs means the audio lags the video",
            "distractorTimes": "times in audio.wav of hits from a neighbouring court; no matching contact exists",
        },
        "frameCount": len(clip.frames),
        "cameras": [_camera_entry(camera) for camera in clip.cameras],
        "shuttle": [None if position is None else [round(float(v), 5) for v in position] for position in clip.shuttle],
        "contacts": [
            {
                "timeMs": round(contact.time * 1000, 2),
                "audioTimeMs": round(heard * 1000, 2),
                "hitter": contact.hitter,
                "kind": contact.kind,
                "position": [round(v, 5) for v in contact.position],
            }
            for contact, heard in zip(clip.rally.contacts, clip.contact_audio_times)
        ],
        "landing": [round(v, 5) for v in clip.rally.landing],
        "rallyEndMs": round(clip.rally.end_time * 1000, 2),
        "terminalVelocity": clip.rally.terminal_velocity,
        "audio": {
            "sampleRate": clip.sample_rate,
            "offsetMs": round(clip.audio_offset_s * 1000, 3),
            "distractorTimesMs": [round(t * 1000, 2) for t in clip.distractor_times],
        },
    }


def write_clip(clip: SyntheticClip, directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"video": directory / "video.mp4", "audio": directory / "audio.wav", "truth": directory / "truth.json"}
    writer = cv2.VideoWriter(str(paths["video"]), cv2.VideoWriter_fourcc(*"mp4v"), clip.fps, clip.image_size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an mp4v video writer")
    for frame in clip.frames:
        writer.write(frame)
    writer.release()
    wavfile.write(paths["audio"], clip.sample_rate, (np.clip(clip.audio, -1.0, 1.0) * 32767).astype(np.int16))
    paths["truth"].write_text(json.dumps(truth_dict(clip)), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a synthetic singles rally with ground truth.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tripod", action="store_true", help="still camera instead of a steadily held phone")
    parser.add_argument("--distractors", type=int, default=2, help="quieter hits from a neighbouring court")
    parser.add_argument("--audio-offset-ms", type=float, default=0.0)
    args = parser.parse_args(argv)
    size = (args.width, args.height)
    # Behind the near baseline, 3 m up: the recommended tier-A placement; focal ~ 26 mm-equivalent.
    camera = Camera.look_at((2.59, -3.5, 3.0), (2.59, 7.0, 0.0), 1300.0 * args.width / 1920, size)
    clip = make_clip(
        camera, size, fps=args.fps, handheld=not args.tripod, seed=args.seed,
        distractors=args.distractors, audio_offset_s=args.audio_offset_ms / 1000,
    )
    paths = write_clip(clip, args.out)
    print(f"wrote {len(clip.frames)} frames, {len(clip.rally.contacts)} contacts -> {paths['video'].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
