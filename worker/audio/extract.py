# worker/audio/extract.py
"""The sound track of a video file, as mono float samples, read with ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000


def read_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Mono float32 samples at `sample_rate`; None when the file has no sound track."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is needed to read the sound track of a video")
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=False)
    message = result.stderr.decode("utf-8", "replace")
    if result.returncode != 0 and "does not contain any stream" not in message:
        raise RuntimeError(f"ffmpeg could not read the sound track of {path.name}: {message.strip()[-300:]}")
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype="<f4").copy()
