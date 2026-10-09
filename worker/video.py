# worker/video.py
"""Reading a video file's frames in order, each with its container timestamp."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoFile:
    """Frames are read in order, never by seeking: phone files seek unevenly."""

    def __init__(self, path: Path):
        self.path = Path(path)
        probe = cv2.VideoCapture(str(self.path))
        self.opened = probe.isOpened()
        self.fps = probe.get(cv2.CAP_PROP_FPS)
        self.size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))  # from the container; may be approximate
        probe.release()

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield capture.get(cv2.CAP_PROP_POS_MSEC), frame
        finally:
            capture.release()
