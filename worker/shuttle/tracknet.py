# worker/shuttle/tracknet.py
"""Shuttle position in every frame with TrackNetV3.

TrackNet (a U-Net) sees 8 consecutive frames together with the clip's background and outputs a
heatmap per frame, so a blurred streak or a shuttle a few pixels wide is still found from its
motion. Frames run through overlapping windows and each frame's heatmaps are averaged, weighted
towards the middle of each window. InpaintNet then fills short gaps along the flight (a shuttle
hidden by a player or lost in the background) from the positions around them.

Architecture, preprocessing and weights: TrackNetV3 by qaz812345 (MIT licence,
github.com/qaz812345/TrackNetV3); see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

INPUT_SIZE = (512, 288)  # width, height the networks work at
TRACKNET_FILE = "TrackNet_best.pt"
INPAINT_FILE = "InpaintNet_best.pt"
HEAT_THRESHOLD = 0.5
GAP_TOP_FRACTION = 0.05  # a shuttle last seen this near the top of the frame left it; no gap to fill
MAX_GAP_FRAMES = 8  # half InpaintNet's window; longer gaps are left empty rather than guessed
INVISIBLE_NORMALISED = 50 / math.hypot(*INPUT_SIZE)  # filled positions this near (0, 0) mean "not there"
BACKGROUND_SAMPLES = 120


@dataclass(frozen=True)
class ShuttleDetection:
    x: float  # pixels in the original frame
    y: float
    score: float  # peak heatmap value of the shuttle's blob (0.5-1); 0 for a filled gap
    inpainted: bool = False


# --- Networks (TrackNetV3 architecture; parameter names match the published weights) ---


class _Conv2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv2DStack(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, count: int):
        super().__init__()
        for index in range(count):
            setattr(self, f"conv_{index + 1}", _Conv2D(in_dim if index == 0 else out_dim, out_dim))
        self.count = count

    def forward(self, x):
        for index in range(self.count):
            x = getattr(self, f"conv_{index + 1}")(x)
        return x


class TrackNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.down_block_1 = _Conv2DStack(in_dim, 64, 2)
        self.down_block_2 = _Conv2DStack(64, 128, 2)
        self.down_block_3 = _Conv2DStack(128, 256, 3)
        self.bottleneck = _Conv2DStack(256, 512, 3)
        self.up_block_1 = _Conv2DStack(768, 256, 3)
        self.up_block_2 = _Conv2DStack(384, 128, 2)
        self.up_block_3 = _Conv2DStack(192, 64, 2)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(self.pool(x1))
        x3 = self.down_block_3(self.pool(x2))
        x = self.bottleneck(self.pool(x3))
        x = self.up_block_1(torch.cat([self.up(x), x3], dim=1))
        x = self.up_block_2(torch.cat([self.up(x), x2], dim=1))
        x = self.up_block_3(torch.cat([self.up(x), x1], dim=1))
        return torch.sigmoid(self.predictor(x))


class _Conv1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding="same", bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class _Conv1DPair(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = _Conv1D(in_dim, out_dim)
        self.conv_2 = _Conv1D(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class InpaintNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = _Conv1D(3, 32)
        self.down_2 = _Conv1D(32, 64)
        self.down_3 = _Conv1D(64, 128)
        self.buttleneck = _Conv1DPair(128, 256)  # (sic) the published weights use this name
        self.up_1 = _Conv1D(384, 128)
        self.up_2 = _Conv1D(192, 64)
        self.up_3 = _Conv1D(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding="same")

    def forward(self, coordinates, mask):
        x = torch.cat([coordinates, mask], dim=2).permute(0, 2, 1)
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x = self.buttleneck(x3)
        x = self.up_1(torch.cat([x, x3], dim=1))
        x = self.up_2(torch.cat([x, x2], dim=1))
        x = self.up_3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.predictor(x)).permute(0, 2, 1)


# --- Pure steps (tested without the networks) ---


def to_input(frame: np.ndarray) -> np.ndarray:
    """A BGR frame as the networks see it: RGB, 512x288 (bicubic, as in training), channels first."""
    image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])).resize(INPUT_SIZE, Image.BICUBIC)
    return np.moveaxis(np.asarray(image), -1, 0)


def ensemble_weights(seq_len: int) -> np.ndarray:
    """Weights by position in a window, highest in the middle, summing to 1."""
    weights = np.array([min(k + 1, seq_len - k) for k in range(seq_len)], dtype=np.float64)
    return weights / weights.sum()


def ensembled_heatmaps(
    inputs: Iterable[np.ndarray],
    predict: Callable[[list[list[np.ndarray]]], np.ndarray],
    seq_len: int,
    overlap: bool = True,
    batch_size: int = 8,
) -> Iterator[np.ndarray]:
    """One heatmap per input frame, in order, as frames stream in.

    `predict` maps a batch of windows (each seq_len inputs) to heatmaps (batch, seq_len, H, W).
    With overlap a window starts at every frame and a frame's heatmap is the weighted average over
    the windows that saw it (renormalised at the clip's ends); without overlap the windows tile the
    clip. A window running past the end is padded with the last frame.
    """
    weights = ensemble_weights(seq_len) if overlap else np.ones(seq_len)
    stride = 1 if overlap else seq_len
    buffer: dict[int, np.ndarray] = {}  # inputs a queued or future window still needs
    sums: dict[int, np.ndarray] = {}
    totals: dict[int, float] = {}
    pending: list[int] = []  # starts of windows waiting for the network
    next_start = 0
    done = 0
    count = 0

    def run(starts: list[int]) -> None:
        heat = predict([[buffer[min(start + k, count - 1)] for k in range(seq_len)] for start in starts])
        for start, maps in zip(starts, heat):
            for k in range(min(seq_len, count - start)):
                sums[start + k] = sums.get(start + k, 0.0) + weights[k] * maps[k]
                totals[start + k] = totals.get(start + k, 0.0) + weights[k]

    def release(final_before: int) -> Iterator[np.ndarray]:
        nonlocal done
        while done < final_before:
            yield (sums.pop(done) / totals.pop(done)).astype(np.float32)
            done += 1
        for index in [index for index in buffer if index < next_start]:
            del buffer[index]

    for image in inputs:
        buffer[count] = image
        count += 1
        if count - next_start >= seq_len:
            pending.append(next_start)
            next_start += stride
        if len(pending) >= batch_size:
            last = pending[-1]
            run(pending)
            pending.clear()
            yield from release(last + 1 if overlap else last + seq_len)
    if count == 0:
        return
    if next_start == 0 or (not overlap and next_start < count):
        pending.append(next_start)  # a clip shorter than a window, or the last partial tile
        next_start += stride
    for first in range(0, len(pending), batch_size):
        run(pending[first : first + batch_size])
    yield from release(count)


def decode_heatmap(heatmap: np.ndarray, frame_size: tuple[int, int]) -> ShuttleDetection | None:
    """The shuttle where the heatmap's largest blob above 0.5 is (bounding-box centre, as the
    authors decode), scaled to the original frame; None when nothing passes."""
    mask = (heatmap > HEAT_THRESHOLD).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(contour)
    width, height = frame_size
    region = np.zeros_like(mask)
    cv2.drawContours(region, [contour], -1, 1, thickness=-1)
    score = float(heatmap[region.astype(bool)].max())
    return ShuttleDetection((x + w / 2) * width / INPUT_SIZE[0], (y + h / 2) * height / INPUT_SIZE[1], score)


def gap_mask(ys: np.ndarray, visible: np.ndarray, frame_height: int) -> np.ndarray:
    """Frames to fill: short runs without a shuttle between two sightings, unless it was seen so
    near the top of the frame that it may have flown out over it. Nothing is invented before the
    first sighting or after the last (the authors fill the run before the first sighting, which
    suits clips cut at the serve but would invent seconds of flight in a whole video)."""
    threshold = GAP_TOP_FRACTION * frame_height
    mask = np.zeros(len(visible), dtype=int)
    seen = np.flatnonzero(visible)
    for before, after in zip(seen[:-1], seen[1:]):
        if 0 < after - before - 1 <= MAX_GAP_FRAMES and ys[before] > threshold and ys[after] > threshold:
            mask[before + 1 : after] = 1
    return mask


# --- The detector ---


class ShuttleDetector:
    def __init__(self, tracknet: TrackNet, inpaintnet: InpaintNet, seq_len: int, inpaint_len: int, device: str):
        self.tracknet = tracknet.eval()
        self.inpaintnet = inpaintnet.eval()
        self.seq_len = seq_len
        self.inpaint_len = inpaint_len
        self.device = device

    @classmethod
    def load(cls, weights_dir: Path, device: str | None = None) -> "ShuttleDetector":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        files = {name: Path(weights_dir) / name for name in (TRACKNET_FILE, INPAINT_FILE)}
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(f"shuttle detector weights missing: {path}")
        # weights_only: the published checkpoints are plain tensors and settings, never code.
        tracknet_ckpt = torch.load(files[TRACKNET_FILE], map_location=device, weights_only=True)
        inpaint_ckpt = torch.load(files[INPAINT_FILE], map_location=device, weights_only=True)
        seq_len = int(tracknet_ckpt["param_dict"]["seq_len"])
        if tracknet_ckpt["param_dict"]["bg_mode"] != "concat":
            raise ValueError("expected a TrackNetV3 checkpoint trained with the background image concatenated")
        tracknet = TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
        tracknet.load_state_dict(tracknet_ckpt["model"])
        inpaintnet = InpaintNet()
        inpaintnet.load_state_dict(inpaint_ckpt["model"])
        return cls(tracknet.to(device), inpaintnet.to(device), seq_len, int(inpaint_ckpt["param_dict"]["seq_len"]), device)

    def background(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """The clip's empty court: the median of the given frames (pass a spread-out sample, e.g.
        BACKGROUND_SAMPLES of them). Each is shrunk to the network's size as it arrives, so a
        generator over a long 4K video holds only small images."""
        return np.median(np.stack([to_input(frame) for frame in frames]), axis=0).astype(np.uint8)

    def _predict(self, background: np.ndarray) -> Callable[[list[list[np.ndarray]]], np.ndarray]:
        def predict(windows: list[list[np.ndarray]]) -> np.ndarray:
            stacks = np.stack([np.concatenate([background, *window], axis=0) for window in windows]).astype(np.float32) / 255.0
            with torch.no_grad():
                return self.tracknet(torch.from_numpy(stacks).to(self.device)).cpu().numpy()

        return predict

    def detect(
        self,
        frames: Iterable[np.ndarray],
        background: np.ndarray,
        frame_size: tuple[int, int],
        overlap: bool = True,
        fill_gaps: bool = True,
        batch_size: int = 8,
    ) -> list[ShuttleDetection | None]:
        """A detection (or None) for every frame, in order."""
        heatmaps = ensembled_heatmaps((to_input(frame) for frame in frames), self._predict(background), self.seq_len, overlap, batch_size)
        track = [decode_heatmap(heatmap, frame_size) for heatmap in heatmaps]
        return self.fill_gaps(track, frame_size) if fill_gaps else track

    def fill_gaps(self, track: list[ShuttleDetection | None], frame_size: tuple[int, int]) -> list[ShuttleDetection | None]:
        """InpaintNet over sliding windows of normalised positions; only masked frames change."""
        count = len(track)
        if count == 0:
            return track
        width, height = frame_size
        visible = np.array([detection is not None for detection in track])
        coords = np.array([[d.x / width, d.y / height] if d is not None else [0.0, 0.0] for d in track], dtype=np.float32)
        mask = gap_mask(coords[:, 1] * height, visible, height).astype(np.float32)
        if not mask.any():
            return track
        length = self.inpaint_len
        starts = list(range(0, max(1, count - length + 1)))
        index = np.array([[min(start + k, count - 1) for k in range(length)] for start in starts])
        with torch.no_grad():
            out = self.inpaintnet(
                torch.from_numpy(coords[index]).to(self.device), torch.from_numpy(mask[index][..., None]).to(self.device)
            ).cpu().numpy()
        weights = ensemble_weights(length)
        sums = np.zeros((count, 2))
        totals = np.zeros(count)
        for start, filled in zip(starts, out):
            for k in range(min(length, count - start)):
                sums[start + k] += weights[k] * filled[k]
                totals[start + k] += weights[k]
        result = list(track)
        for frame in np.flatnonzero(mask):
            x, y = sums[frame] / totals[frame]
            if x < INVISIBLE_NORMALISED and y < INVISIBLE_NORMALISED:
                continue
            result[frame] = ShuttleDetection(float(x * width), float(y * height), 0.0, inpainted=True)
        return result
