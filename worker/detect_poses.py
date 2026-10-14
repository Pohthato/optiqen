# worker/detect_poses.py
"""Every person in a rally video's frames: their 2D joints and their pose in 3D, from MediaPipe's
pose landmarker (BlazePose GHUM, Apache 2.0).

    python detect_poses.py --video clip.mp4 --track clip.track.json --model pose_landmarker_heavy.task --out clip.poses.json

The landmarker is made for a person who fills much of the picture, and a far player in a
court-wide shot is a hundred pixels tall. So besides the whole frame it looks again in square
crops: around each person on the court in the frame before, and every RESCAN_FRAMES frames (or
when a half of the court has nobody) in crops tiling everywhere a player could stand, each
sized from the camera so that a player there is 1/CROP_PLAYERS of its height. A person found
in more than one look is kept once, from the look that saw them best.

The output is track_players.py's --poses format with each person's pose in 3D added:
    {"frames": [{"timeMs": 0.0, "people": [{"keypoints": [[x, y, score], ... 17 COCO joints],
        "box": [x0, y0, x1, y1], "world": [[x, y, z], ... 13 joints in skeleton.JOINTS order] or null}]}]}
"world" is in metres about the middle of the hips, along the camera's axes (x right, y down,
z away from the camera), as geometry.skeleton.fit_bodies takes it. It is null where the
landmarker's 3D pose does not agree with its own 2D joints.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from find_hits import _cameras
from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, NET_Y_M, SINGLES_WIDTH_M
from geometry.skeleton import JOINTS
from players import CHASE_BACK_M, CHASE_SIDE_M

# BlazePose's 33 landmarks for the 17 COCO joints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
BLAZEPOSE_FOR_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
PLAYER_HEIGHT_M = 1.8
CROP_PLAYERS = 2.5  # a crop is this many players tall
GRID_M = 1.0  # where a player could stand, sampled this finely over each half
BAND = 1.5  # players whose sizes differ by less than this share a tiling
BIG = 0.6  # a person this much of the frame's height is seen well enough in the whole frame
RESCAN_FRAMES = 10
SAME_PERSON_IOU = 0.4
AGREE_SCORE = 0.5  # the joints used to check the 3D pose against the 2D one
AGREE_RMS = 0.15  # of the person's height: how far the 3D pose, seen flat, may miss the 2D joints
FULL_FRAME_POSES = 4
CROP_POSES = 2

# One look: an RGB image in, per person found its 33 landmarks (x, y normalised to the image,
# visibility) and its 33 world landmarks (metres), or None for the world ones.
Look = Callable[[np.ndarray, int], list[tuple[np.ndarray, np.ndarray | None]]]


@dataclass(frozen=True)
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Pose:
    keypoints: np.ndarray  # (17, 3) frame pixels and score, COCO order
    box: np.ndarray  # x0, y0, x1, y1
    world: np.ndarray | None  # (13, 3) metres about the hips' middle, camera axes, skeleton.JOINTS order

    @property
    def score(self) -> float:
        return float(np.mean(self.keypoints[:, 2]))


def _square(centre: np.ndarray, side: float, size: tuple[int, int]) -> Crop:
    """A square crop about this centre, moved inside the frame."""
    side = int(round(side))
    x0 = int(np.clip(round(centre[0] - side / 2), 0, max(size[0] - side, 0)))
    y0 = int(np.clip(round(centre[1] - side / 2), 0, max(size[1] - side, 0)))
    return Crop(x0, y0, min(x0 + side, size[0]), min(y0 + side, size[1]))


def court_crops(camera: Camera, size: tuple[int, int]) -> list[Crop]:
    """Square crops in which every place a player could stand is seen whole, the player about
    1/CROP_PLAYERS of the crop's height (no crop taller than the frame): players of about one
    size (within BAND) share a tiling. A player most of the frame's height tall is left to the
    whole frame."""
    width, height = size
    xs = np.arange(-DOUBLES_MARGIN_M - CHASE_SIDE_M, SINGLES_WIDTH_M + DOUBLES_MARGIN_M + CHASE_SIDE_M + 1e-9, GRID_M)
    ys = np.arange(-CHASE_BACK_M, COURT_LENGTH_M + CHASE_BACK_M + 1e-9, GRID_M)
    spots = []
    for y in ys:
        for x in xs:
            feet, head = camera.project(np.array([[x, y, 0.0], [x, y, PLAYER_HEIGHT_M]]))
            if not (np.isfinite(feet).all() and np.isfinite(head).all()):
                continue
            tall = float(np.hypot(*(feet - head)))
            centre = (feet + head) / 2
            if 8 <= tall and -tall / 2 < centre[0] < width + tall / 2 and -tall / 2 < centre[1] < height + tall / 2:
                spots.append((tall, centre))
    spots.sort(key=lambda spot: spot[0])
    crops: list[Crop] = []
    while spots:
        band = [spot for spot in spots if spot[0] <= BAND * spots[0][0]]
        spots = spots[len(band):]
        tall = max(spot[0] for spot in band)
        if tall > BIG * height:
            break
        side = min(CROP_PLAYERS * tall, height)
        centres = np.array([spot[1] for spot in band])
        low, high = centres.min(axis=0), centres.max(axis=0)
        stride = side - tall  # a player box (tall across) wholly inside some crop
        counts = np.maximum(np.ceil((high - low) / stride).astype(int) + 1, 1)
        for i in range(counts[0]):
            for j in range(counts[1]):
                centre = low + (high - low) * np.array([i / max(counts[0] - 1, 1), j / max(counts[1] - 1, 1)])
                if counts[0] == 1:
                    centre[0] = (low[0] + high[0]) / 2
                if counts[1] == 1:
                    centre[1] = (low[1] + high[1]) / 2
                if np.min(np.hypot(*(centres - centre).T)) <= side / 2:
                    crops.append(_square(centre, side, size))
    return crops


def person_crops(poses: Sequence[Pose], camera: Camera, size: tuple[int, int]) -> list[Crop]:
    """A crop about each person found standing on the court, sized to them."""
    from players import Person, floor_position, half_of

    crops = []
    for pose in poses:
        floor = floor_position(Person(pose.keypoints[:, :2], pose.keypoints[:, 2], pose.box), camera)
        if floor is None or half_of(floor) is None:
            continue
        tall = max(pose.box[3] - pose.box[1], pose.box[2] - pose.box[0], 16.0)
        if tall <= BIG * size[1]:
            centre = np.array([(pose.box[0] + pose.box[2]) / 2, (pose.box[1] + pose.box[3]) / 2])
            crops.append(_square(centre, min(CROP_PLAYERS * tall, size[1]), size))
    return crops


def on_court_halves(poses: Sequence[Pose], camera: Camera) -> set[str]:
    from players import Person, floor_position, half_of

    halves = set()
    for pose in poses:
        floor = floor_position(Person(pose.keypoints[:, :2], pose.keypoints[:, 2], pose.box), camera)
        if floor is not None and half_of(floor) is not None:
            halves.add(half_of(floor))
    return halves


def _pose(landmarks: np.ndarray, world: np.ndarray | None, crop: Crop) -> Pose:
    """A person from one look's landmarks, put back into the frame's pixels."""
    span = np.array([crop.x1 - crop.x0, crop.y1 - crop.y0], dtype=float)
    pixels = np.array([crop.x0, crop.y0]) + landmarks[:, :2] * span
    keypoints = np.column_stack([pixels[list(BLAZEPOSE_FOR_COCO)], np.clip(landmarks[list(BLAZEPOSE_FOR_COCO), 2], 0.0, 1.0)])
    box = np.array([*pixels.min(axis=0), *pixels.max(axis=0)])
    lifted = None
    if world is not None:
        coco = np.asarray(world, dtype=float)[list(BLAZEPOSE_FOR_COCO)]
        lifted = coco[list(JOINTS)] - (coco[11] + coco[12]) / 2
        if not agrees(keypoints, lifted):
            lifted = None
    return Pose(keypoints, box, lifted)


def agrees(keypoints: np.ndarray, world: np.ndarray) -> bool:
    """Whether a 3D pose, seen flat along the camera's axes (a scale and a shift, no turn: they
    share the camera's axes), lands on the 2D joints."""
    seen = keypoints[list(JOINTS), 2] >= AGREE_SCORE
    if seen.sum() < 6:
        return False
    flat, image = world[seen, :2], keypoints[list(JOINTS)][seen, :2]
    flat_c, image_c = flat - flat.mean(axis=0), image - image.mean(axis=0)
    scale = float(np.sum(flat_c * image_c) / max(np.sum(flat_c**2), 1e-12))
    if scale <= 0:
        return False
    rms = float(np.sqrt(np.mean(np.sum((scale * flat_c - image_c) ** 2, axis=1))))
    tall = float(np.ptp(image[:, 1])) or 1.0
    return rms <= AGREE_RMS * tall


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge(poses: Sequence[Pose]) -> list[Pose]:
    """Each person once: of the poses that overlap, the one seen best."""
    kept: list[Pose] = []
    for pose in sorted(poses, key=lambda p: -p.score):
        if all(_iou(pose.box, other.box) < SAME_PERSON_IOU for other in kept):
            kept.append(pose)
    return kept


def detect_people(look: Look, frame_bgr: np.ndarray, crops: Sequence[Crop]) -> list[Pose]:
    """Everyone in one frame: the whole frame, then every crop."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    poses = [_pose(lm, world, Crop(0, 0, width, height)) for lm, world in look(rgb, FULL_FRAME_POSES)]
    for crop in crops:
        part = np.ascontiguousarray(rgb[crop.y0 : crop.y1, crop.x0 : crop.x1])
        poses += [_pose(lm, world, crop) for lm, world in look(part, CROP_POSES)]
    return merge(poses)


def mediapipe_look(model: Path) -> Look:
    """A look through MediaPipe's pose landmarker, one landmarker per number of people sought."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    landmarkers: dict[int, object] = {}

    def look(rgb: np.ndarray, people: int):
        if people not in landmarkers:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.IMAGE, num_poses=people,
                min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
            )
            landmarkers[people] = vision.PoseLandmarker.create_from_options(options)
        result = landmarkers[people].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        found = []
        for k, landmarks in enumerate(result.pose_landmarks):
            flat = np.array([[p.x, p.y, p.visibility if p.visibility is not None else 1.0] for p in landmarks], dtype=float)
            world = None
            if k < len(result.pose_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[k]], dtype=float)
            found.append((flat, world))
        return found

    return look


def _person(pose: Pose) -> dict:
    return {
        "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(s), 3)] for x, y, s in pose.keypoints],
        "box": [round(float(v), 1) for v in pose.box],
        "world": None if pose.world is None else [[round(float(v), 4) for v in joint] for joint in pose.world],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find every person's 2D joints and 3D pose in every frame of a rally video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True, help="track_camera.py output for the same video (where the court halves are)")
    parser.add_argument("--model", type=Path, required=True, help="MediaPipe pose landmarker model (.task)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from video import VideoFile

    if not args.model.is_file():
        print(f"no pose model at {args.model}", file=sys.stderr)
        return 1
    video = VideoFile(args.video)
    if not video.opened:
        print(f"cannot open {args.video}", file=sys.stderr)
        return 1
    track_doc = json.loads(args.track.read_text(encoding="utf-8"))
    cameras = _cameras(track_doc)
    look = mediapipe_look(args.model)
    frames, people, lifted = [], 0, 0
    poses: list[Pose] = []
    tiles: dict[int, list[Crop]] = {}
    for index, (time_ms, frame) in enumerate(video.frames()):
        camera = cameras[index] if index < len(cameras) else None
        crops: list[Crop] = []
        if camera is not None:
            crops = person_crops(poses, camera, video.size)
            if index % RESCAN_FRAMES == 0 or len(on_court_halves(poses, camera)) < 2:
                key = index // RESCAN_FRAMES  # the camera moves little within a rescan's frames
                crops += tiles.setdefault(key, court_crops(camera, video.size))
        poses = detect_people(look, frame, crops)
        people += len(poses)
        lifted += sum(p.world is not None for p in poses)
        frames.append({"timeMs": round(float(time_ms), 3), "people": [_person(p) for p in poses]})
    result = {
        "summary": {"frames": len(frames), "people": people, "with3d": lifted, "model": args.model.name},
        "conventions": {
            "keypoints": "pixels and score per COCO joint",
            "world": "metres about the hips' middle along the camera's axes (x right, y down, z away), skeleton.JOINTS order",
        },
        "frames": frames,
    }
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
