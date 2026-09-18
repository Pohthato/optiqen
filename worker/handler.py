"""Netoval RunPod worker.

Pose and shuttle outputs are confidence-gated. Shot labels require observed
contact plus a post-contact shuttle track. Warmup jobs load models without
running a video so a customer calibration can hide GPU boot time.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import runpod
from ultralytics import YOLO

from badminton import (
    COURT_LENGTH_M,
    build_event_layer,
    capture_warnings,
    court_width_m,
)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/netoval/models"))
POSE_MODEL_PATH = MODEL_DIR / os.environ.get("POSE_MODEL_FILE", "badminton_pose.pt")
SHUTTLE_MODEL_PATH = MODEL_DIR / os.environ.get("SHUTTLE_MODEL_FILE", "shuttlecock_yolov8n.pt")
RACKET_MODEL_PATH = MODEL_DIR / os.environ.get("RACKET_MODEL_FILE", "badminton_racket.pt")
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "120"))
POSE_SAMPLE_FPS = float(os.environ.get("POSE_SAMPLE_FPS", "12"))
SHUTTLE_SAMPLE_FPS = float(os.environ.get("SHUTTLE_SAMPLE_FPS", "30"))
MIN_CONFIDENCE = float(os.environ.get("MIN_DETECTION_CONFIDENCE", "0.55"))
PROCESSING_VERSION = os.environ.get("PROCESSING_VERSION", "netoval-badminton-worker-2.0.0")
# RunPod workers handle one job per process by default. Setting concurrency lets
# ONE warm GPU accept several small video analyses in parallel — the three YOLO
# nano/medium-class models share a 12-24 GB card comfortably at 2-3 processes.
HANDLER_CONCURRENCY = int(os.environ.get("HANDLER_CONCURRENCY", "2"))


@dataclass
class Models:
    pose: YOLO | None = None
    shuttle: YOLO | None = None
    racket: YOLO | None = None
    warnings: list[str] = field(default_factory=list)
    ready: bool = False


def load_model(path: Path, name: str) -> YOLO | None:
    if not path.is_file():
        MODELS.warnings.append(f"{name} model is not installed at {path.name}; this layer was not inferred.")
        return None
    return YOLO(str(path))


MODELS = Models()


def initialise_models() -> None:
    """Load once at container boot; never trigger a model download on a job."""
    if MODELS.ready:
        return
    MODELS.pose = load_model(POSE_MODEL_PATH, "Badminton pose")
    MODELS.shuttle = load_model(SHUTTLE_MODEL_PATH, "Shuttlecock")
    MODELS.racket = load_model(RACKET_MODEL_PATH, "Racket")
    MODELS.ready = True


def elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def error(message: str) -> dict[str, dict[str, str]]:
    return {"error": message}


def validate_job(data: dict[str, Any]) -> None:
    if not isinstance(data.get("analysisId"), str) or not data["analysisId"]:
        raise ValueError("analysisId is required")
    source_url = data.get("sourceVideoUrl")
    if not isinstance(source_url, str) or urlparse(source_url).scheme != "https":
        raise ValueError("sourceVideoUrl must be a short-lived HTTPS download URL")
    if data.get("selectedPlayer") not in ("near", "far"):
        raise ValueError("selectedPlayer must be near or far")
    if data.get("courtType", "singles") not in ("singles", "doubles"):
        raise ValueError("courtType must be singles or doubles")
    layers = data.get("requestedLayers")
    if not isinstance(layers, list) or not layers or any(layer not in {"skeleton", "shuttle", "racket", "courtMap"} for layer in layers):
        raise ValueError("requestedLayers must contain supported analysis layers")
    calibration = data.get("calibration")
    corners = calibration.get("corners") if isinstance(calibration, dict) else None
    if not isinstance(corners, list) or len(corners) not in (3, 4):
        raise ValueError("calibration must contain three or four labelled corners")


def download_source(url: str, destination: Path) -> None:
    response = requests.get(url, stream=True, timeout=(10, 180))
    response.raise_for_status()
    size = 0
    with destination.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > 2_000_000_000:
                raise ValueError("source video exceeds the 2 GB worker safety limit")
            handle.write(chunk)


def normalise_point(point: np.ndarray | list[float], width: int, height: int, confidence: float) -> dict[str, float]:
    return {
        "x": round(float(np.clip(point[0] / max(width, 1) * 100, 0, 100)), 3),
        "y": round(float(np.clip(point[1] / max(height, 1) * 100, 0, 100)), 3),
        "confidence": round(float(np.clip(confidence, 0, 1)), 3),
    }


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    ab, cb = a - b, c - b
    denominator = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denominator < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(ab, cb) / denominator, -1, 1))))


def confidence_mean(values: np.ndarray) -> float:
    return float(np.clip(np.mean(values), 0, 1)) if values.size else 0.0


class PlayerSelector:
    """Identity from court depth, then motion and stature continuity."""

    def __init__(self, side: str, frame_diagonal: float):
        self.side = side
        self.previous: np.ndarray | None = None
        self.height = 0.0
        self.frame_diagonal = frame_diagonal

    @staticmethod
    def anchor(keypoints: np.ndarray, scores: np.ndarray, box: np.ndarray) -> np.ndarray:
        hips = keypoints[11:13]
        hip_scores = scores[11:13]
        if np.count_nonzero(hip_scores >= MIN_CONFIDENCE):
            return np.mean(hips[hip_scores >= MIN_CONFIDENCE], axis=0)
        return np.array([(box[0] + box[2]) / 2, box[3]], dtype=np.float32)

    def select(self, candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        if not candidates:
            return None
        anchors = [self.anchor(keypoints, scores, box) for keypoints, scores, box, _ in candidates]
        heights = [float(box[3] - box[1]) for _, _, box, _ in candidates]
        if self.previous is None:
            index = int(np.argmax([anchor[1] for anchor in anchors]) if self.side == "near" else np.argmin([anchor[1] for anchor in anchors]))
        else:
            costs = []
            for anchor, height in zip(anchors, heights):
                motion = float(np.linalg.norm(anchor - self.previous)) / self.frame_diagonal
                stature = abs(height - self.height) / max(self.height, 1.0) if self.height else 0.0
                costs.append(motion + 0.35 * stature)
            index = int(np.argmin(costs))
            if costs[index] > 0.42:
                return None
        self.previous = anchors[index]
        self.height = heights[index]
        return candidates[index]


class ShuttleTracker:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.misses = 0

    def pick(self, candidates: list[tuple[np.ndarray, float]], diagonal: float) -> tuple[np.ndarray, float] | None:
        if not candidates:
            self.misses += 1
            if self.misses >= 8:
                self.previous = None
                self.velocity = np.zeros(2, dtype=np.float32)
            return None
        if self.previous is None:
            point, confidence = max(candidates, key=lambda item: item[1])
        else:
            expected = self.previous + self.velocity
            point, confidence = min(candidates, key=lambda item: np.linalg.norm(item[0] - expected) - item[1] * diagonal * 0.15)
            if np.linalg.norm(point - expected) > diagonal * 0.22:
                self.misses += 1
                return None
        self.misses = 0
        if self.previous is not None:
            self.velocity = 0.65 * self.velocity + 0.35 * (point - self.previous)
        self.previous = point
        return point, confidence


def pose_candidates(frame: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if MODELS.pose is None:
        return []
    result = MODELS.pose(frame, conf=MIN_CONFIDENCE, verbose=False, device=0)[0]
    if result.boxes is None or result.keypoints is None:
        return []
    keypoints = result.keypoints.xy.cpu().numpy()
    scores = result.keypoints.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    box_scores = result.boxes.conf.cpu().numpy()
    return [(keypoints[index], scores[index], boxes[index], float(box_scores[index])) for index in range(len(boxes)) if len(keypoints[index]) >= 17]


def object_candidates(model: YOLO | None, frame: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
    if model is None:
        return []
    result = model(frame, conf=MIN_CONFIDENCE, verbose=False, device=0)[0]
    if result.boxes is None:
        return []
    output: list[tuple[np.ndarray, float, np.ndarray]] = []
    for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        output.append((np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), float(confidence), box))
    return output


def calibration_homography(calibration: dict[str, Any], frame: np.ndarray, court_type: str) -> tuple[np.ndarray | None, dict[str, Any], float | None]:
    corners = calibration["corners"]
    by_label = {corner.get("label"): corner for corner in corners}
    required = ("nearLeft", "nearRight", "farRight", "farLeft")
    width_m = court_width_m(court_type)
    if not all(label in by_label for label in required):
        return None, {**calibration, "confidence": "image_space_only", "supportsCourtMapping": False, "guidance": "Fewer than four named court intersections were supplied; movement remains image-space only."}, None
    height, width = frame.shape[:2]
    points = np.float32([[by_label[label]["x"] * width / 100, by_label[label]["y"] * height / 100] for label in required])
    court = np.float32([[0, 0], [width_m, 0], [width_m, COURT_LENGTH_M], [0, COURT_LENGTH_M]])
    homography = cv2.getPerspectiveTransform(points, court)
    edge = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 160)
    support = []
    for index in range(4):
        start, end = points[index], points[(index + 1) % 4]
        samples = np.linspace(start, end, 100).astype(int)
        values = []
        for x, y in samples:
            y0, y1 = max(0, y - 2), min(height, y + 3)
            x0, x1 = max(0, x - 2), min(width, x + 3)
            values.append(bool(np.any(edge[y0:y1, x0:x1])))
        support.append(float(np.mean(values)))
    line_support = float(np.mean(support))
    if line_support < 0.42:
        return None, {**calibration, "confidence": "image_space_only", "supportsCourtMapping": False, "guidance": "Manual corners did not have enough court-line evidence in the source frame. Court movement is withheld; image-space pose remains available."}, line_support
    return homography, {**calibration, "confidence": "validated", "supportsCourtMapping": True, "guidance": f"Four labelled intersections passed court-line support on a BWF {court_type} court ({width_m:.2f} x {COURT_LENGTH_M:.2f} m). Court-plane movement is available; this does not establish shuttle height or 3D body depth."}, line_support


def project_to_court(point: np.ndarray, homography: np.ndarray | None, width_m: float) -> np.ndarray | None:
    if homography is None:
        return None
    projected = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), homography)[0][0]
    if not np.isfinite(projected).all() or not (-1 <= projected[0] <= width_m + 1 and -1 <= projected[1] <= COURT_LENGTH_M + 1):
        return None
    return projected


def metric(metric: str, values: list[float], unit: str, confidence: float, evidence: list[dict[str, Any]], direction: str = "contextual") -> dict[str, Any] | None:
    if not values or not evidence:
        return None
    return {"metric": metric, "value": round(float(np.mean(values)), 3), "unit": unit, "direction": direction, "confidence": round(confidence, 3), "evidenceFrames": evidence[:12]}


def calculate_metrics(samples: list[dict[str, Any]], has_court: bool, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stance, knees, coverage_steps = [], [], []
    evidence: list[dict[str, Any]] = []
    last_court: np.ndarray | None = None
    for sample in samples:
        keypoints, scores = sample["keypoints"], sample["scores"]
        confidence = sample["confidence"]
        if confidence < MIN_CONFIDENCE:
            continue
        evidence.append({"frame": sample["frame"], "timeMs": sample["timeMs"], "confidence": round(confidence, 3), "source": "pose"})
        shoulder = np.linalg.norm(keypoints[5] - keypoints[6])
        ankle = np.linalg.norm(keypoints[15] - keypoints[16])
        if shoulder > 1 and min(scores[5], scores[6], scores[15], scores[16]) >= MIN_CONFIDENCE:
            stance.append(float(ankle / shoulder))
        for a, b, c, triplet_scores in ((keypoints[11], keypoints[13], keypoints[15], scores[[11, 13, 15]]), (keypoints[12], keypoints[14], keypoints[16], scores[[12, 14, 16]])):
            if float(np.min(triplet_scores)) >= MIN_CONFIDENCE:
                value = angle_degrees(a, b, c)
                if value is not None:
                    knees.append(value)
        current = sample.get("court")
        if has_court and current is not None:
            if last_court is not None:
                step = float(np.linalg.norm(current - last_court))
                if 0.025 <= step <= 0.65:
                    coverage_steps.append(step)
            last_court = current
    confidence = float(np.mean([item["confidence"] for item in evidence])) if evidence else 0.0
    result = [
        metric("base width", stance, "normalized shoulder-width", confidence, evidence),
        metric("knee flexion angle (2D)", knees, "degrees", confidence, evidence),
    ]
    if has_court:
        result.append(metric("court coverage", [sum(coverage_steps)], "metres travelled", confidence, evidence))
    contacts = [item for item in events if item["type"] == "contact" and item["confidence"] >= 0.65]
    split_steps = [item for item in events if item["type"] == "split_step" and item["confidence"] >= 0.65]
    recoveries = [item for item in events if item["type"] == "recovery_complete" and item["confidence"] >= 0.65]
    if split_steps and contacts:
        offsets = []
        for step in split_steps:
            following = next((contact for contact in contacts if contact["timeMs"] > step["timeMs"]), None)
            if following:
                offsets.append(following["timeMs"] - step["timeMs"])
        if offsets:
            result.append(metric("split-step to contact", offsets, "ms", float(np.mean([item["confidence"] for item in split_steps])), split_steps, "contextual"))
    if contacts and recoveries:
        durations = []
        evidence_frames = []
        for contact in contacts:
            recovery = next((item for item in recoveries if item["timeMs"] > contact["timeMs"]), None)
            if recovery:
                durations.append(recovery["timeMs"] - contact["timeMs"])
                evidence_frames.extend([contact, recovery])
        if durations:
            result.append(metric("recovery time", durations, "ms", min(len(durations) / max(len(contacts), 1), 1.0), evidence_frames, "lower_is_better"))
    if contacts:
        result.append(metric("verified contacts", [float(len(contacts))], "count", float(np.mean([item["confidence"] for item in contacts])), contacts, "contextual"))
    return [item for item in result if item is not None]


def warmup_result() -> dict[str, Any]:
    initialise_models()
    return {
        "warmup": True,
        "processingVersion": PROCESSING_VERSION,
        "modelVersions": {
            "pose": POSE_MODEL_PATH.name if MODELS.pose else "unavailable",
            "shuttle": SHUTTLE_MODEL_PATH.name if MODELS.shuttle else "unavailable",
            "racket": RACKET_MODEL_PATH.name if MODELS.racket else "unavailable",
        },
        "warnings": MODELS.warnings[:8],
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    data = job.get("input", {})
    try:
        if data.get("warmup"):
            return warmup_result()
        validate_job(data)
        initialise_models()
        court_type = data.get("courtType", "singles")
        width_m = court_width_m(court_type)
        timings = {"download": 0.0, "decode": 0.0, "inference": 0.0, "render": 0.0, "total": 0.0}
        warnings = list(MODELS.warnings)
        requested = set(data["requestedLayers"])
        with tempfile.TemporaryDirectory(prefix="netoval-") as temp:
            video_path = Path(temp) / "source.mp4"
            phase_start = time.perf_counter()
            download_source(data["sourceVideoUrl"], video_path)
            timings["download"] = elapsed(phase_start)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError("OpenCV could not decode the submitted video")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if source_fps <= 0 or width <= 0 or height <= 0:
                raise ValueError("source video is missing readable frame metadata")
            duration = frame_count / source_fps
            if duration > MAX_VIDEO_SECONDS:
                raise ValueError(f"source video is {duration:.1f}s; upload a rally or drill up to {MAX_VIDEO_SECONDS:.0f}s")
            pose_fps = min(POSE_SAMPLE_FPS, source_fps)
            shuttle_fps = min(SHUTTLE_SAMPLE_FPS, source_fps)
            pose_stride = max(1, round(source_fps / pose_fps))
            shuttle_stride = max(1, round(source_fps / shuttle_fps))
            selector = PlayerSelector(data["selectedPlayer"], math.hypot(width, height))
            shuttle_tracker = ShuttleTracker()
            overlays, samples = [], []
            shuttle_obs, racket_obs = [], []
            pose_detected = shuttle_detected = 0
            usable = sampled = 0
            homography: np.ndarray | None = None
            accepted_calibration = data["calibration"]
            line_support: float | None = None
            frame_index = 0
            first_frame: np.ndarray | None = None
            phase_start = time.perf_counter()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if first_frame is None:
                    first_frame = frame.copy()
                    homography, accepted_calibration, line_support = calibration_homography(data["calibration"], first_frame, court_type)
                    if homography is None:
                        warnings.append(accepted_calibration["guidance"])
                run_pose = frame_index % pose_stride == 0
                run_shuttle = frame_index % shuttle_stride == 0
                if not run_pose and not run_shuttle:
                    frame_index += 1
                    continue
                sampled += 1
                time_ms = round(frame_index / source_fps * 1000)
                overlay: dict[str, Any] = {"timeMs": time_ms}
                errors: dict[str, str] = {}
                infer_start = time.perf_counter()
                selected = selector.select(pose_candidates(frame)) if run_pose and "skeleton" in requested else None
                shuttle_candidates = object_candidates(MODELS.shuttle, frame) if run_shuttle and "shuttle" in requested else []
                racket_candidates = object_candidates(MODELS.racket, frame) if run_shuttle and "racket" in requested else []
                timings["inference"] += elapsed(infer_start)
                foot: np.ndarray | None = None
                if selected is not None:
                    keypoints, scores, box, box_confidence = selected
                    pose_confidence = min(box_confidence, confidence_mean(scores))
                    if pose_confidence >= MIN_CONFIDENCE:
                        overlay["skeleton"] = [normalise_point(point, width, height, float(score)) for point, score in zip(keypoints[:17], scores[:17])]
                        foot = keypoints[15] if scores[15] >= scores[16] else keypoints[16]
                        court = project_to_court(foot, homography, width_m)
                        if court is not None:
                            overlay["courtPosition"] = {"x": round(float(court[0] / width_m * 100), 3), "y": round(float(court[1] / COURT_LENGTH_M * 100), 3), "confidence": round(pose_confidence, 3)}
                        samples.append({"frame": frame_index, "timeMs": time_ms, "keypoints": keypoints, "scores": scores, "confidence": pose_confidence, "court": court, "xy": foot})
                        pose_detected += 1
                        usable += 1
                    else:
                        errors["Skeleton"] = "Pose confidence was below the analysis threshold."
                elif run_pose and "skeleton" in requested:
                    errors["Skeleton"] = "Selected player was not confidently tracked in this frame."
                if run_shuttle and "shuttle" in requested:
                    shuttle = shuttle_tracker.pick([(point, confidence) for point, confidence, _ in shuttle_candidates], math.hypot(width, height))
                    if shuttle is not None:
                        point, confidence = shuttle
                        overlay["shuttle"] = normalise_point(point, width, height, confidence)
                        shuttle_court = project_to_court(point, homography, width_m)
                        shuttle_obs.append({"frame": frame_index, "timeMs": time_ms, "xy": point, "court": shuttle_court, "confidence": confidence})
                        shuttle_detected += 1
                    else:
                        errors["Shuttle"] = "Shuttlecock was not confidently detected or temporally consistent."
                if run_shuttle and "racket" in requested:
                    if racket_candidates:
                        point, confidence, box = max(racket_candidates, key=lambda item: item[1])
                        overlay["racket"] = {"grip": normalise_point(np.array([box[0], box[3]]), width, height, confidence), "head": normalise_point(point, width, height, confidence)}
                        racket_obs.append({"frame": frame_index, "timeMs": time_ms, "xy": point, "confidence": confidence})
                    else:
                        errors["Racket"] = "Racket model is unavailable or the racket is occluded."
                if "courtMap" in requested and homography is None:
                    errors["Court map"] = accepted_calibration["guidance"]
                if errors:
                    overlay["errors"] = errors
                overlays.append(overlay)
                if sampled % 20 == 0:
                    runpod.serverless.progress_update(job, {"stage": "tracking", "sampledFrames": sampled, "estimatedProgress": round(min(0.96, frame_index / max(frame_count, 1)), 2)})
                frame_index += 1
            capture.release()
            timings["decode"] = elapsed(phase_start) - timings["inference"]
        if MODELS.pose is None and "skeleton" in requested:
            warnings.append("No pose model was installed; no technique or movement metrics were created.")
        if MODELS.shuttle is None and "shuttle" in requested:
            warnings.append("No shuttlecock model was installed; contact, trajectory, and shot labels were withheld.")
        if MODELS.racket is None and "racket" in requested:
            warnings.append("No racket model was installed; racket-path evidence was withheld.")
        if line_support is not None:
            warnings.append(f"Court-line support score: {line_support:.2f} (threshold 0.42) on a {court_type} court.")
        events = build_event_layer(shuttle_obs, samples, racket_obs, math.hypot(width, height), court_type, data["selectedPlayer"], homography is not None)
        warnings.extend(capture_warnings(source_fps, duration, shuttle_detected / max(sampled, 1), usable / max(sampled, 1)))
        if not events["shotDistribution"]:
            warnings.append("No verified stroke labels were produced. Contacts without a confident post-impact shuttle track stay unverified.")
        if "racket" in requested and not [item for item in events["events"] if item["type"] == "racket_swing"]:
            warnings.append("No racket swing was detected near any contact. The racket was occluded, slow, or the model layer is not reliable on this view.")
        timings["total"] = elapsed(started)
        return {
            "processingVersion": PROCESSING_VERSION,
            "calibration": accepted_calibration,
            "courtType": court_type,
            "quality": {"usableFrameRatio": round(usable / max(sampled, 1), 3), "poseTrackConfidence": round(float(np.mean([sample["confidence"] for sample in samples])) if samples else 0, 3), "shuttleTrackConfidence": round(shuttle_detected / max(sampled, 1), 3)},
            "metrics": [*calculate_metrics(samples, homography is not None, events["events"]), *events["metrics"]],
            "shotDistribution": events["shotDistribution"],
            "events": events["events"][:240],
            "rallies": events["rallies"][:48],
            "shots": events["shots"][:240],
            "overlays": overlays,
            "diagnostics": {
                "sampledFrames": sampled,
                "sourceFps": round(source_fps, 3),
                "sampleFps": round(shuttle_fps, 3),
                "poseSampleFps": round(pose_fps, 3),
                "shuttleSampleFps": round(shuttle_fps, 3),
                "racketSampleFps": round(shuttle_fps, 3),
                "timingsMs": timings,
                "modelVersions": {"pose": POSE_MODEL_PATH.name if MODELS.pose else "unavailable", "shuttle": SHUTTLE_MODEL_PATH.name if MODELS.shuttle else "unavailable", "racket": RACKET_MODEL_PATH.name if MODELS.racket else "unavailable"},
                "warnings": warnings[:32],
            },
        }
    except Exception as exc:
        return error(str(exc)[:500])


if __name__ == "__main__":
    initialise_models()
    runpod.serverless.start({"handler": handler, "concurrency": HANDLER_CONCURRENCY})
