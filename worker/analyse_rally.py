# worker/analyse_rally.py
"""A rally video understood in one command: the camera of every frame, the shuttle in every frame,
the hits, each flight in 3D, the two players' bodies in 3D, and every shot's facts with the
hitter's body at the hit.

    python analyse_rally.py clip.mp4 --golden clip.json --out clip.analysis.json
        [--court singles|doubles] [--overlay clip.analysis.mp4] [--diagram clip.court.png]
        [--track clip.track.json] [--shuttle clip.shuttle.json] [--poses clip.poses.json]
        [--pose-model weights/pose_landmarker_heavy.task] [--terminal-velocity 6.9 --terminal-sigma 0.05]

--golden is the labeller export with court points placed on one frame (see track_camera.py). An
existing --track, --shuttle or --poses file is used as it is; otherwise the camera track, the
shuttle detections and everyone's poses are computed and written next to --out. Without the pose
model the rally is analysed without the players, and the summary says why. The overlay draws
each fitted flight into the video with its hit, the players' fitted bodies and, for the rally's
last shot, where it landed and the call; the diagram is the court from above with every shot's
path, where both players stood at each hit, and the landing's uncertainty. The shuttle's drag
(terminal velocity) is measured from the rally's own long flights unless given: measured once
over a whole session's rallies it is surer, and close line calls sharpen with it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import detect_poses
import detect_shuttle
import track_camera
from body import BodyFacts, rally_body_facts, standing_at, standing_reach
from find_hits import hear, hits_document, load_detections, rounded
from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, SINGLES_WIDTH_M, court_lines, net_tape_lines
from geometry.skeleton import BONES, Body
from hits import find_hits
from lift_players import fit_players
from players import HALVES
from shots import ShotFacts, rally_shots
from track_players import players_document
from video import VideoFile

HITTER_COLOURS = {"near": (255, 160, 40), "far": (40, 200, 255)}  # BGR
CALL_COLOURS = {"in": (60, 200, 60), "out": (60, 60, 230), "too close to call": (0, 200, 230)}
TRAIL_MS = 600.0  # how much of a flight the overlay shows behind the shuttle
DIAGRAM_PX_PER_M = 60
DIAGRAM_MARGIN_M = 1.5
ELLIPSE_SCALE = float(np.sqrt(5.991))  # 95 % of a 2D Gaussian
POSE_MODEL = Path(__file__).resolve().parent / "weights" / "pose_landmarker_heavy.task"
DRAWN_BONES = [pair for name, pairs in BONES.items() if name != "diagonal" for pair in pairs]

Bodies = dict[str, list[Body | None]]
Stood = dict[str, tuple[np.ndarray, np.ndarray] | None]  # each half's player at a hit: where, and its covariance


def _estimate(estimate) -> dict:
    return {"value": round(estimate.value, 3), "sigma": round(estimate.sigma, 3)}


def _clean(value):
    """The result with any number that is not finite as null, so it always writes as JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _pixels(camera, points) -> np.ndarray:
    """Points drawn through the camera: those in front of it, as int32 pixels kept near the image."""
    pixels = camera.project(np.asarray(points, dtype=float).reshape(-1, 3))
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    return np.clip(pixels, -10_000, 10_000).astype(np.int32)


def shot_entry(number: int, shot: ShotFacts) -> dict:
    """One shot's facts as the result reports them (metres, m/s, degrees, ms)."""
    return {
        "shot": number,
        "hitMs": round(shot.hit_ms, 1),
        "hitter": shot.hitter,
        "serve": shot.serve,
        "contact": rounded(shot.contact, 3),
        "contactSigma": rounded(shot.contact_sigma, 3),
        "speedMS": _estimate(shot.speed_m_s),
        "elevationDeg": _estimate(shot.elevation_deg),
        "apexM": _estimate(shot.apex_m),
        "net": None if shot.net is None else {"xM": round(shot.net.x_m, 3), "heightM": _estimate(shot.net.height_m), "clearanceM": _estimate(shot.net.clearance_m)},
        "flightMs": round(shot.flight_ms, 1),
        "landing": None if shot.landing is None else {
            "point": rounded(shot.landing.point, 3),
            "covariance": [rounded(row, 6) for row in shot.landing.covariance],
            "timeMs": round(shot.landing.time_ms, 1),
            "call": shot.landing.call,
            "pIn": round(shot.landing.p_in, 3),
        },
        "path": [rounded(point, 3) for point in shot.path],
    }


def body_entry(facts: BodyFacts | None) -> dict | None:
    """The hitter's body at one shot as the result reports it (metres, degrees, ms), or None."""
    if facts is None:
        return None
    ready = facts.ready
    return {
        "arm": facts.arm,
        "aboveReachM": _estimate(facts.above_reach_m),
        "aheadM": _estimate(facts.ahead_m),
        "asideM": _estimate(facts.aside_m),
        "elbowDeg": _estimate(facts.elbow_deg),
        "sideOnDeg": _estimate(facts.side_on_deg),
        "hipsHeightM": _estimate(facts.hips_height_m),
        "frontKneeDeg": _estimate(facts.front_knee_deg),
        "offFloorM": _estimate(facts.off_floor_m),
        "weightAheadM": _estimate(facts.weight_ahead_m),
        "ready": None if ready is None else {
            "fromBaseM": _estimate(ready.from_base_m),
            "splitDropM": _estimate(ready.split_drop_m),
            "splitLowestMs": None if ready.split_lowest_ms is None else _estimate(ready.split_lowest_ms),
        },
    }


def stood_entry(stood: Stood) -> dict:
    """Where each half's player stood at a hit (the middle of the hips over the floor), or None."""
    return {
        half: None if where is None else {"point": rounded(where[0], 3), "covariance": [rounded(row, 6) for row in where[1]]}
        for half, where in stood.items()
    }


def find_players(video: Path, track_path: Path, poses_path: Path, detect: bool, model: Path, cameras, times_ms) -> tuple[Bodies | None, str | None]:
    """Each half's fitted body per frame, from everyone's poses (detected first, into poses_path,
    when `detect`); or None and why not."""
    if detect:
        if not model.is_file():
            return None, f"no pose model at {model}"
        try:
            if detect_poses.main(["--video", str(video), "--track", str(track_path), "--model", str(model), "--out", str(poses_path)]) != 0:
                return None, "the people could not be detected"
        except ImportError as error:
            return None, f"no pose detector: {error}"
    try:
        players = players_document(json.loads(Path(poses_path).read_text(encoding="utf-8")), json.loads(Path(track_path).read_text(encoding="utf-8")))
    except ValueError as error:
        return None, str(error)
    return fit_players(players, cameras, times_ms)[0], None


def _draw_body(frame: np.ndarray, camera: Camera, body: Body, colour: tuple[int, int, int]) -> None:
    pixels = camera.project(body.joints)
    for a, b in DRAWN_BONES:
        if np.isfinite(pixels[[a, b]]).all() and (np.abs(pixels[[a, b]]) < 10_000).all():
            cv2.line(frame, tuple(int(v) for v in pixels[a]), tuple(int(v) for v in pixels[b]), colour, 2, cv2.LINE_AA)


def write_overlay(video: VideoFile, cameras, shots: list[ShotFacts], path: Path, bodies: Bodies | None = None) -> None:
    """The video with the players' fitted bodies, each shot's fitted flight drawn behind the
    shuttle, its hit marked, and the landing and call once the rally's last shot comes down."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), video.fps or 30.0, video.size)
    if not writer.isOpened():
        raise OSError(f"cannot write {path}")
    for index, (time_ms, frame) in enumerate(video.frames()):
        camera = cameras[index] if index < len(cameras) else None
        if camera is not None and bodies is not None:
            for half in HALVES:
                body = bodies[half][index] if index < len(bodies[half]) else None
                if body is not None:
                    _draw_body(frame, camera, body, HITTER_COLOURS[half])
        if camera is not None:
            for number, shot in enumerate(shots, start=1):
                if not shot.path or time_ms < shot.hit_ms:
                    continue
                pixels = _pixels(camera, [p[1:] for p in shot.path if time_ms - TRAIL_MS <= p[0] <= time_ms])
                if len(pixels) >= 2:
                    cv2.polylines(frame, [pixels.reshape(-1, 1, 2)], False, HITTER_COLOURS[shot.hitter], 3, cv2.LINE_AA)
                at = _pixels(camera, [shot.contact])
                if time_ms - shot.hit_ms <= 400.0 and len(at):
                    x, y = (int(v) for v in at[0])
                    cv2.circle(frame, (x, y), 10, HITTER_COLOURS[shot.hitter], 2, cv2.LINE_AA)
                    label = f"{number} {'serve' if shot.serve else ''} {shot.speed_m_s.value * 3.6:.0f} km/h".replace("  ", " ")
                    cv2.putText(frame, label, (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                landed = _pixels(camera, [[*shot.landing.point, 0.0]]) if shot.landing is not None else []
                if len(landed) and time_ms >= shot.landing.time_ms:
                    x, y = (int(v) for v in landed[0])
                    colour = CALL_COLOURS[shot.landing.call]
                    cv2.drawMarker(frame, (x, y), colour, cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
                    cv2.putText(frame, shot.landing.call.upper(), (x + 14, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def draw_diagram(shots: list[ShotFacts], court: str, stood: list[Stood] | None = None) -> np.ndarray:
    """The court from above (near baseline at the bottom): each shot's path in its hitter's
    colour, numbered at its contact, where both players stood at the hit (a ring with its 95 %
    region; the hitter's joined to the contact), and the landing with its 95 % region and the
    call."""
    left, right = -DOUBLES_MARGIN_M - DIAGRAM_MARGIN_M, SINGLES_WIDTH_M + DOUBLES_MARGIN_M + DIAGRAM_MARGIN_M
    bottom, top = -DIAGRAM_MARGIN_M, COURT_LENGTH_M + DIAGRAM_MARGIN_M
    width, height = int((right - left) * DIAGRAM_PX_PER_M), int((top - bottom) * DIAGRAM_PX_PER_M)
    image = np.full((height, width, 3), (40, 90, 45), np.uint8)

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round((x - left) * DIAGRAM_PX_PER_M)), int(round((top - y) * DIAGRAM_PX_PER_M))

    for _, start, end in court_lines():
        cv2.line(image, px(start[0], start[1]), px(end[0], end[1]), (235, 235, 235), 2, cv2.LINE_AA)
    for _, start, end in net_tape_lines():
        cv2.line(image, px(start[0], start[1]), px(end[0], end[1]), (200, 200, 200), 4, cv2.LINE_AA)
    def region(centre: tuple[int, int], covariance, colour: tuple[int, int, int], thickness: int) -> None:
        values, vectors = np.linalg.eigh(np.asarray(covariance))
        axes = tuple(max(2, min(10_000, int(round(ELLIPSE_SCALE * np.sqrt(max(v, 0.0)) * DIAGRAM_PX_PER_M)))) for v in values)
        angle = float(np.degrees(np.arctan2(-vectors[1, 0], vectors[0, 0])))  # image y runs down
        cv2.ellipse(image, centre, axes, angle, 0, 360, colour, thickness, cv2.LINE_AA)

    cv2.putText(image, court, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    for number, shot in enumerate(shots, start=1):
        colour = HITTER_COLOURS[shot.hitter]
        if len(shot.path) >= 2:
            cv2.polylines(image, [np.array([px(p[1], p[2]) for p in shot.path], np.int32).reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
        at = px(shot.contact[0], shot.contact[1])
        for half, where in (stood[number - 1] if stood else {}).items():
            if where is None:
                continue
            centre = px(*where[0])
            region(centre, where[1], HITTER_COLOURS[half], 1)
            cv2.circle(image, centre, 6, HITTER_COLOURS[half], 2, cv2.LINE_AA)
            if half == shot.hitter:
                cv2.line(image, centre, at, colour, 1, cv2.LINE_AA)
        cv2.circle(image, at, 7, colour, -1, cv2.LINE_AA)
        cv2.putText(image, str(number), (at[0] + 9, at[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if shot.landing is not None:
            colour = CALL_COLOURS[shot.landing.call]
            centre = px(*shot.landing.point)
            region(centre, shot.landing.covariance, colour, 2)
            cv2.drawMarker(image, centre, colour, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(image, shot.landing.call, (centre[0] + 12, centre[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return image


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a rally video: camera, shuttle, hits, flights, the players' bodies and shot facts.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--golden", type=Path, help="labeller export with court points placed on one frame (needed without --track)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--court", choices=("singles", "doubles"), default="singles")
    parser.add_argument("--track", type=Path, help="an existing track_camera.py output for this video")
    parser.add_argument("--shuttle", type=Path, help="an existing detect_shuttle.py output for this video")
    parser.add_argument("--poses", type=Path, help="an existing detect_poses.py output for this video")
    parser.add_argument("--pose-model", type=Path, default=POSE_MODEL, help="MediaPipe pose landmarker model (.task), to detect the poses")
    parser.add_argument("--overlay", type=Path, help="also write the video with the players, flights, hits and calls drawn on it")
    parser.add_argument("--diagram", type=Path, help="also write the court from above with every shot and where the players stood (PNG)")
    parser.add_argument("--terminal-velocity", type=float, help="the shuttle's terminal velocity in m/s, when known")
    parser.add_argument("--terminal-sigma", type=float, help="its standard deviation (with --terminal-velocity; 0.05 if not given)")
    args = parser.parse_args(argv)
    if args.terminal_sigma is not None and args.terminal_velocity is None:
        return _fail("--terminal-sigma goes with --terminal-velocity")
    inputs = {Path(p).resolve() for p in (args.video, args.golden, args.track, args.shuttle, args.poses) if p is not None}
    if args.out.resolve() in inputs or any(p is not None and Path(p).resolve() in inputs for p in (args.overlay, args.diagram)):
        return _fail("an output would overwrite an input: choose other names for --out, --overlay or --diagram")

    stem = args.out.name.removesuffix(".json").removesuffix(".analysis")
    track_path = args.track or args.out.with_name(f"{stem}.track.json")
    shuttle_path = args.shuttle or args.out.with_name(f"{stem}.shuttle.json")
    poses_path = args.poses or args.out.with_name(f"{stem}.poses.json")
    if args.track is None:
        if args.golden is None:
            return _fail("give --golden (court points to track the camera from) or an existing --track")
        if track_camera.main([str(args.video), "--golden", str(args.golden), "--out", str(track_path)]) != 0:
            return _fail("the camera could not be tracked")
    if args.shuttle is None and detect_shuttle.main([str(args.video), "--out", str(shuttle_path)]) != 0:
        return _fail("the shuttle could not be detected")

    shuttle_doc = json.loads(Path(shuttle_path).read_text(encoding="utf-8"))
    try:
        track, times_ms, cameras = load_detections(shuttle_doc, json.loads(Path(track_path).read_text(encoding="utf-8")))
    except ValueError as error:
        return _fail(str(error))
    onsets, sound_error = hear(args.video)
    if sound_error:
        print(f"no sound used: {sound_error}", file=sys.stderr)
    tracked = any(camera is not None for camera in cameras)
    analysis = find_hits(track, times_ms, cameras, onsets) if tracked else None
    drag = (args.terminal_velocity, args.terminal_sigma if args.terminal_sigma is not None else 0.05) if args.terminal_velocity is not None else None
    shots = rally_shots(analysis, track, times_ms, cameras, court=args.court, terminal_velocity=drag) if analysis is not None else []
    bodies, players_error = find_players(args.video, track_path, poses_path, args.poses is None, args.pose_model, cameras, times_ms) if tracked else (None, "no camera")
    if players_error:
        print(f"no players: {players_error}", file=sys.stderr)
    facts = rally_body_facts(shots, bodies, times_ms) if bodies is not None else [None] * len(shots)
    stood = [{half: standing_at(bodies[half], times_ms, shot.hit_ms) for half in HALVES} if bodies is not None else None for shot in shots]

    result = hits_document(args.video, shuttle_doc, analysis, onsets, sound_error, times_ms)
    calls = [shot.landing.call for shot in shots if shot.landing is not None]
    result["court"] = args.court
    result["summary"]["shots"] = len(shots)
    result["summary"]["calls"] = {call: calls.count(call) for call in sorted(set(calls))}
    result["summary"]["terminalVelocity"] = None
    if shots:
        flights = shots[0].terminal_flights
        source = "given" if drag else (f"this rally's {flights} long flights" if flights else "the usual feather shuttle (no long flight to measure it)")
        result["summary"]["terminalVelocity"] = {**_estimate(shots[0].terminal_velocity), "from": source}
    result["summary"]["players"] = None if bodies is None else {
        half: {
            "frames": sum(body is not None for body in bodies[half]),
            "standingReachM": _estimate(standing_reach(bodies[half])) if any(body is not None for body in bodies[half]) else None,
        }
        for half in HALVES
    }
    result["summary"]["playersError"] = players_error
    result["shots"] = [
        {**shot_entry(number, shot), "body": body_entry(fact), "players": None if at is None else stood_entry(at)}
        for number, (shot, fact, at) in enumerate(zip(shots, facts, stood), start=1)
    ]
    result["sources"] = {"track": Path(track_path).name, "shuttle": Path(shuttle_path).name, "poses": Path(poses_path).name if bodies is not None else None}
    args.out.write_text(json.dumps(_clean(result), allow_nan=False), encoding="utf-8")
    if args.diagram:
        cv2.imwrite(str(args.diagram), draw_diagram(shots, args.court, stood if bodies is not None else None))
    if args.overlay:
        video = VideoFile(args.video)
        if not video.opened:
            return _fail(f"cannot read {args.video} to draw the overlay")
        try:
            write_overlay(video, cameras, shots, args.overlay, bodies)
        except OSError as error:
            return _fail(str(error))
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
