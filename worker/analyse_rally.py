# worker/analyse_rally.py
"""A rally video understood in one command: the camera of every frame, the shuttle in every frame,
the hits, each flight in 3D and every shot's facts.

    python analyse_rally.py clip.mp4 --golden clip.json --out clip.analysis.json
        [--court singles|doubles] [--overlay clip.analysis.mp4] [--diagram clip.court.png]
        [--track clip.track.json] [--shuttle clip.shuttle.json] [--terminal-velocity 6.9 --terminal-sigma 0.05]

--golden is the labeller export with court points placed on one frame (see track_camera.py). An
existing --track or --shuttle file is used as it is; otherwise the camera track and the shuttle
detections are computed and written next to --out. The overlay draws each fitted flight into the
video with its hit and, for the rally's last shot, where it landed and the call; the diagram is
the court from above with every shot's path and the landing's uncertainty. The shuttle's drag
(terminal velocity) is measured from the rally's own long flights unless given: measured once
over a whole session's rallies, it is far surer, and close line calls sharpen with it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import detect_shuttle
import track_camera
from find_hits import hear, hits_document, load_detections, rounded
from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, SINGLES_WIDTH_M, court_lines, net_tape_lines
from hits import find_hits
from shots import ShotFacts, rally_shots
from video import VideoFile

HITTER_COLOURS = {"near": (255, 160, 40), "far": (40, 200, 255)}  # BGR
CALL_COLOURS = {"in": (60, 200, 60), "out": (60, 60, 230), "too close to call": (0, 200, 230)}
TRAIL_MS = 600.0  # how much of a flight the overlay shows behind the shuttle
DIAGRAM_PX_PER_M = 60
DIAGRAM_MARGIN_M = 1.5
ELLIPSE_SCALE = float(np.sqrt(5.991))  # 95 % of a 2D Gaussian


def _estimate(estimate) -> dict:
    return {"value": round(estimate.value, 3), "sigma": round(estimate.sigma, 3)}


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


def write_overlay(video: VideoFile, cameras, shots: list[ShotFacts], path: Path) -> None:
    """The video with each shot's fitted flight drawn behind the shuttle, its hit marked, and the
    landing and call once the rally's last shot comes down."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), video.fps or 30.0, video.size)
    for index, (time_ms, frame) in enumerate(video.frames()):
        camera = cameras[index] if index < len(cameras) else None
        if camera is not None:
            for number, shot in enumerate(shots, start=1):
                if not shot.path or time_ms < shot.hit_ms:
                    continue
                points = np.array([p[1:] for p in shot.path if time_ms - TRAIL_MS <= p[0] <= time_ms])
                if len(points) >= 2:
                    pixels = camera.project(points).astype(np.int32)
                    cv2.polylines(frame, [pixels.reshape(-1, 1, 2)], False, HITTER_COLOURS[shot.hitter], 3, cv2.LINE_AA)
                if time_ms - shot.hit_ms <= 400.0:
                    at = camera.project(np.array([shot.contact]))[0].astype(int)
                    cv2.circle(frame, tuple(at), 10, HITTER_COLOURS[shot.hitter], 2, cv2.LINE_AA)
                    label = f"{number} {'serve' if shot.serve else ''} {shot.speed_m_s.value * 3.6:.0f} km/h".replace("  ", " ")
                    cv2.putText(frame, label, (int(at[0]) + 12, int(at[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if shot.landing is not None and time_ms >= shot.landing.time_ms:
                    at = camera.project(np.array([[*shot.landing.point, 0.0]]))[0].astype(int)
                    colour = CALL_COLOURS[shot.landing.call]
                    cv2.drawMarker(frame, tuple(at), colour, cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
                    cv2.putText(frame, shot.landing.call.upper(), (int(at[0]) + 14, int(at[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def draw_diagram(shots: list[ShotFacts], court: str) -> np.ndarray:
    """The court from above (near baseline at the bottom): each shot's path in its hitter's
    colour, numbered at its contact, and the landing with its 95 % region and the call."""
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
    cv2.putText(image, court, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    for number, shot in enumerate(shots, start=1):
        colour = HITTER_COLOURS[shot.hitter]
        if len(shot.path) >= 2:
            cv2.polylines(image, [np.array([px(p[1], p[2]) for p in shot.path], np.int32).reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
        at = px(shot.contact[0], shot.contact[1])
        cv2.circle(image, at, 7, colour, -1, cv2.LINE_AA)
        cv2.putText(image, str(number), (at[0] + 9, at[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if shot.landing is not None:
            colour = CALL_COLOURS[shot.landing.call]
            values, vectors = np.linalg.eigh(np.asarray(shot.landing.covariance))
            axes = tuple(max(2, int(round(ELLIPSE_SCALE * np.sqrt(max(v, 0.0)) * DIAGRAM_PX_PER_M))) for v in values)
            angle = float(np.degrees(np.arctan2(-vectors[1, 0], vectors[0, 0])))  # image y runs down
            centre = px(*shot.landing.point)
            cv2.ellipse(image, centre, axes, angle, 0, 360, colour, 2, cv2.LINE_AA)
            cv2.drawMarker(image, centre, colour, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(image, shot.landing.call, (centre[0] + 12, centre[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return image


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse a rally video: camera, shuttle, hits, flights and shot facts.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--golden", type=Path, help="labeller export with court points placed on one frame (needed without --track)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--court", choices=("singles", "doubles"), default="singles")
    parser.add_argument("--track", type=Path, help="an existing track_camera.py output for this video")
    parser.add_argument("--shuttle", type=Path, help="an existing detect_shuttle.py output for this video")
    parser.add_argument("--overlay", type=Path, help="also write the video with the flights, hits and calls drawn on it")
    parser.add_argument("--diagram", type=Path, help="also write the court from above with every shot (PNG)")
    parser.add_argument("--terminal-velocity", type=float, help="the shuttle's terminal velocity in m/s, when known")
    parser.add_argument("--terminal-sigma", type=float, default=0.05, help="its standard deviation (with --terminal-velocity)")
    args = parser.parse_args(argv)

    stem = args.out.name.removesuffix(".json").removesuffix(".analysis")
    track_path = args.track or args.out.with_name(f"{stem}.track.json")
    shuttle_path = args.shuttle or args.out.with_name(f"{stem}.shuttle.json")
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
    analysis = find_hits(track, times_ms, cameras, onsets) if any(camera is not None for camera in cameras) else None
    drag = (args.terminal_velocity, args.terminal_sigma) if args.terminal_velocity else None
    shots = rally_shots(analysis, track, times_ms, cameras, court=args.court, drag=drag) if analysis is not None else []

    result = hits_document(args.video, shuttle_doc, analysis, onsets, sound_error, times_ms)
    calls = [shot.landing.call for shot in shots if shot.landing is not None]
    result["court"] = args.court
    result["summary"]["shots"] = len(shots)
    result["summary"]["calls"] = {call: calls.count(call) for call in sorted(set(calls))}
    if shots:
        result["summary"]["terminalVelocity"] = {**_estimate(shots[0].terminal_velocity), "from": "given" if drag else "this rally"}
    result["shots"] = [shot_entry(number, shot) for number, shot in enumerate(shots, start=1)]
    result["sources"] = {"track": Path(track_path).name, "shuttle": Path(shuttle_path).name}
    args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    if args.diagram:
        cv2.imwrite(str(args.diagram), draw_diagram(shots, args.court))
    if args.overlay:
        video = VideoFile(args.video)
        if not video.opened:
            return _fail(f"cannot read {args.video} to draw the overlay")
        write_overlay(video, cameras, shots, args.overlay)
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
