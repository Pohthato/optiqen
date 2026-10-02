# worker/geometry/court_model.py
"""BWF singles court in a metric world frame.

Frame: x runs across the court from the left singles sideline (0) to the right
singles sideline (5.18); y runs along the court from the near baseline (0) to
the far baseline (13.40); z is up from the floor. The doubles sidelines sit
0.46 m outside the singles sidelines and stay visible during singles play, so
they are part of the model.
"""
from __future__ import annotations

import numpy as np

COURT_LENGTH_M = 13.40
SINGLES_WIDTH_M = 5.18
DOUBLES_MARGIN_M = 0.46
NET_Y_M = COURT_LENGTH_M / 2
NET_POST_HEIGHT_M = 1.55
NET_CENTRE_HEIGHT_M = 1.524
SHORT_SERVICE_M = 1.98
LONG_SERVICE_INSET_M = 0.76

COLUMNS: dict[str, float] = {
    "dl": -DOUBLES_MARGIN_M,
    "sl": 0.0,
    "c": SINGLES_WIDTH_M / 2,
    "sr": SINGLES_WIDTH_M,
    "dr": SINGLES_WIDTH_M + DOUBLES_MARGIN_M,
}
ROWS: dict[str, float] = {
    "back0": 0.0,
    "long0": LONG_SERVICE_INSET_M,
    "short0": NET_Y_M - SHORT_SERVICE_M,
    "short1": NET_Y_M + SHORT_SERVICE_M,
    "long1": COURT_LENGTH_M - LONG_SERVICE_INSET_M,
    "back1": COURT_LENGTH_M,
}


def _build_keypoints() -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for row_name, y in ROWS.items():
        for column_name, x in COLUMNS.items():
            points[f"{row_name}_{column_name}"] = np.array([x, y, 0.0])
    left, right = COLUMNS["dl"], COLUMNS["dr"]
    points["post_left_base"] = np.array([left, NET_Y_M, 0.0])
    points["post_left_top"] = np.array([left, NET_Y_M, NET_POST_HEIGHT_M])
    points["post_right_base"] = np.array([right, NET_Y_M, 0.0])
    points["post_right_top"] = np.array([right, NET_Y_M, NET_POST_HEIGHT_M])
    points["net_centre_top"] = np.array([COLUMNS["c"], NET_Y_M, NET_CENTRE_HEIGHT_M])
    return points


KEYPOINTS: dict[str, np.ndarray] = _build_keypoints()
FLOOR_KEYPOINT_NAMES: tuple[str, ...] = tuple(
    name for name, point in KEYPOINTS.items() if point[2] == 0.0 and not name.startswith("post_")
)

# The four singles corners the existing calibration UI asks the player to tap.
CORNER_LABEL_TO_KEYPOINT: dict[str, str] = {
    "nearLeft": "back0_sl",
    "nearRight": "back0_sr",
    "farRight": "back1_sr",
    "farLeft": "back1_sl",
}


def court_lines() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Painted floor lines as (name, start, end) in world metres."""
    left, right = COLUMNS["dl"], COLUMNS["dr"]
    lines: list[tuple[str, np.ndarray, np.ndarray]] = []
    for row_name, y in ROWS.items():
        lines.append((row_name, np.array([left, y, 0.0]), np.array([right, y, 0.0])))
    for column_name, x in COLUMNS.items():
        if column_name == "c":
            continue
        lines.append((f"side_{column_name}", np.array([x, 0.0, 0.0]), np.array([x, COURT_LENGTH_M, 0.0])))
    centre = COLUMNS["c"]
    lines.append(("centre0", np.array([centre, 0.0, 0.0]), np.array([centre, ROWS["short0"], 0.0])))
    lines.append(("centre1", np.array([centre, ROWS["short1"], 0.0]), np.array([centre, COURT_LENGTH_M, 0.0])))
    return lines
