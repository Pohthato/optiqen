# worker/evaluation/shuttleset.py
"""ShuttleSet's professional rallies on the app's court, to learn what a position is worth from.

ShuttleSet (Wang, Huang, Ik and Peng, "ShuttleSet: A Human-Annotated Stroke-Level Singles
Dataset for Badminton Tactical Analysis", KDD 2023) is 44 professional singles matches from 2018
to 2021, labelled by hand from broadcast video. It comes from github.com/wywyWang/CoachAI-Projects
under the MIT licence (copyright 2022 Wei-Yao Wang) and is kept out of the repository, like the
clips: `worker/ShuttleSet/` holds its `set/` folder, README and LICENSE.

Every shot gives where both players stood when it was hit and where it went: where it was met,
or, for the rally's last shot, where it came down. These are pixels in the broadcast frame, and
each video's four doubles-court corners map them to the floor in geometry.court_model's frame
(x across from the left singles sideline, y along from the near baseline). A click is taken to
be CLICK_PX out either way in the image, carried through that mapping, so the far half is less
sure than the near.

Players were clicked a little above their feet, which puts a far player deeper and a near player
nearer the net than they stood. The clicks on players are moved down by `ABOVE_FEET_PX`, fitted
so that servers stand as far from the net in either half; receivers, left out of the fit, check
it.

The rally's last shot says how it ended (`ENDS`) and who won. A serve ShuttleSet does not know
was not labelled, mostly without positions, so its rally starts at the return. A rally is set
aside, and counted, when it has no end, ended in a fault, has another shot ShuttleSet does not
know, or misses a position.

The test matches are fixed here, before any model is fitted (`held_out`): every match of one man
and one woman drawn at random among those with two to five matches, so that test sees players
never learnt from without taking a quarter of the data alone, and one in four of the other
matches.

    python -m evaluation.shuttleset ShuttleSet --out shuttleset.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geometry.court_model import COURT_LENGTH_M, DOUBLES_MARGIN_M, NET_Y_M, SINGLES_WIDTH_M

CLICK_PX = 3.0
ABOVE_FEET_PX = 3.42  # fitted by `fit_above_feet` on all 44 matches; see the plan's 7a results
FRAME_RATES = (25.0, 29.97, 30.0, 50.0, 59.94, 60.0)
# ShuttleSet's 18 types, as the app names them (shot_names.NAMES, and the serve).
NAMES = {
    "發短球": "serve",  # short service
    "發長球": "serve",  # long service
    "長球": "clear",
    "殺球": "smash",
    "點扣": "smash",  # wrist smash
    "撲球": "smash",  # rush: hit down from the net
    "切球": "drop",
    "過度切球": "drop",  # passive drop
    "平球": "drive",
    "小平球": "drive",  # driven flight
    "後場抽平球": "drive",  # back-court drive
    "防守回抽": "drive",  # defensive return drive
    "推球": "push",
    "挑球": "lift",  # lob
    "防守回挑": "lift",  # defensive return lob
    "放小球": "net",
    "勾球": "net",  # cross-court net shot
    "擋小球": "block",  # return net
}
UNKNOWN = "未知球種"
# Why the rally was lost, as ShuttleSet gives it, as the app's end and how it came about.
ENDS = {
    "對手落地致勝": ("in", "winner"),  # the opponent's shot came down in
    "落點判斷失誤": ("in", "let in"),  # misjudged where it would come down
    "對手落地判斷失誤": ("in", "let in"),  # the same, in a few rallies
    "出界": ("out", "out"),
    "掛網": ("net", "net"),  # met the net
    "未過網": ("net", "not over"),  # did not reach it
}
FAULT = "犯規"
# ShuttleSet does not say which matches are women's.
WOMEN = frozenset(
    {
        "An Se Young", "Busanan ONGBAMRUNGPHAN", "Carolina MARIN", "Evgeniya KOSETSKAYA", "Mia BLICHFELDT",
        "Michelle LI", "Neslihan YIGIT", "Pornpawee CHOCHUWONG", "PUSARLA V. Sindhu", "Ratchanok INTANON",
        "Supanida KATETHONG",
    }
)
HELD_OUT_SEED = 7
HELD_OUT_SHARE = 0.25
HELD_OUT_PLAYED = (2, 5)  # matches played by a player who may be held out
# The doubles court's corners as ShuttleSet names them, in metres.
CORNERS_M = {
    "upleft": (-DOUBLES_MARGIN_M, COURT_LENGTH_M),
    "upright": (SINGLES_WIDTH_M + DOUBLES_MARGIN_M, COURT_LENGTH_M),
    "downleft": (-DOUBLES_MARGIN_M, 0.0),
    "downright": (SINGLES_WIDTH_M + DOUBLES_MARGIN_M, 0.0),
}


@dataclass(frozen=True)
class Spot:
    point: np.ndarray  # on the floor, metres
    covariance: np.ndarray


@dataclass(frozen=True)
class Hit:
    hitter: str  # "A" won the match, "B" lost it
    time_ms: float
    name: str  # one of shot_names.NAMES, or "serve"
    hitter_at: Spot  # where the hitter stood at the hit
    receiver_at: Spot
    goes_to: Spot  # where it was met next, or where it came down


@dataclass(frozen=True)
class Rally:
    match: str
    game: int
    number: int
    hits: list[Hit]
    end: str  # in, out or net
    how: str  # winner, let in, out, net or not over
    winner: str
    score: tuple[int, int]  # A's and B's in the game, after the rally


@dataclass
class Match:
    video: str
    players: tuple[str, str]  # A, B
    fps: float
    rallies: list[Rally] = field(default_factory=list)
    set_aside: Counter = field(default_factory=Counter)
    score_agrees: int = 0  # rallies whose score after them follows from the one before and the winner
    score_checked: int = 0


def homography(image: list[tuple[float, float]], floor: list[tuple[float, float]]) -> np.ndarray:
    """The 3x3 map from image pixels to the floor through four or more point pairs."""
    rows = []
    for (x, y), (u, v) in zip(image, floor):
        rows += [[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u], [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]]
    h = np.linalg.svd(np.asarray(rows))[2][-1]
    return (h / h[-1]).reshape(3, 3)


def to_floor(H: np.ndarray, x: float, y: float, click_px: float = CLICK_PX) -> Spot:
    """A clicked pixel on the floor, its covariance from `click_px` either way in the image."""
    p = H @ np.array([x, y, 1.0])
    point = p[:2] / p[2]
    jacobian = (H[:2, :2] - np.outer(point, H[2, :2])) / p[2]
    return Spot(point, click_px**2 * jacobian @ jacobian.T)


def frame_rate(times_s: list[float], frames: list[float]) -> float:
    """The video's frame rate: frame numbers against the clock, to the nearest usual rate."""
    slope = float(np.polyfit(times_s, frames, 1)[0])
    return min(FRAME_RATES, key=lambda rate: abs(rate - slope))


def _seconds(clock: str) -> float:
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    return hours * 3600.0 + minutes * 60.0 + seconds


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _spot(H: np.ndarray, row: dict[str, str], key: str, above_feet_px: float = 0.0) -> Spot | None:
    if row[f"{key}_x"] == "" or row[f"{key}_y"] == "":
        return None
    return to_floor(H, float(row[f"{key}_x"]), float(row[f"{key}_y"]) + above_feet_px)


def _rally(match: Match, H: np.ndarray, game: int, number: int, rows: list[dict[str, str]], above_feet_px: float) -> Rally | str:
    """The rally, or why it is set aside."""
    rows = sorted(rows, key=lambda row: float(row["ball_round"]))
    last = rows[-1]
    if not last["lose_reason"]:
        return "no end"
    if last["lose_reason"] == FAULT:
        return "fault"
    if last["lose_reason"] not in ENDS:
        return f"end not known: {last['lose_reason']}"
    if rows[0]["type"] == UNKNOWN:
        rows = rows[1:]
    if not rows or any(row["type"] == UNKNOWN for row in rows):
        return "a shot not known"
    hits = []
    for row in rows:
        spots = (_spot(H, row, "player_location", above_feet_px), _spot(H, row, "opponent_location", above_feet_px), _spot(H, row, "landing"))
        if any(spot is None for spot in spots):
            return "a position missing"
        hits.append(Hit(row["player"], float(row["frame_num"]) / match.fps * 1000.0, NAMES[row["type"]], *spots))
    end, how = ENDS[last["lose_reason"]]
    winner = last["getpoint_player"]
    if winner != (last["player"] if end == "in" else {"A": "B", "B": "A"}[last["player"]]):
        return "winner not the end's"
    return Rally(match.video, game, number, hits, end, how, winner, (int(last["roundscore_A"]), int(last["roundscore_B"])))


def read_match(root: Path, info: dict[str, str], corners: dict[str, str], above_feet_px: float = ABOVE_FEET_PX) -> Match:
    folder = root / "set" / info["video"]
    games = {int(path.stem.removeprefix("set")): _read_csv(path) for path in sorted(folder.glob("set*.csv"))}
    clock = [(_seconds(row["time"]), float(row["frame_num"])) for rows in games.values() for row in rows if row["time"] and row["frame_num"]]
    match = Match(info["video"], (info["winner"].strip(), info["loser"].strip()), frame_rate(*zip(*clock)))
    H = homography([(float(corners[f"{name}_x"]), float(corners[f"{name}_y"])) for name in CORNERS_M], list(CORNERS_M.values()))
    for game, rows in sorted(games.items()):
        by_rally: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_rally[int(float(row["rally"]))].append(row)
        before: dict[int, tuple[int, int]] = {0: (0, 0)}
        for number, rally_rows in sorted(by_rally.items()):
            before[number] = (int(rally_rows[-1]["roundscore_A"]), int(rally_rows[-1]["roundscore_B"]))
        for number, rally_rows in sorted(by_rally.items()):
            rally = _rally(match, H, game, number, rally_rows, above_feet_px)
            if isinstance(rally, str):
                match.set_aside[rally] += 1
                continue
            match.rallies.append(rally)
            if number - 1 in before:
                gained = np.subtract(rally.score, before[number - 1])
                match.score_checked += 1
                match.score_agrees += tuple(gained) == ((1, 0) if rally.winner == "A" else (0, 1))
    return match


def read(root: Path, above_feet_px: float = ABOVE_FEET_PX) -> list[Match]:
    corners = {row["video"]: row for row in _read_csv(root / "set" / "homography.csv")}
    return [read_match(root, info, corners[info["video"]], above_feet_px) for info in _read_csv(root / "set" / "match.csv")]


def held_out(matches: list[Match]) -> dict[str, Any]:
    """The test matches: every match of one man and one woman drawn at random among those with
    `HELD_OUT_PLAYED` matches, and one in four of the rest. Training has neither."""
    rng = np.random.default_rng(HELD_OUT_SEED)
    played = Counter(name for match in matches for name in match.players)
    players = sorted(name for name, count in played.items() if HELD_OUT_PLAYED[0] <= count <= HELD_OUT_PLAYED[1])
    chosen = [str(rng.choice([name for name in players if name not in WOMEN])), str(rng.choice([name for name in players if name in WOMEN]))]
    stressed = sorted(match.video for match in matches if set(match.players) & set(chosen))
    rest = sorted(match.video for match in matches if match.video not in stressed)
    count = round(HELD_OUT_SHARE * len(rest))
    test = sorted(str(video) for video in rng.choice(rest, size=count, replace=False))
    return {"players": chosen, "stressed": stressed, "test": test, "train": [video for video in rest if video not in test]}


def depths_at_serve(matches: list[Match]) -> dict[str, list[float]]:
    """How far from the net the server and the receiver stood at each serve, by half."""
    out: dict[str, list[float]] = defaultdict(list)
    for match in matches:
        for rally in match.rallies:
            serve = rally.hits[0]
            if serve.name != "serve":
                continue
            for who, spot in (("server", serve.hitter_at), ("receiver", serve.receiver_at)):
                half = "near" if spot.point[1] < NET_Y_M else "far"
                out[f"{who} {half}"].append(abs(float(spot.point[1]) - NET_Y_M))
    return out


def fit_above_feet(root: Path, low: float = 0.0, high: float = 10.0) -> float:
    """How far above their feet the players were clicked: the offset at which servers stand as
    far from the net in either half (medians), by bisection."""
    def gap(px: float) -> float:
        depths = depths_at_serve(read(root, px))
        return float(np.median(depths["server far"]) - np.median(depths["server near"]))

    for _ in range(20):
        middle = (low + high) / 2
        low, high = (middle, high) if gap(middle) > 0 else (low, middle)
    return round((low + high) / 2, 2)


def _beyond(spot: Spot) -> tuple[float, float]:
    """How far the spot is outside the singles court (negative inside), and its sigma that way."""
    x, y = spot.point
    edges = [(-x, np.array([-1.0, 0.0])), (x - SINGLES_WIDTH_M, np.array([1.0, 0.0])), (-y, np.array([0.0, -1.0])), (y - COURT_LENGTH_M, np.array([0.0, 1.0]))]
    distance, normal = max(edges, key=lambda edge: edge[0])
    return float(distance), float(np.sqrt(normal @ spot.covariance @ normal))


def _own_side(hit: Hit) -> bool:
    """Whether the shot came down on its hitter's side of the net, or within twice its sigma of
    the net."""
    y, sigma = float(hit.goes_to.point[1]), float(np.sqrt(hit.goes_to.covariance[1, 1]))
    return (y < NET_Y_M) == (float(hit.hitter_at.point[1]) < NET_Y_M) or abs(y - NET_Y_M) <= 2 * sigma


def check(matches: list[Match]) -> dict[str, Any]:
    """The checks 7a is accepted on, and the counts."""
    rallies = [rally for match in matches for rally in match.rallies]
    ends: dict[str, Any] = {}
    for how in ("winner", "let in", "out", "net", "not over"):
        landed = [_beyond(rally.hits[-1].goes_to) for rally in rallies if rally.how == how]
        if not landed:
            continue
        distance, sigma = np.array(landed).T
        if how in ("winner", "let in"):
            agrees = distance <= 2 * sigma
        elif how == "out":
            agrees = distance >= -2 * sigma
        else:  # on the hitter's side of the net, or at it
            agrees = np.array([_own_side(rally.hits[-1]) for rally in rallies if rally.how == how])
        ends[how] = {"rallies": len(landed), "agree": int(agrees.sum()), "share": round(float(agrees.mean()), 3), "beyondMedianM": round(float(np.median(distance)), 2)}
    depths = depths_at_serve(matches)
    serves = {key: round(float(np.median(value)), 2) for key, value in sorted(depths.items())}
    set_aside = sum((match.set_aside for match in matches), Counter())
    return {
        "matches": len(matches),
        "rallies": len(rallies),
        "hits": sum(len(rally.hits) for rally in rallies),
        "setAside": dict(set_aside.most_common()),
        "scoreAgrees": [sum(match.score_agrees for match in matches), sum(match.score_checked for match in matches)],
        "ends": ends,
        "fromNetAtServeM": serves,
        "frameRates": dict(Counter(match.fps for match in matches)),
        "names": dict(Counter(hit.name for rally in rallies for hit in rally.hits).most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read ShuttleSet onto the court and check it.")
    parser.add_argument("root", type=Path, help="the folder holding ShuttleSet's set/")
    parser.add_argument("--fit-above-feet", action="store_true", help="fit ABOVE_FEET_PX instead")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.fit_above_feet:
        print(f"above the feet: {fit_above_feet(args.root)} px")
        return 0
    matches = read(args.root)
    report = check(matches)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    report["heldOut"] = held = held_out(matches)
    print(f"held out: players {held['players']}; {len(held['stressed'])} of their matches, {len(held['test'])} others; {len(held['train'])} to learn from")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
