# worker/test_shuttleset.py
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.shuttleset import CORNERS_M, Match, check, frame_rate, held_out, homography, read, to_floor

PIXELS = {"upleft": (410.0, 310.0), "upright": (870.0, 310.0), "downleft": (300.0, 670.0), "downright": (980.0, 670.0)}
FLOOR_FROM_IMAGE = homography(list(PIXELS.values()), [CORNERS_M[name] for name in PIXELS])
COLUMNS = (
    "rally,ball_round,time,frame_num,roundscore_A,roundscore_B,player,server,type,aroundhead,backhand,hit_height,hit_area,hit_x,hit_y,"
    "landing_height,landing_area,landing_x,landing_y,lose_reason,win_reason,getpoint_player,flaw,player_location_area,"
    "player_location_x,player_location_y,opponent_location_area,opponent_location_x,opponent_location_y,db"
).split(",")


def pixel(x, y):
    p = np.linalg.inv(FLOOR_FROM_IMAGE) @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def stroke(rally, order, frame, score, player, kind, stood, receiver, to, lose="", win=""):
    """One row of a set file, the positions given in metres."""
    row = dict.fromkeys(COLUMNS, "")
    seconds = int(frame // 25)
    row.update(rally=rally, ball_round=f"{order}.0", time=f"00:{seconds // 60:02d}:{seconds % 60:02d}", frame_num=f"{frame}.0",
               roundscore_A=score[0], roundscore_B=score[1], player=player, server=2, type=kind, db=0)
    for key, at in (("player_location", stood), ("opponent_location", receiver), ("landing", to)):
        if at is not None:
            row[f"{key}_x"], row[f"{key}_y"] = (f"{v:.3f}" for v in pixel(*at))
    if lose:
        row.update(lose_reason=lose, win_reason=win, getpoint_player={"A": "A", "B": "B"}[score[2]])
    return row


def write_match(root: Path, rows):
    video = "A_B_Open_2020_Final"
    (root / "set" / video).mkdir(parents=True)
    with open(root / "set" / "match.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "video", "tournament", "round", "year", "month", "day", "set", "duration", "winner", "loser", "downcourt", "url"])
        writer.writerow([1, video, "Open", "Final", 2020, 1, 1, 1, 40, "Player A", "Player B", 0, ""])
    with open(root / "set" / "homography.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        names = ["upleft", "upright", "downleft", "downright"]
        writer.writerow(["id", "video", "homography_matrix", *[f"{n}_x" for n in names], *[f"{n}_y" for n in names]])
        writer.writerow([1, video, "[]", *[PIXELS[n][0] for n in names], *[PIXELS[n][1] for n in names]])
    with open(root / "set" / video / "set1.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def rallies():
    near, far = (2.6, 3.5), (2.6, 9.9)
    return [
        # 1: A serves from the near half, B nets, A's smash comes down in.
        stroke(1, 1, 250, (1, 0, "A"), "A", "發短球", near, far, (2.0, 8.8)),
        stroke(1, 2, 275, (1, 0, "A"), "B", "放小球", (2.2, 8.2), (2.4, 4.0), (2.5, 5.5)),
        stroke(1, 3, 300, (1, 0, "A"), "A", "殺球", (2.5, 5.0), (2.3, 8.2), (4.9, 12.0), "對手落地致勝", "落地致勝"),
        # 2: the serve not known and not placed; B's lift goes out.
        stroke(2, 1, 500, (2, 0, "A"), "A", "未知球種", None, None, None),
        stroke(2, 2, 525, (2, 0, "A"), "B", "挑球", (2.0, 8.0), (2.4, 4.0), (2.0, -0.5), "出界", "對手出界"),
        # 3: no end.
        stroke(3, 1, 700, (2, 1, "B"), "B", "發短球", far, near, (2.0, 4.5)),
        # 4: a shot in the middle not known.
        stroke(4, 1, 900, (2, 2, "B"), "B", "發短球", far, near, (2.0, 4.5)),
        stroke(4, 2, 925, (2, 2, "B"), "A", "未知球種", (2.0, 4.5), (2.6, 9.9), (2.0, 11.0)),
        stroke(4, 3, 950, (2, 2, "B"), "B", "長球", (2.0, 11.0), (2.0, 4.5), (2.0, 6.0), "掛網", "對手掛網"),
        # 5: B's push meets the net, but the point is B's.
        stroke(5, 1, 1100, (2, 3, "B"), "A", "發短球", near, far, (2.0, 8.8)),
        stroke(5, 2, 1125, (2, 3, "B"), "B", "推球", (2.0, 8.8), (2.6, 3.0), (2.4, 6.6), "掛網", "對手掛網"),
    ]


class FloorTests(unittest.TestCase):
    def test_the_corners_land_on_the_doubles_court(self):
        for name, (x, y) in PIXELS.items():
            spot = to_floor(FLOOR_FROM_IMAGE, x, y)
            np.testing.assert_allclose(spot.point, CORNERS_M[name], atol=1e-3)

    def test_a_click_is_less_sure_at_the_far_end(self):
        near, far = to_floor(FLOOR_FROM_IMAGE, *pixel(2.6, 1.0)), to_floor(FLOOR_FROM_IMAGE, *pixel(2.6, 12.0))
        self.assertGreater(far.covariance[1, 1], 2 * near.covariance[1, 1])
        step = to_floor(FLOOR_FROM_IMAGE, *(pixel(2.6, 12.0) + np.array([0.0, 3.0])))
        self.assertAlmostEqual(abs(step.point[1] - far.point[1]), np.sqrt(far.covariance[1, 1]), delta=0.1 * np.sqrt(far.covariance[1, 1]))

    def test_the_frame_rate_is_the_nearest_usual_one(self):
        seconds = np.arange(0.0, 300.0, 7.0)
        self.assertEqual(frame_rate(list(seconds), list(seconds * 25.03 + 40)), 25.0)
        self.assertEqual(frame_rate(list(seconds), list(seconds * 29.97)), 29.97)


class ReadTests(unittest.TestCase):
    def read(self, above_feet_px=0.0):
        with tempfile.TemporaryDirectory() as folder:
            write_match(Path(folder), rallies())
            return read(Path(folder), above_feet_px)

    def test_rallies_with_their_ends_and_winners(self):
        (match,) = self.read()
        self.assertEqual(match.players, ("Player A", "Player B"))
        self.assertEqual(match.fps, 25.0)
        first, second = match.rallies
        self.assertEqual([hit.name for hit in first.hits], ["serve", "net", "smash"])
        self.assertEqual((first.end, first.how, first.winner, first.score), ("in", "winner", "A", (1, 0)))
        self.assertAlmostEqual(first.hits[1].time_ms - first.hits[0].time_ms, 1000.0)
        np.testing.assert_allclose(first.hits[2].hitter_at.point, (2.5, 5.0), atol=1e-3)
        np.testing.assert_allclose(first.hits[2].goes_to.point, (4.9, 12.0), atol=1e-3)
        # A serve not known starts the rally at the return.
        self.assertEqual([hit.name for hit in second.hits], ["lift"])
        self.assertEqual((second.end, second.winner), ("out", "A"))

    def test_rallies_set_aside_and_the_score_checked(self):
        (match,) = self.read()
        self.assertEqual(dict(match.set_aside), {"no end": 1, "a shot not known": 1, "winner not the end's": 1})
        self.assertEqual((match.score_agrees, match.score_checked), (2, 2))

    def test_players_clicked_above_their_feet_are_put_back_on_them(self):
        rows = rallies()[:3]
        for row in rows:
            for key in ("player_location_y", "opponent_location_y"):
                row[key] = f"{float(row[key]) - 4.0:.3f}"
        with tempfile.TemporaryDirectory() as folder:
            write_match(Path(folder), rows)
            (raw,), (moved,) = read(Path(folder), 0.0), read(Path(folder), 4.0)
        serve = moved.rallies[0].hits[0]
        np.testing.assert_allclose(serve.hitter_at.point, (2.6, 3.5), atol=1e-3)
        self.assertGreater(raw.rallies[0].hits[0].hitter_at.point[1], 3.5)  # a near player looks nearer the net
        self.assertGreater(raw.rallies[0].hits[0].receiver_at.point[1], 9.9)  # a far one deeper
        np.testing.assert_allclose(serve.goes_to.point, (2.0, 8.8), atol=1e-3)  # the shuttle is not moved

    def test_the_checks(self):
        report = check(self.read())
        self.assertEqual(report["rallies"], 2)
        self.assertEqual(report["ends"]["winner"]["agree"], 1)
        self.assertEqual(report["ends"]["out"]["agree"], 1)
        self.assertEqual(report["scoreAgrees"], [2, 2])


class HeldOutTests(unittest.TestCase):
    def test_test_matches_are_fixed_and_apart_from_training(self):
        men, women = ["Man 1", "Man 2", "Man 3", "Man 4"], ["Carolina MARIN", "An Se Young", "Michelle LI"]
        pairs = [(men[i % 4], men[(i + 1) % 4]) for i in range(8)] + [(women[i % 3], women[(i + 1) % 3]) for i in range(6)]
        matches = [Match(f"match {i}", pair, 25.0) for i, pair in enumerate(pairs)]
        split = held_out(matches)
        self.assertEqual(split, held_out(list(reversed(matches))))
        self.assertEqual(len(split["players"]), 2)
        self.assertIn(split["players"][1], women)
        everything = split["stressed"] + split["test"] + split["train"]
        self.assertEqual(sorted(everything), sorted(match.video for match in matches))
        by_video = {match.video: match for match in matches}
        for video in split["test"] + split["train"]:
            self.assertFalse(set(by_video[video].players) & set(split["players"]))
        self.assertEqual(len(split["test"]), round(0.25 * (len(matches) - len(split["stressed"]))))


if __name__ == "__main__":
    unittest.main()
