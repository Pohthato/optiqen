# worker/test_golden.py
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from evaluation.golden import COCO_JOINTS, GOLDEN_SHOT_LABELS, keypoint_frame, load_golden_dir, validate_golden

VALID = {
    "schemaVersion": 1,
    "clipId": "clip-001",
    "sourceFps": 60,
    "imageSize": [1920, 1080],
    "courtKeypoints": [{"name": "back1_sl", "x": 612.0, "y": 301.5}],
    "contacts": [{"timeMs": 1200}, {"timeMs": 2100}],
    "shots": [
        {"timeMs": 1200, "label": "serve", "landing": {"x": 2.4, "y": 9.8}},
        {"timeMs": 2100, "label": "clear", "landing": None},
    ],
    "rallies": [{"startMs": 800, "endMs": 4300, "winner": "near"}],
}


def corrupted(**changes):
    doc = copy.deepcopy(VALID)
    doc.update(changes)
    return doc


class ValidateGoldenTests(unittest.TestCase):
    def test_valid_document_has_no_errors(self):
        self.assertEqual(validate_golden(VALID), [])

    def test_labels_include_the_worker_shot_classes(self):
        for label in ("smash", "clear", "drop", "net", "lift", "drive", "push", "serve"):
            self.assertIn(label, GOLDEN_SHOT_LABELS)

    def test_reports_each_kind_of_problem(self):
        cases = {
            "schemaVersion": corrupted(schemaVersion=2),
            "clipId": corrupted(clipId=" "),
            "sourceFps": corrupted(sourceFps=0),
            "imageSize": corrupted(imageSize=[1920]),
            "keypoint": corrupted(courtKeypoints=[{"name": "nope", "x": 1, "y": 1}]),
            "outside": corrupted(courtKeypoints=[{"name": "back1_sl", "x": 5000, "y": 1}]),
            "contact": corrupted(contacts=[{"timeMs": -5}]),
            "label": corrupted(shots=[{"timeMs": 10, "label": "bogus", "landing": None}]),
            "landing": corrupted(shots=[{"timeMs": 10, "label": "clear", "landing": {"x": 90, "y": 1}}]),
            "before endms": corrupted(rallies=[{"startMs": 500, "endMs": 400, "winner": "near"}]),
            "winner": corrupted(rallies=[{"startMs": 1, "endMs": 400, "winner": "left"}]),
        }
        for needle, doc in cases.items():
            errors = validate_golden(doc)
            self.assertTrue(any(needle.lower() in error.lower() for error in errors), (needle, errors))

    def test_non_object_and_missing_lists_are_rejected(self):
        self.assertTrue(validate_golden([]))
        broken = copy.deepcopy(VALID)
        del broken["contacts"]
        self.assertTrue(any("contacts" in error for error in validate_golden(broken)))

    def test_keypoint_frame_time_is_optional_but_must_be_valid(self):
        with_time = corrupted(courtKeypoints=[{"name": "back1_sl", "x": 612.0, "y": 301.5, "timeMs": 4000}])
        self.assertEqual(validate_golden(with_time), [])
        negative = corrupted(courtKeypoints=[{"name": "back1_sl", "x": 612.0, "y": 301.5, "timeMs": -1}])
        self.assertTrue(any("timeMs" in error for error in validate_golden(negative)))

    def test_bool_and_nan_are_not_numbers(self):
        self.assertTrue(validate_golden(corrupted(sourceFps=True)))
        self.assertTrue(validate_golden(corrupted(contacts=[{"timeMs": float("nan")}])))


def pose(**changes):
    item = {"timeMs": 1200, "player": "near", "keypoints": [[900.0 + i, 500.0 + i, 2] for i in range(17)]}
    item.update(changes)
    return item


class PoseLabelTests(unittest.TestCase):
    def test_coco_joint_order(self):
        self.assertEqual(len(COCO_JOINTS), 17)
        self.assertEqual((COCO_JOINTS[0], COCO_JOINTS[9], COCO_JOINTS[16]), ("nose", "left_wrist", "right_ankle"))

    def test_poses_and_selected_player_are_optional_but_validated(self):
        self.assertEqual(validate_golden(corrupted(selectedPlayer="near", poses=[pose()])), [])
        hidden = pose(keypoints=[[0.0, 0.0, 0]] * 17)
        self.assertEqual(validate_golden(corrupted(poses=[hidden])), [])

    def test_reports_each_kind_of_pose_problem(self):
        cases = {
            "selectedplayer": corrupted(selectedPlayer="left"),
            "player": corrupted(poses=[pose(player="umpire")]),
            "17 coco": corrupted(poses=[pose(keypoints=[[1.0, 1.0, 2]] * 16)]),
            "visibility": corrupted(poses=[pose(keypoints=[[1.0, 1.0, 3]] * 17)]),
            "outside the image": corrupted(poses=[pose(keypoints=[[5000.0, 1.0, 2]] * 17)]),
            "timems": corrupted(poses=[pose(timeMs=-1)]),
        }
        for needle, doc in cases.items():
            errors = validate_golden(doc)
            self.assertTrue(any(needle in error.lower() for error in errors), (needle, errors))


class AnswerKeyConsistencyTests(unittest.TestCase):
    def test_a_pose_must_sit_on_a_labelled_hit(self):
        errors = validate_golden(corrupted(poses=[pose(timeMs=1300)]))
        self.assertTrue(any("contact" in error for error in errors), errors)

    def test_a_hit_has_at_most_one_pose(self):
        errors = validate_golden(corrupted(poses=[pose(), pose()]))
        self.assertTrue(any("duplicate" in error for error in errors), errors)

    def test_poses_belong_to_the_selected_player(self):
        errors = validate_golden(corrupted(selectedPlayer="near", poses=[pose(player="far")]))
        self.assertTrue(any("selectedplayer" in error.lower() for error in errors), errors)

    def test_court_point_source_is_one_of_the_known_kinds(self):
        good = corrupted(courtKeypoints=[{"name": "back1_sl", "x": 612.0, "y": 301.5, "timeMs": 0, "source": "proposed"}])
        self.assertEqual(validate_golden(good), [])
        bad = corrupted(courtKeypoints=[{"name": "back1_sl", "x": 612.0, "y": 301.5, "source": "guessed"}])
        self.assertTrue(any("source" in error for error in validate_golden(bad)))


class ShuttlePointTests(unittest.TestCase):
    def test_shuttle_points_are_optional_but_validated(self):
        points = [{"timeMs": 1167, "x": 640.5, "y": 210.0}, {"timeMs": 1233, "visible": False}]
        self.assertEqual(validate_golden(corrupted(shuttlePoints=points)), [])

    def test_reports_each_kind_of_shuttle_problem(self):
        cases = {
            "outside the image": [{"timeMs": 100, "x": 5000.0, "y": 10.0}],
            "x and y": [{"timeMs": 100, "x": 5.0}],
            "timems": [{"timeMs": -5, "x": 5.0, "y": 5.0}],
            "duplicate": [{"timeMs": 100, "x": 5.0, "y": 5.0}, {"timeMs": 100, "visible": False}],
            "visible": [{"timeMs": 100, "visible": "no"}],
            "not seen": [{"timeMs": 100, "visible": False, "x": 5.0, "y": 5.0}],
        }
        for needle, points in cases.items():
            errors = validate_golden(corrupted(shuttlePoints=points))
            self.assertTrue(any(needle in error.lower() for error in errors), (needle, errors))
        self.assertTrue(validate_golden(corrupted(shuttlePoints="all of them")))


class KeypointFrameTests(unittest.TestCase):
    def test_picks_the_frame_with_the_most_placed_points(self):
        doc = corrupted(courtKeypoints=[
            {"name": "back0_sl", "x": 10.0, "y": 20.0, "timeMs": 500},
            {"name": "back0_sl", "x": 11.0, "y": 21.0, "timeMs": 1000, "source": "clicked"},
            {"name": "back0_sr", "x": 30.0, "y": 20.0, "timeMs": 1000, "source": "adjusted"},
            {"name": "back1_sr", "x": 30.0, "y": 5.0, "timeMs": 500, "source": "proposed"},
            {"name": "back1_sl", "x": 10.0, "y": 5.0, "timeMs": 500, "source": "proposed"},
        ])
        time_ms, points = keypoint_frame(doc)
        self.assertEqual(time_ms, 1000)
        self.assertEqual(points, {"back0_sl": (11.0, 21.0), "back0_sr": (30.0, 20.0)})

    def test_proposed_points_alone_are_no_frame(self):
        doc = corrupted(courtKeypoints=[{"name": "back0_sl", "x": 1.0, "y": 2.0, "timeMs": 0, "source": "proposed"}])
        self.assertIsNone(keypoint_frame(doc))
        self.assertIsNone(keypoint_frame(corrupted(courtKeypoints=[])))


class LoadGoldenDirTests(unittest.TestCase):
    def test_loads_valid_files_and_reports_all_bad_ones(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "a.json").write_text(json.dumps(VALID))
            self.assertEqual([doc["clipId"] for doc in load_golden_dir(directory)], ["clip-001"])
            (directory / "b.json").write_text(json.dumps(corrupted(sourceFps=-1)))
            (directory / "c.json").write_text("{not json")
            with self.assertRaises(ValueError) as raised:
                load_golden_dir(directory)
            message = str(raised.exception)
            self.assertIn("b.json", message)
            self.assertIn("c.json", message)


class GoldenGuideTests(unittest.TestCase):
    def test_the_json_example_in_the_guide_is_valid(self):
        guide = Path(__file__).resolve().parent.parent / "docs" / "golden-set.md"
        match = re.search(r"```json\n(.*?)\n```", guide.read_text(encoding="utf-8"), re.S)
        self.assertIsNotNone(match)
        self.assertEqual(validate_golden(json.loads(match.group(1))), [])


if __name__ == "__main__":
    unittest.main()
