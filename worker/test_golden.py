# worker/test_golden.py
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from evaluation.golden import COCO_JOINTS, GOLDEN_SHOT_LABELS, load_golden_dir, validate_golden

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
