# worker/test_detect_poses.py
import unittest

import numpy as np

from detect_poses import BIG, BLAZEPOSE_FOR_COCO, Crop, Pose, _pose, agrees, court_crops, detect_people, merge, person_crops
from evaluation.hits_benchmark import FOCAL_PX, SIZE, VIEWS
from geometry.camera import Camera
from geometry.court_model import COURT_LENGTH_M, SINGLES_WIDTH_M
from geometry.skeleton import JOINTS
from simulation.people import skeleton


def frame_of_places(size):
    """A BGR frame whose every pixel says where it is (x * 1024 + y over its three bytes), so a
    pretend landmarker can tell which part of the frame it was shown."""
    ys, xs = np.mgrid[0 : size[1], 0 : size[0]]
    code = xs * 1024 + ys
    rgb = np.stack([(code >> 16) & 255, (code >> 8) & 255, code & 255], axis=-1).astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def landmarker_that_sees(people, camera, smallest=0.3):
    """A pretend landmarker: in whatever part of the frame it is shown, it finds each of these
    bodies that is wholly inside and at least `smallest` of its height, as BlazePose would
    report them (33 landmarks normalised to what it was shown; world ones about the hips'
    middle, along the camera's axes)."""

    def look(rgb, most):
        code = (int(rgb[0, 0, 0]) << 16) | (int(rgb[0, 0, 1]) << 8) | int(rgb[0, 0, 2])
        x0, y0 = code // 1024, code % 1024
        height, width = rgb.shape[:2]
        found = []
        for joints in people:
            pixels = camera.project(joints)
            box = np.array([*pixels.min(axis=0), *pixels.max(axis=0)])
            if box[0] < x0 or box[1] < y0 or box[2] > x0 + width or box[3] > y0 + height:
                continue
            if box[3] - box[1] < smallest * height:
                continue
            landmarks = np.zeros((33, 3))
            landmarks[:, :2] = ((pixels.mean(axis=0) - (x0, y0)) / (width, height))
            landmarks[list(BLAZEPOSE_FOR_COCO), :2] = (pixels - (x0, y0)) / (width, height)
            landmarks[list(BLAZEPOSE_FOR_COCO), 2] = 0.95
            in_camera = joints @ camera.rotation.T
            world = np.zeros((33, 3))
            world[list(BLAZEPOSE_FOR_COCO)] = in_camera - (in_camera[11] + in_camera[12]) / 2
            found.append((landmarks, world))
        return found[:most]

    return look


class PoseTests(unittest.TestCase):
    def setUp(self):
        position, target = VIEWS["behind"]
        self.camera = Camera.look_at(position, target, FOCAL_PX, SIZE)
        self.joints = skeleton((2.0, 3.0), np.pi / 2)

    def test_a_crop_s_landmarks_come_back_in_the_frame_s_pixels(self):
        low, high = self.camera.project(self.joints).min(axis=0), self.camera.project(self.joints).max(axis=0)
        crop = Crop(int(low[0]) - 40, int(low[1]) - 30, int(low[0]) + 360, int(low[1]) + 370)
        self.assertTrue(high[0] < crop.x1 and high[1] < crop.y1)
        look = landmarker_that_sees([self.joints], self.camera, smallest=0.0)
        found = look(frame_of_places(SIZE)[crop.y0 : crop.y1, crop.x0 : crop.x1, ::-1].copy(), 1)
        pose = _pose(*found[0], crop)
        np.testing.assert_allclose(pose.keypoints[:, :2], self.camera.project(self.joints), atol=1e-6)
        self.assertTrue(np.all(pose.keypoints[:, 2] == 0.95))

    def test_the_3d_pose_is_about_the_hips_middle_in_the_skeleton_s_joints(self):
        in_camera = self.joints @ self.camera.rotation.T
        world = np.zeros((33, 3))
        world[list(BLAZEPOSE_FOR_COCO)] = in_camera + 5.0  # wherever the landmarker puts its origin
        landmarks = np.zeros((33, 3))
        landmarks[list(BLAZEPOSE_FOR_COCO), :2] = self.camera.project(self.joints) / SIZE
        landmarks[:, 2] = 0.9
        pose = _pose(landmarks, world, Crop(0, 0, *SIZE))
        expected = in_camera[list(JOINTS)] - (in_camera[11] + in_camera[12]) / 2
        np.testing.assert_allclose(pose.world, expected, atol=1e-9)

    def test_a_3d_pose_must_agree_with_its_2d_joints(self):
        pixels = self.camera.project(self.joints)
        keypoints = np.column_stack([pixels, np.full(17, 0.9)])
        in_camera = self.joints @ self.camera.rotation.T
        world = in_camera[list(JOINTS)] - (in_camera[11] + in_camera[12]) / 2
        self.assertTrue(agrees(keypoints, world))
        self.assertFalse(agrees(keypoints, world * (-1, 1, 1)))  # left and right swapped
        self.assertFalse(agrees(keypoints, world * (1, -1, 1)))  # upside down
        self.assertFalse(agrees(keypoints, world[::-1]))  # the joints in the wrong order

    def test_one_person_seen_twice_is_kept_once_as_seen_best(self):
        keypoints = np.column_stack([np.zeros((17, 2)), np.full(17, 0.6)])
        a = Pose(keypoints, np.array([100, 100, 200, 300.0]), None)
        b = Pose(keypoints + (0, 0, 0.2), np.array([105, 98, 204, 305.0]), None)
        c = Pose(keypoints, np.array([400, 100, 500, 300.0]), None)
        kept = merge([a, b, c])
        self.assertEqual(len(kept), 2)
        self.assertIs(kept[0], b)
        self.assertTrue(any(k is c for k in kept))


class CropTests(unittest.TestCase):
    def test_every_place_a_player_could_stand_is_seen_whole_in_some_crop(self):
        for view, (position, target) in VIEWS.items():
            camera = Camera.look_at(position, target, FOCAL_PX, SIZE)
            crops = court_crops(camera, SIZE)
            self.assertLessEqual(len(crops), 40, view)
            for crop in crops:
                self.assertEqual(crop.x1 - crop.x0, crop.y1 - crop.y0)
                self.assertTrue(0 <= crop.x0 and crop.x1 <= SIZE[0] and 0 <= crop.y0 and crop.y1 <= SIZE[1])
            for x in np.linspace(-0.5, SINGLES_WIDTH_M + 0.5, 7):
                for y in np.linspace(-1.0, COURT_LENGTH_M + 1.0, 15):
                    pixels = camera.project(skeleton((x, y), np.pi / 2))
                    low, high = pixels.min(axis=0), pixels.max(axis=0)
                    tall = high[1] - low[1]
                    if low[0] < 0 or low[1] < 0 or high[0] > SIZE[0] or high[1] > SIZE[1] or tall > BIG * SIZE[1] / 1.5:
                        continue  # out of the frame, or big enough for the whole frame
                    inside = [c for c in crops if c.x0 <= low[0] and c.y0 <= low[1] and high[0] <= c.x1 and high[1] <= c.y1]
                    self.assertTrue(inside, f"{view}: nobody looks at ({x:.1f}, {y:.1f})")
                    self.assertGreaterEqual(max(tall / (c.y1 - c.y0) for c in inside), 0.2, f"{view}: ({x:.1f}, {y:.1f}) seen too small")

    def test_a_person_on_the_court_is_looked_at_closely_next_frame(self):
        position, target = VIEWS["behind"]
        camera = Camera.look_at(position, target, FOCAL_PX, SIZE)
        pixels = camera.project(skeleton((3.0, 10.0), -np.pi / 2))
        pose = Pose(np.column_stack([pixels, np.full(17, 0.9)]), np.array([*pixels.min(axis=0), *pixels.max(axis=0)]), None)
        crops = person_crops([pose], camera, SIZE)
        self.assertEqual(len(crops), 1)
        crop = crops[0]
        self.assertTrue(crop.x0 <= pose.box[0] and crop.y0 <= pose.box[1] and pose.box[2] <= crop.x1 and pose.box[3] <= crop.y1)
        off_court = Pose(pose.keypoints + (900.0, 0.0, 0.0), pose.box + (900.0, 0.0, 900.0, 0.0), None)
        self.assertEqual(person_crops([off_court], camera, SIZE), [])


class DetectTests(unittest.TestCase):
    def test_the_far_player_is_found_in_a_crop_when_the_whole_frame_misses_them(self):
        position, target = VIEWS["behind"]
        camera = Camera.look_at(position, target, FOCAL_PX, SIZE)
        near, far = skeleton((2.5, 2.0), np.pi / 2), skeleton((3.0, 11.0), -np.pi / 2)
        look = landmarker_that_sees([near, far], camera, smallest=0.2)  # near: a quarter of the frame; far: an eighth
        frame = frame_of_places(SIZE)
        whole_only = detect_people(look, frame, [])
        self.assertEqual(len(whole_only), 1)
        poses = detect_people(look, frame, court_crops(camera, SIZE))
        self.assertEqual(len(poses), 2)
        for joints in (near, far):
            truth = camera.project(joints)
            best = min(poses, key=lambda p: np.abs(p.keypoints[:, :2] - truth).max())
            np.testing.assert_allclose(best.keypoints[:, :2], truth, atol=1e-6)
            self.assertIsNotNone(best.world)


if __name__ == "__main__":
    unittest.main()
