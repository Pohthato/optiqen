# worker/test_quality.py
import unittest

from geometry.camera import Camera
from geometry.quality import assess_geometry, court_visibility
from geometry.synthetic import LANDSCAPE, standard_scenes


class GeometryTierTests(unittest.TestCase):
    def tier(self, name: str) -> str:
        camera, size = standard_scenes()[name]
        return assess_geometry(camera, size).tier

    def test_elevated_views_with_the_whole_court_are_tier_a(self):
        for name in ("behind_elevated", "corner_elevated", "behind_portrait"):
            self.assertEqual(self.tier(name), "A", name)

    def test_low_camera_is_tier_b_and_says_to_raise_the_phone(self):
        camera, size = standard_scenes()["behind_low"]
        quality = assess_geometry(camera, size)
        self.assertEqual(quality.tier, "B")
        self.assertTrue(any("Raise the phone" in reason for reason in quality.reasons))

    def test_side_view_missing_part_of_the_court_is_tier_b(self):
        # side_elevated has ~83% of the court in frame, just under the tier A bar.
        camera, size = standard_scenes()["side_elevated"]
        quality = assess_geometry(camera, size)
        self.assertEqual(quality.tier, "B")
        self.assertTrue(any("out of frame" in reason for reason in quality.reasons))

    def test_close_camera_seeing_little_court_is_tier_c(self):
        camera = Camera.look_at((2.59, 1.0, 2.0), (2.59, 3.0, 0.0), 1300.0, LANDSCAPE)
        quality = assess_geometry(camera, LANDSCAPE)
        self.assertEqual(quality.tier, "C")
        self.assertLess(quality.visible_fraction, 0.40)

    def test_floor_level_camera_is_tier_c(self):
        camera = Camera.look_at((2.59, -3.0, 0.3), (2.59, 7.0, 0.3), 1300.0, LANDSCAPE)
        quality = assess_geometry(camera, LANDSCAPE)
        self.assertEqual(quality.tier, "C")
        self.assertLess(quality.elevation_deg, 5.0)

    def test_visibility_is_a_fraction(self):
        for camera, size in standard_scenes().values():
            self.assertTrue(0.0 <= court_visibility(camera, size) <= 1.0)


if __name__ == "__main__":
    unittest.main()
