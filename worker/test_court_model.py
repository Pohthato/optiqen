# worker/test_court_model.py
import unittest

import numpy as np

from geometry.court_model import (
    CORNER_LABEL_TO_KEYPOINT,
    FLOOR_KEYPOINT_NAMES,
    KEYPOINTS,
    LINE_WIDTH_M,
    NET_Y_M,
    court_lines,
    net_tape_lines,
)


def on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    segment, offset = end - start, point - start
    if np.linalg.norm(np.cross(segment, offset)) > 1e-9:
        return False
    t = float(np.dot(offset, segment) / np.dot(segment, segment))
    return -1e-9 <= t <= 1 + 1e-9


class CourtModelTests(unittest.TestCase):
    def test_bwf_dimensions_run_to_the_outer_edges_of_the_lines(self):
        # Keypoints are line centres; adding one line width (half each side) gives the BWF size.
        self.assertAlmostEqual(LINE_WIDTH_M, 0.04)
        self.assertAlmostEqual(KEYPOINTS["back1_dr"][1] - KEYPOINTS["back0_dl"][1] + LINE_WIDTH_M, 13.40)
        self.assertAlmostEqual(KEYPOINTS["back0_dr"][0] - KEYPOINTS["back0_dl"][0] + LINE_WIDTH_M, 6.10)
        self.assertAlmostEqual(KEYPOINTS["back0_sr"][0] - KEYPOINTS["back0_sl"][0] + LINE_WIDTH_M, 5.18)
        np.testing.assert_allclose(KEYPOINTS["back0_sl"][:2], [0.02, 0.02])

    def test_service_lines(self):
        # Short service line: net-side edge 1.98 m from the net. Long service line: back edge
        # 0.76 m inside the baseline's outer edge.
        self.assertAlmostEqual(NET_Y_M - (KEYPOINTS["short0_c"][1] + 0.02), 1.98)
        self.assertAlmostEqual((KEYPOINTS["short1_c"][1] - 0.02) - NET_Y_M, 1.98)
        self.assertAlmostEqual(KEYPOINTS["long0_c"][1] - 0.02, 0.76)
        self.assertAlmostEqual(13.40 - (KEYPOINTS["long1_c"][1] + 0.02), 0.76)

    def test_centre_line_is_mid_court(self):
        self.assertAlmostEqual(KEYPOINTS["back0_c"][0], 2.59)

    def test_net_geometry(self):
        self.assertAlmostEqual(KEYPOINTS["post_left_top"][2], 1.55)
        self.assertAlmostEqual(KEYPOINTS["post_right_top"][2], 1.55)
        self.assertAlmostEqual(KEYPOINTS["net_centre_top"][2], 1.524)
        for name in ("post_left_base", "post_left_top", "post_right_base", "post_right_top", "net_centre_top"):
            self.assertAlmostEqual(KEYPOINTS[name][1], NET_Y_M)
        # The posts stand on the doubles sidelines.
        self.assertAlmostEqual(KEYPOINTS["post_left_base"][0], -0.44)
        self.assertAlmostEqual(KEYPOINTS["post_right_base"][0], 5.62)

    def test_keypoint_counts(self):
        self.assertEqual(len(FLOOR_KEYPOINT_NAMES), 30)
        self.assertEqual(len(KEYPOINTS), 35)
        self.assertTrue(all(KEYPOINTS[name][2] == 0.0 for name in FLOOR_KEYPOINT_NAMES))

    def test_every_floor_keypoint_is_a_line_intersection(self):
        lines = court_lines()
        for name in FLOOR_KEYPOINT_NAMES:
            count = sum(1 for _, start, end in lines if on_segment(KEYPOINTS[name], start, end))
            self.assertGreaterEqual(count, 2, name)

    def test_net_tape_runs_post_to_centre_half_a_tape_below_the_net_top(self):
        (_, left, centre), (_, centre_again, right) = net_tape_lines()
        np.testing.assert_allclose(centre, centre_again)
        np.testing.assert_allclose(left, KEYPOINTS["post_left_top"] - [0, 0, 0.0375])
        np.testing.assert_allclose(right, KEYPOINTS["post_right_top"] - [0, 0, 0.0375])
        np.testing.assert_allclose(centre, KEYPOINTS["net_centre_top"] - [0, 0, 0.0375])

    def test_ui_corner_labels_are_the_singles_corners(self):
        self.assertEqual(
            {label: tuple(KEYPOINTS[name][:2]) for label, name in CORNER_LABEL_TO_KEYPOINT.items()},
            {"nearLeft": (0.02, 0.02), "nearRight": (5.16, 0.02), "farRight": (5.16, 13.38), "farLeft": (0.02, 13.38)},
        )


if __name__ == "__main__":
    unittest.main()
