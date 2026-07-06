import unittest

import numpy as np

from skku_autocar.estimation.lane_geometry import LaneGeometry, MaskLaneGeometryEstimator
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from skku_autocar.planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig


class YoloLaneGeometryTest(unittest.TestCase):
    def test_centered_mask_has_near_zero_error(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[45:95, 70:130] = 255

        lane = MaskLaneGeometryEstimator().estimate(mask, (100, 200, 3))

        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.lateral_error_norm, 0.0, delta=0.01)

    def test_right_shifted_mask_requests_right_steering(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[45:95, 110:170] = 255

        lane = MaskLaneGeometryEstimator().estimate(mask, (100, 200, 3))
        command = YoloLaneFollower(
            YoloLaneFollowerConfig(steering_rate_limit=120)
        ).plan(lane)

        self.assertTrue(lane.lateral_error_norm > 0)
        self.assertTrue(command.steering > 0)

    def test_missing_mask_stops(self):
        lane = MaskLaneGeometryEstimator().estimate(None, (100, 200, 3))
        command = YoloLaneFollower().plan(lane)

        self.assertTrue(command.brake)
        self.assertEqual(command.speed, 0)

    def test_center_and_right_side_build_drive_corridor(self):
        center = np.zeros((100, 200), dtype=np.uint8)
        right = np.zeros((100, 200), dtype=np.uint8)
        center[:, 80:84] = 255
        right[:, 140:144] = 255

        segmenter = object.__new__(YoloLaneSegmenter)
        segmenter.config = YoloLaneConfig()
        candidates = [
            ("center", 0.8, 0, center, int(center.sum()), 2, "lane-center"),
            ("side", 0.8, 1, right, int(right.sum()), 3, "lane-side"),
        ]

        selected = segmenter._select_group(candidates, (100, 200))
        lane = MaskLaneGeometryEstimator().estimate(selected["mask"], (100, 200, 3))

        self.assertEqual(selected["class_name"], "lane-center+right-lane-side")
        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.center_x, 112.0, delta=2.0)

    def test_center_without_right_side_offsets_into_right_lane(self):
        center = np.zeros((100, 200), dtype=np.uint8)
        center[:, 80:84] = 255

        segmenter = object.__new__(YoloLaneSegmenter)
        segmenter.config = YoloLaneConfig(fallback_lane_width_ratio=0.30)
        candidates = [
            ("center", 0.8, 0, center, int(center.sum()), 2, "lane-center"),
        ]

        selected = segmenter._select_group(candidates, (100, 200))
        lane = MaskLaneGeometryEstimator().estimate(selected["mask"], (100, 200, 3))

        self.assertEqual(selected["class_name"], "lane-center+virtual-right-side")
        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.center_x, 112.0, delta=2.0)

    def test_right_side_without_center_offsets_left_into_lane(self):
        right = np.zeros((100, 200), dtype=np.uint8)
        right[:, 140:144] = 255

        segmenter = object.__new__(YoloLaneSegmenter)
        segmenter.config = YoloLaneConfig(fallback_lane_width_ratio=0.30)
        candidates = [
            ("side", 0.8, 0, right, int(right.sum()), 3, "lane-side"),
        ]

        selected = segmenter._select_group(candidates, (100, 200))
        lane = MaskLaneGeometryEstimator().estimate(selected["mask"], (100, 200, 3))

        self.assertEqual(selected["class_name"], "virtual-lane-center+right-lane-side")
        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.center_x, 112.0, delta=2.0)

    def test_lane_side_left_of_vehicle_still_offsets_left_into_road(self):
        side = np.zeros((100, 200), dtype=np.uint8)
        side[:, 80:84] = 255

        segmenter = object.__new__(YoloLaneSegmenter)
        segmenter.config = YoloLaneConfig(fallback_lane_width_ratio=0.30)
        candidates = [
            ("side", 0.8, 0, side, int(side.sum()), 3, "lane-side"),
        ]

        selected = segmenter._select_group(candidates, (100, 200))
        lane = MaskLaneGeometryEstimator().estimate(selected["mask"], (100, 200, 3))

        self.assertEqual(selected["class_name"], "virtual-lane-center+right-lane-side")
        self.assertTrue(lane.found)
        self.assertLess(lane.center_x, 82.0)
        self.assertAlmostEqual(lane.center_x, 52.0, delta=2.0)

    def test_pd_steering_adds_derivative_when_error_changes(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=100.0,
                kd_lateral=40.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                max_steering=500,
            )
        )
        first = lane_geometry(lateral_error_norm=0.10, heading_error=0.0)
        second = lane_geometry(lateral_error_norm=0.30, heading_error=0.0)

        first_command = follower.plan(first)
        second_command = follower.plan(second)

        self.assertEqual(first_command.steering, 10)
        self.assertEqual(second_command.steering, 38)


def lane_geometry(lateral_error_norm: float, heading_error: float) -> LaneGeometry:
    return LaneGeometry(
        found=True,
        center_x=0.0,
        vehicle_center_x=0.0,
        target_y=0.0,
        lateral_error_px=0.0,
        lateral_error_norm=lateral_error_norm,
        heading_error=heading_error,
        confidence=1.0,
        reason="test",
    )


if __name__ == "__main__":
    unittest.main()
