import unittest

import numpy as np

from skku_autocar.estimation.lane_geometry import LaneGeometry, LaneGeometryConfig, MaskLaneGeometryEstimator
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from skku_autocar.planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig


class YoloLaneGeometryTest(unittest.TestCase):
    def test_centered_mask_has_near_zero_error(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[45:95, 70:130] = 255

        lane = MaskLaneGeometryEstimator().estimate(mask, (100, 200, 3))

        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.lateral_error_norm, 0.0, delta=0.01)

    def test_positive_vehicle_center_offset_requests_left_steering(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[45:95, 70:130] = 255

        lane = MaskLaneGeometryEstimator(
            LaneGeometryConfig(vehicle_center_x_offset_ratio=0.05)
        ).estimate(mask, (100, 200, 3))
        command = YoloLaneFollower(
            YoloLaneFollowerConfig(steering_rate_limit=120)
        ).plan(lane)

        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.vehicle_center_x, 110.0, delta=0.1)
        self.assertLess(lane.lateral_error_norm, 0.0)
        self.assertLess(command.steering, 0)

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
                min_steering_rate_limit=500,
                max_steering=500,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                center_recovery_error_threshold=1.0,
            )
        )
        first = lane_geometry(lateral_error_norm=0.10, heading_error=0.0)
        second = lane_geometry(lateral_error_norm=0.30, heading_error=0.0)

        first_command = follower.plan(first)
        second_command = follower.plan(second)

        self.assertEqual(first_command.steering, 10)
        self.assertEqual(second_command.steering, 38)

    def test_lateral_target_overrides_conflicting_heading(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=170.0,
                kd_lateral=0.0,
                kp_heading=55.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                max_steering=500,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
            )
        )
        lane = lane_geometry(lateral_error_norm=0.40, heading_error=-1.0)

        command = follower.plan(lane)

        self.assertGreater(command.steering, 0)

    def test_curve_strength_ramps_steering_response(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=100.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                max_steering=500,
                curve_strength_alpha=0.5,
                straight_steering_scale=0.4,
                curve_steering_scale=1.0,
            )
        )
        lane = lane_geometry(lateral_error_norm=0.50, heading_error=0.0)

        first_command = follower.plan(lane)
        second_command = follower.plan(lane)

        self.assertLess(first_command.steering, second_command.steering)

    def test_curve_slows_speed_before_steering_ramp_finishes(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                base_speed=100,
                min_curve_speed=40,
                speed_curve_slowdown=50,
                kp_lateral=0.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
            )
        )
        straight = lane_geometry(lateral_error_norm=0.0, heading_error=0.0)
        curve = lane_geometry(lateral_error_norm=0.10, heading_error=1.0)

        straight_command = follower.plan(straight)
        curve_command = follower.plan(curve)

        self.assertLess(curve_command.speed, straight_command.speed)

    def test_center_recovery_forces_minimum_steering(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=10.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                center_recovery_error_threshold=0.10,
                center_recovery_min_steering=80,
                center_recovery_steering_boost=1.0,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        lane = lane_geometry(lateral_error_norm=-0.65, heading_error=0.0)

        command = follower.plan(lane)

        self.assertLessEqual(command.steering, -80)

    def test_center_recovery_limits_speed(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                base_speed=100,
                min_curve_speed=20,
                speed_curve_slowdown=0,
                center_recovery_error_threshold=0.10,
                center_recovery_max_speed=45,
            )
        )
        lane = lane_geometry(lateral_error_norm=0.65, heading_error=0.0)

        command = follower.plan(lane)

        self.assertLessEqual(command.speed, 45)

    def test_release_rate_limit_slows_unwinding_same_direction(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=200.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=20,
                center_recovery_error_threshold=1.0,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        first = lane_geometry(lateral_error_norm=0.50, heading_error=0.0)
        second = lane_geometry(lateral_error_norm=0.05, heading_error=0.0)

        first_command = follower.plan(first)
        second_command = follower.plan(second)

        self.assertEqual(first_command.steering, 100)
        self.assertEqual(second_command.steering, 80)

    def test_release_rate_limit_does_not_block_opposite_turn(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=200.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=20,
                center_recovery_error_threshold=1.0,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        first = lane_geometry(lateral_error_norm=0.50, heading_error=0.0)
        second = lane_geometry(lateral_error_norm=-0.50, heading_error=0.0)

        follower.plan(first)
        second_command = follower.plan(second)

        self.assertEqual(second_command.steering, -100)


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
