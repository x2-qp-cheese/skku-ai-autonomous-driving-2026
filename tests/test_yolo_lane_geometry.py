import unittest

import numpy as np

from skku_autocar.estimation.lane_geometry import LaneGeometry, LaneGeometryConfig, MaskLaneGeometryEstimator
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from skku_autocar.planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig
from skku_autocar.runtime.yolo_drive_app import (
    CommandSafetyFilter,
    build_bev_lane_config,
    build_follower_config,
    parse_args,
)
from skku_autocar.types import ControlCommand


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

    def test_center_lock_forces_minimum_steering_near_center(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=0.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                center_recovery_error_threshold=1.0,
                center_lock_enabled=True,
                center_lock_error_threshold=0.05,
                center_lock_min_steering=80,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        lane = lane_geometry(lateral_error_norm=0.06, heading_error=0.0)

        command = follower.plan(lane)

        self.assertEqual(command.steering, 80)
        self.assertIn("center_lock", command.reason)

    def test_center_lock_keeps_deadband_unforced(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=0.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                center_recovery_error_threshold=1.0,
                center_lock_enabled=True,
                center_lock_error_threshold=0.05,
                center_lock_min_steering=80,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        lane = lane_geometry(lateral_error_norm=0.03, heading_error=0.0)

        command = follower.plan(lane)

        self.assertEqual(command.steering, 0)

    def test_center_lock_ignores_conflicting_heading(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=0.0,
                kd_lateral=0.0,
                kp_heading=200.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                center_recovery_error_threshold=1.0,
                center_lock_enabled=True,
                center_lock_error_threshold=0.05,
                center_lock_min_steering=80,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        lane = lane_geometry(lateral_error_norm=-0.06, heading_error=1.0)

        command = follower.plan(lane)

        self.assertEqual(command.steering, -80)

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

    def test_lane_lost_hold_releases_cached_steering(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                base_speed=100,
                speed_curve_slowdown=0,
                kp_lateral=100.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                center_recovery_error_threshold=1.0,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
                lane_lost_steering_release_rate_limit=30,
            )
        )
        follower.plan(lane_geometry(lateral_error_norm=1.0, heading_error=0.0))
        lost = LaneGeometry(
            found=False,
            center_x=0.0,
            vehicle_center_x=0.0,
            target_y=0.0,
            lateral_error_px=0.0,
            lateral_error_norm=0.0,
            heading_error=0.0,
            confidence=0.0,
            reason="no_sampled_rows",
        )

        command = follower.plan(lost)

        self.assertEqual(command.speed, 100)
        self.assertEqual(command.steering, 70)

    def test_bev_runtime_uses_common_parameter_aliases(self):
        args = parse_args(
            [
                "--lookahead",
                "0.70",
                "--sample-top",
                "0.35",
                "--sample-bottom",
                "0.76",
                "--lateral-priority-threshold",
                "0.25",
                "--curve-strength-alpha",
                "0.25",
                "--center-lock",
                "on",
                "--center-lock-error-threshold",
                "0.04",
                "--center-lock-min-steering",
                "95",
            ]
        )

        bev_config = build_bev_lane_config(args)
        follower_config = build_follower_config(args)

        self.assertAlmostEqual(bev_config.lookahead_y_ratio, 0.70)
        self.assertAlmostEqual(bev_config.sample_top_y_ratio, 0.35)
        self.assertAlmostEqual(bev_config.sample_bottom_y_ratio, 0.76)
        self.assertAlmostEqual(follower_config.lateral_priority_threshold, 0.25)
        self.assertAlmostEqual(follower_config.curve_strength_alpha, 0.25)
        self.assertTrue(follower_config.center_lock_enabled)
        self.assertAlmostEqual(follower_config.center_lock_error_threshold, 0.04)
        self.assertEqual(follower_config.center_lock_min_steering, 95)

    def test_virtual_lane_command_is_capped(self):
        args = parse_args(
            [
                "--virtual-lane-max-steering",
                "110",
                "--virtual-lane-speed-cap",
                "220",
                "--virtual-lane-warmup-frames",
                "0",
                "--virtual-lane-min-reliable-frames",
                "0",
            ]
        )
        safety = CommandSafetyFilter(args)
        lane = lane_geometry(lateral_error_norm=-0.5, heading_error=-1.0)
        command = ControlCommand(speed=255, steering=-150, brake=False, reason="test")

        guarded = safety.apply(DummyMask("virtual-lane-center+right-lane-side"), lane, command, True)

        self.assertEqual(guarded.speed, 220)
        self.assertEqual(guarded.steering, -110)
        self.assertIn("virtual_cap", guarded.reason)

    def test_virtual_lane_warmup_reuses_last_reliable_steering(self):
        args = parse_args(
            [
                "--virtual-lane-max-steering",
                "110",
                "--virtual-lane-speed-cap",
                "220",
                "--virtual-lane-warmup-frames",
                "2",
                "--virtual-lane-min-reliable-frames",
                "0",
            ]
        )
        safety = CommandSafetyFilter(args)
        lane = lane_geometry(lateral_error_norm=0.1, heading_error=0.0)
        safety.apply(DummyMask("lane-center+right-lane-side"), lane, ControlCommand(255, 35), True)

        guarded = safety.apply(
            DummyMask("virtual-lane-center+right-lane-side"),
            lane,
            ControlCommand(255, -150),
            True,
        )

        self.assertEqual(guarded.speed, 220)
        self.assertEqual(guarded.steering, 35)
        self.assertIn("virtual_hold", guarded.reason)

    def test_lane_lost_command_speed_is_capped(self):
        args = parse_args(["--lane-lost-speed-cap", "200"])
        safety = CommandSafetyFilter(args)
        lost = LaneGeometry(
            found=False,
            center_x=0.0,
            vehicle_center_x=0.0,
            target_y=0.0,
            lateral_error_px=0.0,
            lateral_error_norm=0.0,
            heading_error=0.0,
            confidence=0.0,
            reason="no_sampled_rows",
        )

        guarded = safety.apply(
            DummyMask("lane-center+right-lane-side"),
            lost,
            ControlCommand(255, 40, brake=False, reason="lane_lost_hold"),
            True,
        )

        self.assertEqual(guarded.speed, 200)
        self.assertEqual(guarded.steering, 40)
        self.assertIn("lane_lost_speed_cap", guarded.reason)

    def test_fixed_speed_overrides_safety_speed_caps(self):
        args = parse_args(
            [
                "--speed",
                "255",
                "--fixed-speed",
                "on",
                "--virtual-lane-speed-cap",
                "210",
                "--lane-lost-speed-cap",
                "190",
                "--virtual-lane-warmup-frames",
                "0",
                "--virtual-lane-min-reliable-frames",
                "0",
            ]
        )
        safety = CommandSafetyFilter(args)
        lane = lane_geometry(lateral_error_norm=0.1, heading_error=0.0)

        reliable = safety.apply(
            DummyMask("lane-center+right-lane-side"),
            lane,
            ControlCommand(180, 20, brake=False, reason="test"),
            True,
        )
        virtual = safety.apply(
            DummyMask("virtual-lane-center+right-lane-side"),
            lane,
            ControlCommand(170, 30, brake=False, reason="test"),
            True,
        )

        self.assertEqual(reliable.speed, 255)
        self.assertEqual(virtual.speed, 255)
        self.assertIn("fixed_speed", reliable.reason)

    def test_virtual_lane_blends_and_rate_limits_steering(self):
        args = parse_args(
            [
                "--virtual-lane-warmup-frames",
                "0",
                "--virtual-lane-steering-blend",
                "0.25",
                "--virtual-lane-max-steering-step",
                "10",
                "--virtual-lane-max-steering",
                "110",
                "--virtual-lane-min-reliable-frames",
                "0",
            ]
        )
        safety = CommandSafetyFilter(args)
        lane = lane_geometry(lateral_error_norm=0.1, heading_error=0.0)
        safety.apply(DummyMask("lane-center+right-lane-side"), lane, ControlCommand(255, 20), True)

        guarded = safety.apply(
            DummyMask("virtual-lane-center+right-lane-side"),
            lane,
            ControlCommand(255, -150, brake=False, reason="test"),
            True,
        )

        self.assertEqual(guarded.steering, 10)
        self.assertIn("virtual_blend", guarded.reason)

    def test_virtual_lane_scales_center_lock_steering(self):
        args = parse_args(
            [
                "--virtual-lane-warmup-frames",
                "0",
                "--virtual-lane-steering-blend",
                "1.0",
                "--virtual-lane-max-steering-step",
                "0",
                "--virtual-lane-max-steering",
                "150",
                "--virtual-lane-center-lock-scale",
                "0.5",
                "--virtual-lane-min-reliable-frames",
                "0",
            ]
        )
        safety = CommandSafetyFilter(args)
        lane = lane_geometry(lateral_error_norm=-0.2, heading_error=0.0)

        guarded = safety.apply(
            DummyMask("virtual-lane-center+right-lane-side"),
            lane,
            ControlCommand(255, -120, brake=False, reason="yolo_lane_follow:center_lock"),
            True,
        )

        self.assertEqual(guarded.steering, -60)
        self.assertIn("virtual_center_lock_scale", guarded.reason)

    def test_virtual_lane_bootstrap_holds_last_reliable_command(self):
        args = parse_args(
            [
                "--virtual-lane-min-reliable-frames",
                "3",
                "--virtual-lane-bootstrap-speed-cap",
                "140",
                "--virtual-lane-max-steering",
                "90",
            ]
        )
        safety = CommandSafetyFilter(args)
        lane = lane_geometry(lateral_error_norm=0.1, heading_error=0.0)
        safety.apply(DummyMask("lane-center+right-lane-side"), lane, ControlCommand(210, 35), True)

        guarded = safety.apply(
            DummyMask("lane-center+virtual-right-side"),
            lane,
            ControlCommand(210, 120, brake=False, reason="yolo_lane_follow"),
            True,
        )

        self.assertEqual(guarded.speed, 140)
        self.assertEqual(guarded.steering, 35)
        self.assertIn("virtual_bootstrap", guarded.reason)


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


class DummyMask:
    def __init__(self, class_name: str):
        self.class_name = class_name


if __name__ == "__main__":
    unittest.main()
