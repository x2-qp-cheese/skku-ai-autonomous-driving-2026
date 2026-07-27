import unittest

from skku_autocar.estimation.lane_geometry import LaneGeometry
from skku_autocar.planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig
from skku_autocar.planning.lane_change import LaneChangeConfig, LaneChangeController
from skku_autocar.runtime.obstacle_mode import (
    build_lane_change_config,
    handle_lane_change_key,
)
from skku_autocar.runtime.yolo_drive_app import (
    CommandSafetyFilter,
    DrivePriorityController,
    build_bev_corridor_config,
    build_follower_config,
    parse_args,
)
from skku_autocar.types import ControlCommand


class YoloLaneGeometryTest(unittest.TestCase):
    def test_full_path_tracking_does_not_chase_extreme_far_dot(self):
        lane = LaneGeometry(
            found=True,
            center_x=250.0,
            vehicle_center_x=400.0,
            target_y=160.0,
            lateral_error_px=-150.0,
            lateral_error_norm=-0.375,
            heading_error=-0.04,
            confidence=1.0,
            reason="corridor",
            height=500.0,
            near_center_x=380.0,
            near_target_y=440.0,
            near_lateral_error_px=-20.0,
            near_lateral_error_norm=-0.05,
            path_points=tuple(
                (250.0 + 130.0 * index / 9.0, 160.0 + 280.0 * index / 9.0)
                for index in range(10)
            ),
        )
        path_follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )
        dot_follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                pure_pursuit=True,
                pure_pursuit_gain=330.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )

        path_command = path_follower.plan(lane)
        dot_command = dot_follower.plan(lane)

        self.assertLess(path_command.steering, 0)
        self.assertLess(
            abs(path_command.steering),
            abs(dot_command.steering),
        )
        self.assertIn("whole_centerline", path_command.reason)

    def test_path_tracking_uses_rise_response_on_direction_reversal(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=100.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=0.05,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )

        follower.plan(
            lane_geometry(
                lateral_error_norm=0.20,
                heading_error=0.0,
                near_lateral_error_norm=0.20,
            )
        )
        reversed_command = follower.plan(
            lane_geometry(
                lateral_error_norm=-0.20,
                heading_error=0.0,
                near_lateral_error_norm=-0.20,
            )
        )

        self.assertEqual(reversed_command.steering, -20)

    def test_path_center_recovery_acts_immediately_on_reliable_straight(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_error_threshold=0.10,
                path_center_recovery_heading_limit=0.12,
                path_center_recovery_min_steering=70.0,
                path_center_recovery_alpha=1.0,
                path_center_recovery_rate_limit=120,
                path_steering_rise_alpha=0.10,
                steering_rate_limit=35,
                min_steering_rate_limit=35,
                max_steering=500,
            )
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=-0.20,
                heading_error=-0.02,
                near_lateral_error_norm=-0.20,
                reason="corridor_tier1",
            )
        )

        self.assertEqual(command.steering, -70)

    def test_path_center_recovery_does_not_override_crosswalk_cache(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_min_steering=70.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=-0.20,
                heading_error=0.0,
                near_lateral_error_norm=-0.20,
                reason="corridor_tier1:crosswalk_priority_hold",
            )
        )

        self.assertEqual(command.steering, 0)

    def test_path_reversal_crosses_zero_in_one_frame_on_coherent_s_curve(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=120.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_error_threshold=2.0,
                path_steering_rise_alpha=0.20,
                path_steering_release_alpha=0.20,
                path_reversal_alpha=0.90,
                path_reversal_min_steering=25.0,
                path_reversal_min_geometry=0.08,
                path_reversal_rate_limit=160,
                steering_rate_limit=80,
                min_steering_rate_limit=35,
                steering_release_rate_limit=55,
                max_steering=500,
            )
        )
        follower.accept_applied_command(
            ControlCommand(speed=255, steering=-120, reason="previous_curve")
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=1.0,
                heading_error=0.0,
                near_lateral_error_norm=1.0,
                reason="corridor_tier1",
            )
        )

        self.assertGreater(command.steering, 0)

    def test_path_reversal_ignores_small_opposite_sign_noise(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=100.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_error_threshold=2.0,
                path_steering_rise_alpha=0.20,
                path_steering_release_alpha=0.20,
                path_reversal_alpha=0.90,
                path_reversal_min_steering=25.0,
                path_reversal_min_geometry=0.08,
                path_reversal_rate_limit=160,
                steering_rate_limit=80,
                min_steering_rate_limit=35,
                steering_release_rate_limit=55,
                max_steering=500,
            )
        )
        follower.accept_applied_command(
            ControlCommand(speed=255, steering=-120, reason="previous_curve")
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=0.05,
                heading_error=0.0,
                near_lateral_error_norm=0.05,
                reason="corridor_tier1",
            )
        )

        self.assertLess(command.steering, 0)

    def test_path_near_conflict_blends_toward_near_path_without_hard_zero(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=120.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_error_threshold=2.0,
                path_near_conflict_error_threshold=0.01,
                path_near_conflict_release_alpha=1.0,
                path_reversal_min_steering=25.0,
                path_reversal_rate_limit=220,
                steering_rate_limit=80,
                min_steering_rate_limit=35,
                steering_release_rate_limit=55,
                max_steering=500,
            )
        )
        follower.accept_applied_command(
            ControlCommand(speed=255, steering=-120, reason="previous_curve")
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=1.0,
                heading_error=0.0,
                near_lateral_error_norm=-0.12,
                reason="corridor_tier1",
            )
        )

        self.assertGreater(command.steering, -120)
        self.assertLess(command.steering, 0)

    def test_path_reversal_waits_for_near_center_before_opposite_turn(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=225.0,
                path_heading_gain=65.0,
                path_derivative_gain=0.0,
                path_near_weight=1.0,
                path_far_weight=1.0,
                path_heading_lead_gain=170.0,
                path_heading_lead_span=0.16,
                path_heading_lead_max_steering=32.0,
                path_center_recovery_error_threshold=2.0,
                path_near_conflict_error_threshold=0.06,
                path_near_conflict_release_alpha=0.90,
                path_near_conflict_heading_limit=0.18,
                path_reversal_alpha=0.90,
                path_reversal_min_steering=25.0,
                path_reversal_min_geometry=0.05,
                path_reversal_output_min_steering=70.0,
                path_reversal_rate_limit=160,
                path_reversal_near_guard_error=0.025,
                path_reversal_near_full_error=0.12,
                steering_rate_limit=80,
                min_steering_rate_limit=35,
                steering_release_rate_limit=55,
                max_steering=150,
            )
        )
        follower.accept_applied_command(
            ControlCommand(speed=255, steering=-105, reason="previous_curve")
        )

        guarded = follower.plan(
            lane_geometry(
                lateral_error_norm=0.12,
                heading_error=0.36,
                near_lateral_error_norm=-0.13,
                reason="corridor_tier1",
            )
        )

        self.assertLess(guarded.steering, 0)
        self.assertIn("curve_transition", guarded.reason)

        released = follower.plan(
            lane_geometry(
                lateral_error_norm=0.12,
                heading_error=0.36,
                near_lateral_error_norm=-0.01,
                reason="corridor_tier1",
            )
        )

        self.assertGreater(released.steering, 0)

    def test_path_reversal_guard_does_not_delay_curve_entry_from_straight(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=225.0,
                path_heading_gain=65.0,
                path_derivative_gain=0.0,
                path_near_weight=1.0,
                path_far_weight=1.0,
                path_heading_lead_gain=170.0,
                path_heading_lead_span=0.16,
                path_heading_lead_max_steering=32.0,
                path_center_recovery_error_threshold=2.0,
                path_reversal_near_guard_error=0.025,
                path_reversal_near_full_error=0.12,
                path_steering_rise_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=0.12,
                heading_error=0.36,
                near_lateral_error_norm=-0.13,
                reason="corridor_tier1",
            )
        )

        self.assertGreater(command.steering, 0)

    def test_path_curve_guard_caps_inner_cut_while_near_field_is_centered(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=500.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_error_threshold=2.0,
                path_curve_guard_heading_threshold=0.25,
                path_curve_guard_near_error=0.08,
                path_curve_guard_steering_limit=110.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=-0.20,
                heading_error=-0.50,
                near_lateral_error_norm=0.0,
                reason="corridor_tier1",
            )
        )

        self.assertEqual(command.steering, -110)

    def test_path_curve_guard_blends_continuously_as_center_error_grows(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=500.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_center_recovery_error_threshold=2.0,
                path_curve_guard_heading_threshold=0.25,
                path_curve_guard_near_error=0.10,
                path_curve_guard_release_error=0.24,
                path_curve_guard_steering_limit=105.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=150,
            )
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=-0.20,
                heading_error=-0.50,
                near_lateral_error_norm=-0.17,
                reason="corridor_tier1",
            )
        )

        self.assertGreater(abs(command.steering), 105)
        self.assertLess(abs(command.steering), 150)

    def test_path_heading_lead_anticipates_curve_before_lateral_drift(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=100.0,
                path_heading_lead_span=0.20,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )
        curve_entry = lane_geometry(
            lateral_error_norm=-0.02,
            heading_error=-0.20,
            near_lateral_error_norm=-0.04,
        )

        command = follower.plan(curve_entry)

        self.assertLessEqual(command.steering, -15)
        self.assertIn("curve_entry", command.reason)

    def test_path_heading_lead_is_bounded_during_s_curve_transition(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=500.0,
                path_heading_lead_span=0.01,
                path_heading_lead_max_steering=30.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )

        command = follower.plan(
            lane_geometry(
                lateral_error_norm=0.01,
                heading_error=0.60,
                near_lateral_error_norm=0.01,
            )
        )

        self.assertEqual(command.steering, 30)

    def test_path_integral_removes_persistent_straight_centering_bias(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_integral_gain=40.0,
                path_integral_limit=0.25,
                path_integral_decay=0.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )
        biased_straight = lane_geometry(
            lateral_error_norm=-0.02,
            heading_error=0.0,
            near_lateral_error_norm=-0.02,
            reason="corridor_tier1",
        )

        command = None
        for _ in range(20):
            command = follower.plan(biased_straight)

        self.assertIsNotNone(command)
        self.assertEqual(command.steering, -10)

    def test_path_integral_releases_outside_reliable_straight(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=0.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_integral_gain=40.0,
                path_integral_limit=0.25,
                path_integral_decay=0.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=1.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )
        biased_straight = lane_geometry(
            lateral_error_norm=-0.02,
            heading_error=0.0,
            near_lateral_error_norm=-0.02,
            reason="corridor_tier1",
        )
        for _ in range(20):
            follower.plan(biased_straight)

        curve = follower.plan(
            lane_geometry(
                lateral_error_norm=0.0,
                heading_error=0.50,
                near_lateral_error_norm=0.0,
                reason="corridor_tier1",
            )
        )

        self.assertEqual(curve.steering, 0)

    def test_path_filter_tracks_actual_brake_steering_before_restart(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                path_tracking=True,
                path_lateral_gain=100.0,
                path_heading_gain=0.0,
                path_derivative_gain=0.0,
                path_heading_lead_gain=0.0,
                path_steering_rise_alpha=1.0,
                path_steering_release_alpha=0.05,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
            )
        )
        follower.plan(
            lane_geometry(
                lateral_error_norm=0.30,
                heading_error=0.0,
                near_lateral_error_norm=0.30,
            )
        )
        follower.accept_applied_command(
            ControlCommand.stop("traffic_light:red_contact")
        )

        restarted = follower.plan(
            lane_geometry(
                lateral_error_norm=-0.20,
                heading_error=0.0,
                near_lateral_error_norm=-0.20,
            )
        )

        self.assertEqual(restarted.steering, -20)

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

    def test_curve_strength_releases_slower_than_it_rises(self):
        slow_release = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=100.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
                curve_strength_alpha=1.0,
                curve_strength_release_alpha=0.10,
                straight_steering_scale=0.4,
                curve_steering_scale=1.0,
            )
        )
        fast_release = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=100.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                steering_release_rate_limit=500,
                max_steering=500,
                curve_strength_alpha=1.0,
                curve_strength_release_alpha=1.0,
                straight_steering_scale=0.4,
                curve_steering_scale=1.0,
            )
        )
        curve = lane_geometry(lateral_error_norm=0.65, heading_error=0.0)
        exit_curve = lane_geometry(lateral_error_norm=0.30, heading_error=0.0)

        slow_release.plan(curve)
        fast_release.plan(curve)
        slow_command = slow_release.plan(exit_curve)
        fast_command = fast_release.plan(exit_curve)

        self.assertGreater(slow_command.steering, fast_command.steering)

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

    def test_near_error_keeps_center_lock_active(self):
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
                center_lock_error_threshold=0.055,
                center_lock_min_steering=75,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        lane = lane_geometry(
            lateral_error_norm=-0.03,
            heading_error=0.0,
            near_lateral_error_norm=-0.12,
        )

        command = follower.plan(lane)

        self.assertEqual(command.steering, -75)
        self.assertIn("center_lock", command.reason)

    def test_near_error_can_override_opposite_far_error_for_centering(self):
        follower = YoloLaneFollower(
            YoloLaneFollowerConfig(
                kp_lateral=100.0,
                kd_lateral=0.0,
                kp_heading=0.0,
                kd_heading=0.0,
                steering_rate_limit=500,
                min_steering_rate_limit=500,
                center_recovery_error_threshold=1.0,
                center_lock_enabled=True,
                center_lock_error_threshold=0.055,
                center_lock_min_steering=75,
                straight_steering_scale=1.0,
                curve_steering_scale=1.0,
                max_steering=500,
            )
        )
        lane = lane_geometry(
            lateral_error_norm=0.04,
            heading_error=0.0,
            near_lateral_error_norm=-0.12,
        )

        command = follower.plan(lane)

        self.assertEqual(command.steering, -75)

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

    def test_bev_corridor_runtime_uses_lookahead_alias(self):
        args = parse_args(
            [
                "--lookahead",
                "0.70",
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
                "--corridor-max-heading-jump",
                "0.16",
            ]
        )

        bev_config = build_bev_corridor_config(args)
        follower_config = build_follower_config(args)

        self.assertAlmostEqual(bev_config.lookahead_y_ratio, 0.70)
        self.assertAlmostEqual(follower_config.lateral_priority_threshold, 0.25)
        self.assertAlmostEqual(follower_config.curve_strength_alpha, 0.25)
        self.assertTrue(follower_config.center_lock_enabled)
        self.assertAlmostEqual(follower_config.center_lock_error_threshold, 0.04)
        self.assertEqual(follower_config.center_lock_min_steering, 95)
        self.assertAlmostEqual(bev_config.max_heading_jump, 0.16)

    def test_path_stage_control_cli_values_reach_follower_config(self):
        args = parse_args(
            [
                "--path-center-recovery-error-threshold",
                "0.11",
                "--path-center-recovery-heading-limit",
                "0.13",
                "--path-center-recovery-min-steering",
                "72",
                "--path-center-recovery-alpha",
                "0.91",
                "--path-center-recovery-rate-limit",
                "125",
                "--path-reversal-alpha",
                "0.92",
                "--path-reversal-min-steering",
                "27",
                "--path-reversal-min-geometry",
                "0.09",
                "--path-reversal-rate-limit",
                "165",
                "--path-reversal-near-guard-error",
                "0.03",
                "--path-reversal-near-full-error",
                "0.14",
                "--path-near-conflict-heading-limit",
                "0.17",
                "--path-curve-guard-heading-threshold",
                "0.26",
                "--path-curve-guard-near-error",
                "0.07",
                "--path-curve-guard-release-error",
                "0.21",
                "--path-curve-guard-steering-limit",
                "108",
            ]
        )

        config = build_follower_config(args)

        self.assertAlmostEqual(config.path_center_recovery_error_threshold, 0.11)
        self.assertAlmostEqual(config.path_center_recovery_heading_limit, 0.13)
        self.assertAlmostEqual(config.path_center_recovery_min_steering, 72.0)
        self.assertAlmostEqual(config.path_center_recovery_alpha, 0.91)
        self.assertEqual(config.path_center_recovery_rate_limit, 125)
        self.assertAlmostEqual(config.path_reversal_alpha, 0.92)
        self.assertAlmostEqual(config.path_reversal_min_steering, 27.0)
        self.assertAlmostEqual(config.path_reversal_min_geometry, 0.09)
        self.assertEqual(config.path_reversal_rate_limit, 165)
        self.assertAlmostEqual(config.path_reversal_near_guard_error, 0.03)
        self.assertAlmostEqual(config.path_reversal_near_full_error, 0.14)
        self.assertAlmostEqual(config.path_near_conflict_heading_limit, 0.17)
        self.assertAlmostEqual(config.path_curve_guard_heading_threshold, 0.26)
        self.assertAlmostEqual(config.path_curve_guard_near_error, 0.07)
        self.assertAlmostEqual(config.path_curve_guard_release_error, 0.21)
        self.assertAlmostEqual(config.path_curve_guard_steering_limit, 108.0)

    def test_crosswalk_cache_defaults_hold_preliminary_run_geometry(self):
        args = parse_args([])

        bev_config = build_bev_corridor_config(args)
        follower_config = build_follower_config(args)

        self.assertEqual(bev_config.max_coast_frames, 3)
        self.assertAlmostEqual(bev_config.max_center_jump_px, 150.0)
        self.assertAlmostEqual(bev_config.crosswalk_max_center_jump_px, 150.0)
        self.assertTrue(bev_config.virtual_hold)
        self.assertEqual(follower_config.lane_lost_hold_frames, 3)
        self.assertEqual(follower_config.lane_lost_steering_release_rate_limit, 35)

    def test_default_lane_following_uses_last_known_good_drive_tuning(self):
        args = parse_args([])

        bev_config = build_bev_corridor_config(args)
        follower_config = build_follower_config(args)

        self.assertAlmostEqual(bev_config.lookahead_y_ratio, 0.45)
        self.assertAlmostEqual(bev_config.centerline_bias, 0.46)
        self.assertAlmostEqual(bev_config.vehicle_center_x_offset_ratio, 0.04)
        self.assertAlmostEqual(bev_config.max_heading_jump, 0.45)
        self.assertEqual(follower_config.base_speed, 255)
        self.assertEqual(follower_config.max_speed, 255)
        self.assertEqual(follower_config.min_curve_speed, 255)
        self.assertEqual(follower_config.max_steering, 150)
        self.assertAlmostEqual(follower_config.kp_lateral, 205.0)
        self.assertAlmostEqual(follower_config.kd_lateral, 75.0)
        self.assertAlmostEqual(follower_config.curve_strength_alpha, 0.60)
        self.assertAlmostEqual(follower_config.curve_strength_release_alpha, 0.18)
        self.assertAlmostEqual(follower_config.straight_steering_scale, 0.50)
        self.assertAlmostEqual(follower_config.curve_steering_scale, 1.68)
        self.assertAlmostEqual(follower_config.center_recovery_error_threshold, 0.08)
        self.assertAlmostEqual(follower_config.center_recovery_steering_boost, 1.35)
        self.assertEqual(follower_config.center_recovery_min_steering, 85)
        self.assertTrue(follower_config.center_lock_enabled)
        self.assertAlmostEqual(follower_config.center_lock_error_threshold, 0.055)
        self.assertEqual(follower_config.center_lock_min_steering, 75)

    def test_lane_change_cli_defaults_are_external_ready_and_aggressive(self):
        args = parse_args([])
        lane_change_config = build_lane_change_config(args)

        self.assertEqual(lane_change_config.mode, "external")
        self.assertTrue(lane_change_config.smooth_avoidance)
        self.assertGreater(lane_change_config.return_duration_scale, 1.0)
        self.assertLess(lane_change_config.transition_seconds, 2.0)
        self.assertGreaterEqual(lane_change_config.speed_cap, 85)
        self.assertGreaterEqual(lane_change_config.steering_min, 100)

    def test_lane_change_key_requests_and_returns(self):
        controller = LaneChangeController(
            LaneChangeConfig(mode="external", transition_seconds=1.0, stable_required_frames=0)
        )

        action, _ = handle_lane_change_key(controller, True)
        controller.update(lane_geometry(0.0, 0.0), 150.0, 800.0, 0.0, True)
        controller.update(lane_geometry(0.0, 0.0), 150.0, 800.0, 1.0, True)
        return_action, _ = handle_lane_change_key(controller, True)

        self.assertEqual(action, "request")
        self.assertEqual(controller.state, "lane1")
        self.assertEqual(return_action, "return")
        self.assertEqual(controller.return_source, "operator_return")

    def test_virtual_lane_command_is_capped(self):
        args = parse_args(
            [
                "--virtual-lane-max-steering",
                "110",
                "--virtual-lane-speed-cap",
                "220",
                "--fixed-speed",
                "off",
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
                "--fixed-speed",
                "off",
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
        args = parse_args(["--fixed-speed", "off", "--lane-lost-speed-cap", "200"])
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
                "--fixed-speed",
                "off",
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

    def test_drive_priority_restores_fixed_speed_after_late_obstacle_cap(self):
        args = parse_args(["--speed", "255", "--fixed-speed", "on"])
        policy = DrivePriorityController(
            CommandSafetyFilter(args),
            traffic_light_enabled=False,
        )
        lane = lane_geometry(lateral_error_norm=0.0, heading_error=0.0)

        command = policy.apply(
            ControlCommand(180, 12, brake=False, reason="lane"),
            DummyMask("lane-center+right-lane-side"),
            lane,
            True,
            DummyObstacleMode(late_speed_cap=80),
            DummyTrafficLight(),
        )

        self.assertEqual(command.speed, 255)
        self.assertEqual(command.steering, 12)
        self.assertIn("fixed_speed", command.reason)
        self.assertIn("late_obstacle_cap", command.reason)

    def test_drive_priority_never_overrides_brake_with_fixed_speed(self):
        args = parse_args(["--speed", "255", "--fixed-speed", "on"])
        policy = DrivePriorityController(
            CommandSafetyFilter(args),
            traffic_light_enabled=True,
        )
        lane = lane_geometry(lateral_error_norm=0.0, heading_error=0.0)

        command = policy.apply(
            ControlCommand(255, 0, brake=False, reason="lane"),
            DummyMask("lane-center+right-lane-side"),
            lane,
            True,
            DummyObstacleMode(),
            DummyTrafficLight(stop=True),
        )

        self.assertTrue(command.brake)
        self.assertEqual(command.speed, 0)
        self.assertEqual(command.reason, "traffic_light:red_contact")


def lane_geometry(
    lateral_error_norm: float,
    heading_error: float,
    near_lateral_error_norm: float = None,
    reason: str = "test",
) -> LaneGeometry:
    return LaneGeometry(
        found=True,
        center_x=0.0,
        vehicle_center_x=0.0,
        target_y=0.0,
        lateral_error_px=0.0,
        lateral_error_norm=lateral_error_norm,
        heading_error=heading_error,
        confidence=1.0,
        reason=reason,
        near_lateral_error_norm=near_lateral_error_norm,
    )


class DummyMask:
    def __init__(self, class_name: str):
        self.class_name = class_name


class DummyObstacleMode:
    blocks_light_stop = False

    def __init__(self, late_speed_cap: int = 0):
        self.late_speed_cap = late_speed_cap

    def apply_steering(self, command: ControlCommand) -> ControlCommand:
        return command

    def apply_speed_cap(self, command: ControlCommand) -> ControlCommand:
        return command

    def apply_safety(self, command: ControlCommand, running: bool) -> ControlCommand:
        if not running or command.brake or self.late_speed_cap <= 0:
            return command
        return ControlCommand(
            speed=min(command.speed, self.late_speed_cap),
            steering=command.steering,
            brake=False,
            reason="%s:late_obstacle_cap" % command.reason,
        )


class DummyTrafficLight:
    def __init__(self, stop: bool = False):
        self.stop = stop

    def apply(self, command: ControlCommand, running: bool) -> ControlCommand:
        if self.stop and running:
            return ControlCommand.stop("traffic_light:red_contact")
        return command


if __name__ == "__main__":
    unittest.main()
