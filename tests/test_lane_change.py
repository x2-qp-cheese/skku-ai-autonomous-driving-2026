import unittest
from dataclasses import replace

from skku_autocar.estimation.lane_geometry import LaneGeometry
from skku_autocar.planning.lane_change import LaneChangeConfig, LaneChangeController
from skku_autocar.types import ControlCommand


def lane(heading=0.0):
    return LaneGeometry(
        found=True,
        center_x=400.0,
        vehicle_center_x=400.0,
        target_y=200.0,
        lateral_error_px=0.0,
        lateral_error_norm=0.0,
        heading_error=heading,
        confidence=1.0,
        reason="corridor",
        height=500.0,
    )


def lane_for_shifted_lane1(heading=0.0):
    return LaneGeometry(
        found=True,
        center_x=550.0,
        vehicle_center_x=400.0,
        target_y=200.0,
        lateral_error_px=150.0,
        lateral_error_norm=150.0 / 400.0,
        heading_error=heading,
        confidence=1.0,
        reason="corridor",
        height=500.0,
    )


def lane_for_target_error(lateral_error_norm=0.0, heading=0.0, lane_width_px=150.0, bev_width_px=800.0):
    vehicle_center_x = 400.0
    target_center_x = vehicle_center_x + lateral_error_norm * (bev_width_px / 2.0)
    return LaneGeometry(
        found=True,
        center_x=target_center_x + lane_width_px,
        vehicle_center_x=vehicle_center_x,
        target_y=200.0,
        lateral_error_px=target_center_x + lane_width_px - vehicle_center_x,
        lateral_error_norm=(target_center_x + lane_width_px - vehicle_center_x) / (bev_width_px / 2.0),
        heading_error=heading,
        confidence=1.0,
        reason="corridor",
        height=500.0,
    )


class LaneChangeControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="timed",
                transition_seconds=2.0,
                hold_seconds=3.0,
                max_straight_heading=0.08,
                speed_cap=70,
                stable_required_frames=0,
            )
        )

    def update(self, now, geometry=None):
        return self.controller.update(geometry or lane(), 150.0, 800.0, now, True)

    def test_timed_mode_moves_lane2_to_lane1_and_returns(self):
        self.update(0.0)
        self.assertTrue(self.controller.request())

        start = self.update(1.0)
        halfway_left = self.update(2.0)
        lane1 = self.update(3.0)
        return_start = self.update(6.0)
        halfway_right = self.update(7.0)
        completed = self.update(8.0)

        self.assertEqual(start.state, "changing_to_lane1")
        self.assertAlmostEqual(halfway_left.offset_px, -75.0)
        self.assertAlmostEqual(halfway_left.lane.center_x, 325.0)
        self.assertEqual(lane1.state, "lane1")
        self.assertAlmostEqual(lane1.offset_px, -150.0)
        self.assertEqual(return_start.state, "changing_to_lane2")
        self.assertAlmostEqual(halfway_right.offset_px, -75.0)
        self.assertEqual(completed.state, "completed")
        self.assertAlmostEqual(completed.offset_px, 0.0)

    def test_parallel_path_translation_is_not_clipped_at_bev_edge(self):
        controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                max_straight_heading=0.30,
                stable_required_frames=0,
            )
        )
        base = replace(
            lane(),
            center_x=40.0,
            lateral_error_px=-360.0,
            lateral_error_norm=-0.9,
            heading_error=0.25,
            path_points=((20.0, 10.0), (40.0, 200.0), (80.0, 490.0)),
        )
        controller.request("test")
        controller.update(base, 150.0, 800.0, 0.0, True)
        shifted = controller.update(base, 150.0, 800.0, 1.0, True).lane

        self.assertAlmostEqual(shifted.center_x, -110.0)
        self.assertEqual(
            shifted.path_points,
            ((-130.0, 10.0), (-110.0, 200.0), (-70.0, 490.0)),
        )
        self.assertAlmostEqual(shifted.heading_error, base.heading_error)

    def test_request_waits_until_straight(self):
        self.update(0.0)
        self.controller.request()

        armed = self.update(1.0, lane(heading=0.2))
        started = self.update(2.0, lane(heading=0.02))

        self.assertEqual(armed.state, "armed")
        self.assertEqual(started.state, "changing_to_lane1")

    def test_timed_mode_triggers_after_delay(self):
        controller = LaneChangeController(
            LaneChangeConfig(mode="timed", trigger_seconds=5.0)
        )
        before = controller.update(lane(), 150.0, 800.0, 4.9, True)
        after = controller.update(lane(), 150.0, 800.0, 10.0, True)

        self.assertEqual(before.state, "lane2")
        self.assertEqual(after.state, "changing_to_lane1")

    def test_speed_is_capped_only_while_active(self):
        command = ControlCommand(speed=105, steering=20, reason="lane")
        capped = self.controller.apply_speed_cap(command, active=True)
        unchanged = self.controller.apply_speed_cap(command, active=False)

        self.assertEqual(capped.speed, 70)
        self.assertEqual(capped.steering, 20)
        self.assertEqual(unchanged, command)

    def test_steering_assist_forces_directional_minimum_while_changing(self):
        self.update(0.0)
        self.controller.request()
        changing = self.update(1.0)
        command = ControlCommand(speed=105, steering=15, reason="lane")

        assisted = self.controller.apply_control_adjustments(command, changing)

        self.assertEqual(assisted.speed, 70)
        self.assertEqual(assisted.steering, -100)
        self.assertIn("lane_change_steer", assisted.reason)

    def test_steering_assist_is_not_applied_while_holding_lane1(self):
        self.update(0.0)
        self.controller.request()
        self.update(1.0)
        lane1 = self.update(3.0)
        command = ControlCommand(speed=105, steering=0, reason="lane")

        assisted = self.controller.apply_control_adjustments(command, lane1)

        self.assertEqual(lane1.state, "lane1")
        self.assertEqual(assisted.speed, 105)
        self.assertEqual(assisted.steering, 0)
        self.assertNotIn("lane_change_steer", assisted.reason)

    def test_armed_state_caps_speed_before_steering_can_start(self):
        controller = LaneChangeController(
            LaneChangeConfig(mode="external", speed_cap=80, stable_required_frames=0)
        )
        controller.request("obstacle")
        result = controller.update(lane(heading=0.2), 150.0, 800.0, 1.0, True)
        command = ControlCommand(speed=255, steering=0, reason="lane")

        capped = controller.apply_control_adjustments(command, result)

        self.assertEqual(result.state, "armed")
        self.assertEqual(capped.speed, 80)
        self.assertEqual(capped.steering, 0)

    def test_transition_requires_center_stability_before_lane1_is_final(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                stable_required_frames=3,
                stable_lateral_error=0.12,
                stable_heading_error=0.18,
                speed_cap=70,
            )
        )
        self.controller.request("obstacle")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        unstable = self.controller.update(lane(heading=0.25), 150.0, 800.0, 1.0, True)
        stable1 = self.controller.update(lane_for_shifted_lane1(heading=0.05), 150.0, 800.0, 1.1, True)
        stable2 = self.controller.update(lane_for_shifted_lane1(heading=0.04), 150.0, 800.0, 1.2, True)
        stable3 = self.controller.update(lane_for_shifted_lane1(heading=0.03), 150.0, 800.0, 1.3, True)

        self.assertEqual(unstable.state, "stabilizing_lane1")
        self.assertEqual(unstable.stable_frames, 0)
        self.assertEqual(stable1.state, "stabilizing_lane1")
        self.assertEqual(stable2.state, "stabilizing_lane1")
        self.assertEqual(stable3.state, "lane1")
        self.assertEqual(stable3.stable_frames, 3)

    def test_stabilizing_caps_speed_but_lets_lane_follower_balance(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                stable_required_frames=3,
                speed_cap=70,
                steering_override=True,
                steering_cap=150,
            )
        )
        self.controller.request("obstacle")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        stabilizing = self.controller.update(lane(heading=0.25), 150.0, 800.0, 1.0, True)
        command = ControlCommand(speed=255, steering=35, reason="lane")

        adjusted = self.controller.apply_control_adjustments(command, stabilizing)

        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertEqual(stabilizing.direction, 0)
        self.assertEqual(adjusted.speed, 70)
        self.assertEqual(adjusted.steering, 35)

    def test_unreliable_transition_caps_speed_and_steering(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                speed_cap=135,
                steering_cap=150,
                unreliable_speed_cap=70,
                unreliable_steering_cap=90,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        result = self.controller.update(
            lane(), 150.0, 800.0, 0.0, True, lane_reliable=False
        )
        command = ControlCommand(speed=255, steering=-150, reason="lane")

        adjusted = self.controller.apply_control_adjustments(command, result)

        self.assertEqual(result.state, "armed")
        self.assertEqual(adjusted.speed, 70)
        self.assertEqual(adjusted.steering, -90)
        self.assertIn("lane_change_unreliable", adjusted.reason)

    def test_avoidance_stabilizing_releases_steering_to_lane_feedback(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                stable_required_frames=5,
                steering_override=True,
                steering_cap=150,
                target_capture_frames=1,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        stabilizing = self.controller.update(
            lane_for_target_error(lateral_error_norm=0.06, heading=0.30),
            150.0,
            800.0,
            1.0,
            True,
        )
        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=255, steering=-30, reason="lane"),
            stabilizing,
        )

        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertEqual(stabilizing.direction, 0)
        self.assertEqual(adjusted.steering, -30)

    def test_avoidance_stability_follows_target_geometry_on_a_curve(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                target_capture_frames=1,
                stable_required_frames=3,
                stable_lateral_error=0.12,
                stable_near_lateral_error=0.18,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        curved_target = lane_for_target_error(
            lateral_error_norm=0.06,
            heading=0.45,
        )

        first = self.controller.update(curved_target, 150.0, 800.0, 0.1, True)
        second = self.controller.update(curved_target, 150.0, 800.0, 0.2, True)
        stable = self.controller.update(curved_target, 150.0, 800.0, 0.3, True)

        self.assertEqual(first.state, "stabilizing_lane1")
        self.assertEqual(second.state, "stabilizing_lane1")
        self.assertEqual(stable.state, "lane1")

    def test_avoidance_uses_complete_target_with_feedback_steering(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                stable_required_frames=2,
                stable_lateral_error=0.12,
                stable_heading_error=0.18,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        still_approaching = self.controller.update(lane(), 150.0, 800.0, 1.5, True)
        target_center = self.controller.update(
            lane_for_shifted_lane1(heading=0.30),
            150.0,
            800.0,
            1.6,
            True,
        )
        arrived = self.controller.update(
            lane_for_target_error(lateral_error_norm=0.06, heading=0.30),
            150.0,
            800.0,
            1.7,
            True,
        )
        stable1 = self.controller.update(
            lane_for_shifted_lane1(heading=0.05),
            150.0,
            800.0,
            1.8,
            True,
        )
        stable2 = self.controller.update(
            lane_for_shifted_lane1(heading=0.04),
            150.0,
            800.0,
            1.9,
            True,
        )

        self.assertEqual(still_approaching.state, "changing_to_lane1")
        self.assertEqual(still_approaching.direction, -1)
        self.assertEqual(target_center.state, "changing_to_lane1")
        self.assertEqual(target_center.direction, -1)
        self.assertEqual(arrived.state, "stabilizing_lane1")
        self.assertEqual(arrived.direction, 0)
        self.assertEqual(stable1.state, "lane1")
        self.assertEqual(stable2.state, "lane1")

    def test_avoidance_uses_fixed_width_instead_of_narrow_live_measurement(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                target_lane_width_px=150.0,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 110.0, 800.0, 0.0, True)

        halfway = self.controller.update(lane(), 105.0, 800.0, 0.5, True)

        self.assertAlmostEqual(halfway.offset_px, -150.0)

    def test_avoidance_waits_for_near_vehicle_target_before_feedback_control(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                target_lane_width_px=150.0,
                stable_required_frames=2,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        lookahead_only = replace(
            lane_for_target_error(lateral_error_norm=0.06, heading=0.30),
            near_center_x=470.0,
            near_target_y=440.0,
            near_lateral_error_px=70.0,
            near_lateral_error_norm=70.0 / 400.0,
        )
        body_arrived = replace(
            lookahead_only,
            near_center_x=574.0,
            near_lateral_error_px=174.0,
            near_lateral_error_norm=174.0 / 400.0,
        )

        still_changing = self.controller.update(lookahead_only, 110.0, 800.0, 1.5, True)
        arrived = self.controller.update(body_arrived, 110.0, 800.0, 1.6, True)

        self.assertEqual(still_changing.state, "changing_to_lane1")
        self.assertEqual(still_changing.direction, -1)
        self.assertEqual(arrived.state, "stabilizing_lane1")
        self.assertEqual(arrived.direction, 0)

    def test_avoidance_preserves_directional_minimum_until_target_arrival(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                steering_override=False,
                steering_min=100,
                steering_cap=150,
                stable_required_frames=2,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        approaching = self.controller.update(lane(), 150.0, 800.0, 1.5, True)

        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=255, steering=-20, reason="lane"),
            approaching,
        )

        self.assertEqual(approaching.state, "changing_to_lane1")
        self.assertAlmostEqual(approaching.lane.lateral_error_norm, -0.375)
        self.assertEqual(adjusted.steering, -100)
        self.assertIn("lane_change_steer", adjusted.reason)

    def test_avoidance_rejects_countersteer_until_target_arrival(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                steering_min=150,
                steering_boost=35,
                steering_cap=150,
                target_capture_error=0.20,
                target_capture_frames=2,
                stable_required_frames=5,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        changing = self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=255, steering=62, reason="lane"),
            changing,
        )

        self.assertEqual(changing.state, "changing_to_lane1")
        self.assertEqual(changing.direction, -1)
        self.assertEqual(adjusted.steering, -150)

    def test_avoidance_releases_feedback_in_target_approach_zone(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                steering_min=150,
                steering_boost=35,
                steering_cap=150,
                target_approach_error=0.32,
                target_capture_error=0.20,
                target_capture_frames=2,
                stable_required_frames=5,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        approaching = self.controller.update(
            lane_for_target_error(lateral_error_norm=-0.211, heading=-0.54),
            150.0,
            800.0,
            0.1,
            True,
        )

        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=135, steering=62, reason="lane"),
            approaching,
        )

        self.assertEqual(approaching.state, "changing_to_lane1")
        self.assertEqual(approaching.direction, -1)
        self.assertEqual(adjusted.steering, 62)
        self.assertIn("lane_change_capture_feedback", adjusted.reason)

    def test_avoidance_waits_for_reliable_geometry_before_starting(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(mode="external")
        )
        self.controller.request_avoidance("obstacle_fusion")

        unreliable = self.controller.update(
            lane(), 150.0, 800.0, 0.0, True, lane_reliable=False
        )
        reliable = self.controller.update(
            lane(), 150.0, 800.0, 0.1, True, lane_reliable=True
        )

        self.assertEqual(unreliable.state, "armed")
        self.assertEqual(reliable.state, "changing_to_lane1")

    def test_lane2_keeps_reason_clean_when_no_offset_is_active(self):
        result = self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        self.assertEqual(result.state, "lane2")
        self.assertEqual(result.offset_px, 0.0)
        self.assertEqual(result.lane.reason, "corridor")

    def test_avoidance_holds_last_reliable_target_on_unreliable_geometry(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                target_lane_width_px=150.0,
                stable_required_frames=5,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        reliable = self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        unreliable_lane = replace(
            lane_for_shifted_lane1(heading=0.55),
            confidence=0.30,
            reason="virtual_hold:heading_jump(3)",
        )

        held = self.controller.update(
            unreliable_lane,
            150.0,
            800.0,
            0.1,
            True,
            lane_reliable=False,
        )

        self.assertEqual(reliable.state, "changing_to_lane1")
        self.assertEqual(held.state, "changing_to_lane1")
        self.assertFalse(held.lane_reliable)
        self.assertAlmostEqual(held.lane.center_x, reliable.lane.center_x)
        self.assertAlmostEqual(held.lane.near_lateral_error_norm or 0.0, reliable.lane.near_lateral_error_norm or 0.0)
        self.assertIn("lane_change_hold_unreliable", held.lane.reason)

    def test_unreliable_avoidance_keeps_directional_steering(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                target_lane_width_px=150.0,
                steering_min=150,
                steering_cap=150,
                unreliable_steering_cap=90,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        result = self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        result = replace(result, lane_reliable=False)

        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=255, steering=20, reason="lane"),
            result,
        )

        self.assertEqual(result.state, "changing_to_lane1")
        self.assertEqual(result.direction, -1)
        self.assertEqual(adjusted.speed, 70)
        self.assertEqual(adjusted.steering, -90)
        self.assertIn("lane_change_unreliable_steer", adjusted.reason)

    def test_unreliable_lane1_hold_caps_path_feedback(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                unreliable_steering_cap=90,
            )
        )
        self.controller.state = "lane1"
        lane1 = self.controller.update(
            lane(),
            150.0,
            800.0,
            0.0,
            True,
            lane_reliable=True,
        )
        unreliable = replace(lane1, lane_reliable=False)

        adjusted = self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=121, reason="lane"),
            unreliable,
        )

        self.assertEqual(unreliable.state, "lane1")
        self.assertEqual(adjusted.steering, 90)
        self.assertIn("lane_change_unreliable", adjusted.reason)

    def test_unreliable_geometry_cannot_confirm_target_capture(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                target_capture_error=0.20,
                target_capture_frames=1,
                stable_required_frames=5,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        result = self.controller.update(
            lane_for_target_error(lateral_error_norm=0.0),
            150.0,
            800.0,
            0.1,
            True,
            lane_reliable=False,
        )

        self.assertEqual(result.state, "changing_to_lane1")
        self.assertFalse(result.lane_reliable)

    def test_avoidance_return_rejects_countersteer_until_target_arrival(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                steering_min=150,
                steering_boost=35,
                steering_cap=150,
                target_capture_error=0.20,
                target_capture_frames=2,
                stable_required_frames=5,
            )
        )
        self.controller.state = "lane1"
        self.assertTrue(
            self.controller.request_avoidance_return("obstacle_fusion")
        )
        current_lane1 = lane_for_shifted_lane1()
        self.controller.update(current_lane1, 150.0, 800.0, 0.0, True)
        changing = self.controller.update(
            current_lane1, 150.0, 800.0, 0.1, True
        )

        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=255, steering=-40, reason="lane"),
            changing,
        )

        self.assertEqual(changing.state, "changing_to_lane2")
        self.assertEqual(changing.direction, 1)
        self.assertEqual(adjusted.steering, 150)

    def test_urgent_avoidance_return_uses_full_transition_rate_and_cap(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                smooth_avoidance=True,
                steering_override=True,
                steering_min=150,
                steering_cap=150,
                return_duration_scale=1.35,
                return_steering_cap=115,
                stable_required_frames=0,
            )
        )
        self.controller.state = "lane1"
        self.assertTrue(
            self.controller.request_avoidance_return("obstacle_fusion")
        )
        started = self.controller.update(
            lane_for_shifted_lane1(), 150.0, 800.0, 0.0, True
        )
        changing = self.controller.update(
            lane_for_shifted_lane1(), 150.0, 800.0, 0.1, True
        )
        adjusted = self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=-40, reason="lane"),
            changing,
        )
        finished = self.controller.update(
            lane_for_shifted_lane1(), 150.0, 800.0, 1.1, True
        )

        self.assertEqual(started.state, "changing_to_lane2")
        self.assertEqual(changing.state, "changing_to_lane2")
        self.assertEqual(adjusted.steering, 150)
        self.assertEqual(finished.state, "completed")

    def test_normal_return_keeps_slow_return_profile(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                smooth_avoidance=True,
                return_duration_scale=1.35,
                stable_required_frames=0,
            )
        )
        self.controller.state = "lane1"
        self.assertTrue(self.controller.request_return("obstacle_clear"))
        self.controller.update(
            lane_for_shifted_lane1(), 150.0, 800.0, 0.0, True
        )

        returning = self.controller.update(
            lane_for_shifted_lane1(), 150.0, 800.0, 1.0, True
        )

        self.assertEqual(returning.state, "changing_to_lane2")

    def test_smooth_avoidance_does_not_jump_to_full_offset_on_commit(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=0.60,
                smooth_avoidance=True,
                stable_required_frames=0,
            )
        )
        self.assertTrue(
            self.controller.request_avoidance("obstacle_fusion")
        )

        started = self.controller.update(
            lane(), 150.0, 800.0, 0.0, True
        )
        progressing = self.controller.update(
            lane(), 150.0, 800.0, 0.30, True
        )
        finished = self.controller.update(
            lane(), 150.0, 800.0, 0.60, True
        )

        self.assertEqual(started.state, "changing_to_lane1")
        self.assertAlmostEqual(started.offset_px, 0.0)
        self.assertLess(abs(progressing.offset_px), 150.0)
        self.assertEqual(finished.state, "lane1")

    def test_avoidance_steering_slew_is_continuous_through_stabilization(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                smooth_avoidance=True,
                steering_min=80,
                steering_boost=25,
                steering_cap=150,
                steering_slew_limit=35,
                stabilizing_steering_min=0,
                stable_required_frames=5,
            )
        )
        idle = self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=0, reason="lane"),
            idle,
        )
        self.controller.request_avoidance("obstacle_fusion")
        changing1 = self.controller.update(lane(), 150.0, 800.0, 0.1, True)
        output1 = self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=-150, reason="lane"),
            changing1,
        )
        changing2 = self.controller.update(lane(), 150.0, 800.0, 0.2, True)
        output2 = self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=-150, reason="lane"),
            changing2,
        )
        stabilizing = self.controller.update(lane(), 150.0, 800.0, 1.2, True)
        output3 = self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=80, reason="lane"),
            stabilizing,
        )

        self.assertEqual(output1.steering, -35)
        self.assertEqual(output2.steering, -70)
        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertEqual(output3.steering, -35)
        self.assertEqual(output3.speed, 255)
        self.assertIn("lane_change_slew", output3.reason)

    def test_avoidance_releases_max_steering_inside_target_capture_zone(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                steering_cap=150,
                target_capture_error=0.20,
                target_capture_frames=2,
                stable_required_frames=5,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        first_capture = self.controller.update(
            lane_for_target_error(lateral_error_norm=-0.197, heading=0.32),
            150.0,
            800.0,
            1.0,
            True,
        )
        stabilizing = self.controller.update(
            lane_for_target_error(lateral_error_norm=-0.139, heading=0.37),
            150.0,
            800.0,
            1.1,
            True,
        )
        adjusted = self.controller.apply_steering_assist(
            ControlCommand(speed=120, steering=47, reason="lane"),
            stabilizing,
        )

        self.assertEqual(first_capture.state, "changing_to_lane1")
        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertEqual(stabilizing.direction, 0)
        self.assertEqual(adjusted.steering, 70)
        self.assertIn("lane_change_stabilize", adjusted.reason)

    def test_avoidance_stabilization_amplifies_weak_lane_feedback(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                stabilizing_steering_min=70,
                target_approach_error=0.32,
                target_capture_error=0.20,
                target_capture_frames=1,
                stable_required_frames=5,
                steering_cap=150,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        stabilizing = self.controller.update(
            lane_for_target_error(lateral_error_norm=0.16, heading=0.46),
            150.0,
            800.0,
            0.1,
            True,
        )

        adjusted = self.controller.apply_steering_assist(
            ControlCommand(speed=135, steering=44, reason="lane"),
            stabilizing,
        )

        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertEqual(adjusted.steering, 70)
        self.assertIn("lane_change_stabilize", adjusted.reason)

    def test_unreliable_avoidance_stabilization_uses_bounded_path_feedback(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                stabilizing_steering_min=90,
                steering_cap=150,
                target_capture_error=0.20,
                target_capture_frames=1,
                stable_required_frames=5,
                unreliable_steering_cap=90,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        captured = self.controller.update(
            lane_for_target_error(lateral_error_norm=0.05, heading=0.30),
            150.0,
            800.0,
            0.1,
            True,
            lane_reliable=True,
        )
        stabilizing = self.controller.update(
            lane_for_target_error(lateral_error_norm=0.17, heading=0.43),
            150.0,
            800.0,
            0.2,
            True,
            lane_reliable=False,
        )

        adjusted = self.controller.apply_steering_assist(
            ControlCommand(speed=255, steering=40, reason="lane"),
            stabilizing,
        )

        self.assertEqual(captured.state, "stabilizing_lane1")
        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertFalse(stabilizing.lane_reliable)
        self.assertEqual(adjusted.steering, 40)
        self.assertIn("lane_change_unreliable", adjusted.reason)

    def test_target_capture_requires_consecutive_frames(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                target_capture_error=0.20,
                target_capture_frames=2,
            )
        )
        self.controller.request_avoidance("obstacle_fusion")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)

        self.controller.update(
            lane_for_target_error(lateral_error_norm=-0.10),
            150.0,
            800.0,
            1.0,
            True,
        )
        reset = self.controller.update(
            lane_for_target_error(lateral_error_norm=-0.30),
            150.0,
            800.0,
            1.1,
            True,
        )

        self.assertEqual(reset.state, "changing_to_lane1")

    def test_return_request_waits_until_lane1_is_stable(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=1.0,
                stable_required_frames=2,
            )
        )
        self.controller.request("obstacle")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        stabilizing = self.controller.update(lane(heading=0.25), 150.0, 800.0, 1.0, True)

        accepted = self.controller.request_return("obstacle")

        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertFalse(accepted)

    def test_avoidance_return_waits_until_stabilizing_finishes(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                steering_override=False,
                steering_cap=150,
                stable_required_frames=5,
            )
        )
        self.controller.state = "stabilizing_lane1"

        accepted = self.controller.request_avoidance_return("obstacle_fusion")
        changing = self.controller.update(
            lane_for_shifted_lane1(heading=0.40),
            150.0,
            800.0,
            1.0,
            True,
        )
        adjusted = self.controller.apply_control_adjustments(
            ControlCommand(speed=255, steering=20, reason="lane"),
            changing,
        )

        self.assertFalse(accepted)
        self.assertEqual(changing.state, "stabilizing_lane1")
        self.assertEqual(changing.direction, 0)
        self.assertEqual(adjusted.steering, 20)

    def test_generic_request_uses_same_controller_entrypoint(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(mode="external", max_straight_heading=0.08)
        )
        self.update(0.0)

        accepted = self.controller.request("obstacle")
        result = self.update(1.0)

        self.assertTrue(accepted)
        self.assertEqual(self.controller.request_source, "obstacle")
        self.assertEqual(result.state, "changing_to_lane1")

    def test_external_mode_waits_for_return_request(self):
        self.controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                transition_seconds=2.0,
                hold_seconds=0.0,
                max_straight_heading=0.08,
                stable_required_frames=0,
            )
        )
        self.update(0.0)
        self.controller.request("obstacle")
        self.update(1.0)
        lane1 = self.update(3.0)
        still_lane1 = self.update(20.0)

        accepted = self.controller.request_return("obstacle_clear")
        returning = self.update(21.0)

        self.assertEqual(lane1.state, "lane1")
        self.assertEqual(still_lane1.state, "lane1")
        self.assertTrue(accepted)
        self.assertEqual(self.controller.return_source, "obstacle_clear")
        self.assertEqual(returning.state, "changing_to_lane2")

    def test_crosswalk_pause_does_not_advance_transition_clock(self):
        self.update(0.0)
        self.controller.request()
        self.update(1.0)

        self.controller.pause(1.5)
        self.controller.resume(11.5)
        halfway = self.update(12.0)

        self.assertEqual(halfway.state, "changing_to_lane1")
        self.assertAlmostEqual(halfway.offset_px, -75.0)

    def test_pause_resets_controller(self):
        self.update(0.0)
        self.controller.request()
        self.update(1.0)
        result = self.controller.update(lane(), 150.0, 800.0, 2.0, False)

        self.assertEqual(result.state, "lane2")
        self.assertEqual(result.lane, lane())


if __name__ == "__main__":
    unittest.main()
