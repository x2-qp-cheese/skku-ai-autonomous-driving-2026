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

    def test_avoidance_forces_max_steering_until_target_arrival(self):
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
        self.assertEqual(adjusted.steering, -20)
        self.assertIn("lane_change_feedback", adjusted.reason)

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
        self.assertEqual(adjusted.steering, 47)

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

    def test_pause_resets_controller(self):
        self.update(0.0)
        self.controller.request()
        self.update(1.0)
        result = self.controller.update(lane(), 150.0, 800.0, 2.0, False)

        self.assertEqual(result.state, "lane2")
        self.assertEqual(result.lane, lane())


if __name__ == "__main__":
    unittest.main()
