import unittest

from skku_autocar.estimation.lane_geometry import LaneGeometry
from skku_autocar.planning.lane_change_test import LaneChangeTestConfig, LaneChangeTestController
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


class LaneChangeTestControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="manual",
                transition_seconds=2.0,
                hold_seconds=3.0,
                max_straight_heading=0.08,
                speed_cap=70,
                stable_required_frames=0,
            )
        )

    def update(self, now, geometry=None):
        return self.controller.update(geometry or lane(), 150.0, 800.0, now, True)

    def test_manual_request_moves_lane2_to_lane1_and_returns(self):
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
        controller = LaneChangeTestController(
            LaneChangeTestConfig(mode="timed", trigger_seconds=5.0)
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

    def test_settle_keeps_hard_steering_after_timed_shift_finishes(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=1.0,
                settle_seconds=0.7,
                steering_min=120,
                steering_boost=0,
                steering_cap=150,
                steering_override=True,
                speed_cap=70,
                stable_required_frames=0,
            )
        )
        self.update(0.0)
        self.controller.request("obstacle")
        self.update(0.1)
        settling = self.update(1.1)
        command = ControlCommand(speed=255, steering=0, reason="lane")

        assisted = self.controller.apply_control_adjustments(command, settling)

        self.assertEqual(settling.state, "lane1")
        self.assertEqual(settling.direction, -1)
        self.assertEqual(assisted.speed, 70)
        self.assertEqual(assisted.steering, -150)

    def test_settle_releases_to_normal_control_after_timeout(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=1.0,
                settle_seconds=0.5,
                steering_min=120,
                steering_cap=150,
                steering_override=True,
                stable_required_frames=0,
            )
        )
        self.update(0.0)
        self.controller.request("obstacle")
        self.update(0.1)
        self.update(1.1)
        lane1 = self.update(1.7)
        command = ControlCommand(speed=255, steering=0, reason="lane")

        assisted = self.controller.apply_control_adjustments(command, lane1)

        self.assertEqual(lane1.state, "lane1")
        self.assertEqual(lane1.direction, 0)
        self.assertEqual(assisted.steering, 0)
        self.assertEqual(assisted.speed, 255)

    def test_armed_state_caps_speed_before_steering_can_start(self):
        controller = LaneChangeTestController(
            LaneChangeTestConfig(mode="keyboard", speed_cap=80, stable_required_frames=0)
        )
        controller.request("obstacle")
        result = controller.update(lane(heading=0.2), 150.0, 800.0, 1.0, True)
        command = ControlCommand(speed=255, steering=0, reason="lane")

        capped = controller.apply_control_adjustments(command, result)

        self.assertEqual(result.state, "armed")
        self.assertEqual(capped.speed, 80)
        self.assertEqual(capped.steering, 0)

    def test_transition_requires_center_stability_before_lane1_is_final(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=1.0,
                stable_required_frames=3,
                stable_lateral_error=0.12,
                stable_heading_error=0.18,
                stabilize_timeout_seconds=0.0,
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
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=1.0,
                stable_required_frames=3,
                stabilize_timeout_seconds=0.0,
                speed_cap=70,
                steering_override=True,
                steering_cap=150,
                recenter_steering=False,
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

    def test_stabilizing_counter_steers_after_left_lane_change(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=1.0,
                stable_required_frames=5,
                stabilize_timeout_seconds=0.0,
                speed_cap=60,
                steering_override=True,
                steering_cap=150,
                recenter_steering=True,
                recenter_lateral_error=0.35,
                recenter_heading_error=0.12,
            )
        )
        self.controller.request("obstacle")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        stabilizing = self.controller.update(
            lane_for_target_error(lateral_error_norm=-0.20, heading=0.30),
            150.0,
            800.0,
            1.0,
            True,
        )
        command = ControlCommand(speed=255, steering=-30, reason="lane")

        adjusted = self.controller.apply_control_adjustments(command, stabilizing)

        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertEqual(stabilizing.direction, 1)
        self.assertEqual(adjusted.speed, 60)
        self.assertEqual(adjusted.steering, 150)

    def test_return_request_waits_until_lane1_is_stable(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=1.0,
                stable_required_frames=2,
                stabilize_timeout_seconds=0.0,
            )
        )
        self.controller.request("obstacle")
        self.controller.update(lane(), 150.0, 800.0, 0.0, True)
        stabilizing = self.controller.update(lane(heading=0.25), 150.0, 800.0, 1.0, True)

        accepted = self.controller.request_return("obstacle")

        self.assertEqual(stabilizing.state, "stabilizing_lane1")
        self.assertFalse(accepted)

    def test_future_obstacle_detector_uses_same_request_entrypoint(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(mode="external", max_straight_heading=0.08)
        )
        self.update(0.0)

        accepted = self.controller.request("obstacle")
        result = self.update(1.0)

        self.assertTrue(accepted)
        self.assertEqual(self.controller.request_source, "obstacle")
        self.assertEqual(result.state, "changing_to_lane1")

    def test_external_obstacle_mode_waits_for_clear_request_before_returning(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
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

    def test_keyboard_mode_waits_for_second_key_before_returning(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="keyboard",
                transition_seconds=2.0,
                hold_seconds=0.0,
                max_straight_heading=0.08,
                stable_required_frames=0,
            )
        )
        self.update(0.0)
        self.controller.request("keyboard_obstacle")
        self.update(1.0)
        lane1 = self.update(3.0)
        still_lane1 = self.update(20.0)

        self.controller.request_return("keyboard_clear")
        returning = self.update(21.0)

        self.assertEqual(lane1.state, "lane1")
        self.assertEqual(still_lane1.state, "lane1")
        self.assertEqual(returning.state, "changing_to_lane2")

    def test_pause_resets_test(self):
        self.update(0.0)
        self.controller.request()
        self.update(1.0)
        result = self.controller.update(lane(), 150.0, 800.0, 2.0, False)

        self.assertEqual(result.state, "lane2")
        self.assertEqual(result.lane, lane())


if __name__ == "__main__":
    unittest.main()
