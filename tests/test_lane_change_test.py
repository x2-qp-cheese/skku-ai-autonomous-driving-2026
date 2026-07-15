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


class LaneChangeTestControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = LaneChangeTestController(
            LaneChangeTestConfig(
                mode="manual",
                transition_seconds=2.0,
                hold_seconds=3.0,
                max_straight_heading=0.08,
                speed_cap=70,
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

    def test_pause_resets_test(self):
        self.update(0.0)
        self.controller.request()
        self.update(1.0)
        result = self.controller.update(lane(), 150.0, 800.0, 2.0, False)

        self.assertEqual(result.state, "lane2")
        self.assertEqual(result.lane, lane())


if __name__ == "__main__":
    unittest.main()
