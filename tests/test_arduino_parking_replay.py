import math
import unittest
from pathlib import Path

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.parking_config import load_parking_config
from skku_autocar.planning.t_parking_planner import TParkingPlanner
from skku_autocar.runtime.arduino_parking_replay import (
    ArduinoReplayConstants,
    ArduinoParkingControllerReplay,
    CameraSample,
    ReplayParkingState,
    SharedParkingPlannerReplay,
    latest_camera_sample,
)


ROOT = Path(__file__).resolve().parents[1]


def lidar_observation(
    *,
    confirmed=False,
    found=False,
    center_y_back_mm=None,
    center_x_right_mm=None,
    depth_x=None,
    depth_y=None,
    entry_target_y_back_mm=None,
) -> LidarParkingObservation:
    return LidarParkingObservation(
        valid=True,
        gap_found=found,
        gap_confirmed=confirmed,
        gap_center_y_back_mm=center_y_back_mm,
        gap_center_x_right_mm=center_x_right_mm,
        slot_depth_x_right=depth_x,
        slot_depth_y_back=depth_y,
        entry_target_y_back_mm=entry_target_y_back_mm,
    )


class ArduinoParkingControllerReplayTests(unittest.TestCase):
    def test_actual_replay_adapter_uses_same_planner_and_commands_as_parking_runtime(self):
        config = load_parking_config(str(ROOT / "configs" / "parking.json"))
        replay = SharedParkingPlannerReplay(config)
        live = TParkingPlanner(config.planner, config.path)
        live.start(0.0)
        first_car = LidarParkingObservation(
            valid=True,
            car_count=1,
            first_car_seen=True,
            first_car_confirmed=True,
            first_car_turn_error_mm=-5.0,
            first_car_turn_reached=True,
        )
        geometry = ParkingGeometry(reason="not_visible_yet")

        replay_command = replay.update(first_car, geometry, 0.1)
        live_plan = live.update(geometry, first_car, 0.1)

        self.assertIsInstance(replay.planner, TParkingPlanner)
        self.assertEqual(replay.planner_state, live_plan.state)
        self.assertEqual(replay_command.speed, live_plan.command.speed)
        self.assertEqual(replay_command.steering_deg, live_plan.command.steering)
        self.assertEqual(replay_command.event, live_plan.reason)

    def test_state_flow_and_rear_lidar_finish(self):
        controller = ArduinoParkingControllerReplay(
            ArduinoReplayConstants(prealign_enabled=False)
        )
        empty = lidar_observation()
        command = controller.update(empty, 100.0, None)
        self.assertEqual(command.state, ReplayParkingState.SEARCHING)
        self.assertEqual(command.speed, 28)

        gap = lidar_observation(confirmed=True, found=True, center_y_back_mm=180.0)
        for _ in range(3):
            command = controller.update(gap, 100.0, None)
        self.assertEqual(command.state, ReplayParkingState.POSITIONING)
        self.assertEqual(command.speed, 0)

        for _ in range(5):
            command = controller.update(gap, 100.0, None)
        self.assertEqual(command.state, ReplayParkingState.REVERSING)

        for _ in range(3):
            command = controller.update(gap, 20.0, 10.0)
        self.assertEqual(command.state, ReplayParkingState.FINISHED)
        self.assertEqual(command.speed, 0)

    def test_positioning_moves_forward_for_gap_ahead(self):
        controller = ArduinoParkingControllerReplay()
        controller.state = ReplayParkingState.POSITIONING
        gap = lidar_observation(found=True, center_y_back_mm=-300.0)
        command = controller.update(gap, 100.0, None)
        self.assertEqual(command.speed, 16)

    def test_camera_and_side_ultrasonic_combine_during_reverse(self):
        controller = ArduinoParkingControllerReplay()
        controller.state = ReplayParkingState.REVERSING
        command = controller.update(
            lidar_observation(),
            rear_lidar_cm=100.0,
            line_error_px=20.0,
            left_ultrasonic_cm=40.0,
            right_ultrasonic_cm=30.0,
        )
        self.assertLess(command.steering_deg, 0)
        self.assertEqual(command.speed, -22)

    def test_missing_values_do_not_trigger_emergency(self):
        controller = ArduinoParkingControllerReplay()
        command = controller.update(
            lidar_observation(),
            rear_lidar_cm=None,
            line_error_px=math.nan,
        )
        self.assertEqual(command.state, ReplayParkingState.SEARCHING)

    def test_zero_distance_latches_emergency(self):
        controller = ArduinoParkingControllerReplay()
        command = controller.update(lidar_observation(), 0.0, None)
        self.assertEqual(command.state, ReplayParkingState.EMERGENCY_STOP)
        self.assertEqual(command.speed, 0)

    def test_camera_replay_never_uses_a_future_sample(self):
        samples = [
            CameraSample(1.0, 10.0, True, "ok", 1.0),
            CameraSample(2.0, 20.0, True, "ok", 1.0),
        ]
        self.assertIsNone(latest_camera_sample(samples, 0.5))
        self.assertEqual(latest_camera_sample(samples, 1.5).line_error_px, 10.0)

    def test_positioning_prealign_uses_max_left_then_reverses_when_aligned(self):
        constants = ArduinoReplayConstants(
            position_confirm_cycles=1,
            prealign_enabled=True,
            prealign_steer_settle_s=0.0,
            prealign_confirm_cycles=2,
        )
        controller = ArduinoParkingControllerReplay(constants)
        controller.state = ReplayParkingState.POSITIONING
        aligned_position = lidar_observation(
            confirmed=True,
            found=True,
            center_y_back_mm=constants.position_target_cm * 10.0,
            center_x_right_mm=1000.0,
            depth_x=1.0,
            depth_y=0.0,
            entry_target_y_back_mm=constants.position_target_cm * 10.0,
        )

        entered = controller.update(
            aligned_position,
            100.0,
            None,
            elapsed_s=1.0,
        )
        moving = controller.update(
            aligned_position,
            100.0,
            None,
            elapsed_s=1.1,
        )
        direct_pose = lidar_observation(
            confirmed=True,
            found=True,
            center_y_back_mm=(constants.position_target_cm + 90.0) * 10.0,
            center_x_right_mm=50.0,
            depth_x=0.05,
            depth_y=1.0,
            entry_target_y_back_mm=constants.position_target_cm * 10.0,
        )
        confirming = controller.update(
            direct_pose,
            100.0,
            None,
            elapsed_s=1.2,
        )
        ready = controller.update(
            direct_pose,
            100.0,
            None,
            elapsed_s=1.3,
        )

        self.assertEqual(entered.state, ReplayParkingState.POSITIONING)
        self.assertEqual(entered.steering_deg, -45)
        self.assertEqual(moving.speed, constants.prealign_speed)
        self.assertEqual(moving.steering_deg, -45)
        self.assertEqual(confirming.state, ReplayParkingState.POSITIONING)
        self.assertEqual(ready.state, ReplayParkingState.REVERSING)
        self.assertEqual(ready.event, "prealign_direct_reverse_ready")


if __name__ == "__main__":
    unittest.main()
