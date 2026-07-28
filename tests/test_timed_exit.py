from dataclasses import replace
import unittest

from skku_autocar.config import PaperControllerConfig, RearLidarConfig
from skku_autocar.perception.rear_lidar import (
    RearLidarObservation,
    RearLidarPerception,
)
from skku_autocar.planning.paper_controller import PaperParkingController
from skku_autocar.sensors.lidar import LidarPoint, LidarScan
from skku_autocar.types import ParkingState


class TimedExitTest(unittest.TestCase):
    def test_perception_preserves_closest_cd_y_coordinates(self):
        perception = RearLidarPerception(RearLidarConfig())
        observation = perception.observe(
            LidarScan(
                timestamp=1.0,
                points=(
                    LidarPoint(15, 350.0, 1000.0),
                    LidarPoint(15, 190.0, 1000.0),
                ),
            )
        )

        self.assertAlmostEqual(observation.c_y_back_mm, 173.65, places=2)
        self.assertAlmostEqual(observation.d_y_back_mm, 173.65, places=2)

    def test_both_side_distance_jumps_finish_parking(self):
        config = PaperControllerConfig()
        controller = PaperParkingController(config)
        controller.state = ParkingState.REVERSE_STRAIGHT
        controller._direct_reverse_committed = True

        command = controller.update(
            RearLidarObservation(
                timestamp=1.0,
                valid=True,
                dist_c_mm=650.0,
                dist_d_mm=850.0,
            ),
            1.0,
        )
        self.assertEqual(controller.state, ParkingState.REVERSE_STRAIGHT)
        self.assertEqual(command.speed, config.inside_reverse_speed)

        command = controller.update(
            RearLidarObservation(
                timestamp=2.0,
                valid=True,
                dist_c_mm=755.0,
                dist_d_mm=855.0,
            ),
            2.0,
        )
        self.assertEqual(controller.state, ParkingState.REVERSE_STRAIGHT)

        command = controller.update(
            RearLidarObservation(
                timestamp=3.0,
                valid=True,
                dist_c_mm=760.0,
                dist_d_mm=960.0,
            ),
            3.0,
        )

        self.assertEqual(controller.state, ParkingState.PARKED)
        self.assertTrue(command.brake)
        self.assertEqual(
            command.reason,
            "paper_both_side_distance_jumps_finish",
        )

    def test_parked_runs_forward_right_then_straight_without_lidar(self):
        config = replace(
            PaperControllerConfig(),
            park_hold_s=4.0,
            exit_speed=50,
            exit_forward_s=4.0,
            exit_turn_steering=150,
            exit_turn_right_s=5.0,
            actuator_steering_offset=-28,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.PARKED
        controller._parked_started_at = 0.0
        no_lidar = RearLidarObservation(timestamp=0.0, valid=False)

        command = controller.update(no_lidar, 3.99)
        self.assertTrue(command.brake)

        command = controller.update(no_lidar, 4.0)
        self.assertEqual(controller.state, ParkingState.EXIT_FORWARD)
        self.assertEqual((command.speed, command.steering), (50, -28))

        command = controller.update(no_lidar, 8.0)
        self.assertEqual(controller.state, ParkingState.EXIT_TURN_RIGHT)
        self.assertEqual((command.speed, command.steering), (50, 150))

        command = controller.update(no_lidar, 13.0)
        self.assertEqual(controller.state, ParkingState.EXIT_STRAIGHT)
        self.assertEqual((command.speed, command.steering), (50, -28))

        command = controller.update(no_lidar, 20.0)
        self.assertEqual((command.speed, command.steering), (50, -28))


if __name__ == "__main__":
    unittest.main()
