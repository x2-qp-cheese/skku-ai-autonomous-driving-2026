from dataclasses import replace
import unittest

from skku_autocar.config import PaperControllerConfig, RearLidarConfig
from skku_autocar.perception.rear_lidar import (
    RearLidarObservation,
    RearLidarPerception,
    TangentPair,
)
from skku_autocar.planning.paper_controller import PaperParkingController
from skku_autocar.sensors.lidar import LidarPoint, LidarScan
from skku_autocar.types import ParkingState


class PaperParkingFlowTest(unittest.TestCase):
    def make_controller(self) -> PaperParkingController:
        config = replace(
            PaperControllerConfig(),
            actuator_steering_offset=-28,
            dist_bias_cd_threshold_mm=220.0,
            recovery_forward_s=3.0,
        )
        controller = PaperParkingController(config)
        controller.state = ParkingState.CENTER_CHECK
        return controller

    def test_cd_bias_below_threshold_starts_straight_reverse(self):
        controller = self.make_controller()
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
        self.assertEqual(
            (command.speed, command.steering),
            (controller.config.reverse_speed, -28),
        )

    def test_cd_bias_at_threshold_starts_paper_recovery(self):
        controller = self.make_controller()
        command = controller.update(
            RearLidarObservation(
                timestamp=1.0,
                valid=True,
                dist_c_mm=650.0,
                dist_d_mm=870.0,
            ),
            1.0,
        )

        self.assertEqual(controller.state, ParkingState.RECOVERY_FORWARD)
        self.assertEqual(
            (command.speed, command.steering),
            (controller.config.forward_speed, -28),
        )

    def test_both_cd_none_starts_paper_recovery(self):
        controller = self.make_controller()
        command = controller.update(
            RearLidarObservation(timestamp=1.0, valid=True),
            1.0,
        )

        self.assertEqual(controller.state, ParkingState.RECOVERY_FORWARD)
        self.assertEqual(command.speed, controller.config.forward_speed)

    def test_one_cd_none_starts_paper_recovery_before_centering(self):
        controller = self.make_controller()
        command = controller.update(
            RearLidarObservation(
                timestamp=1.0,
                valid=True,
                dist_c_mm=None,
                dist_d_mm=1006.0,
            ),
            1.0,
        )

        self.assertEqual(controller.state, ParkingState.RECOVERY_FORWARD)
        self.assertEqual(command.speed, controller.config.forward_speed)
        self.assertIn("paper_recovery_forward", command.reason)

    def test_reverse_uses_paper_equation_five(self):
        controller = self.make_controller()
        controller.state = ParkingState.REVERSE_ALIGN
        command = controller.update(
            RearLidarObservation(
                timestamp=1.0,
                valid=True,
                near=False,
                pair=TangentPair(
                    valid=True,
                    angle_a_deg=20.0,
                    angle_b_deg=40.0,
                    dist_a_mm=900.0,
                    dist_b_mm=950.0,
                    angle_bisector_deg=30.0,
                    reason="test_pair",
                ),
            ),
            1.0,
        )

        self.assertEqual(command.speed, controller.config.reverse_speed)
        self.assertEqual(command.steering, 149)
        self.assertIn("paper_eq5_reverse", command.reason)
        self.assertAlmostEqual(
            controller.debug.paper_steering,
            6.9513888889,
        )
        self.assertEqual(
            controller.debug.applied_paper_steering,
            controller.debug.paper_steering,
        )

    def test_reverse_steers_toward_detected_right_entry(self):
        controller = self.make_controller()
        controller.state = ParkingState.REVERSE_ALIGN
        command = controller.update(
            RearLidarObservation(
                timestamp=1.0,
                valid=True,
                near=False,
                pair=TangentPair(
                    valid=True,
                    angle_a_deg=16.0,
                    angle_b_deg=52.0,
                    dist_a_mm=2791.0,
                    dist_b_mm=1584.0,
                    angle_bisector_deg=34.0,
                    reason="recorded_right_entry",
                ),
            ),
            1.0,
        )

        self.assertEqual(command.speed, controller.config.reverse_speed)
        self.assertEqual(command.steering, 150)
        self.assertEqual(controller.debug.paper_steering, 7.0)
        self.assertEqual(
            controller.debug.applied_paper_steering,
            7.0,
        )

    def test_one_cd_none_finishes_after_centered_reverse(self):
        controller = self.make_controller()
        controller.state = ParkingState.REVERSE_STRAIGHT
        command = controller.update(
            RearLidarObservation(
                timestamp=1.0,
                valid=True,
                dist_c_mm=None,
                dist_d_mm=850.0,
            ),
            1.0,
        )

        self.assertEqual(controller.state, ParkingState.PARKED)
        self.assertTrue(command.brake)
        self.assertEqual(
            command.reason,
            "paper_dist_c_or_d_none_finish",
        )

    def test_parked_remains_stopped_during_hold(self):
        controller = self.make_controller()
        controller.state = ParkingState.PARKED
        controller._parked_started_at = 0.0
        command = controller.update(
            RearLidarObservation(timestamp=1.0, valid=False),
            controller.config.park_hold_s - 0.01,
        )

        self.assertEqual(controller.state, ParkingState.PARKED)
        self.assertTrue(command.brake)

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


if __name__ == "__main__":
    unittest.main()
