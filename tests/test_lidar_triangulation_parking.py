import unittest

from skku_autocar.estimation.lidar_triangulation import (
    decision_triangle_from_observation,
)
from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.model_based_parking import (
    ModelBasedParkingConfig,
    ModelBasedTParkingPlanner,
)
from skku_autocar.planning.t_parking_planner import ParkingState


class LidarDecisionTriangleTest(unittest.TestCase):
    def test_law_of_cosines_and_rear_axis_correction(self):
        observation = LidarParkingObservation(
            valid=True,
            gap_pair_observed=True,
            second_car_seen=True,
            gap_near_edge_x_right_mm=300.0,
            gap_near_edge_y_back_mm=0.0,
            gap_far_edge_x_right_mm=300.0,
            gap_far_edge_y_back_mm=400.0,
            slot_depth_x_right=1.0,
            slot_depth_y_back=0.0,
        )

        triangle = decision_triangle_from_observation(observation)

        self.assertTrue(triangle.valid)
        self.assertAlmostEqual(triangle.lidar_to_car1_mm, 300.0)
        self.assertAlmostEqual(triangle.lidar_to_car2_mm, 500.0)
        self.assertAlmostEqual(triangle.car_gap_mm, 400.0)
        self.assertAlmostEqual(triangle.decision_angle_deg, 90.0)
        self.assertAlmostEqual(triangle.correction_angle_deg, 90.0)

    def test_coasted_pair_cannot_create_triangle(self):
        triangle = decision_triangle_from_observation(
            LidarParkingObservation(
                valid=True,
                gap_pair_observed=True,
                second_car_seen=True,
                coasted=True,
                gap_near_edge_x_right_mm=100.0,
                gap_near_edge_y_back_mm=100.0,
                gap_far_edge_x_right_mm=1000.0,
                gap_far_edge_y_back_mm=100.0,
            )
        )

        self.assertFalse(triangle.valid)
        self.assertEqual(triangle.reason, "fresh_two_car_pair_required")


class LidarOnlyEntryTriggerTest(unittest.TestCase):
    def test_first_car_close_then_stable_loss_starts_left_setup(self):
        planner = ModelBasedTParkingPlanner(
            ModelBasedParkingConfig(
                lidar_only_enabled=True,
                lidar_first_car_confirm_scans=3,
                lidar_first_car_lost_scans=3,
                auto_exit_enabled=False,
            )
        )
        planner.start(0.0)
        geometry = ParkingGeometry()

        for index in range(3):
            plan = planner.update(
                geometry,
                LidarParkingObservation(
                    timestamp=1.0 + index,
                    valid=True,
                    first_car_confirmed=True,
                    first_car_slot_edge_x_right_mm=600.0,
                    first_car_slot_edge_y_back_mm=1200.0,
                ),
                None,
                1.0 + index,
            )
        self.assertEqual(plan.state, ParkingState.TRACK_GAP)

        # Repeating one scan must not satisfy the fresh-scan debounce.
        repeated = planner.update(
            geometry,
            LidarParkingObservation(timestamp=3.0, valid=True),
            None,
            3.1,
        )
        self.assertEqual(repeated.state, ParkingState.TRACK_GAP)
        self.assertIn("0/3", repeated.reason)

        for index in range(3):
            plan = planner.update(
                geometry,
                LidarParkingObservation(
                    timestamp=4.0 + index,
                    valid=True,
                ),
                None,
                4.0 + index,
            )

        self.assertEqual(plan.state, ParkingState.ENTRY_SETUP)
        self.assertGreater(plan.command.speed, 0)
        self.assertLess(plan.command.steering, 0)
        self.assertIn("lidar_open_gap_confirmed", plan.reason)
        self.assertTrue(planner.consume_lidar_reset_request())

    def test_mechanical_neutral_mapping_preserves_end_stops(self):
        planner = ModelBasedTParkingPlanner(
            ModelBasedParkingConfig(
                straight_steering_trim=-33,
                max_steering_command=150,
            )
        )

        self.assertEqual(planner._physical_steering_command(0), -33)
        self.assertEqual(planner._physical_steering_command(-150), -150)
        self.assertEqual(planner._physical_steering_command(150), 150)


if __name__ == "__main__":
    unittest.main()
