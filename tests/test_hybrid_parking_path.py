import math
import unittest

from skku_autocar.planning.hybrid_parking_path import (
    HybridAStarParkingPathPlanner,
    PathPose,
    VehicleModel,
    build_slot_maneuver_model,
    polygons_intersect,
    pure_pursuit_curvature,
    vehicle_footprint,
)


SENSOR_TO_AXLE_Y_BACK_MM = -300.0


def lidar_polygon_for_right_bay(
    entrance_x_mm=900.0,
    entrance_center_y_mm=0.0,
):
    planning_polygon = (
        (entrance_x_mm, entrance_center_y_mm - 475.0),
        (entrance_x_mm, entrance_center_y_mm + 475.0),
        (entrance_x_mm + 1500.0, entrance_center_y_mm + 475.0),
        (entrance_x_mm + 1500.0, entrance_center_y_mm - 475.0),
    )
    return tuple(
        (x, SENSOR_TO_AXLE_Y_BACK_MM - y)
        for x, y in planning_polygon
    )


def lidar_polygon_when_parked():
    planning_polygon = (
        (-475.0, 1180.0),
        (475.0, 1180.0),
        (475.0, -320.0),
        (-475.0, -320.0),
    )
    return tuple(
        (x, SENSOR_TO_AXLE_Y_BACK_MM - y)
        for x, y in planning_polygon
    )


class HybridParkingPathTest(unittest.TestCase):
    def setUp(self):
        self.vehicle = VehicleModel()
        self.planner = HybridAStarParkingPathPlanner(self.vehicle)

    def test_plans_forward_setup_then_reverse_into_right_bay(self):
        model = build_slot_maneuver_model(
            lidar_polygon_for_right_bay(),
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
            vehicle=self.vehicle,
        )
        self.assertIsNotNone(model)

        path = self.planner.plan(model.parking_goal, model.obstacles)

        self.assertTrue(path.found, path.reason)
        self.assertEqual(path.first_gear, 1)
        self.assertIn(-1, {pose.gear for pose in path.poses})
        self.assertLessEqual(
            math.hypot(
                path.poses[-1].x_right_mm - model.parking_goal.x_right_mm,
                path.poses[-1].y_forward_mm - model.parking_goal.y_forward_mm,
            ),
            self.planner.config.goal_position_tolerance_mm,
        )
        for pose in path.poses:
            footprint = vehicle_footprint(pose, self.vehicle)
            self.assertFalse(
                any(
                    polygons_intersect(footprint, obstacle)
                    for obstacle in model.obstacles
                )
            )

    def test_changed_longitudinal_start_positions_still_plan(self):
        for center_y in (-600.0, 0.0, 600.0):
            with self.subTest(center_y=center_y):
                model = build_slot_maneuver_model(
                    lidar_polygon_for_right_bay(
                        entrance_center_y_mm=center_y,
                    ),
                    sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
                    vehicle=self.vehicle,
                )
                path = self.planner.plan(
                    model.parking_goal,
                    model.obstacles,
                )
                self.assertTrue(path.found, path.reason)

    def test_slot_coordinate_axis_is_stable_when_border_car_order_swaps(self):
        polygon = lidar_polygon_for_right_bay()
        swapped = (polygon[1], polygon[0], polygon[3], polygon[2])
        first = build_slot_maneuver_model(
            polygon,
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
            vehicle=self.vehicle,
        )
        second = build_slot_maneuver_model(
            swapped,
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
            vehicle=self.vehicle,
        )

        self.assertAlmostEqual(first.slot_axis[0], second.slot_axis[0])
        self.assertAlmostEqual(first.slot_axis[1], second.slot_axis[1])
        self.assertAlmostEqual(
            first.depth_direction[0],
            second.depth_direction[0],
        )
        self.assertAlmostEqual(
            first.depth_direction[1],
            second.depth_direction[1],
        )

    def test_reverse_pure_pursuit_steers_toward_right_hand_target(self):
        target = PathPose(
            x_right_mm=250.0,
            y_forward_mm=-500.0,
            heading_rad=0.0,
            gear=-1,
        )
        self.assertGreater(pure_pursuit_curvature(target), 0.0)

    def test_exit_is_planned_forward_to_lane_pose(self):
        model = build_slot_maneuver_model(
            lidar_polygon_when_parked(),
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
            vehicle=self.vehicle,
        )

        path = self.planner.plan(
            model.exit_goal,
            model.obstacles,
            initial_gear=1,
            allowed_gears=(1,),
        )

        self.assertTrue(path.found, path.reason)
        self.assertEqual({pose.gear for pose in path.poses[1:]}, {1})

    def test_forward_exit_still_plans_with_small_final_pose_error(self):
        global_slot = (
            (900.0, -475.0),
            (900.0, 475.0),
            (2400.0, 475.0),
            (2400.0, -475.0),
        )
        vehicle_x = 1970.0
        vehicle_y = -23.0
        vehicle_heading = math.radians(-96.0)

        def current_frame(point):
            dx = point[0] - vehicle_x
            dy = point[1] - vehicle_y
            return (
                dx * math.cos(vehicle_heading)
                - dy * math.sin(vehicle_heading),
                dx * math.sin(vehicle_heading)
                + dy * math.cos(vehicle_heading),
            )

        lidar_polygon = tuple(
            (x, SENSOR_TO_AXLE_Y_BACK_MM - y)
            for x, y in (current_frame(point) for point in global_slot)
        )
        model = build_slot_maneuver_model(
            lidar_polygon,
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
            vehicle=self.vehicle,
        )

        path = self.planner.plan(
            model.exit_goal,
            model.obstacles,
            initial_gear=1,
            allowed_gears=(1,),
        )

        self.assertTrue(path.found, path.reason)
        self.assertEqual({pose.gear for pose in path.poses[1:]}, {1})


if __name__ == "__main__":
    unittest.main()
