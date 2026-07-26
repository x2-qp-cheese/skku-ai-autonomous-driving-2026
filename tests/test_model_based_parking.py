import math
import unittest

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.lidar_slot_geometry import point_in_convex_polygon
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.hybrid_parking_path import (
    PathPose,
    VehicleModel,
    vehicle_footprint,
    wrap_angle,
)
from skku_autocar.planning.model_based_parking import (
    ModelBasedParkingConfig,
    ModelBasedTParkingPlanner,
)
from skku_autocar.planning.t_parking_planner import ParkingState


SENSOR_TO_AXLE_Y_BACK_MM = -300.0


def lidar_polygon_from_planning(points):
    return tuple(
        (x, SENSOR_TO_AXLE_Y_BACK_MM - y)
        for x, y in points
    )


def right_bay():
    return lidar_polygon_from_planning(
        (
            (900.0, -475.0),
            (900.0, 475.0),
            (2400.0, 475.0),
            (2400.0, -475.0),
        )
    )


def current_pose_is_parked_bay():
    return lidar_polygon_from_planning(
        (
            (-475.0, 1180.0),
            (475.0, 1180.0),
            (475.0, -320.0),
            (-475.0, -320.0),
        )
    )


def confirmed_lidar(unsafe=False):
    return LidarParkingObservation(
        valid=True,
        unsafe=unsafe,
        gap_found=True,
        gap_confirmed=True,
        gap_pair_observed=True,
    )


class ModelBasedParkingPlannerTest(unittest.TestCase):
    def make_planner(self, **changes):
        config = ModelBasedParkingConfig(
            slot_lock_confirm_scans=2,
            gear_change_stop_frames=2,
            auto_exit_enabled=False,
            emergency_stop_enabled=True,
            **changes,
        )
        return ModelBasedTParkingPlanner(
            config,
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
        )

    def test_motion_is_path_and_pose_driven_without_entry_seconds(self):
        planner = self.make_planner()
        geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            confidence=0.95,
        )
        lidar = confirmed_lidar()
        polygon = right_bay()
        planner.start(0.0)

        first = planner.update(geometry, lidar, polygon, 0.1)
        second = planner.update(geometry, lidar, polygon, 0.2)
        armed = planner.update(geometry, lidar, polygon, 0.3)
        settle_one = planner.update(geometry, lidar, polygon, 0.4)
        settle_two = planner.update(geometry, lidar, polygon, 0.5)
        moving = planner.update(geometry, lidar, polygon, 0.6)

        self.assertEqual(first.state, ParkingState.TRACK_GAP)
        self.assertEqual(second.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(settle_one.command.speed, 0)
        self.assertEqual(settle_two.command.speed, 0)
        self.assertGreater(moving.command.speed, 0)
        self.assertIsNotNone(moving.world_path)
        self.assertTrue(moving.world_path.found)

    def test_front_center_sensor_latches_emergency_stop(self):
        planner = self.make_planner()
        planner.start(0.0)

        plan = planner.update(
            ParkingGeometry(),
            LidarParkingObservation(valid=True),
            None,
            0.1,
            front_center_ultrasonic_mm=90.0,
        )

        self.assertEqual(plan.state, ParkingState.EMERGENCY_STOP)
        self.assertEqual(plan.command.speed, 0)

    def test_parking_completion_requires_pose_and_full_footprint(self):
        planner = self.make_planner(parking_complete_confirm_scans=2)
        geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            vehicle_inside_ratio=1.0,
            vehicle_fully_inside=True,
            confidence=0.95,
        )
        lidar = confirmed_lidar()
        polygon = current_pose_is_parked_bay()
        planner.start(0.0)
        planner.update(geometry, lidar, polygon, 0.1)
        planner.update(geometry, lidar, polygon, 0.2)
        planner.update(geometry, lidar, polygon, 0.3)
        confirming = planner.update(geometry, lidar, polygon, 0.4)
        parked = planner.update(geometry, lidar, polygon, 0.5)

        self.assertIn("confirming", confirming.reason)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertEqual(parked.command.speed, 0)
        still_parked = planner.update(geometry, lidar, polygon, 3.7)
        self.assertEqual(still_parked.state, ParkingState.PARKED)
        self.assertIn("3_to_5", still_parked.reason)
        hold_complete = planner.update(geometry, lidar, polygon, 4.0)
        self.assertIn("auto_exit_disabled", hold_complete.reason)

    def test_closed_loop_kinematic_simulation_parks_without_gear_chatter(self):
        vehicle = VehicleModel()
        planner = ModelBasedTParkingPlanner(
            ModelBasedParkingConfig(
                slot_lock_confirm_scans=1,
                auto_exit_enabled=False,
            ),
            vehicle,
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
        )
        global_slot = (
            (900.0, -475.0),
            (900.0, 475.0),
            (2400.0, 475.0),
            (2400.0, -475.0),
        )
        vehicle_pose = [0.0, 0.0, 0.0]
        now = 0.0
        dt = 0.12
        gears = []
        planner.start(now)

        for _ in range(220):
            heading = vehicle_pose[2]

            def current_frame(point):
                dx = point[0] - vehicle_pose[0]
                dy = point[1] - vehicle_pose[1]
                return (
                    dx * math.cos(heading) - dy * math.sin(heading),
                    dx * math.sin(heading) + dy * math.cos(heading),
                )

            current_slot = tuple(current_frame(point) for point in global_slot)
            lidar_slot = tuple(
                (x, SENSOR_TO_AXLE_Y_BACK_MM - y)
                for x, y in current_slot
            )
            footprint = vehicle_footprint(
                PathPose(
                    vehicle_pose[0],
                    vehicle_pose[1],
                    vehicle_pose[2],
                ),
                vehicle,
            )
            fully_inside = all(
                point_in_convex_polygon(point, global_slot)
                for point in footprint
            )
            geometry = ParkingGeometry(
                found=True,
                has_side_pair=True,
                has_back_line=True,
                vehicle_inside_ratio=1.0 if fully_inside else 0.0,
                vehicle_fully_inside=fully_inside,
                confidence=0.95,
            )
            plan = planner.update(
                geometry,
                confirmed_lidar(),
                lidar_slot,
                now,
            )
            if plan.command.speed:
                gear = 1 if plan.command.speed > 0 else -1
                if not gears or gears[-1] != gear:
                    gears.append(gear)
            if plan.state == ParkingState.PARKED:
                break
            self.assertNotEqual(plan.state, ParkingState.ABORTED, plan.reason)

            travel_mm = plan.command.speed * 3.6 * dt
            steering_rad = (
                math.radians(vehicle.max_steering_angle_deg)
                * plan.command.steering
                / planner.config.max_steering_command
            )
            curvature = math.tan(steering_rad) / vehicle.wheelbase_mm
            middle_heading = vehicle_pose[2] + 0.5 * curvature * travel_mm
            vehicle_pose[0] += math.sin(middle_heading) * travel_mm
            vehicle_pose[1] += math.cos(middle_heading) * travel_mm
            vehicle_pose[2] = wrap_angle(
                vehicle_pose[2] + curvature * travel_mm
            )
            now += dt

        self.assertEqual(planner.state, ParkingState.PARKED)
        self.assertEqual(gears, [1, -1])


if __name__ == "__main__":
    unittest.main()
