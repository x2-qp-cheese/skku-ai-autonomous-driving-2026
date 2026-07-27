import math
import unittest

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.lidar_slot_geometry import point_in_convex_polygon
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.hybrid_parking_path import (
    HybridParkingPath,
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


def confirmed_lidar(unsafe=False, first_car_confirmed=False):
    return LidarParkingObservation(
        valid=True,
        unsafe=unsafe,
        gap_found=True,
        gap_confirmed=True,
        gap_pair_observed=True,
        first_car_confirmed=first_car_confirmed,
    )


def missing_path(reason="no_reverse_path"):
    return HybridParkingPath(reason=reason)


def reverse_path():
    goal = PathPose(900.0, -900.0, -math.pi / 2.0)
    return HybridParkingPath(
        found=True,
        poses=(
            PathPose(0.0, 0.0, 0.0),
            PathPose(250.0, -250.0, -0.20, gear=-1, steering_rad=0.25),
            PathPose(650.0, -650.0, -0.90, gear=-1, steering_rad=0.25),
        ),
        goal=goal,
        reason="hybrid_astar_ready",
    )


def looping_reverse_path():
    goal = PathPose(900.0, -900.0, -math.pi / 2.0)
    return HybridParkingPath(
        found=True,
        poses=(
            PathPose(0.0, 0.0, 0.0),
            PathPose(-320.0, 120.0, 0.15, gear=-1, steering_rad=-0.25),
            PathPose(-700.0, -250.0, -0.40, gear=-1, steering_rad=-0.25),
            PathPose(900.0, -900.0, -1.57, gear=-1, steering_rad=0.25),
        ),
        goal=goal,
        reason="hybrid_astar_ready",
    )


class FakePathPlanner:
    def __init__(self, *paths):
        self.paths = list(paths)
        self.calls = []

    def plan(self, goal, obstacles, *, initial_gear=0, allowed_gears=(-1, 1)):
        self.calls.append(tuple(allowed_gears))
        if self.paths:
            return self.paths.pop(0)
        return reverse_path()


class ModelBasedParkingPlannerTest(unittest.TestCase):
    def make_planner(self, **changes):
        values = dict(
            slot_lock_confirm_scans=2,
            gear_change_stop_frames=2,
            auto_exit_enabled=False,
            emergency_stop_enabled=False,
        )
        values.update(changes)
        config = ModelBasedParkingConfig(**values)
        return ModelBasedTParkingPlanner(
            config,
            sensor_to_rear_axle_y_back_mm=SENSOR_TO_AXLE_Y_BACK_MM,
        )

    def test_entry_setup_left_until_reverse_only_path_is_available(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=1,
            entry_setup_path_confirm_scans=1,
            gear_change_stop_frames=1,
        )
        fake = FakePathPlanner(missing_path(), reverse_path())
        planner.path_planner = fake
        geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            confidence=0.95,
        )
        lidar = confirmed_lidar()
        polygon = right_bay()
        planner.start(0.0)

        locked = planner.update(geometry, lidar, polygon, 0.1)
        armed_setup = planner.update(geometry, lidar, polygon, 0.2)
        setup = planner.update(geometry, lidar, polygon, 0.3)
        armed_reverse = planner.update(geometry, lidar, polygon, 0.4)
        settle = planner.update(geometry, lidar, polygon, 0.5)
        moving = planner.update(geometry, lidar, polygon, 0.6)

        self.assertEqual(locked.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(armed_setup.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(setup.state, ParkingState.ENTRY_SETUP)
        self.assertGreater(setup.command.speed, 0)
        self.assertLess(setup.command.steering, 0)
        self.assertIn("entry_setup_left", setup.reason)
        self.assertEqual(armed_reverse.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertIn("reverse_only", armed_reverse.reason)
        self.assertEqual(settle.command.speed, 0)
        self.assertLess(moving.command.speed, 0)
        self.assertTrue(all(call == (-1,) for call in fake.calls))

    def test_entry_setup_rejects_looping_reverse_path(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=1,
            entry_setup_path_confirm_scans=1,
        )
        fake = FakePathPlanner(looping_reverse_path())
        planner.path_planner = fake
        geometry = ParkingGeometry(
            found=True,
            has_side_pair=True,
            has_back_line=True,
            confidence=0.95,
        )
        lidar = confirmed_lidar()
        polygon = right_bay()
        planner.start(0.0)

        planner.update(geometry, lidar, polygon, 0.1)
        planner.update(geometry, lidar, polygon, 0.2)
        rejected = planner.update(geometry, lidar, polygon, 0.3)

        self.assertEqual(rejected.state, ParkingState.ENTRY_SETUP)
        self.assertGreater(rejected.command.speed, 0)
        self.assertLess(rejected.command.steering, 0)
        self.assertIn("reverse_path_moves_left_of_vehicle", rejected.reason)

    def test_right_ultrasonic_first_car_lost_starts_entry_setup(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=5,
            right_ultrasonic_slot_confirm_enabled=True,
            right_ultrasonic_confirm_scans=2,
            right_ultrasonic_lidar_fallback_scans=99,
            right_ultrasonic_first_car_speed=50,
        )
        geometry = ParkingGeometry(found=True, confidence=0.95)
        lidar = confirmed_lidar(first_car_confirmed=True)
        polygon = right_bay()
        planner.start(0.0)

        first_close = planner.update(
            geometry,
            lidar,
            polygon,
            0.1,
            right_ultrasonic_mm=950.0,
        )
        first_open = planner.update(
            geometry,
            lidar,
            polygon,
            0.2,
            right_ultrasonic_mm=None,
        )

        self.assertEqual(first_close.state, ParkingState.TRACK_GAP)
        self.assertEqual(first_close.command.speed, 50)
        self.assertEqual(first_open.state, ParkingState.ENTRY_SETUP)
        self.assertIn("first_car_lost", first_open.reason)

    def test_lidar_gap_does_not_lock_before_ultrasonic_open_gate(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=1,
            right_ultrasonic_slot_confirm_enabled=True,
            right_ultrasonic_lidar_fallback_scans=1,
            right_ultrasonic_first_car_speed=50,
        )
        geometry = ParkingGeometry(found=True, confidence=0.95)
        lidar = confirmed_lidar(first_car_confirmed=True)
        polygon = right_bay()
        planner.start(0.0)

        for index in range(5):
            plan = planner.update(
                geometry,
                lidar,
                polygon,
                0.1 + 0.1 * index,
                right_ultrasonic_mm=950.0,
            )

        self.assertEqual(plan.state, ParkingState.TRACK_GAP)
        self.assertEqual(plan.command.speed, 50)
        self.assertNotIn("slot_pose_locked", plan.reason)

    def test_right_ultrasonic_close_before_open_gap_slows_until_lost(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=5,
            right_ultrasonic_slot_confirm_enabled=True,
            right_ultrasonic_confirm_scans=1,
            right_ultrasonic_lidar_fallback_scans=99,
            right_ultrasonic_first_car_speed=50,
        )
        geometry = ParkingGeometry(found=True, confidence=0.95)
        lidar = confirmed_lidar(first_car_confirmed=True)
        polygon = right_bay()
        planner.start(0.0)

        first_close = planner.update(
            geometry,
            lidar,
            polygon,
            0.1,
            right_ultrasonic_mm=950.0,
        )
        still_close = planner.update(
            geometry,
            lidar,
            polygon,
            0.2,
            right_ultrasonic_mm=950.0,
        )
        first_open = planner.update(
            geometry,
            lidar,
            polygon,
            0.3,
            right_ultrasonic_mm=None,
        )

        self.assertEqual(first_close.state, ParkingState.TRACK_GAP)
        self.assertEqual(still_close.state, ParkingState.TRACK_GAP)
        self.assertEqual(first_close.command.speed, 50)
        self.assertEqual(still_close.command.speed, 50)
        self.assertEqual(first_open.state, ParkingState.ENTRY_SETUP)
        self.assertIn("first_car_lost", first_open.reason)

    def test_ultrasonic_first_car_lost_starts_entry_setup_before_slot_pose(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=5,
            right_ultrasonic_slot_confirm_enabled=True,
            right_ultrasonic_confirm_scans=1,
            right_ultrasonic_lidar_fallback_scans=99,
            entry_setup_speed=90,
        )
        geometry = ParkingGeometry(found=False)
        lidar = LidarParkingObservation(
            valid=True,
            first_car_confirmed=True,
            first_car_seen=True,
        )
        planner.start(0.0)

        planner.update(
            geometry,
            lidar,
            None,
            0.1,
            right_ultrasonic_mm=950.0,
        )
        first_open = planner.update(
            geometry,
            lidar,
            None,
            0.2,
            right_ultrasonic_mm=None,
        )
        waiting_slot_pose = planner.update(
            geometry,
            lidar,
            None,
            0.3,
            right_ultrasonic_mm=None,
        )

        self.assertEqual(first_open.state, ParkingState.ENTRY_SETUP)
        self.assertGreater(first_open.command.speed, 0)
        self.assertLess(first_open.command.steering, 0)
        self.assertEqual(waiting_slot_pose.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(waiting_slot_pose.command.speed, 90)
        self.assertIn("until_two_lidar_cars_confirmed", waiting_slot_pose.reason)

    def test_entry_setup_keeps_moving_on_stale_lidar_until_lidar_confirms_slot(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=5,
            right_ultrasonic_slot_confirm_enabled=True,
            entry_setup_speed=90,
        )
        geometry = ParkingGeometry(found=False)
        lidar = LidarParkingObservation(valid=True)
        planner.start(0.0)
        planner.update(
            geometry,
            lidar,
            None,
            0.1,
            right_ultrasonic_mm=950.0,
        )
        planner.update(
            geometry,
            lidar,
            None,
            0.2,
            right_ultrasonic_mm=None,
        )

        moving = planner.update(
            geometry,
            LidarParkingObservation(valid=False, reason="stale_scan"),
            None,
            0.3,
        )

        self.assertEqual(moving.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(moving.command.speed, 90)
        self.assertIn("until_two_lidar_cars_confirmed", moving.reason)

    def test_entry_setup_does_not_arm_reverse_from_camera_polygon_without_lidar_gap(self):
        planner = self.make_planner(
            slot_lock_confirm_scans=5,
            right_ultrasonic_slot_confirm_enabled=True,
            entry_setup_speed=90,
        )
        geometry = ParkingGeometry(found=True, confidence=0.95)
        lidar = LidarParkingObservation(valid=True)
        polygon = right_bay()
        planner.start(0.0)
        planner.update(
            geometry,
            lidar,
            polygon,
            0.1,
            right_ultrasonic_mm=950.0,
        )
        planner.update(
            geometry,
            lidar,
            polygon,
            0.2,
            right_ultrasonic_mm=None,
        )
        triggered = planner.update(
            geometry,
            lidar,
            polygon,
            0.3,
            right_ultrasonic_mm=None,
        )
        still_setup = planner.update(
            geometry,
            lidar,
            polygon,
            0.4,
            right_ultrasonic_mm=None,
        )

        self.assertEqual(triggered.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(still_setup.state, ParkingState.ENTRY_SETUP)
        self.assertGreater(still_setup.command.speed, 0)
        self.assertIn("until_two_lidar_cars_confirmed", still_setup.reason)

    def test_front_center_sensor_no_longer_latches_emergency_stop(self):
        planner = self.make_planner()
        planner.start(0.0)

        plan = planner.update(
            ParkingGeometry(),
            LidarParkingObservation(valid=True),
            None,
            0.1,
            front_center_ultrasonic_mm=90.0,
        )

        self.assertEqual(plan.state, ParkingState.SEARCH_CARS)
        self.assertGreater(plan.command.speed, 0)

    def test_parking_completion_requires_pose_and_full_footprint(self):
        planner = self.make_planner(
            parking_complete_confirm_scans=2,
            entry_setup_enabled=False,
        )
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
