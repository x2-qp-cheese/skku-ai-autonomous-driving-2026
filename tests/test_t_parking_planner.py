import unittest
import math

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.reverse_parking_path import ReversePathConfig
from skku_autocar.planning.t_parking_planner import (
    ParkingPlannerConfig,
    ParkingState,
    TParkingPlanner,
)


def geometry(heading=20.0, lateral=0.4, remaining=100.0):
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=remaining is not None,
        heading_error_deg=heading,
        lateral_error_norm=lateral,
        depth_remaining_px=remaining,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        slot_direction_x=0.0,
        slot_direction_y=-1.0,
        stop_target_x_px=350.0,
        stop_target_y_px=100.0,
        confidence=0.9,
        reason="parking_bay",
    )


def lidar_gap(entry_error=200.0, reached=False, unsafe=False):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        unsafe=unsafe,
        observed_points=20,
        car_count=2,
        first_car_seen=True,
        second_car_seen=True,
        gap_found=True,
        gap_confirmed=True,
        gap_width_mm=1375.0,
        gap_center_y_back_mm=380.0 if not reached else 180.0,
        entry_target_y_back_mm=180.0,
        entry_error_mm=entry_error,
        entry_reached=reached,
        reason="gap_confirmed",
    )


def prealign_lidar(slot_heading_deg=90.0, entry_bearing_deg=90.0, distance_mm=1000.0):
    slot_angle = math.radians(slot_heading_deg)
    bearing = math.radians(entry_bearing_deg)
    rear_axle_y = -300.0
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        observed_points=20,
        car_count=2,
        gap_found=True,
        gap_confirmed=True,
        gap_width_mm=1375.0,
        gap_center_x_right_mm=math.sin(bearing) * distance_mm,
        gap_center_y_back_mm=rear_axle_y + math.cos(bearing) * distance_mm,
        entry_target_y_back_mm=rear_axle_y,
        slot_depth_x_right=math.sin(slot_angle),
        slot_depth_y_back=math.cos(slot_angle),
        reason="gap_confirmed",
    )


def first_car_lidar(turn_reached=False, turn_error=100.0):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        observed_points=10,
        car_count=1,
        first_car_seen=True,
        first_car_confirmed=True,
        first_car_slot_edge_x_right_mm=1500.0,
        first_car_slot_edge_y_back_mm=-650.0 - turn_error,
        first_car_turn_error_mm=turn_error,
        first_car_turn_reached=turn_reached,
        reason="first_car_confirmed",
    )


class TParkingPlannerTest(unittest.TestCase):
    def make_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )

    def make_prealign_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=True,
                prealign_speed=14,
                prealign_steering=-150,
                prealign_steer_settle_s=0.0,
                prealign_timeout_s=2.0,
                prealign_confirm_frames=2,
                verify_hold_s=0.0,
                search_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
            )
        )

    @staticmethod
    def enter_prealign(planner):
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)
        return planner.update(geometry(), lidar_gap(0.0, reached=True), 0.1)

    def arm_reverse(self, planner):
        planner.start(0.0)
        one_car = LidarParkingObservation(
            timestamp=0.0,
            valid=True,
            observed_points=10,
            car_count=1,
            first_car_seen=True,
            reason="one_parked_car",
        )
        planner.update(geometry(), one_car, 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)
        return planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)

    def test_complete_sequence_stops_at_camera_back_line(self):
        planner = self.make_planner()
        planner.start(0.0)
        one_car = LidarParkingObservation(
            timestamp=0.0,
            valid=True,
            observed_points=10,
            car_count=1,
            first_car_seen=True,
            reason="one_parked_car",
        )

        tracking = planner.update(geometry(), one_car, 0.0)
        positioned = planner.update(geometry(), lidar_gap(), 0.1)
        verifying = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        path_plan = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)
        armed = planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)
        aligned = planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        parked = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(positioned.state, ParkingState.POSITION_REAR_AXLE)
        self.assertEqual(verifying.state, ParkingState.VERIFY_PARKING_LINES)
        self.assertEqual(path_plan.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertIsNotNone(armed.path)
        self.assertEqual(aligned.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(aligned.command.speed, 0)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertTrue(parked.command.brake)

    def test_search_drives_forward_while_waiting_for_lidar(self):
        planner = self.make_planner()
        planner.start(0.0)

        searching = planner.update(
            geometry(),
            LidarParkingObservation(reason="no_scan"),
            0.1,
        )

        self.assertEqual(searching.state, ParkingState.SEARCH_CARS)
        self.assertEqual(searching.command.speed, planner.config.search_speed)
        self.assertEqual(searching.command.steering, 0)
        self.assertFalse(searching.command.brake)
        self.assertEqual(searching.reason, "searching_for_lidar")

    def test_lidar_obstacle_latches_emergency_stop(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(geometry(), lidar_gap(unsafe=True), 0.5)
        still_stopped = planner.update(geometry(), lidar_gap(), 0.6)

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(still_stopped.state, ParkingState.EMERGENCY_STOP)

    def test_missing_lidar_never_allows_reverse_motion(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(
            geometry(),
            LidarParkingObservation(reason="stale_scan"),
            0.5,
        )

        self.assertEqual(stopped.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(stopped.reason, "lidar_unavailable_during_reverse")

    def test_legacy_rear_axle_positioning_corrects_in_both_directions(self):
        planner = self.make_planner()
        planner.start(0.0)
        planner.update(geometry(), lidar_gap(), 0.0)

        forward = planner.update(geometry(), lidar_gap(entry_error=-150.0), 0.1)

        other = self.make_planner()
        other.start(0.0)
        other.update(geometry(), lidar_gap(), 0.0)
        reverse = other.update(geometry(), lidar_gap(entry_error=150.0), 0.1)

        self.assertGreater(forward.command.speed, 0)
        self.assertLess(reverse.command.speed, 0)

    def test_prealign_moves_forward_with_maximum_left_steering(self):
        planner = self.make_prealign_planner()
        entered = self.enter_prealign(planner)

        moving = planner.update(
            geometry(),
            prealign_lidar(slot_heading_deg=70.0, entry_bearing_deg=65.0),
            0.2,
        )

        self.assertEqual(entered.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(moving.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(moving.command.speed, 14)
        self.assertEqual(moving.command.steering, -150)

    def test_first_car_slows_then_starts_left_turn_before_gap_confirmation(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                first_car_approach_speed=10,
                prealign_speed=35,
                prealign_steering=-150,
                prealign_steer_settle_s=0.4,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        creeping = planner.update(geometry(), first_car_lidar(), 0.1)
        settling = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-5.0),
            0.2,
        )
        turning = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.7,
        )

        self.assertEqual(creeping.state, ParkingState.TRACK_GAP)
        self.assertEqual(creeping.command.speed, 10)
        self.assertEqual(settling.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(settling.command.speed, 0)
        self.assertEqual(settling.command.steering, -150)
        self.assertEqual(turning.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(turning.command.speed, 35)
        self.assertEqual(turning.command.steering, -150)
        self.assertIn("waiting_for_second_car", turning.reason)

    def test_prealign_direct_reverse_requires_stable_heading_and_bearing(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)
        aligned_lidar = prealign_lidar(
            slot_heading_deg=5.0,
            entry_bearing_deg=8.0,
            distance_mm=900.0,
        )

        confirming = planner.update(geometry(), aligned_lidar, 0.2)
        ready = planner.update(geometry(), aligned_lidar, 0.3)

        self.assertEqual(confirming.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(ready.state, ParkingState.VERIFY_PARKING_LINES)
        self.assertEqual(ready.reason, "prealign_direct_reverse_ready")
        self.assertTrue(ready.command.brake)

    def test_prealign_timeout_stops_then_uses_camera_curve_fallback(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        fallback = planner.update(
            geometry(),
            prealign_lidar(slot_heading_deg=60.0, entry_bearing_deg=55.0),
            2.2,
        )

        self.assertEqual(fallback.state, ParkingState.VERIFY_PARKING_LINES)
        self.assertEqual(fallback.reason, "prealign_fallback:timeout")
        self.assertTrue(fallback.command.brake)

    def test_side_ultrasonic_emergency_stop_is_latched(self):
        planner = self.make_planner()
        planner.start(0.0)

        stopped = planner.update(
            geometry(),
            lidar_gap(),
            0.1,
            left_ultrasonic_mm=95.0,
            right_ultrasonic_mm=500.0,
        )
        still_stopped = planner.update(geometry(), lidar_gap(), 0.2)

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(still_stopped.state, ParkingState.EMERGENCY_STOP)

    def test_side_ultrasonic_p_control_is_bounded(self):
        planner = self.make_planner()

        self.assertEqual(planner._ultrasonic_correction(500.0, 500.0), 0)
        self.assertEqual(planner._ultrasonic_correction(400.0, 800.0), 35)
        self.assertEqual(planner._ultrasonic_correction(800.0, 400.0), -35)
        self.assertEqual(planner._ultrasonic_correction(None, 400.0), 0)


if __name__ == "__main__":
    unittest.main()
