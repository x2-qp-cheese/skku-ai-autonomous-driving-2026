import unittest
import math

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.reverse_parking_path import ReversePath, ReversePathConfig
from skku_autocar.planning.t_parking_planner import (
    ParkingPlannerConfig,
    ParkingState,
    TParkingPlanner,
)


def geometry(heading=20.0, lateral=0.4, remaining=100.0, reason="parking_bay"):
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
        reason=reason,
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
                start_forward_s=0.0,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                position_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
                exit_straight_s=3.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )

    def make_prealign_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=0.0,
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

    def make_correction_planner(self):
        return TParkingPlanner(
            ParkingPlannerConfig(
                prealign_enabled=False,
                start_forward_s=0.0,
                verify_hold_s=0.0,
                aligned_confirm_frames=1,
                correction_enabled=True,
                correction_steer_settle_s=0.0,
                correction_forward_s=0.5,
                correction_reverse_s=0.5,
                correction_min_reverse_s=0.0,
                correction_depth_trigger_px=1000.0,
                correction_heading_trigger_deg=15.0,
                correction_lateral_trigger_norm=0.30,
                correction_trigger_frames=2,
                correction_max_attempts=2,
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
        hold = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.5,
            right_ultrasonic_mm=500.0,
        )
        exit_right = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.7,
            right_ultrasonic_mm=500.0,
        )
        exit_straight = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            5.4,
            right_ultrasonic_mm=500.0,
        )
        exit_done = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            8.5,
            right_ultrasonic_mm=500.0,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(positioned.state, ParkingState.POSITION_REAR_AXLE)
        self.assertEqual(verifying.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(path_plan.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertIsNotNone(armed.path)
        self.assertEqual(aligned.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(aligned.command.speed, 0)
        self.assertEqual(aligned.command.steering, 0)
        self.assertEqual(aligned.reason, "following_slot_center:straight")
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertTrue(parked.command.brake)
        self.assertEqual(hold.state, ParkingState.PARKED)
        self.assertEqual(hold.reason, "parked_hold")
        self.assertTrue(hold.command.brake)
        self.assertEqual(exit_right.state, ParkingState.EXIT_RIGHT)
        self.assertEqual(exit_right.command.speed, planner.config.exit_speed)
        self.assertEqual(exit_right.command.steering, planner.config.exit_turn_steering)
        self.assertEqual(exit_straight.state, ParkingState.EXIT_STRAIGHT)
        self.assertEqual(exit_straight.command.speed, planner.config.exit_speed)
        self.assertEqual(exit_straight.command.steering, 0)
        self.assertEqual(exit_done.state, ParkingState.EXIT_DONE)
        self.assertTrue(exit_done.command.brake)

    def test_exit_right_waits_when_right_side_is_too_close(self):
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

        planner.update(geometry(), one_car, 0.0)
        planner.update(geometry(), lidar_gap(), 0.1)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.2)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.3)
        planner.update(geometry(), lidar_gap(0.0, reached=True), 0.4)
        planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_gap(0.0, reached=True),
            0.5,
        )
        planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            0.6,
        )

        blocked = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.7,
            right_ultrasonic_mm=150.0,
        )
        moving = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_gap(0.0, reached=True),
            3.8,
            right_ultrasonic_mm=500.0,
        )

        self.assertEqual(blocked.state, ParkingState.EXIT_RIGHT)
        self.assertTrue(blocked.command.brake)
        self.assertIn("exit_right_blocked", blocked.reason)
        self.assertEqual(moving.state, ParkingState.EXIT_RIGHT)
        self.assertFalse(moving.command.brake)
        self.assertEqual(moving.command.steering, planner.config.exit_turn_steering)

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

    def test_start_rollout_drives_straight_even_with_immediate_first_car(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.8,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        rollout = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.2,
            left_ultrasonic_mm=80.0,
        )
        after_rollout = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.9,
        )
        after_delay = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            2.0,
        )

        self.assertEqual(rollout.state, ParkingState.SEARCH_CARS)
        self.assertEqual(rollout.command.speed, planner.config.search_speed)
        self.assertEqual(rollout.command.steering, 0)
        self.assertFalse(rollout.command.brake)
        self.assertEqual(rollout.reason, "start_forward_rollout")
        self.assertEqual(after_rollout.state, ParkingState.TRACK_GAP)
        self.assertEqual(after_rollout.command.speed, planner.config.first_car_approach_speed)
        self.assertEqual(after_rollout.command.steering, 0)
        self.assertEqual(after_rollout.reason, "first_car_straight_delay")
        self.assertEqual(after_delay.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(after_delay.command.speed, 0)
        self.assertEqual(after_delay.command.steering, planner.config.prealign_steering)

    def test_prealign_keeps_searching_until_second_car_without_timeout(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                first_car_preemptive_turn_enabled=True,
                start_forward_s=0.0,
                first_car_straight_s=0.0,
                prealign_enabled=True,
                prealign_speed=35,
                prealign_steering=-150,
                prealign_steer_settle_s=0.0,
                prealign_gap_acquire_timeout_s=0.0,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        tracking = planner.update(geometry(), first_car_lidar(), 0.1)
        turn_point = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            0.2,
        )
        still_searching = planner.update(
            geometry(),
            LidarParkingObservation(
                timestamp=1.0,
                valid=True,
                observed_points=8,
                car_count=1,
                first_car_seen=True,
                first_car_confirmed=False,
                reason="one_parked_car",
            ),
            60.0,
        )

        self.assertEqual(tracking.state, ParkingState.TRACK_GAP)
        self.assertEqual(turn_point.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(still_searching.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(turn_point.command.speed, 0)
        self.assertEqual(still_searching.command.speed, planner.config.prealign_speed)
        self.assertEqual(turn_point.command.steering, planner.config.prealign_steering)
        self.assertEqual(still_searching.command.steering, planner.config.prealign_steering)
        self.assertFalse(turn_point.command.brake)
        self.assertFalse(still_searching.command.brake)
        self.assertEqual(still_searching.reason, "prealign_left_waiting_for_second_car")

    def test_prealign_missing_tracked_slot_keeps_moving(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        waiting = planner.update(
            geometry(),
            lidar_gap(0.0, reached=True),
            0.2,
        )

        self.assertEqual(waiting.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(waiting.command.speed, planner.config.prealign_speed)
        self.assertEqual(waiting.command.steering, planner.config.prealign_steering)
        self.assertFalse(waiting.command.brake)
        self.assertEqual(waiting.reason, "prealign_waiting_for_tracked_slot")

    def test_prealign_switches_to_reverse_setup_when_lidar_slot_box_is_visible(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        seen = planner.update(
            geometry(
                heading=47.0,
                lateral=-0.42,
                remaining=655.0,
                reason="lidar_slot_box",
            ),
            prealign_lidar(
                slot_heading_deg=47.0,
                entry_bearing_deg=46.0,
                distance_mm=1590.0,
            ),
            0.2,
        )

        self.assertEqual(seen.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertTrue(seen.command.brake)
        self.assertEqual(seen.reason, "lidar_slot_box_seen")

    def test_reverse_entry_switches_to_right_steering_after_left_prealign(self):
        planner = self.make_prealign_planner()
        path = ReversePath(
            found=True,
            points=((300.0, 570.0), (360.0, 500.0), (420.0, 320.0)),
            lookahead_point=(360.0, 500.0),
            curvature_per_px=planner.path_generator.config.full_steering_curvature_per_px,
            reason="reverse_path_ready",
        )

        steering = planner._path_steering(path)

        self.assertEqual(planner.config.prealign_steering, -150)
        self.assertGreater(steering, 0)

    def test_reverse_entry_uses_minimum_visible_steering_for_small_curve(self):
        planner = self.make_planner()
        path = ReversePath(
            found=True,
            points=((300.0, 570.0), (320.0, 500.0), (350.0, 320.0)),
            lookahead_point=(320.0, 500.0),
            curvature_per_px=0.001,
            reason="reverse_path_ready",
        )

        steering = planner._entry_curve_steering(path)

        self.assertEqual(steering, planner.config.reverse_entry_min_steering)

    def test_misaligned_reverse_runs_forward_then_reverse_correction(self):
        planner = self.make_correction_planner()
        self.arm_reverse(planner)
        off_center = geometry(
            heading=30.0,
            lateral=-0.45,
            remaining=700.0,
            reason="lidar_slot_box",
        )

        first_reverse = planner.update(off_center, lidar_gap(0.0, reached=True), 0.5)
        correction_start = planner.update(off_center, lidar_gap(0.0, reached=True), 0.6)
        correcting_forward = planner.update(off_center, lidar_gap(0.0, reached=True), 0.7)
        reverse_settle = planner.update(off_center, lidar_gap(0.0, reached=True), 1.2)
        correcting_reverse = planner.update(off_center, lidar_gap(0.0, reached=True), 1.3)
        aligned = planner.update(
            geometry(
                heading=0.0,
                lateral=0.0,
                remaining=600.0,
                reason="lidar_slot_box",
            ),
            lidar_gap(0.0, reached=True),
            1.4,
        )

        self.assertEqual(first_reverse.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(first_reverse.command.steering, planner.config.reverse_entry_min_steering)
        self.assertEqual(correction_start.state, ParkingState.CORRECT_FORWARD)
        self.assertEqual(correction_start.command.speed, 0)
        self.assertLess(correction_start.command.steering, 0)
        self.assertEqual(correcting_forward.state, ParkingState.CORRECT_FORWARD)
        self.assertGreater(correcting_forward.command.speed, 0)
        self.assertLess(correcting_forward.command.steering, 0)
        self.assertEqual(reverse_settle.state, ParkingState.CORRECT_REVERSE)
        self.assertEqual(reverse_settle.command.speed, 0)
        self.assertGreater(reverse_settle.command.steering, 0)
        self.assertEqual(correcting_reverse.state, ParkingState.CORRECT_REVERSE)
        self.assertLess(correcting_reverse.command.speed, 0)
        self.assertGreater(correcting_reverse.command.steering, 0)
        self.assertEqual(aligned.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(aligned.command.speed, 0)
        self.assertEqual(aligned.command.steering, 0)
        self.assertEqual(aligned.reason, "parking_correction_aligned:straight")

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
                start_forward_s=0.0,
                first_car_approach_speed=10,
                first_car_straight_s=1.0,
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
        delayed = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-5.0),
            1.0,
        )
        turning = planner.update(
            geometry(),
            first_car_lidar(turn_reached=True, turn_error=-20.0),
            1.2,
        )

        self.assertEqual(creeping.state, ParkingState.TRACK_GAP)
        self.assertEqual(creeping.command.speed, 10)
        self.assertEqual(settling.state, ParkingState.TRACK_GAP)
        self.assertEqual(settling.command.speed, 10)
        self.assertEqual(settling.command.steering, 0)
        self.assertEqual(settling.reason, "first_car_straight_delay")
        self.assertEqual(delayed.state, ParkingState.TRACK_GAP)
        self.assertEqual(delayed.command.speed, 10)
        self.assertEqual(delayed.command.steering, 0)
        self.assertEqual(turning.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual(turning.command.speed, 0)
        self.assertEqual(turning.command.steering, -150)
        self.assertEqual(turning.reason, "first_car_straight_elapsed:settling_max_left")

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
        self.assertEqual(ready.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(ready.reason, "prealign_direct_reverse_ready")
        self.assertTrue(ready.command.brake)

    def test_prealign_timeout_stops_then_uses_lidar_box_curve_fallback(self):
        planner = self.make_prealign_planner()
        self.enter_prealign(planner)

        fallback = planner.update(
            geometry(),
            prealign_lidar(slot_heading_deg=60.0, entry_bearing_deg=55.0),
            2.2,
        )

        self.assertEqual(fallback.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(fallback.reason, "prealign_fallback:timeout")
        self.assertTrue(fallback.command.brake)

    def test_side_ultrasonic_emergency_stop_is_latched(self):
        planner = self.make_planner()
        self.arm_reverse(planner)

        stopped = planner.update(
            geometry(),
            lidar_gap(),
            0.5,
            left_ultrasonic_mm=95.0,
            right_ultrasonic_mm=500.0,
        )
        still_stopped = planner.update(geometry(), lidar_gap(), 0.6)

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
