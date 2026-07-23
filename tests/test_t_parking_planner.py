import unittest

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.reverse_parking_path import ReversePath, ReversePathConfig
from skku_autocar.planning.t_parking_planner import (
    ParkingPlannerConfig,
    ParkingState,
    TParkingPlanner,
)


def geometry(
    heading=20.0,
    lateral=0.4,
    remaining=100.0,
    reason="lidar_slot_box",
    fully_inside=True,
):
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
        stop_target_x_px=300.0 + 125.0 * lateral,
        stop_target_y_px=100.0,
        vehicle_fully_inside=fully_inside,
        confidence=0.9,
        reason=reason,
    )


def missing_geometry(reason="lidar_slot_box_unavailable"):
    return ParkingGeometry(reason=reason)


def lidar_search(unsafe=False):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        unsafe=unsafe,
        observed_points=20,
        car_count=1,
        reason="searching_for_parked_cars",
    )


def lidar_candidate(*, pair=False):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        observed_points=24,
        car_count=2,
        gap_found=True,
        gap_confirmed=False,
        gap_pair_observed=pair,
        reason="gap_confirming:1/3",
    )


def lidar_first_car_turn_reached():
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        observed_points=18,
        car_count=1,
        first_car_seen=True,
        first_car_confirmed=True,
        first_car_slot_edge_x_right_mm=700.0,
        first_car_slot_edge_y_back_mm=-700.0,
        first_car_turn_error_mm=50.0,
        first_car_turn_reached=True,
        reason="one_parked_car",
    )


def lidar_confirmed(unsafe=False):
    return LidarParkingObservation(
        timestamp=1.0,
        valid=True,
        unsafe=unsafe,
        observed_points=30,
        car_count=2,
        first_car_seen=True,
        second_car_seen=True,
        gap_found=True,
        gap_confirmed=True,
        gap_pair_observed=True,
        gap_width_mm=1375.0,
        gap_center_x_right_mm=650.0,
        gap_center_y_back_mm=380.0,
        slot_depth_x_right=0.0,
        slot_depth_y_back=1.0,
        entry_target_y_back_mm=-300.0,
        reason="gap_confirmed",
    )


class TParkingPlannerTest(unittest.TestCase):
    def make_planner(self, *, emergency_stop_enabled=False):
        return TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=1,
                reverse_entry_steer_settle_s=0.0,
                reverse_entry_release_confirm_frames=1,
                aligned_confirm_frames=1,
                emergency_stop_enabled=emergency_stop_enabled,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
                exit_straight_s=3.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )

    def arm_reverse(self, planner):
        planner.start(0.0)
        planner.update(missing_geometry(), lidar_search(), 0.0)
        planner.update(missing_geometry(), lidar_candidate(), 0.1)
        planner.update(geometry(), lidar_confirmed(), 0.2)
        planner.update(geometry(), lidar_confirmed(), 0.3)
        return planner.update(geometry(), lidar_confirmed(), 0.4)

    def test_straight_search_locks_slot_then_reverses_into_bay(self):
        planner = self.make_planner()
        planner.start(0.0)

        searching = planner.update(missing_geometry(), lidar_search(), 0.0)
        candidate = planner.update(missing_geometry(), lidar_candidate(), 0.1)
        detected = planner.update(geometry(), lidar_confirmed(), 0.2)
        verified = planner.update(geometry(), lidar_confirmed(), 0.3)
        armed = planner.update(geometry(), lidar_confirmed(), 0.4)
        entering = planner.update(geometry(), lidar_confirmed(), 0.5)
        centered = planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_confirmed(),
            0.6,
        )
        parked = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_confirmed(),
            0.7,
        )

        self.assertEqual(searching.state, ParkingState.SEARCH_CARS)
        self.assertEqual(searching.command.speed, planner.config.search_speed)
        self.assertEqual(searching.command.steering, planner.config.straight_steering_trim)
        self.assertEqual(searching.reason, "straight_searching_for_slot")
        self.assertEqual(candidate.state, ParkingState.TRACK_GAP)
        self.assertEqual(candidate.command.speed, planner.config.gap_tracking_speed)
        self.assertEqual(candidate.command.steering, planner.config.straight_steering_trim)
        self.assertEqual(detected.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertTrue(detected.command.brake)
        self.assertEqual(detected.reason, "slot_detected")
        self.assertEqual(verified.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(verified.reason, "slot_geometry_verified")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertIsNotNone(armed.path)
        self.assertEqual(armed.reason, "reverse_path_armed")
        self.assertEqual(entering.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertLess(entering.command.speed, 0)
        self.assertGreater(abs(entering.command.steering), 0)
        self.assertEqual(centered.state, ParkingState.FOLLOW_SLOT_CENTER)
        self.assertLess(centered.command.speed, 0)
        self.assertEqual(centered.command.steering, 0)
        self.assertEqual(parked.state, ParkingState.PARKED)
        self.assertTrue(parked.command.brake)

    def test_confirmed_lidar_starts_entry_setup_while_slot_geometry_arrives(self):
        planner = self.make_planner()
        planner.start(0.0)

        waiting = planner.update(missing_geometry(), lidar_confirmed(), 0.0)
        still_waiting = planner.update(missing_geometry(), lidar_confirmed(), 0.4)
        detected = planner.update(geometry(), lidar_confirmed(), 1.1)
        armed = planner.update(geometry(), lidar_confirmed(), 1.2)

        self.assertEqual(waiting.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(waiting.command.speed, 0)
        self.assertEqual(waiting.reason, "slot_lidar_confirmed_entry_setup")
        self.assertEqual(still_waiting.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(still_waiting.command.speed, planner.config.entry_setup_speed)
        self.assertEqual(still_waiting.reason, "entry_setup_waiting_for_full_geometry")
        self.assertEqual(detected.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(detected.reason, "entry_setup_angle_ready")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)

    def test_gap_candidate_starts_entry_setup_before_lidar_confirmation(self):
        planner = self.make_planner()
        planner.start(0.0)

        setup = planner.update(missing_geometry(), lidar_candidate(pair=True), 0.0)
        waiting_confirmation = planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_candidate(pair=True),
            0.4,
        )
        confirmed = planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_confirmed(),
            1.1,
        )

        self.assertEqual(setup.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(setup.command.speed, 0)
        self.assertEqual(setup.reason, "early_entry_setup:gap_candidate")
        self.assertEqual(waiting_confirmation.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(waiting_confirmation.command.speed, planner.config.entry_setup_speed)
        self.assertEqual(
            waiting_confirmation.reason,
            "entry_setup_waiting_for_lidar_confirmation",
        )
        self.assertEqual(confirmed.state, ParkingState.PLAN_REVERSE_PATH)

    def test_first_car_turn_trigger_starts_entry_setup_before_second_car(self):
        planner = self.make_planner()
        planner.start(0.0)

        setup = planner.update(
            missing_geometry(),
            lidar_first_car_turn_reached(),
            0.0,
        )

        self.assertEqual(setup.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(setup.command.speed, 0)
        self.assertEqual(setup.reason, "early_entry_setup:first_car_turn_reached")

    def test_start_rollout_does_not_ignore_lidar_slot_cue(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=1.0,
                entry_setup_steer_settle_s=0.0,
                search_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        setup = planner.update(missing_geometry(), lidar_candidate(pair=True), 0.2)

        self.assertEqual(setup.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(setup.reason, "early_entry_setup:gap_candidate")

    def test_slot_entry_setup_turns_left_before_reverse_when_angle_is_not_ready(self):
        planner = self.make_planner()
        planner.start(0.0)
        bad_angle = geometry(heading=99.0, lateral=-2.0)

        planner.update(missing_geometry(), lidar_candidate(), 0.0)
        detected = planner.update(bad_angle, lidar_confirmed(), 0.1)
        setup_settle = planner.update(bad_angle, lidar_confirmed(), 0.2)
        setup_forward = planner.update(bad_angle, lidar_confirmed(), 0.6)
        ready = planner.update(
            geometry(heading=45.0, lateral=-0.2),
            lidar_confirmed(),
            1.3,
        )
        armed = planner.update(
            geometry(heading=45.0, lateral=-0.2),
            lidar_confirmed(),
            1.4,
        )

        self.assertEqual(detected.state, ParkingState.VERIFY_SLOT_BOX)
        self.assertEqual(setup_settle.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(setup_settle.command.speed, 0)
        self.assertEqual(
            setup_settle.command.steering,
            planner.config.entry_setup_steering,
        )
        self.assertEqual(setup_settle.reason, "entry_setup_steering_settle")
        self.assertEqual(setup_forward.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(setup_forward.command.speed, planner.config.entry_setup_speed)
        self.assertEqual(
            setup_forward.command.steering,
            planner.config.entry_setup_steering,
        )
        self.assertEqual(setup_forward.reason, "entry_setup_forward_angle")
        self.assertEqual(ready.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(ready.reason, "entry_setup_angle_ready")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)

    def test_slot_entry_setup_aborts_when_angle_never_becomes_safe(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                verify_hold_s=0.0,
                entry_setup_steer_settle_s=0.0,
                entry_setup_min_s=0.0,
                entry_setup_max_s=0.5,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                verify_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        planner.start(0.0)
        bad_angle = geometry(heading=99.0, lateral=-2.0)

        planner.update(missing_geometry(), lidar_candidate(), 0.0)
        planner.update(bad_angle, lidar_confirmed(), 0.1)
        setup = planner.update(bad_angle, lidar_confirmed(), 0.2)
        aborted = planner.update(bad_angle, lidar_confirmed(), 0.8)

        self.assertEqual(setup.state, ParkingState.ENTRY_SETUP)
        self.assertEqual(aborted.state, ParkingState.ABORTED)
        self.assertIn("entry_setup_angle_not_ready", aborted.reason)

    def test_reverse_path_must_be_confirmed_before_reverse_entry(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=3,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
                entry_curve_timeout_s=100.0,
                center_follow_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        planner.start(0.0)
        planner.update(missing_geometry(), lidar_candidate(), 0.0)
        planner.update(geometry(), lidar_confirmed(), 0.1)
        planner.update(geometry(), lidar_confirmed(), 0.2)

        first = planner.update(geometry(), lidar_confirmed(), 0.3)
        second = planner.update(geometry(), lidar_confirmed(), 0.4)
        armed = planner.update(geometry(), lidar_confirmed(), 0.5)

        self.assertEqual(first.state, ParkingState.PLAN_REVERSE_PATH)
        self.assertEqual(first.reason, "reverse_path_confirming:1/3")
        self.assertEqual(second.reason, "reverse_path_confirming:2/3")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)
        self.assertEqual(armed.reason, "reverse_path_armed")

    def test_reverse_path_confirm_counter_tolerates_brief_path_loss(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.0,
                verify_hold_s=0.0,
                path_confirm_frames=3,
                search_timeout_s=100.0,
                gap_tracking_timeout_s=100.0,
                verify_timeout_s=100.0,
                path_timeout_s=100.0,
            ),
            ReversePathConfig(maximum_curvature_per_px=0.05),
        )
        planner.start(0.0)
        planner.update(missing_geometry(), lidar_candidate(), 0.0)
        planner.update(geometry(), lidar_confirmed(), 0.1)
        planner.update(geometry(), lidar_confirmed(), 0.2)

        first = planner.update(geometry(), lidar_confirmed(), 0.3)
        second = planner.update(geometry(), lidar_confirmed(), 0.4)
        lost = planner.update(missing_geometry(), lidar_confirmed(), 0.5)
        recovered = planner.update(geometry(), lidar_confirmed(), 0.6)
        armed = planner.update(geometry(), lidar_confirmed(), 0.7)

        self.assertEqual(first.reason, "reverse_path_confirming:1/3")
        self.assertEqual(second.reason, "reverse_path_confirming:2/3")
        self.assertIn("confirm=1/3", lost.reason)
        self.assertEqual(recovered.reason, "reverse_path_confirming:2/3")
        self.assertEqual(armed.state, ParkingState.FOLLOW_ENTRY_CURVE)

    def test_search_stops_after_rollout_while_waiting_for_lidar(self):
        planner = self.make_planner()
        planner.start(0.0)

        searching = planner.update(
            missing_geometry(),
            LidarParkingObservation(reason="no_scan"),
            0.1,
        )

        self.assertEqual(searching.state, ParkingState.SEARCH_CARS)
        self.assertTrue(searching.command.brake)
        self.assertEqual(searching.reason, "waiting_for_lidar_scan")

    def test_start_rollout_drives_straight_even_before_lidar_is_ready(self):
        planner = TParkingPlanner(
            ParkingPlannerConfig(
                start_forward_s=0.8,
                straight_steering_trim=-10,
                search_timeout_s=100.0,
            )
        )
        planner.start(0.0)

        rollout = planner.update(
            missing_geometry(),
            LidarParkingObservation(reason="no_scan"),
            0.2,
        )

        self.assertEqual(rollout.state, ParkingState.SEARCH_CARS)
        self.assertEqual(rollout.command.speed, planner.config.search_speed)
        self.assertEqual(rollout.command.steering, -10)
        self.assertEqual(rollout.reason, "straight_search_rollout")

    def test_entry_curve_uses_path_steering_with_minimum_visible_steering(self):
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
        self.assertEqual(stopped.reason, "lidar_unavailable_for_slot")

    def test_lidar_obstacle_latches_emergency_stop_when_enabled(self):
        planner = self.make_planner(emergency_stop_enabled=True)
        self.arm_reverse(planner)

        stopped = planner.update(geometry(), lidar_confirmed(unsafe=True), 0.5)
        still_stopped = planner.update(geometry(), lidar_confirmed(), 0.6)

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertTrue(stopped.command.brake)
        self.assertEqual(still_stopped.state, ParkingState.EMERGENCY_STOP)

    def test_front_ultrasonic_emergency_stops_forward_search(self):
        planner = self.make_planner(emergency_stop_enabled=True)
        planner.start(0.0)

        stopped = planner.update(
            missing_geometry(),
            lidar_search(),
            0.1,
            front_left_ultrasonic_mm=95.0,
            front_right_ultrasonic_mm=500.0,
        )

        self.assertEqual(stopped.state, ParkingState.EMERGENCY_STOP)
        self.assertEqual(stopped.reason, "front_ultrasonic_distance<=100mm")
        self.assertTrue(stopped.command.brake)

    def test_exit_right_waits_when_right_side_is_too_close(self):
        planner = self.make_planner()
        self.arm_reverse(planner)
        planner.update(
            geometry(heading=0.0, lateral=0.0),
            lidar_confirmed(),
            0.5,
        )
        planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_confirmed(),
            0.6,
        )

        blocked = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_confirmed(),
            3.7,
            right_ultrasonic_mm=150.0,
        )
        moving = planner.update(
            geometry(heading=0.0, lateral=0.0, remaining=0.0),
            lidar_confirmed(),
            3.8,
            right_ultrasonic_mm=500.0,
        )

        self.assertEqual(blocked.state, ParkingState.EXIT_RIGHT)
        self.assertTrue(blocked.command.brake)
        self.assertIn("exit_right_blocked", blocked.reason)
        self.assertEqual(moving.state, ParkingState.EXIT_RIGHT)
        self.assertFalse(moving.command.brake)
        self.assertEqual(moving.command.steering, planner.config.exit_turn_steering)


if __name__ == "__main__":
    unittest.main()
