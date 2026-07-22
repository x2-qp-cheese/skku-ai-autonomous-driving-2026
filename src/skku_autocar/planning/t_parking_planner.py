from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees, hypot
from typing import Optional

from ..estimation.parking_geometry import ParkingGeometry
from ..estimation.parking_lidar import LidarParkingObservation
from ..types import ControlCommand
from .reverse_parking_path import (
    ReverseParkingPathGenerator,
    ReversePath,
    ReversePathConfig,
)


class ParkingState(str, Enum):
    IDLE = "idle"
    SEARCH_CARS = "search_cars"
    TRACK_GAP = "track_gap"
    POSITION_REAR_AXLE = "position_rear_axle"
    PREALIGN_LEFT = "prealign_left"
    VERIFY_SLOT_BOX = "verify_slot_box"
    # Backward-compatible alias for old replay integrations.
    VERIFY_PARKING_LINES = "verify_slot_box"
    PLAN_REVERSE_PATH = "plan_reverse_path"
    FOLLOW_ENTRY_CURVE = "follow_entry_curve"
    FOLLOW_SLOT_CENTER = "follow_slot_center"
    CORRECT_FORWARD = "correct_forward"
    CORRECT_REVERSE = "correct_reverse"
    PARKED = "parked"
    EXIT_RIGHT = "exit_right"
    EXIT_STRAIGHT = "exit_straight"
    EXIT_DONE = "exit_done"
    ABORTED = "aborted"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class ParkingPlannerConfig:
    search_speed: int = 35
    start_forward_s: float = 0.8
    gap_tracking_speed: int = 24
    position_speed: int = 18
    first_car_preemptive_turn_enabled: bool = True
    first_car_approach_speed: int = 24
    first_car_straight_s: float = 1.0
    prealign_enabled: bool = True
    prealign_speed: int = 35
    prealign_steering: int = -150
    prealign_steer_settle_s: float = 0.40
    prealign_timeout_s: float = 6.0
    prealign_gap_acquire_timeout_s: float = 0.0
    prealign_slot_heading_tolerance_deg: float = 18.0
    prealign_entry_bearing_tolerance_deg: float = 25.0
    prealign_target_distance_min_mm: float = 250.0
    prealign_target_distance_max_mm: float = 2200.0
    prealign_confirm_frames: int = 3
    prealign_heading_overshoot_deg: float = 25.0
    ultrasonic_kp_steering_per_mm: float = 0.23
    ultrasonic_max_correction: int = 35
    ultrasonic_emergency_mm: float = 100.0
    ultrasonic_max_valid_mm: float = 2500.0
    ultrasonic_stale_after_s: float = 0.8
    reverse_entry_speed: int = -28
    reverse_center_speed: int = -18
    reverse_entry_min_steering: int = 90
    correction_enabled: bool = True
    correction_forward_speed: int = 18
    correction_reverse_speed: int = -24
    correction_steering: int = 130
    correction_steer_settle_s: float = 0.25
    correction_forward_s: float = 0.70
    correction_reverse_s: float = 1.10
    correction_min_reverse_s: float = 0.80
    correction_depth_trigger_px: float = 760.0
    correction_heading_trigger_deg: float = 16.0
    correction_lateral_trigger_norm: float = 0.30
    correction_trigger_frames: int = 3
    correction_max_attempts: int = 3
    park_hold_s: float = 3.0
    exit_speed: int = 24
    exit_turn_steering: int = 80
    exit_turn_s: float = 1.6
    exit_straight_s: float = 0.0
    exit_right_min_clearance_mm: float = 180.0
    max_steering: int = 130
    reverse_steering_sign: float = 1.0
    geometry_confidence_min: float = 0.20
    aligned_heading_deg: float = 8.0
    aligned_lateral_norm: float = 0.18
    aligned_confirm_frames: int = 4
    stop_depth_margin_px: float = 8.0
    verify_hold_s: float = 0.6
    search_timeout_s: float = 60.0
    gap_tracking_timeout_s: float = 20.0
    position_timeout_s: float = 10.0
    verify_timeout_s: float = 5.0
    path_timeout_s: float = 4.0
    entry_curve_timeout_s: float = 12.0
    center_follow_timeout_s: float = 10.0


@dataclass(frozen=True)
class ParkingPlan:
    state: ParkingState
    command: ControlCommand
    reason: str
    path: Optional[ReversePath] = None


class TParkingPlanner:
    """LiDAR gap positioning followed by LiDAR-box-guided reverse parking."""

    def __init__(
        self,
        config: ParkingPlannerConfig = ParkingPlannerConfig(),
        path_config: ReversePathConfig = ReversePathConfig(),
    ):
        self.config = config
        self.path_generator = ReverseParkingPathGenerator(path_config)
        self.state = ParkingState.IDLE
        self._state_started_at = 0.0
        self._aligned_frames = 0
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._prealign_aligned_frames = 0
        self._prealign_gap_acquired_at: Optional[float] = None
        self._reverse_entry_mode = "lidar_box_curve"

    def start(self, now: float) -> bool:
        if self.state not in (
            ParkingState.IDLE,
            ParkingState.ABORTED,
            ParkingState.EMERGENCY_STOP,
            ParkingState.PARKED,
            ParkingState.EXIT_DONE,
        ):
            return False
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._enter(ParkingState.SEARCH_CARS, now)
        return True

    def reset(self, now: float = 0.0) -> None:
        self.state = ParkingState.IDLE
        self._state_started_at = now
        self._aligned_frames = 0
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._prealign_aligned_frames = 0
        self._prealign_gap_acquired_at = None
        self._reverse_entry_mode = "lidar_box_curve"

    @property
    def prealign_confirmed_frames(self) -> int:
        return self._prealign_aligned_frames

    def update(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        enabled: bool = True,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
    ) -> ParkingPlan:
        if not enabled:
            self.reset(now)
            return self._stop("parking_disabled")
        if self.state == ParkingState.IDLE:
            return self._stop("waiting_for_start")
        if self.state == ParkingState.PARKED:
            if self._state_elapsed(now) >= max(0.0, self.config.park_hold_s):
                self._enter(ParkingState.EXIT_RIGHT, now)
                if self._ultrasonic_emergency(left_ultrasonic_mm) or self._ultrasonic_emergency(
                    right_ultrasonic_mm
                ):
                    self._enter(ParkingState.EMERGENCY_STOP, now)
                    return self._stop(
                        "side_ultrasonic_distance<=%.0fmm"
                        % self.config.ultrasonic_emergency_mm
                    )
                return self._exit_right_plan(now, right_ultrasonic_mm)
            return self._stop("parked_hold")
        if self.state == ParkingState.EXIT_DONE:
            return self._stop("exit_done")
        if self.state == ParkingState.ABORTED:
            return self._stop("parking_aborted")
        if self.state == ParkingState.EMERGENCY_STOP:
            return self._stop("emergency_stop_latched")

        ultrasonic_emergency_states = (
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.PLAN_REVERSE_PATH,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
            ParkingState.EXIT_RIGHT,
            ParkingState.EXIT_STRAIGHT,
        )
        if (
            self.state in ultrasonic_emergency_states
            and (
                self._ultrasonic_emergency(left_ultrasonic_mm)
                or self._ultrasonic_emergency(right_ultrasonic_mm)
            )
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("side_ultrasonic_distance<=%.0fmm" % self.config.ultrasonic_emergency_mm)

        slot_and_reverse_states = (
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.PLAN_REVERSE_PATH,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
        )
        if self.state in slot_and_reverse_states and not lidar.valid:
            return self._stop("lidar_unavailable_during_reverse")
        if lidar.unsafe and self.state in slot_and_reverse_states:
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("lidar_safety_obstacle")

        if self.state == ParkingState.PREALIGN_LEFT:
            if not lidar.valid:
                return self._prealign_drive(
                    self._state_elapsed(now),
                    "prealign_waiting_for_lidar",
                )
            if lidar.unsafe:
                self._enter(ParkingState.EMERGENCY_STOP, now)
                return self._stop("lidar_safety_obstacle_during_prealign")

        if self.state == ParkingState.EXIT_RIGHT:
            return self._exit_right_plan(now, right_ultrasonic_mm)

        if self.state == ParkingState.EXIT_STRAIGHT:
            if (
                self.config.exit_straight_s > 0.0
                and self._state_elapsed(now) >= self.config.exit_straight_s
            ):
                self._enter(ParkingState.EXIT_DONE, now)
                return self._stop("exit_complete")
            return self._drive(self._exit_speed(), 0, "exit_straight")

        if self.state == ParkingState.SEARCH_CARS:
            if self._expired(now, self.config.search_timeout_s):
                return self._abort(now, "parked_car_search_timeout")
            if self._state_elapsed(now) < max(0.0, self.config.start_forward_s):
                return self._drive(self.config.search_speed, 0, "start_forward_rollout")
            if not lidar.valid:
                return self._drive(self.config.search_speed, 0, "searching_for_lidar")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_turn_reached
            ):
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    0,
                    "first_car_straight_delay",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_confirmed
            ):
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    0,
                    "first_car_confirmed:creeping_to_turn_point",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_seen
            ):
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    0,
                    "first_car_detected:confirming_at_creep_speed",
                )
            if lidar.gap_confirmed:
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            if lidar.first_car_seen or lidar.gap_found:
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(self.config.gap_tracking_speed, 0, "tracking_parked_cars")
            return self._drive(self.config.search_speed, 0, "searching_for_parked_cars")

        if self.state == ParkingState.TRACK_GAP:
            if self._expired(now, self.config.gap_tracking_timeout_s):
                return self._abort(now, "two_car_gap_timeout")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
            ):
                if self._state_elapsed(now) < max(0.0, self.config.first_car_straight_s):
                    return self._drive(
                        self.config.first_car_approach_speed,
                        0,
                        "first_car_straight_delay",
                    )
                if (
                    not lidar.valid
                    or lidar.first_car_turn_reached
                    or lidar.gap_confirmed
                    or lidar.first_car_confirmed
                    or lidar.first_car_seen
                    or lidar.car_count > 0
                    or lidar.gap_found
                ):
                    self._enter(ParkingState.PREALIGN_LEFT, now)
                    if lidar.gap_confirmed:
                        self._prealign_gap_acquired_at = now
                    return self._drive(
                        0,
                        self._prealign_steering(),
                        "first_car_straight_elapsed:settling_max_left",
                    )
                return self._drive(
                    self.config.first_car_approach_speed,
                    0,
                    "first_car_temporarily_lost:creeping_to_turn_point",
                )
            if lidar.gap_confirmed:
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            return self._drive(self.config.gap_tracking_speed, 0, "confirming_two_car_gap")

        if self.state == ParkingState.POSITION_REAR_AXLE:
            if self._expired(now, self.config.position_timeout_s):
                return self._abort(now, "rear_axle_position_timeout")
            if not lidar.valid or not lidar.gap_confirmed or lidar.entry_error_mm is None:
                return self._stop("rear_axle_waiting_for_gap")
            if lidar.entry_reached:
                if self.config.prealign_enabled:
                    self._enter(ParkingState.PREALIGN_LEFT, now)
                    self._prealign_gap_acquired_at = now
                    steering = self._prealign_steering()
                    return self._drive(
                        0,
                        steering,
                        "rear_axle_at_gap_center:settling_max_left",
                    )
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop("rear_axle_at_gap_center")
            direction = -1 if lidar.entry_error_mm > 0.0 else 1
            return self._drive(
                direction * abs(self.config.position_speed),
                0,
                "correcting_rear_axle_to_gap",
            )

        if self.state == ParkingState.PREALIGN_LEFT:
            elapsed = now - self._state_started_at
            if self._lidar_slot_box_ready(geometry, lidar):
                self._reverse_entry_mode = "lidar_box_seen"
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop("lidar_slot_box_seen")
            if not lidar.gap_confirmed:
                self._prealign_aligned_frames = 0
                if self._expired(now, self.config.prealign_gap_acquire_timeout_s):
                    return self._abort(now, "second_car_gap_acquire_timeout")
                return self._prealign_drive(
                    elapsed,
                    "prealign_left_waiting_for_second_car",
                )
            if (
                lidar.coasted
                or lidar.gap_center_x_right_mm is None
                or lidar.gap_center_y_back_mm is None
                or lidar.entry_target_y_back_mm is None
                or lidar.slot_depth_x_right is None
                or lidar.slot_depth_y_back is None
            ):
                self._prealign_aligned_frames = 0
                return self._prealign_drive(
                    elapsed,
                    "prealign_waiting_for_tracked_slot",
                )

            if self._prealign_gap_acquired_at is None:
                self._prealign_gap_acquired_at = now

            metrics = self._prealign_metrics(lidar)
            if metrics is None:
                self._prealign_aligned_frames = 0
                return self._prealign_drive(
                    elapsed,
                    "prealign_invalid_slot_pose",
                )
            slot_heading_deg, entry_bearing_deg, target_distance_mm = metrics
            direct_ready = (
                abs(slot_heading_deg)
                <= self.config.prealign_slot_heading_tolerance_deg
                and abs(entry_bearing_deg)
                <= self.config.prealign_entry_bearing_tolerance_deg
                and self.config.prealign_target_distance_min_mm
                <= target_distance_mm
                <= self.config.prealign_target_distance_max_mm
            )
            self._prealign_aligned_frames = (
                self._prealign_aligned_frames + 1 if direct_ready else 0
            )
            if self._prealign_aligned_frames >= max(
                1,
                self.config.prealign_confirm_frames,
            ):
                self._reverse_entry_mode = "direct_aligned"
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop("prealign_direct_reverse_ready")

            overshot = (
                slot_heading_deg < -abs(self.config.prealign_heading_overshoot_deg)
            )
            timed_out = (
                self.config.prealign_timeout_s > 0.0
                and now - self._prealign_gap_acquired_at
                >= self.config.prealign_timeout_s
            )
            if overshot or timed_out:
                self._reverse_entry_mode = "lidar_box_curve_fallback"
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop(
                    "prealign_fallback:%s"
                    % ("heading_overshoot" if overshot else "timeout")
                )

            reason = "prealign_left head=%+.1f bearing=%+.1f dist=%.0fmm" % (
                slot_heading_deg,
                entry_bearing_deg,
                target_distance_mm,
            )
            return self._prealign_drive(elapsed, reason)

        if self.state == ParkingState.VERIFY_SLOT_BOX:
            if self._expired(now, self.config.verify_timeout_s):
                return self._abort(now, "lidar_slot_box_verify_timeout")
            if not self._full_geometry_usable(geometry):
                return self._stop("waiting_for_lidar_slot_box")
            if now - self._state_started_at < self.config.verify_hold_s:
                return self._stop("lidar_slot_box_verify_hold")
            self._enter(ParkingState.PLAN_REVERSE_PATH, now)
            return self._stop("lidar_slot_box_verified")

        path = self.path_generator.generate(geometry)

        if self.state == ParkingState.PLAN_REVERSE_PATH:
            if self._expired(now, self.config.path_timeout_s):
                return self._abort(now, "reverse_path_timeout:%s" % path.reason)
            if not path.found:
                return self._stop("waiting_for_reverse_path:%s" % path.reason, path)
            self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
            return self._stop("reverse_path_armed", path)

        if self.state == ParkingState.CORRECT_FORWARD:
            return self._correction_forward_plan(geometry, path, now)

        if self.state == ParkingState.CORRECT_REVERSE:
            return self._correction_reverse_plan(geometry, path, now)

        if self.state == ParkingState.FOLLOW_ENTRY_CURVE:
            if self._expired(now, self.config.entry_curve_timeout_s):
                return self._abort(now, "entry_curve_timeout")
            stop = self._stop_at_back_line(geometry, now, path)
            if stop is not None:
                return stop
            if not path.found:
                self._aligned_frames = 0
                return self._stop("entry_curve_path_lost:%s" % path.reason, path)
            aligned = self._slot_aligned(geometry)
            self._aligned_frames = self._aligned_frames + 1 if aligned else 0
            if self._aligned_frames >= max(1, self.config.aligned_confirm_frames):
                self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                return self._drive(
                    self.config.reverse_center_speed,
                    0,
                    "following_slot_center:straight",
                    path,
                )
            if self._parking_correction_ready(geometry, now):
                return self._start_parking_correction(geometry, path, now)
            return self._drive(
                self.config.reverse_entry_speed,
                self._entry_curve_steering(
                    path,
                    left_ultrasonic_mm,
                    right_ultrasonic_mm,
                ),
                "following_entry_curve:%s" % self._reverse_entry_mode,
                path,
            )

        if self.state == ParkingState.FOLLOW_SLOT_CENTER:
            if self._expired(now, self.config.center_follow_timeout_s):
                return self._abort(now, "slot_center_follow_timeout")
            stop = self._stop_at_back_line(geometry, now, path)
            if stop is not None:
                return stop
            if not path.found:
                return self._stop("slot_center_path_lost:%s" % path.reason, path)
            if self._parking_correction_ready(geometry, now):
                return self._start_parking_correction(geometry, path, now)
            return self._drive(
                self.config.reverse_center_speed,
                0,
                "following_slot_center:straight",
                path,
            )

        return self._abort(now, "unknown_state")

    def _full_geometry_usable(self, geometry: ParkingGeometry) -> bool:
        return (
            geometry.found
            and geometry.has_side_pair
            and geometry.has_back_line
            and geometry.depth_remaining_px is not None
            and geometry.confidence >= self.config.geometry_confidence_min
        )

    def _slot_aligned(self, geometry: ParkingGeometry) -> bool:
        return (
            abs(geometry.heading_error_deg) <= self.config.aligned_heading_deg
            and abs(geometry.lateral_error_norm) <= self.config.aligned_lateral_norm
        )

    def _parking_correction_ready(
        self,
        geometry: ParkingGeometry,
        now: float,
    ) -> bool:
        if not self.config.correction_enabled:
            self._misaligned_frames = 0
            return False
        if (
            self.config.correction_max_attempts > 0
            and self._correction_attempts >= self.config.correction_max_attempts
        ):
            self._misaligned_frames = 0
            return False
        if self._state_elapsed(now) < max(0.0, self.config.correction_min_reverse_s):
            self._misaligned_frames = 0
            return False
        if not self._full_geometry_usable(geometry):
            self._misaligned_frames = 0
            return False
        if (
            self.config.correction_depth_trigger_px > 0.0
            and geometry.depth_remaining_px is not None
            and geometry.depth_remaining_px > self.config.correction_depth_trigger_px
        ):
            self._misaligned_frames = 0
            return False
        misaligned = (
            abs(geometry.heading_error_deg)
            >= abs(self.config.correction_heading_trigger_deg)
            or abs(geometry.lateral_error_norm)
            >= abs(self.config.correction_lateral_trigger_norm)
        )
        self._misaligned_frames = self._misaligned_frames + 1 if misaligned else 0
        return self._misaligned_frames >= max(1, self.config.correction_trigger_frames)

    def _start_parking_correction(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
    ) -> ParkingPlan:
        self._correction_attempts += 1
        self._misaligned_frames = 0
        self._correction_reverse_steering = self._correction_steering(geometry, path)
        self._enter(ParkingState.CORRECT_FORWARD, now)
        return self._drive(
            0,
            -self._correction_reverse_steering,
            "parking_correction_forward:settling",
            path,
        )

    def _correction_forward_plan(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
    ) -> ParkingPlan:
        if not path.found:
            return self._stop("correction_forward_path_lost:%s" % path.reason, path)
        stop = self._stop_at_back_line(geometry, now, path)
        if stop is not None:
            return stop
        elapsed = self._state_elapsed(now)
        steering = -self._correction_reverse_steering
        if steering == 0:
            steering = -self._correction_steering(geometry, path)
        if elapsed < max(0.0, self.config.correction_steer_settle_s):
            return self._drive(0, steering, "parking_correction_forward:settling", path)
        if elapsed >= (
            max(0.0, self.config.correction_steer_settle_s)
            + max(0.0, self.config.correction_forward_s)
        ):
            self._correction_reverse_steering = self._correction_steering(geometry, path)
            self._enter(ParkingState.CORRECT_REVERSE, now)
            return self._drive(
                0,
                self._correction_reverse_steering,
                "parking_correction_reverse:settling",
                path,
            )
        return self._drive(
            abs(int(self.config.correction_forward_speed)),
            steering,
            "parking_correction_forward",
            path,
        )

    def _correction_reverse_plan(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
    ) -> ParkingPlan:
        if not path.found:
            return self._stop("correction_reverse_path_lost:%s" % path.reason, path)
        stop = self._stop_at_back_line(geometry, now, path)
        if stop is not None:
            return stop
        steering = self._correction_steering(geometry, path)
        self._correction_reverse_steering = steering
        if self._slot_aligned(geometry):
            self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
            return self._drive(
                self.config.reverse_center_speed,
                0,
                "parking_correction_aligned:straight",
                path,
            )
        elapsed = self._state_elapsed(now)
        if elapsed < max(0.0, self.config.correction_steer_settle_s):
            return self._drive(0, steering, "parking_correction_reverse:settling", path)
        if elapsed >= (
            max(0.0, self.config.correction_steer_settle_s)
            + max(0.0, self.config.correction_reverse_s)
        ):
            self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
            return self._drive(
                self.config.reverse_entry_speed,
                self._entry_curve_steering(path),
                "parking_correction_reverse_complete:resume_entry_curve",
                path,
            )
        return self._drive(
            -abs(int(self.config.correction_reverse_speed)),
            steering,
            "parking_correction_reverse",
            path,
        )

    def _lidar_slot_box_ready(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
    ) -> bool:
        return (
            lidar.gap_confirmed
            and self._full_geometry_usable(geometry)
            and geometry.reason in ("lidar_slot_box", "lidar_slot_box_hold")
        )

    def _stop_at_back_line(
        self,
        geometry: ParkingGeometry,
        now: float,
        path: ReversePath,
    ) -> Optional[ParkingPlan]:
        if not self._full_geometry_usable(geometry):
            return self._stop("reverse_waiting_for_full_geometry", path)
        if geometry.depth_remaining_px <= self.config.stop_depth_margin_px:
            self._enter(ParkingState.PARKED, now)
            return self._stop("back_line_clearance_reached", path)
        return None

    def _path_steering(
        self,
        path: ReversePath,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
        minimum_abs: int = 0,
    ) -> int:
        full_scale = max(1e-9, self.path_generator.config.full_steering_curvature_per_px)
        normalized = clip(path.curvature_per_px / full_scale, -1.0, 1.0)
        raw = self.config.reverse_steering_sign * self.config.max_steering * normalized
        steering = round(raw)
        if abs(raw) > 1e-9 and minimum_abs > 0:
            minimum = min(abs(int(minimum_abs)), abs(int(self.config.max_steering)))
            if abs(steering) < minimum:
                steering = minimum if raw > 0.0 else -minimum
        steering += self._ultrasonic_correction(
            left_ultrasonic_mm,
            right_ultrasonic_mm,
        )
        return int(clip(steering, -self.config.max_steering, self.config.max_steering))

    def _entry_curve_steering(
        self,
        path: ReversePath,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
    ) -> int:
        return self._path_steering(
            path,
            left_ultrasonic_mm,
            right_ultrasonic_mm,
            minimum_abs=abs(int(self.config.reverse_entry_min_steering)),
        )

    def _correction_steering(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
    ) -> int:
        steering = self._path_steering(
            path,
            minimum_abs=abs(int(self.config.correction_steering)),
        )
        if steering != 0:
            return steering
        direction = self._geometry_steering_direction(geometry)
        magnitude = min(
            abs(int(self.config.correction_steering)),
            abs(int(self.config.max_steering)),
        )
        return int(direction * magnitude)

    @staticmethod
    def _geometry_steering_direction(geometry: ParkingGeometry) -> int:
        if abs(geometry.heading_error_deg) > 1.0:
            return 1 if geometry.heading_error_deg > 0.0 else -1
        if abs(geometry.lateral_error_norm) > 0.02:
            return 1 if geometry.lateral_error_norm > 0.0 else -1
        return 1

    def _ultrasonic_correction(
        self,
        left_mm: Optional[float],
        right_mm: Optional[float],
    ) -> int:
        if not self._usable_ultrasonic(left_mm) or not self._usable_ultrasonic(right_mm):
            return 0
        correction = self.config.ultrasonic_kp_steering_per_mm * (
            float(right_mm) - float(left_mm)
        )
        limit = abs(self.config.ultrasonic_max_correction)
        return int(round(clip(correction, -limit, limit)))

    def _ultrasonic_emergency(self, value_mm: Optional[float]) -> bool:
        return (
            self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.ultrasonic_emergency_mm
        )

    def _exit_right_blocked(self, value_mm: Optional[float]) -> bool:
        return (
            self.config.exit_right_min_clearance_mm > 0.0
            and self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.exit_right_min_clearance_mm
        )

    def _usable_ultrasonic(self, value_mm: Optional[float]) -> bool:
        return (
            value_mm is not None
            and 0.0 < float(value_mm) <= self.config.ultrasonic_max_valid_mm
        )

    def _prealign_steering(self) -> int:
        return int(self.config.prealign_steering)

    def _prealign_drive(self, elapsed: float, reason: str) -> ParkingPlan:
        steering = self._prealign_steering()
        if elapsed < max(0.0, self.config.prealign_steer_settle_s):
            return self._drive(0, steering, "steering_settle:" + reason)
        return self._drive(self.config.prealign_speed, steering, reason)

    def _exit_speed(self) -> int:
        return abs(int(self.config.exit_speed))

    def _exit_right_plan(
        self,
        now: float,
        right_ultrasonic_mm: Optional[float],
    ) -> ParkingPlan:
        if self._exit_right_blocked(right_ultrasonic_mm):
            self._state_started_at = now
            return self._stop(
                "exit_right_blocked<=%.0fmm"
                % self.config.exit_right_min_clearance_mm
            )
        if self._state_elapsed(now) >= max(0.0, self.config.exit_turn_s):
            self._enter(ParkingState.EXIT_STRAIGHT, now)
            return self._drive(self._exit_speed(), 0, "exit_straight")
        return self._drive(
            self._exit_speed(),
            int(self.config.exit_turn_steering),
            "exit_right_turn",
        )

    @staticmethod
    def _prealign_metrics(
        lidar: LidarParkingObservation,
    ) -> Optional[tuple[float, float, float]]:
        values = (
            lidar.gap_center_x_right_mm,
            lidar.gap_center_y_back_mm,
            lidar.entry_target_y_back_mm,
            lidar.slot_depth_x_right,
            lidar.slot_depth_y_back,
        )
        if any(value is None for value in values):
            return None
        center_x = float(lidar.gap_center_x_right_mm)
        center_y = float(lidar.gap_center_y_back_mm)
        rear_axle_y = float(lidar.entry_target_y_back_mm)
        depth_x = float(lidar.slot_depth_x_right)
        depth_y = float(lidar.slot_depth_y_back)
        depth_length = hypot(depth_x, depth_y)
        if depth_length <= 1e-9:
            return None
        depth_x /= depth_length
        depth_y /= depth_length
        target_x = center_x
        target_y = center_y - rear_axle_y
        target_distance = hypot(target_x, target_y)
        if target_distance <= 1e-9:
            return None
        # Both angles are measured from the vehicle-rear direction (+y_back).
        # Positive values mean the target/slot is still on the vehicle-right.
        slot_heading = degrees(atan2(depth_x, depth_y))
        entry_bearing = degrees(atan2(target_x, target_y))
        return slot_heading, entry_bearing, target_distance

    def _expired(self, now: float, timeout_s: float) -> bool:
        return timeout_s > 0.0 and now - self._state_started_at >= timeout_s

    def _state_elapsed(self, now: float) -> float:
        return max(0.0, now - self._state_started_at)

    def _enter(self, state: ParkingState, now: float) -> None:
        self.state = state
        self._state_started_at = now
        self._aligned_frames = 0
        self._prealign_aligned_frames = 0
        self._prealign_gap_acquired_at = None

    def _abort(self, now: float, reason: str) -> ParkingPlan:
        self._enter(ParkingState.ABORTED, now)
        return self._stop(reason)

    def _stop(self, reason: str, path: Optional[ReversePath] = None) -> ParkingPlan:
        return ParkingPlan(self.state, ControlCommand.stop(reason), reason, path)

    def _drive(
        self,
        speed: int,
        steering: int,
        reason: str,
        path: Optional[ReversePath] = None,
    ) -> ParkingPlan:
        return ParkingPlan(
            self.state,
            ControlCommand(speed=speed, steering=steering, brake=False, reason=reason),
            reason,
            path,
        )


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
