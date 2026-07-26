from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

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
    ENTRY_SETUP = "entry_setup"
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
    search_speed: int = 100
    start_forward_s: float = 0.0
    straight_steering_trim: int = -30
    gap_tracking_speed: int = 100
    position_speed: int = 100
    first_car_preemptive_turn_enabled: bool = False
    first_car_only_prealign_enabled: bool = False
    first_car_approach_speed: int = 100
    first_car_straight_s: float = 0.0
    prealign_enabled: bool = False
    prealign_speed: int = 100
    prealign_steering: int = -150
    prealign_steer_settle_s: float = 0.0
    prealign_timeout_s: float = 0.0
    prealign_gap_acquire_timeout_s: float = 0.0
    prealign_slot_heading_tolerance_deg: float = 12.0
    prealign_entry_bearing_tolerance_deg: float = 12.0
    prealign_center_x_tolerance_mm: float = 180.0
    prealign_curve_slot_heading_tolerance_deg: float = 45.0
    prealign_curve_entry_bearing_tolerance_deg: float = 45.0
    prealign_curve_center_x_tolerance_mm: float = 1400.0
    prealign_target_distance_min_mm: float = 900.0
    prealign_target_distance_max_mm: float = 2600.0
    prealign_confirm_frames: int = 1
    prealign_heading_overshoot_deg: float = 25.0
    ultrasonic_kp_steering_per_mm: float = 0.23
    ultrasonic_max_correction: int = 35
    ultrasonic_emergency_mm: float = 100.0
    emergency_stop_enabled: bool = False
    ultrasonic_max_valid_mm: float = 2500.0
    ultrasonic_stale_after_s: float = 0.8
    ultrasonic_inside_max_mm: float = 500.0
    ultrasonic_inside_confirm_frames: int = 3
    entry_setup_enabled: bool = True
    early_entry_setup_enabled: bool = True
    entry_setup_speed: int = 80
    entry_setup_steering: int = -150
    entry_setup_steer_settle_s: float = 0.30
    entry_setup_min_s: float = 1.00
    entry_setup_max_s: float = 3.00
    entry_setup_target_heading_deg: float = 60.0
    entry_setup_lateral_trigger_norm: float = 0.90
    reverse_entry_speed: int = -100
    reverse_center_speed: int = -100
    reverse_entry_min_steering: int = 90
    reverse_entry_steer_settle_s: float = 0.20
    reverse_entry_release_heading_deg: float = 8.0
    reverse_entry_release_confirm_frames: int = 2
    correction_enabled: bool = False
    correction_forward_speed: int = 100
    correction_reverse_speed: int = -100
    correction_steering: int = 130
    correction_steer_settle_s: float = 0.25
    correction_forward_s: float = 0.70
    correction_reverse_s: float = 1.10
    correction_min_reverse_s: float = 0.80
    correction_depth_trigger_px: float = 760.0
    correction_heading_trigger_deg: float = 35.0
    correction_lateral_trigger_norm: float = 0.30
    correction_trigger_frames: int = 3
    correction_max_attempts: int = 3
    park_hold_s: float = 3.0
    exit_speed: int = 100
    exit_turn_steering: int = 80
    exit_turn_s: float = 1.6
    exit_straight_s: float = 0.0
    exit_right_min_clearance_mm: float = 180.0
    max_steering: int = 150
    reverse_steering_sign: float = 1.0
    geometry_confidence_min: float = 0.20
    aligned_heading_deg: float = 8.0
    aligned_lateral_norm: float = 0.18
    aligned_confirm_frames: int = 3
    stop_depth_margin_px: float = 8.0
    verify_hold_s: float = 0.0
    search_timeout_s: float = 60.0
    gap_tracking_timeout_s: float = 20.0
    position_timeout_s: float = 10.0
    verify_timeout_s: float = 5.0
    path_timeout_s: float = 4.0
    path_confirm_frames: int = 1
    entry_curve_timeout_s: float = 16.0
    center_follow_timeout_s: float = 10.0


@dataclass(frozen=True)
class ParkingPlan:
    state: ParkingState
    command: ControlCommand
    reason: str
    path: Optional[ReversePath] = None
    body_mid_inside: bool = False
    world_path: Optional[Any] = None


class TParkingPlanner:
    """Straight-search T parking planner.

    The mission now stays on a straight forward line until LiDAR confirms the
    empty bay and the rear-camera/LiDAR geometry can produce a reverse target.
    No first-car pre-turn, rear-axle positioning, or fixed steering arc is used.
    """

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
        self._reverse_path_confirm_frames = 0
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False

    def start(self, now: float) -> bool:
        if self.state not in (
            ParkingState.IDLE,
            ParkingState.ABORTED,
            ParkingState.EMERGENCY_STOP,
            ParkingState.PARKED,
            ParkingState.EXIT_DONE,
        ):
            return False
        self._reset_counters()
        self._enter(ParkingState.SEARCH_CARS, now)
        return True

    def reset(self, now: float = 0.0) -> None:
        self.state = ParkingState.IDLE
        self._state_started_at = now
        self._reset_counters()

    @property
    def prealign_confirmed_frames(self) -> int:
        return 0

    def update(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        enabled: bool = True,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
        front_left_ultrasonic_mm: Optional[float] = None,
        front_right_ultrasonic_mm: Optional[float] = None,
    ) -> ParkingPlan:
        if not enabled:
            self.reset(now)
            return self._stop("parking_disabled")
        if self.state == ParkingState.IDLE:
            return self._stop("waiting_for_start")

        self._update_body_mid_inside(left_ultrasonic_mm, right_ultrasonic_mm)

        if self.state == ParkingState.PARKED:
            if self._state_elapsed(now) >= max(0.0, self.config.park_hold_s):
                self._enter(ParkingState.EXIT_RIGHT, now)
                if self._any_ultrasonic_emergency(
                    left_ultrasonic_mm,
                    right_ultrasonic_mm,
                    front_left_ultrasonic_mm,
                    front_right_ultrasonic_mm,
                ):
                    self._enter(ParkingState.EMERGENCY_STOP, now)
                    return self._stop(
                        "ultrasonic_distance<=%.0fmm"
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

        if self._forward_ultrasonic_emergency_state() and (
            self._ultrasonic_emergency(front_left_ultrasonic_mm)
            or self._ultrasonic_emergency(front_right_ultrasonic_mm)
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop(
                "front_ultrasonic_distance<=%.0fmm"
                % self.config.ultrasonic_emergency_mm
            )

        if self._side_ultrasonic_emergency_state() and (
            self._ultrasonic_emergency(left_ultrasonic_mm)
            or self._ultrasonic_emergency(right_ultrasonic_mm)
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop(
                "side_ultrasonic_distance<=%.0fmm"
                % self.config.ultrasonic_emergency_mm
            )

        reverse_or_slot_states = (
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.ENTRY_SETUP,
            ParkingState.PLAN_REVERSE_PATH,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
        )
        if self.state in reverse_or_slot_states and not lidar.valid:
            return self._stop("lidar_unavailable_for_slot")
        if (
            self.config.emergency_stop_enabled
            and lidar.unsafe
            and self.state in reverse_or_slot_states
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("lidar_safety_obstacle")

        if self.state == ParkingState.EXIT_RIGHT:
            return self._exit_right_plan(now, right_ultrasonic_mm)
        if self.state == ParkingState.EXIT_STRAIGHT:
            if (
                self.config.exit_straight_s > 0.0
                and self._state_elapsed(now) >= self.config.exit_straight_s
            ):
                self._enter(ParkingState.EXIT_DONE, now)
                return self._stop("exit_complete")
            return self._drive(
                self._exit_speed(),
                self._straight_steering(),
                "exit_straight",
            )

        if self.state == ParkingState.SEARCH_CARS:
            return self._search_plan(geometry, lidar, now)
        if self.state == ParkingState.TRACK_GAP:
            return self._track_gap_plan(geometry, lidar, now)
        if self.state == ParkingState.VERIFY_SLOT_BOX:
            return self._verify_slot_plan(geometry, now)
        if self.state == ParkingState.ENTRY_SETUP:
            return self._entry_setup_plan(geometry, lidar, now)

        path = self.path_generator.generate(geometry)
        if self.state == ParkingState.PLAN_REVERSE_PATH:
            return self._path_plan(path, now)
        if self.state == ParkingState.FOLLOW_ENTRY_CURVE:
            return self._entry_curve_plan(
                geometry,
                path,
                now,
                left_ultrasonic_mm,
                right_ultrasonic_mm,
            )
        if self.state == ParkingState.FOLLOW_SLOT_CENTER:
            return self._slot_center_plan(
                geometry,
                path,
                now,
                left_ultrasonic_mm,
                right_ultrasonic_mm,
            )

        return self._abort(now, "unsupported_state:%s" % self.state.value)

    def _search_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.search_timeout_s):
            return self._abort(now, "slot_search_timeout")
        if lidar.valid:
            if self._slot_ready(lidar, geometry):
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop("slot_detected")
            if lidar.gap_confirmed:
                return self._begin_entry_setup(
                    now,
                    "slot_lidar_confirmed_entry_setup",
                )
            if self._early_entry_setup_cue(lidar):
                return self._begin_entry_setup(
                    now,
                    self._early_entry_setup_reason(lidar),
                )
            if lidar.gap_found:
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.gap_tracking_speed,
                    self._straight_steering(),
                    "slot_candidate_confirming",
                )
        if self._state_elapsed(now) < max(0.0, self.config.start_forward_s):
            return self._drive(
                self.config.search_speed,
                self._straight_steering(),
                "straight_search_rollout",
            )
        if not lidar.valid:
            return self._stop("waiting_for_lidar_scan")
        return self._drive(
            self.config.search_speed,
            self._straight_steering(),
            "straight_searching_for_slot",
        )

    def _track_gap_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.gap_tracking_timeout_s):
            return self._abort(now, "slot_candidate_timeout")
        if not lidar.valid:
            return self._stop("waiting_for_lidar_scan")
        if self._slot_ready(lidar, geometry):
            self._enter(ParkingState.VERIFY_SLOT_BOX, now)
            return self._stop("slot_detected")
        if lidar.gap_confirmed:
            return self._begin_entry_setup(
                now,
                "slot_lidar_confirmed_entry_setup",
            )
        if self._early_entry_setup_cue(lidar):
            return self._begin_entry_setup(
                now,
                self._early_entry_setup_reason(lidar),
            )
        if not lidar.gap_found:
            self._enter(ParkingState.SEARCH_CARS, now)
            return self._drive(
                self.config.search_speed,
                self._straight_steering(),
                "slot_candidate_lost:straight_search",
            )
        return self._drive(
            self.config.gap_tracking_speed,
            self._straight_steering(),
            "slot_candidate_confirming",
        )

    def _verify_slot_plan(
        self,
        geometry: ParkingGeometry,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.verify_timeout_s):
            return self._abort(now, "slot_geometry_verify_timeout")
        if not self._full_geometry_usable(geometry):
            return self._stop("waiting_for_lidar_camera_slot")
        if self._state_elapsed(now) < max(0.0, self.config.verify_hold_s):
            return self._stop("slot_geometry_verify_hold")
        if self._entry_setup_needed(geometry):
            self._enter(ParkingState.ENTRY_SETUP, now)
            return self._drive(
                0,
                self._entry_setup_steering(),
                "entry_setup_steering_settle",
            )
        self._enter(ParkingState.PLAN_REVERSE_PATH, now)
        return self._stop("slot_geometry_verified")

    def _entry_setup_plan(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        if not lidar.gap_confirmed:
            if self._expired(now, self.config.entry_setup_max_s):
                return self._abort(now, "entry_setup_lidar_not_confirmed")
            return self._entry_setup_drive(
                now,
                "entry_setup_waiting_for_lidar_confirmation",
            )

        if not self._full_geometry_usable(geometry):
            if self._expired(now, self.config.entry_setup_max_s):
                return self._abort(now, "entry_setup_geometry_not_ready")
            return self._entry_setup_drive(
                now,
                "entry_setup_waiting_for_full_geometry",
            )

        elapsed = self._state_elapsed(now)
        if elapsed >= max(0.0, self.config.entry_setup_min_s) and (
            not self._entry_setup_needed(geometry)
        ):
            self._enter(ParkingState.PLAN_REVERSE_PATH, now)
            return self._stop("entry_setup_angle_ready")

        if self._expired(now, self.config.entry_setup_max_s):
            return self._abort(
                now,
                "entry_setup_angle_not_ready:heading=%.1f lateral=%.2f"
                % (geometry.heading_error_deg, geometry.lateral_error_norm),
            )

        return self._entry_setup_drive(now, "entry_setup_forward_angle")

    def _begin_entry_setup(self, now: float, reason: str) -> ParkingPlan:
        self._enter(ParkingState.ENTRY_SETUP, now)
        return self._drive(
            0,
            self._entry_setup_steering(),
            reason,
        )

    def _entry_setup_drive(self, now: float, moving_reason: str) -> ParkingPlan:
        elapsed = self._state_elapsed(now)
        steering = self._entry_setup_steering()
        if elapsed < max(0.0, self.config.entry_setup_steer_settle_s):
            return self._drive(0, steering, "entry_setup_steering_settle")
        return self._drive(
            self.config.entry_setup_speed,
            steering,
            moving_reason,
        )

    def _path_plan(self, path: ReversePath, now: float) -> ParkingPlan:
        if self._expired(now, self.config.path_timeout_s):
            return self._abort(now, "reverse_path_timeout:%s" % path.reason)
        if not path.found:
            self._reverse_path_confirm_frames = max(
                0,
                self._reverse_path_confirm_frames - 1,
            )
            return self._stop(
                "waiting_for_reverse_path:%s confirm=%d/%d"
                % (
                    path.reason,
                    self._reverse_path_confirm_frames,
                    max(1, self.config.path_confirm_frames),
                ),
                path,
            )
        self._reverse_path_confirm_frames += 1
        if self._reverse_path_confirm_frames < max(1, self.config.path_confirm_frames):
            return self._stop(
                "reverse_path_confirming:%d/%d"
                % (
                    self._reverse_path_confirm_frames,
                    max(1, self.config.path_confirm_frames),
                ),
                path,
            )
        self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
        return self._stop("reverse_path_armed", path)

    def _entry_curve_plan(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
        left_ultrasonic_mm: Optional[float],
        right_ultrasonic_mm: Optional[float],
    ) -> ParkingPlan:
        if self._expired(now, self.config.entry_curve_timeout_s):
            return self._abort(now, "entry_curve_timeout")
        stop = self._stop_at_back_line(geometry, now, path)
        if stop is not None:
            return stop
        if not path.found:
            self._aligned_frames = 0
            self._entry_heading_ready_frames = 0
            return self._stop("entry_curve_path_lost:%s" % path.reason, path)

        steering = self._entry_curve_steering(
            path,
            left_ultrasonic_mm,
            right_ultrasonic_mm,
        )
        if self._state_elapsed(now) < max(0.0, self.config.reverse_entry_steer_settle_s):
            return self._drive(0, steering, "entry_curve_steering_settle", path)

        heading_ready = (
            abs(geometry.heading_error_deg)
            <= abs(self.config.reverse_entry_release_heading_deg)
        )
        aligned = self._slot_aligned(geometry)
        self._entry_heading_ready_frames = (
            self._entry_heading_ready_frames + 1
            if heading_ready or aligned
            else 0
        )
        self._aligned_frames = self._aligned_frames + 1 if aligned else 0
        if (
            self._entry_heading_ready_frames
            >= max(1, self.config.reverse_entry_release_confirm_frames)
            or self._aligned_frames >= max(1, self.config.aligned_confirm_frames)
        ):
            self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
            return self._drive(
                self.config.reverse_center_speed,
                self._path_steering(path, left_ultrasonic_mm, right_ultrasonic_mm),
                "following_slot_center:entry_aligned",
                path,
            )

        return self._drive(
            self.config.reverse_entry_speed,
            steering,
            "following_entry_curve:path_target",
            path,
        )

    def _slot_center_plan(
        self,
        geometry: ParkingGeometry,
        path: ReversePath,
        now: float,
        left_ultrasonic_mm: Optional[float],
        right_ultrasonic_mm: Optional[float],
    ) -> ParkingPlan:
        if self._expired(now, self.config.center_follow_timeout_s):
            return self._abort(now, "slot_center_follow_timeout")
        stop = self._stop_at_back_line(geometry, now, path)
        if stop is not None:
            return stop
        if not path.found:
            return self._stop("slot_center_path_lost:%s" % path.reason, path)
        return self._drive(
            self.config.reverse_center_speed,
            self._path_steering(path, left_ultrasonic_mm, right_ultrasonic_mm),
            "following_slot_center:path_target",
            path,
        )

    def _slot_ready(
        self,
        lidar: LidarParkingObservation,
        geometry: ParkingGeometry,
    ) -> bool:
        return lidar.gap_confirmed and self._full_geometry_usable(geometry)

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

    def _entry_setup_needed(self, geometry: ParkingGeometry) -> bool:
        if not self.config.entry_setup_enabled:
            return False
        return (
            abs(geometry.heading_error_deg)
            > abs(self.config.entry_setup_target_heading_deg)
            or abs(geometry.lateral_error_norm)
            > abs(self.config.entry_setup_lateral_trigger_norm)
        )

    def _early_entry_setup_cue(self, lidar: LidarParkingObservation) -> bool:
        if not self.config.entry_setup_enabled:
            return False
        if not self.config.early_entry_setup_enabled:
            return False
        return (
            lidar.first_car_turn_reached
            or (lidar.gap_found and lidar.gap_pair_observed)
        )

    def _early_entry_setup_reason(self, lidar: LidarParkingObservation) -> str:
        if lidar.first_car_turn_reached:
            return "early_entry_setup:first_car_turn_reached"
        return "early_entry_setup:gap_candidate"

    def _stop_at_back_line(
        self,
        geometry: ParkingGeometry,
        now: float,
        path: ReversePath,
    ) -> Optional[ParkingPlan]:
        if not self._full_geometry_usable(geometry):
            return self._stop("reverse_waiting_for_full_geometry", path)
        if geometry.depth_remaining_px <= self.config.stop_depth_margin_px:
            if not geometry.vehicle_fully_inside:
                return self._stop("back_clearance_reached_vehicle_not_fully_inside", path)
            if not self._slot_aligned(geometry):
                return self._stop("back_clearance_reached_vehicle_not_aligned", path)
            self._enter(ParkingState.PARKED, now)
            return self._stop("vehicle_fully_inside_and_aligned", path)
        return None

    def _path_steering(
        self,
        path: ReversePath,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
        minimum_abs: int = 0,
    ) -> int:
        full_scale = max(
            1e-9,
            self.path_generator.config.full_steering_curvature_per_px,
        )
        normalized = clip(path.curvature_per_px / full_scale, -1.0, 1.0)
        raw = self.config.reverse_steering_sign * self.config.max_steering * normalized
        steering = round(raw)
        if abs(raw) > 1e-9 and minimum_abs > 0:
            minimum = min(abs(int(minimum_abs)), abs(int(self.config.max_steering)))
            if abs(steering) < minimum:
                steering = minimum if raw > 0.0 else -minimum
        steering += self._ultrasonic_correction(left_ultrasonic_mm, right_ultrasonic_mm)
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
            self.config.emergency_stop_enabled
            and self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.ultrasonic_emergency_mm
        )

    def _any_ultrasonic_emergency(self, *values_mm: Optional[float]) -> bool:
        return any(self._ultrasonic_emergency(value) for value in values_mm)

    def _usable_ultrasonic(self, value_mm: Optional[float]) -> bool:
        return (
            value_mm is not None
            and 0.0 < float(value_mm) <= self.config.ultrasonic_max_valid_mm
        )

    def _update_body_mid_inside(
        self,
        left_mm: Optional[float],
        right_mm: Optional[float],
    ) -> None:
        if self._body_mid_inside:
            return
        if self.state not in (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
        ):
            self._body_mid_inside_frames = 0
            return
        threshold = max(0.0, self.config.ultrasonic_inside_max_mm)
        detected = (
            threshold > 0.0
            and self._usable_ultrasonic(left_mm)
            and self._usable_ultrasonic(right_mm)
            and float(left_mm) <= threshold
            and float(right_mm) <= threshold
        )
        self._body_mid_inside_frames = (
            self._body_mid_inside_frames + 1 if detected else 0
        )
        if self._body_mid_inside_frames >= max(
            1,
            self.config.ultrasonic_inside_confirm_frames,
        ):
            self._body_mid_inside = True

    def _side_ultrasonic_emergency_state(self) -> bool:
        return self.state in (
            ParkingState.SEARCH_CARS,
            ParkingState.TRACK_GAP,
            ParkingState.VERIFY_SLOT_BOX,
            ParkingState.ENTRY_SETUP,
            ParkingState.PLAN_REVERSE_PATH,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
            ParkingState.EXIT_RIGHT,
            ParkingState.EXIT_STRAIGHT,
        )

    def _forward_ultrasonic_emergency_state(self) -> bool:
        return self.state in (
            ParkingState.SEARCH_CARS,
            ParkingState.TRACK_GAP,
            ParkingState.ENTRY_SETUP,
            ParkingState.EXIT_RIGHT,
            ParkingState.EXIT_STRAIGHT,
        )

    def _straight_steering(self) -> int:
        limit = abs(int(self.config.max_steering))
        return int(clip(self.config.straight_steering_trim, -limit, limit))

    def _entry_setup_steering(self) -> int:
        limit = abs(int(self.config.max_steering))
        return int(clip(self.config.entry_setup_steering, -limit, limit))

    def _exit_speed(self) -> int:
        return abs(int(self.config.exit_speed))

    def _exit_right_blocked(self, value_mm: Optional[float]) -> bool:
        return (
            self.config.exit_right_min_clearance_mm > 0.0
            and self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.exit_right_min_clearance_mm
        )

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
            return self._drive(
                self._exit_speed(),
                self._straight_steering(),
                "exit_straight",
            )
        return self._drive(
            self._exit_speed(),
            int(self.config.exit_turn_steering),
            "exit_right_turn",
        )

    def _expired(self, now: float, timeout_s: float) -> bool:
        return timeout_s > 0.0 and now - self._state_started_at >= timeout_s

    def _state_elapsed(self, now: float) -> float:
        return max(0.0, now - self._state_started_at)

    def _enter(self, state: ParkingState, now: float) -> None:
        self.state = state
        self._state_started_at = now
        self._aligned_frames = 0
        self._reverse_path_confirm_frames = 0
        self._entry_heading_ready_frames = 0

    def _reset_counters(self) -> None:
        self._aligned_frames = 0
        self._reverse_path_confirm_frames = 0
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False

    def _abort(self, now: float, reason: str) -> ParkingPlan:
        self._enter(ParkingState.ABORTED, now)
        return self._stop(reason)

    def _stop(self, reason: str, path: Optional[ReversePath] = None) -> ParkingPlan:
        return ParkingPlan(
            self.state,
            ControlCommand.stop(reason),
            reason,
            path,
            self._body_mid_inside,
        )

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
            self._body_mid_inside,
        )


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
