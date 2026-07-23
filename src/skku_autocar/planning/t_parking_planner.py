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
    IDLE = "idle" #시작 전 대기
    SEARCH_CARS = "search_cars" #첫번째 장애물 찾기
    TRACK_GAP = "track_gap" #첫번째 장애물을 추적하며 두번째 장애물과의 공간 찾기
    POSITION_REAR_AXLE = "position_rear_axle" #후륜을 주차 공간에 맞추는 작업 (선택적 실행)
    PREALIGN_LEFT = "prealign_left" #좌조향 실행
    VERIFY_SLOT_BOX = "verify_slot_box" #라이다로 만든 주차 공간이 안정적인지 확인
    # Backward-compatible alias for old replay integrations.
    VERIFY_PARKING_LINES = "verify_slot_box"
    PLAN_REVERSE_PATH = "plan_reverse_path" #첫 짧은 후진 목표점이 유효한지 확인
    FOLLOW_ENTRY_CURVE = "follow_entry_curve" #곡선 경로를 따라 후진
    FOLLOW_SLOT_CENTER = "follow_slot_center" #직선 경로를 따라 후진
    CORRECT_FORWARD = "correct_forward" #오차가 크면 전진
    CORRECT_REVERSE = "correct_reverse" #다시 후진
    PARKED = "parked"
    EXIT_RIGHT = "exit_right"
    EXIT_STRAIGHT = "exit_straight"
    EXIT_DONE = "exit_done"
    ABORTED = "aborted"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class ParkingPlannerConfig:
    search_speed: int = 100 #첫 차량을 찾으면서 직진하는 속도
    start_forward_s: float = 0.8 #초반에 무조건 직진하는 시간 (초)
    straight_steering_trim: int = -20
    gap_tracking_speed: int = 100 #두 차량 사이 공간을 추적할 때의 모터 속도
    position_speed: int = 100 #후륜축 중심을 공간 중심에 맞출 때 사용하는 속도
    first_car_preemptive_turn_enabled: bool = True
    first_car_only_prealign_enabled: bool = False
    first_car_approach_speed: int = 100 #첫 차량 감지 후 기준 위치까지 접근하는 속도
    first_car_straight_s: float = 1.6 #첫 차량 기준 위치 도달 후 추가 직진 시간 (초)
    prealign_enabled: bool = True
    prealign_speed: int = 100 #최대 좌조향에서 전진하는 속도
    prealign_steering: int = -150 #사전 정렬 좌조향 명령
    prealign_steer_settle_s: float = 0.40 #좌조향 진행 직전 정지 상태로 바퀴만 움직이는 시간
    prealign_timeout_s: float = 6.0
    prealign_gap_acquire_timeout_s: float = 0.0 #두번째 공간을 기다리는 제한 시간
    prealign_slot_heading_tolerance_deg: float = 12.0
    prealign_entry_bearing_tolerance_deg: float = 12.0
    prealign_center_x_tolerance_mm: float = 180.0
    prealign_curve_slot_heading_tolerance_deg: float = 45.0 #주차칸과의 각도
    prealign_curve_entry_bearing_tolerance_deg: float = 45.0 #주차칸과의 각도
    prealign_curve_center_x_tolerance_mm: float = 1400.0
    prealign_target_distance_min_mm: float = 900.0
    prealign_target_distance_max_mm: float = 2600.0
    prealign_confirm_frames: int = 3 #정렬 조건을 만족해야 하는 프레임 수
    prealign_heading_overshoot_deg: float = 25.0
    ultrasonic_kp_steering_per_mm: float = 0.23
    ultrasonic_max_correction: int = 35
    ultrasonic_emergency_mm: float = 100.0
    emergency_stop_enabled: bool = False
    ultrasonic_max_valid_mm: float = 2500.0
    ultrasonic_stale_after_s: float = 0.8
    ultrasonic_inside_max_mm: float = 500.0
    ultrasonic_inside_confirm_frames: int = 3
    reverse_entry_speed: int = -100 #곡선 후진 속도
    reverse_center_speed: int = -100 #정렬된 후 직선 후진 속도
    reverse_entry_min_steering: int = 90 #곡선 진입 시 최소 조향
    reverse_entry_steer_settle_s: float = 0.40
    reverse_entry_release_heading_deg: float = 12.0
    reverse_entry_release_confirm_frames: int = 3
    correction_enabled: bool = True
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
    max_steering: int = 150 #후진 경로 추종 최대 조향 크기
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
    path_confirm_frames: int = 3
    entry_curve_timeout_s: float = 16.0
    center_follow_timeout_s: float = 10.0


@dataclass(frozen=True)
class ParkingPlan:
    state: ParkingState
    command: ControlCommand
    reason: str
    path: Optional[ReversePath] = None
    body_mid_inside: bool = False


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
        self._right_first_car_acquired = False
        self._first_car_turn_reached_at: Optional[float] = None
        self._prealign_aligned_frames = 0
        self._prealign_gap_acquired_at: Optional[float] = None
        self._reverse_path_confirm_frames = 0
        self._reverse_entry_mode = "lidar_box_curve"
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
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._right_first_car_acquired = False
        self._first_car_turn_reached_at = None
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False
        self._enter(ParkingState.SEARCH_CARS, now)
        return True

    def reset(self, now: float = 0.0) -> None:
        self.state = ParkingState.IDLE
        self._state_started_at = now
        self._aligned_frames = 0
        self._misaligned_frames = 0
        self._correction_attempts = 0
        self._correction_reverse_steering = 0
        self._right_first_car_acquired = False
        self._first_car_turn_reached_at = None
        self._prealign_aligned_frames = 0
        self._prealign_gap_acquired_at = None
        self._reverse_path_confirm_frames = 0
        self._reverse_entry_mode = "lidar_box_curve"
        self._entry_heading_ready_frames = 0
        self._body_mid_inside_frames = 0
        self._body_mid_inside = False

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

        side_ultrasonic_emergency_states = (
            ParkingState.SEARCH_CARS,
            ParkingState.TRACK_GAP,
            ParkingState.POSITION_REAR_AXLE,
            ParkingState.PREALIGN_LEFT,
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
            self.state in side_ultrasonic_emergency_states
            and (
                self._ultrasonic_emergency(left_ultrasonic_mm)
                or self._ultrasonic_emergency(right_ultrasonic_mm)
            )
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("side_ultrasonic_distance<=%.0fmm" % self.config.ultrasonic_emergency_mm)

        forward_ultrasonic_emergency_states = (
            ParkingState.SEARCH_CARS,
            ParkingState.TRACK_GAP,
            ParkingState.PREALIGN_LEFT,
            ParkingState.CORRECT_FORWARD,
            ParkingState.EXIT_RIGHT,
            ParkingState.EXIT_STRAIGHT,
        )
        if (
            self.state in forward_ultrasonic_emergency_states
            and (
                self._ultrasonic_emergency(front_left_ultrasonic_mm)
                or self._ultrasonic_emergency(front_right_ultrasonic_mm)
            )
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop(
                "front_ultrasonic_distance<=%.0fmm"
                % self.config.ultrasonic_emergency_mm
            )

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
        if (
            self.config.emergency_stop_enabled
            and lidar.unsafe
            and self.state in slot_and_reverse_states
        ):
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("lidar_safety_obstacle")

        if self.state == ParkingState.PREALIGN_LEFT:
            if not lidar.valid:
                return self._prealign_drive(
                    self._state_elapsed(now),
                    "prealign_waiting_for_lidar",
                )
            if self.config.emergency_stop_enabled and lidar.unsafe:
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
            return self._drive(
                self._exit_speed(),
                self._straight_steering(),
                "exit_straight",
            )

        if self.state == ParkingState.SEARCH_CARS:
            if self._expired(now, self.config.search_timeout_s):
                return self._abort(now, "parked_car_search_timeout")
            if self._state_elapsed(now) < max(0.0, self.config.start_forward_s):
                return self._drive(
                    self.config.search_speed,
                    self._straight_steering(),
                    "start_forward_rollout",
                )
            if not lidar.valid:
                return self._stop("waiting_for_lidar_scan")
            if lidar.gap_confirmed:
                self._right_first_car_acquired = True
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_turn_reached
            ):
                self._right_first_car_acquired = True
                self._first_car_turn_reached_at = now
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_straight_delay",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_confirmed
            ):
                self._right_first_car_acquired = True
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_confirmed:creeping_to_turn_point",
                )
            if lidar.gap_found or lidar.first_car_confirmed:
                self._right_first_car_acquired = True
                self._enter(ParkingState.TRACK_GAP, now)
                return self._drive(
                    self.config.gap_tracking_speed,
                    self._straight_steering(),
                    "tracking_parked_cars",
                )
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
                and lidar.first_car_seen
            ):
                return self._drive(
                    self.config.first_car_approach_speed,
                    self._straight_steering(),
                    "first_car_seen:waiting_for_confirmation",
                )
            return self._drive(
                self.config.search_speed,
                self._straight_steering(),
                "searching_for_parked_cars",
            )

        if self.state == ParkingState.TRACK_GAP:
            if self._expired(now, self.config.gap_tracking_timeout_s):
                return self._abort(now, "two_car_gap_timeout")
            if (
                self.config.prealign_enabled
                and self.config.first_car_preemptive_turn_enabled
            ):
                if not self.config.first_car_only_prealign_enabled:
                    if lidar.gap_confirmed:
                        self._enter(ParkingState.POSITION_REAR_AXLE, now)
                        return self._stop("two_car_gap_confirmed")
                    if lidar.gap_found:
                        return self._drive(
                            self.config.gap_tracking_speed,
                            self._straight_steering(),
                            "gap_confirming_before_prealign",
                        )
                    if (
                        lidar.first_car_turn_reached
                        or lidar.first_car_confirmed
                        or lidar.first_car_seen
                        or self._right_first_car_acquired
                    ):
                        return self._drive(
                            self.config.first_car_approach_speed,
                            self._straight_steering(),
                            "first_car_waiting_for_confirmed_gap",
                        )
                    return self._drive(
                        self.config.first_car_approach_speed,
                        self._straight_steering(),
                        "first_car_temporarily_lost:creeping_to_turn_point",
                    )
                turn_ready = lidar.first_car_turn_reached or lidar.gap_confirmed
                if turn_ready and self._first_car_turn_reached_at is None:
                    self._first_car_turn_reached_at = now
                if self._first_car_turn_reached_at is None:
                    return self._drive(
                        self.config.first_car_approach_speed,
                        self._straight_steering(),
                        "first_car_creeping_to_turn_point",
                    )
                if now - self._first_car_turn_reached_at < max(
                    0.0,
                    self.config.first_car_straight_s,
                ):
                    return self._drive(
                        self.config.first_car_approach_speed,
                        self._straight_steering(),
                        "first_car_straight_delay",
                    )
                if (
                    lidar.first_car_turn_reached
                    or lidar.gap_confirmed
                    or lidar.first_car_confirmed
                    or lidar.first_car_seen
                    or lidar.gap_found
                    or self._right_first_car_acquired
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
                    self._straight_steering(),
                    "first_car_temporarily_lost:creeping_to_turn_point",
                )
            if lidar.gap_confirmed:
                self._enter(ParkingState.POSITION_REAR_AXLE, now)
                return self._stop("two_car_gap_confirmed")
            return self._drive(
                self.config.gap_tracking_speed,
                self._straight_steering(),
                "confirming_two_car_gap",
            )

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
            if direction > 0 and (
                self._ultrasonic_emergency(front_left_ultrasonic_mm)
                or self._ultrasonic_emergency(front_right_ultrasonic_mm)
            ):
                self._enter(ParkingState.EMERGENCY_STOP, now)
                return self._stop(
                    "front_ultrasonic_distance<=%.0fmm"
                    % self.config.ultrasonic_emergency_mm
                )
            return self._drive(
                direction * abs(self.config.position_speed),
                self._straight_steering(),
                "correcting_rear_axle_to_gap",
            )

        if self.state == ParkingState.PREALIGN_LEFT:
            elapsed = now - self._state_started_at
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
            (
                slot_heading_deg,
                entry_bearing_deg,
                target_distance_mm,
                center_x_mm,
            ) = metrics
            prealign_path = (
                self.path_generator.generate(geometry)
                if self._full_geometry_usable(geometry)
                else None
            )
            direct_ready = (
                abs(slot_heading_deg)
                <= self.config.prealign_slot_heading_tolerance_deg
                and abs(entry_bearing_deg)
                <= self.config.prealign_entry_bearing_tolerance_deg
                and abs(center_x_mm)
                <= self.config.prealign_center_x_tolerance_mm
                and self.config.prealign_target_distance_min_mm
                <= target_distance_mm
                <= self.config.prealign_target_distance_max_mm
            )
            curve_ready = (
                not direct_ready
                and self._full_geometry_usable(geometry)
                and not lidar.coasted
                and abs(geometry.heading_error_deg)
                <= self.config.prealign_curve_slot_heading_tolerance_deg
                and abs(entry_bearing_deg)
                <= self.config.prealign_curve_entry_bearing_tolerance_deg
                and abs(center_x_mm)
                <= self.config.prealign_curve_center_x_tolerance_mm
                and self.config.prealign_target_distance_min_mm
                <= target_distance_mm
                <= self.config.prealign_target_distance_max_mm
                and prealign_path is not None
                and prealign_path.found
            )
            ready = direct_ready or curve_ready
            self._prealign_aligned_frames = (
                self._prealign_aligned_frames + 1 if ready else 0
            )
            if self._prealign_aligned_frames >= max(
                1,
                self.config.prealign_confirm_frames,
            ):
                self._reverse_entry_mode = (
                    "direct_aligned" if direct_ready else "lidar_box_curve"
                )
                self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                return self._stop(
                    "prealign_direct_reverse_ready"
                    if direct_ready
                    else "prealign_curve_reverse_ready"
                )

            overshot = (
                slot_heading_deg < -abs(self.config.prealign_heading_overshoot_deg)
            )
            timed_out = (
                self.config.prealign_timeout_s > 0.0
                and now - self._prealign_gap_acquired_at
                >= self.config.prealign_timeout_s
            )
            if overshot or timed_out:
                if curve_ready:
                    self._reverse_entry_mode = "lidar_box_curve"
                    self._enter(ParkingState.VERIFY_SLOT_BOX, now)
                    return self._stop(
                        "prealign_curve_reverse_ready:%s"
                        % ("heading_overshoot" if overshot else "timeout")
                    )
                if timed_out:
                    return self._prealign_drive(
                        elapsed,
                        "prealign_alignment_timeout:continuing",
                    )
                return self._abort(
                    now,
                    "prealign_alignment_heading_overshoot",
                )

            reason = "prealign_left head=%+.1f bearing=%+.1f centerX=%+.0fmm dist=%.0fmm" % (
                slot_heading_deg,
                entry_bearing_deg,
                center_x_mm,
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
            if self._reverse_path_confirm_frames < max(
                1,
                self.config.path_confirm_frames,
            ):
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
                self._entry_heading_ready_frames = 0
                return self._stop("entry_curve_path_lost:%s" % path.reason, path)
            if self._reverse_entry_mode == "lidar_box_curve":
                elapsed = self._state_elapsed(now)
                steering = self._fixed_right_entry_steering()
                if elapsed < max(0.0, self.config.reverse_entry_steer_settle_s):
                    self._entry_heading_ready_frames = 0
                    return self._drive(
                        0,
                        steering,
                        "reverse_entry_max_right:settling",
                        path,
                    )
                heading_ready = (
                    abs(geometry.heading_error_deg)
                    <= abs(self.config.reverse_entry_release_heading_deg)
                )
                self._entry_heading_ready_frames = (
                    self._entry_heading_ready_frames + 1
                    if heading_ready
                    else 0
                )
                required_frames = max(
                    1,
                    self.config.reverse_entry_release_confirm_frames,
                )
                if self._entry_heading_ready_frames >= required_frames:
                    self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                    return self._drive(
                        self.config.reverse_center_speed,
                        self._path_steering(
                            path,
                            left_ultrasonic_mm,
                            right_ultrasonic_mm,
                        ),
                        "following_slot_center:entry_heading_released",
                        path,
                    )
                return self._drive(
                    self.config.reverse_entry_speed,
                    steering,
                    (
                        "following_entry_fixed_max_right:"
                        "heading=%+.1f confirm=%d/%d"
                    )
                    % (
                        geometry.heading_error_deg,
                        self._entry_heading_ready_frames,
                        required_frames,
                    ),
                    path,
                )
            aligned = self._slot_aligned(geometry)
            self._aligned_frames = self._aligned_frames + 1 if aligned else 0
            if self._aligned_frames >= max(1, self.config.aligned_confirm_frames):
                self._enter(ParkingState.FOLLOW_SLOT_CENTER, now)
                return self._drive(
                    self.config.reverse_center_speed,
                    self._path_steering(
                        path,
                        left_ultrasonic_mm,
                        right_ultrasonic_mm,
                    ),
                    "following_slot_center:local_target",
                    path,
                )
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
                self._path_steering(
                    path,
                    left_ultrasonic_mm,
                    right_ultrasonic_mm,
                ),
                "following_slot_center:local_target",
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
                self._straight_steering(),
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

    def _fixed_right_entry_steering(self) -> int:
        direction = 1 if self.config.reverse_steering_sign >= 0.0 else -1
        return direction * abs(int(self.config.max_steering))

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
            self.config.emergency_stop_enabled
            and self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.ultrasonic_emergency_mm
        )

    def _any_ultrasonic_emergency(self, *values_mm: Optional[float]) -> bool:
        return any(self._ultrasonic_emergency(value) for value in values_mm)

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
            ParkingState.CORRECT_FORWARD,
            ParkingState.CORRECT_REVERSE,
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

    def _straight_steering(self) -> int:
        limit = abs(int(self.config.max_steering))
        return int(clip(self.config.straight_steering_trim, -limit, limit))

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

    @staticmethod
    def _prealign_metrics(
        lidar: LidarParkingObservation,
    ) -> Optional[tuple[float, float, float, float]]:
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
        return slot_heading, entry_bearing, target_distance, center_x

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
        self._reverse_path_confirm_frames = 0
        self._entry_heading_ready_frames = 0

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
