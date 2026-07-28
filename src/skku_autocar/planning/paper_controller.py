from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin
from statistics import median
from typing import Deque, Optional, Tuple

from ..config import PaperControllerConfig
from ..perception.rear_lidar import RearLidarObservation, TangentPair
from ..types import ControlCommand, ParkingState


@dataclass(frozen=True)
class PaperParkingDebug:
    state: ParkingState = ParkingState.IDLE
    detected_vehicle_count: int = 0
    paper_steering: Optional[float] = None
    applied_paper_steering: Optional[float] = None
    entry_center_angle_deg: Optional[float] = None
    angle_term: Optional[float] = None
    distance_term: Optional[float] = None
    distance_bias_ab_mm: Optional[float] = None
    distance_bias_cd_mm: Optional[float] = None
    prealign_near_seen: bool = False
    prealign_ready_scans: int = 0
    pair_hold_scans: int = 0
    realigning_after_dropout: bool = False
    center_observation_scans: int = 0
    reason: str = "idle"


class PaperParkingController:
    """Figure-9 controller with scan-confirmed paper measurements."""

    def __init__(self, config: PaperControllerConfig):
        self.config = config
        self.state = ParkingState.IDLE
        self.debug = PaperParkingDebug()
        self.control_pair = TangentPair()
        self._detected_vehicle_count = 0
        self._recovery_started_at = 0.0
        self._reverse_motion_started = False
        self._last_lidar_timestamp: Optional[float] = None
        self._is_new_scan = False
        self._prealign_near_seen = False
        self._prealign_ready_scans = 0
        self._pair_samples: Deque[TangentPair] = deque(
            maxlen=self.config.pair_filter_scans
        )
        self._pair_missing_scans = 0
        self._pair_outlier_rejected = False
        self._realigning_after_dropout = False
        self._center_scan_count = 0
        self._center_c_samples: Deque[float] = deque(
            maxlen=self.config.center_observation_scans
        )
        self._center_d_samples: Deque[float] = deque(
            maxlen=self.config.center_observation_scans
        )
        self._finish_missing_scans = 0

    def start(self) -> None:
        self._detected_vehicle_count = 0
        self._recovery_started_at = 0.0
        self._reverse_motion_started = False
        self._reset_measurement_state()
        self.state = ParkingState.SEARCH_FIRST_CAR
        self._set_debug("paper_search_first_vehicle")

    def reset(self) -> None:
        self._detected_vehicle_count = 0
        self._recovery_started_at = 0.0
        self._reverse_motion_started = False
        self._reset_measurement_state()
        self.state = ParkingState.IDLE
        self._set_debug("idle")

    def update(
        self,
        observation: RearLidarObservation,
        now: float,
    ) -> ControlCommand:
        if self.state == ParkingState.IDLE:
            return self._stop("press_SPACE_to_start")
        if self.state == ParkingState.PARKED:
            return self._stop("paper_parking_complete")
        if not observation.valid:
            return self._stop("waiting_for_rear_lidar")

        self._is_new_scan = (
            observation.timestamp != self._last_lidar_timestamp
        )
        if self._is_new_scan:
            self._last_lidar_timestamp = observation.timestamp

        if self.state == ParkingState.SEARCH_FIRST_CAR:
            return self._search_first_car(observation)
        if self.state == ParkingState.SEARCH_OPEN_SPACE:
            return self._search_open_space(observation)
        if self.state == ParkingState.SEARCH_SECOND_CAR:
            return self._search_second_car(observation)
        if self.state == ParkingState.PREALIGN_LEFT:
            return self._prealign_left(observation)
        if self.state == ParkingState.REVERSE_ALIGN:
            return self._reverse_align(observation)
        if self.state == ParkingState.CENTER_CHECK:
            return self._center_check(observation, now)
        if self.state == ParkingState.REVERSE_STRAIGHT:
            return self._reverse_straight(observation)
        if self.state == ParkingState.RECOVERY_FORWARD:
            return self._recovery_forward(observation, now)
        return self._stop("unknown_state")

    def _search_first_car(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        if observation.right_vehicle_present:
            self._detected_vehicle_count = 1
            self.state = ParkingState.SEARCH_OPEN_SPACE
            return self._forward("paper_first_vehicle_counted")
        return self._forward("paper_search_first_vehicle")

    def _search_open_space(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        if not observation.right_vehicle_present:
            self.state = ParkingState.SEARCH_SECOND_CAR
            return self._forward("paper_between_detected_vehicles")
        return self._forward("paper_passing_first_vehicle")

    def _search_second_car(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        if not observation.right_vehicle_present:
            return self._forward("paper_search_second_vehicle")

        self._detected_vehicle_count = 2
        self._realigning_after_dropout = False
        self._prealign_near_seen = observation.prealign_near
        self._prealign_ready_scans = 0
        self._reset_pair_tracking(observation)
        self.state = ParkingState.PREALIGN_LEFT
        if not observation.pair.valid:
            return self._drive(
                self.config.forward_speed,
                self._paper_to_actuator(
                    -self.config.paper_max_steering
                ),
                "figure7_left_forward_acquiring_valid_ab",
            )
        return self._drive(
            self.config.forward_speed,
            self._paper_to_actuator(
                -self.config.paper_max_steering
            ),
            "figure7_second_vehicle_left_forward",
        )

    def _prealign_left(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        pair, pair_held = self._control_pair(observation)
        if not pair.valid:
            self._prealign_ready_scans = 0
            return self._drive(
                self.config.forward_speed,
                self._paper_to_actuator(
                    -self.config.paper_max_steering
                ),
                "figure7_left_forward_reacquire_ab "
                "missing=%d/%d"
                % (
                    self._pair_missing_scans,
                    self.config.pair_hold_scans,
                )
            )

        prospective_steering, _, _ = self._paper_steering(pair)
        if self._is_new_scan and observation.prealign_near:
            self._prealign_near_seen = True

        near_cycle_complete = (
            self._realigning_after_dropout
            or (
                self._prealign_near_seen
                and not observation.prealign_near
            )
        )
        ready_now = (
            prospective_steering > 0.0
            and near_cycle_complete
            and not pair_held
        )
        if self._is_new_scan:
            if ready_now:
                self._prealign_ready_scans += 1
            else:
                self._prealign_ready_scans = 0

        if (
            self._prealign_ready_scans
            >= self.config.prealign_clear_confirm_scans
        ):
            self._realigning_after_dropout = False
            self.state = ParkingState.REVERSE_ALIGN
            return self._reverse_align(observation)

        return self._drive(
            self.config.forward_speed,
            self._paper_to_actuator(
                -self.config.paper_max_steering
            ),
            (
                "figure7_t_entry_left "
                "nearSeen=%s nearNow=%s "
                "nextEq5=%+.2f confirm=%d/%d"
            )
            % (
                self._prealign_near_seen,
                observation.prealign_near,
                prospective_steering,
                self._prealign_ready_scans,
                self.config.prealign_clear_confirm_scans,
            ),
        )

    def _reverse_align(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        if observation.near:
            if (
                self._reverse_motion_started
                and observation.dist_c_mm is not None
                and observation.dist_d_mm is not None
            ):
                self.state = ParkingState.CENTER_CHECK
                self._reset_center_observation()
                return self._stop(
                    "figure9_near_with_both_cd_stop_and_check"
                )
            return self._start_lidar_realign(
                observation,
                "near_without_both_cd_not_inside_gap",
            )

        pair, pair_held = self._control_pair(observation)
        if not pair.valid:
            if (
                self._reverse_motion_started
                and observation.dist_c_mm is not None
                and observation.dist_d_mm is not None
            ):
                self.state = ParkingState.CENTER_CHECK
                self._reset_center_observation()
                return self._stop(
                    "ab_left_fov_both_cd_visible_center_check"
                )

            return self._start_lidar_realign(
                observation,
                "ab_dropout_not_inside_gap",
            )

        paper_steering, angle_term, distance_term = (
            self._paper_steering(pair)
        )
        entry_steering, entry_center_angle = (
            self._entry_midpoint_steering(pair)
        )
        bias_ab = pair.dist_a_mm - pair.dist_b_mm
        self.debug = PaperParkingDebug(
            state=self.state,
            detected_vehicle_count=self._detected_vehicle_count,
            paper_steering=paper_steering,
            applied_paper_steering=entry_steering,
            entry_center_angle_deg=entry_center_angle,
            angle_term=angle_term,
            distance_term=distance_term,
            distance_bias_ab_mm=bias_ab,
            prealign_near_seen=self._prealign_near_seen,
            prealign_ready_scans=self._prealign_ready_scans,
            pair_hold_scans=self._pair_missing_scans,
            realigning_after_dropout=(
                self._realigning_after_dropout
            ),
            reason="gap_midpoint_reverse",
        )
        self._reverse_motion_started = True
        return ControlCommand(
            speed=self.config.reverse_speed,
            steering=self._apply_steering_offset(
                self._paper_to_actuator(entry_steering)
            ),
            reason=(
                "gap_midpoint_reverse center=%+.1f "
                "entrySteer=%+.2f eq5=%+.2f pair=%s"
            )
            % (
                entry_center_angle,
                entry_steering,
                paper_steering,
                "held" if pair_held else "filtered",
            ),
        )

    def _start_lidar_realign(
        self,
        observation: RearLidarObservation,
        reason: str,
    ) -> ControlCommand:
        self.state = ParkingState.PREALIGN_LEFT
        self._reverse_motion_started = False
        self._realigning_after_dropout = True
        self._prealign_near_seen = observation.prealign_near
        self._prealign_ready_scans = 0
        self._reset_pair_tracking(observation)
        return self._drive(
            self.config.forward_speed,
            self._paper_to_actuator(
                -self.config.paper_max_steering
            ),
            "%s_left_forward C=%s D=%s"
            % (
                reason,
                observation.dist_c_mm is not None,
                observation.dist_d_mm is not None,
            ),
        )

    def _center_check(
        self,
        observation: RearLidarObservation,
        now: float,
    ) -> ControlCommand:
        if self._is_new_scan:
            self._center_scan_count += 1
            if observation.dist_c_mm is not None:
                self._center_c_samples.append(
                    observation.dist_c_mm
                )
            if observation.dist_d_mm is not None:
                self._center_d_samples.append(
                    observation.dist_d_mm
                )

        if (
            self._center_scan_count
            < self.config.center_observation_scans
        ):
            return self._stop(
                "paper_collecting_cd scans=%d/%d C=%d D=%d"
                % (
                    self._center_scan_count,
                    self.config.center_observation_scans,
                    len(self._center_c_samples),
                    len(self._center_d_samples),
                )
            )

        required = max(
            2,
            (self.config.center_observation_scans + 1) // 2,
        )
        dist_c = (
            float(median(self._center_c_samples))
            if len(self._center_c_samples) >= required
            else None
        )
        dist_d = (
            float(median(self._center_d_samples))
            if len(self._center_d_samples) >= required
            else None
        )

        if dist_c is None or dist_d is None:
            missing = (
                "CD"
                if dist_c is None and dist_d is None
                else ("C" if dist_c is None else "D")
            )
            return self._start_recovery(
                now,
                None,
                "paper_cd_not_observed_enough missing=%s" % missing,
            )

        bias_cd = dist_c - dist_d
        if abs(bias_cd) < self.config.dist_bias_cd_threshold_mm:
            self.state = ParkingState.REVERSE_STRAIGHT
            self.debug = PaperParkingDebug(
                state=self.state,
                detected_vehicle_count=self._detected_vehicle_count,
                distance_bias_cd_mm=bias_cd,
                prealign_near_seen=self._prealign_near_seen,
                prealign_ready_scans=self._prealign_ready_scans,
                pair_hold_scans=self._pair_missing_scans,
                center_observation_scans=self._center_scan_count,
                reason="paper_dist_cd_centered",
            )
            self._finish_missing_scans = 0
            return self._drive(
                self.config.reverse_speed,
                0,
                "paper_centered_reverse_straight",
                keep_debug=True,
            )

        return self._start_recovery(
            now,
            bias_cd,
            "paper_abs_dist_bias_cd_over_threshold",
        )

    def _reverse_straight(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        side_missing = (
            observation.dist_c_mm is None
            or observation.dist_d_mm is None
        )
        if self._is_new_scan:
            if side_missing:
                self._finish_missing_scans += 1
            else:
                self._finish_missing_scans = 0
        if (
            self._finish_missing_scans
            >= self.config.finish_missing_confirm_scans
        ):
            self.state = ParkingState.PARKED
            return self._stop(
                "paper_dist_c_or_dist_d_none_finish confirmed=%d"
                % self._finish_missing_scans
            )
        return self._drive(
            self.config.reverse_speed,
            0,
            (
                "paper_centered_reverse_until_side_none "
                "missing=%d/%d"
            )
            % (
                self._finish_missing_scans,
                self.config.finish_missing_confirm_scans,
            ),
        )

    def _start_recovery(
        self,
        now: float,
        bias_cd: Optional[float],
        reason: str,
    ) -> ControlCommand:
        self.state = ParkingState.RECOVERY_FORWARD
        self._reverse_motion_started = False
        self._recovery_started_at = now
        self.debug = PaperParkingDebug(
            state=self.state,
            detected_vehicle_count=self._detected_vehicle_count,
            distance_bias_cd_mm=bias_cd,
            prealign_near_seen=self._prealign_near_seen,
            prealign_ready_scans=self._prealign_ready_scans,
            pair_hold_scans=self._pair_missing_scans,
            center_observation_scans=self._center_scan_count,
            reason=reason,
        )
        return self._drive(
            self.config.forward_speed,
            0,
            "paper_recovery_forward_3_seconds",
            keep_debug=True,
        )

    def _recovery_forward(
        self,
        observation: RearLidarObservation,
        now: float,
    ) -> ControlCommand:
        elapsed = now - self._recovery_started_at
        if elapsed < self.config.recovery_forward_s:
            return self._drive(
                self.config.forward_speed,
                0,
                "paper_recovery_forward:%.2f/%.2fs"
                % (elapsed, self.config.recovery_forward_s),
            )
        self.state = ParkingState.REVERSE_ALIGN
        self._reset_pair_tracking(observation)
        return self._reverse_align(observation)

    def _reset_measurement_state(self) -> None:
        self._last_lidar_timestamp = None
        self._is_new_scan = False
        self._prealign_near_seen = False
        self._prealign_ready_scans = 0
        self._pair_samples.clear()
        self.control_pair = TangentPair()
        self._last_pair_timestamp: Optional[float] = None
        self._pair_missing_scans = 0
        self._pair_outlier_rejected = False
        self._realigning_after_dropout = False
        self._reset_center_observation()
        self._finish_missing_scans = 0

    def _reset_pair_tracking(
        self,
        observation: RearLidarObservation,
    ) -> None:
        self._pair_samples.clear()
        self._pair_missing_scans = 0
        self._pair_outlier_rejected = False
        self._last_pair_timestamp = observation.timestamp
        if observation.pair.valid:
            self._pair_samples.append(observation.pair)
            self.control_pair = observation.pair
        else:
            self.control_pair = TangentPair()

    def _control_pair(
        self,
        observation: RearLidarObservation,
    ) -> Tuple[TangentPair, bool]:
        if observation.timestamp != self._last_pair_timestamp:
            self._last_pair_timestamp = observation.timestamp
            self._pair_outlier_rejected = False
            if (
                observation.pair.valid
                and self._pair_is_continuous(observation.pair)
            ):
                self._pair_samples.append(observation.pair)
                self._pair_missing_scans = 0
            else:
                self._pair_outlier_rejected = (
                    observation.pair.valid
                )
                self._pair_missing_scans += 1

        if (
            not self._pair_samples
            or self._pair_missing_scans
            > self.config.pair_hold_scans
        ):
            self.control_pair = TangentPair(
                reason="filtered_pair_unavailable"
            )
            return self.control_pair, False

        samples = tuple(self._pair_samples)
        filtered = TangentPair(
            valid=True,
            angle_a_deg=float(
                median(item.angle_a_deg for item in samples)
            ),
            angle_b_deg=float(
                median(item.angle_b_deg for item in samples)
            ),
            dist_a_mm=float(
                median(item.dist_a_mm for item in samples)
            ),
            dist_b_mm=float(
                median(item.dist_b_mm for item in samples)
            ),
            angle_bisector_deg=float(
                median(
                    item.angle_bisector_deg for item in samples
                )
            ),
            reason=(
                "rejected_outlier_held_pair"
                if self._pair_outlier_rejected
                else (
                    "held_filtered_pair"
                    if self._pair_missing_scans
                    else "median_filtered_pair"
                )
            ),
        )
        self.control_pair = filtered
        return filtered, self._pair_missing_scans > 0

    def _pair_is_continuous(
        self,
        candidate: TangentPair,
    ) -> bool:
        if not self._pair_samples:
            return True

        previous = self._pair_samples[-1]
        angle_limit = self.config.pair_max_angle_jump_deg
        distance_limit = (
            self.config.pair_max_distance_jump_mm
        )
        return (
            abs(candidate.angle_a_deg - previous.angle_a_deg)
            <= angle_limit
            and abs(
                candidate.angle_b_deg - previous.angle_b_deg
            )
            <= angle_limit
            and abs(candidate.dist_a_mm - previous.dist_a_mm)
            <= distance_limit
            and abs(candidate.dist_b_mm - previous.dist_b_mm)
            <= distance_limit
        )

    def _reset_center_observation(self) -> None:
        self._center_scan_count = 0
        self._center_c_samples.clear()
        self._center_d_samples.clear()

    def _paper_steering(
        self,
        pair: TangentPair,
    ) -> Tuple[float, float, float]:
        distance_bias_ab = pair.dist_a_mm - pair.dist_b_mm
        distance_term = self._f(distance_bias_ab)
        angle_term = self._g(-pair.angle_bisector_deg)
        steering = self._h(angle_term + distance_term)
        return steering, angle_term, distance_term

    def _entry_midpoint_steering(
        self,
        pair: TangentPair,
    ) -> Tuple[float, float]:
        angle_a = radians(pair.angle_a_deg)
        angle_b = radians(pair.angle_b_deg)
        center_x = (
            sin(angle_a) * pair.dist_a_mm
            + sin(angle_b) * pair.dist_b_mm
        ) / 2.0
        center_y = (
            cos(angle_a) * pair.dist_a_mm
            + cos(angle_b) * pair.dist_b_mm
        ) / 2.0
        center_angle = degrees(atan2(center_x, center_y))
        steering = self._h(
            self.config.paper_max_steering
            * center_angle
            / self.config.entry_full_steer_angle_deg
        )
        return steering, center_angle

    def _f(self, value_mm: float) -> float:
        maximum = self.config.paper_max_steering
        scale = self.config.distance_bias_scale_mm
        value = maximum * value_mm * value_mm / (scale * scale)
        return value if value_mm >= 0.0 else -value

    def _g(self, value_deg: float) -> float:
        maximum = self.config.paper_max_steering
        if value_deg > maximum:
            return maximum
        if value_deg < -maximum:
            return -maximum
        return maximum * value_deg / 20.0

    def _h(self, value: float) -> float:
        maximum = self.config.paper_max_steering
        return max(-maximum, min(maximum, value))

    def _paper_to_actuator(self, paper_steering: float) -> int:
        normalized = (
            paper_steering / self.config.paper_max_steering
        )
        return int(
            round(normalized * self.config.actuator_max_steering)
        )

    def _apply_steering_offset(self, steering: int) -> int:
        maximum = self.config.actuator_max_steering
        corrected = (
            int(steering)
            + self.config.actuator_steering_offset
        )
        return max(-maximum, min(maximum, corrected))

    def _forward(self, reason: str) -> ControlCommand:
        return self._drive(self.config.forward_speed, 0, reason)

    def _drive(
        self,
        speed: int,
        steering: int,
        reason: str,
        *,
        keep_debug: bool = False,
    ) -> ControlCommand:
        if not keep_debug:
            self._set_debug(reason)
        return ControlCommand(
            speed=int(speed),
            steering=self._apply_steering_offset(steering),
            reason=reason,
        )

    def _stop(self, reason: str) -> ControlCommand:
        self._set_debug(reason)
        return ControlCommand.stop(reason)

    def _set_debug(self, reason: str) -> None:
        self.debug = PaperParkingDebug(
            state=self.state,
            detected_vehicle_count=self._detected_vehicle_count,
            prealign_near_seen=self._prealign_near_seen,
            prealign_ready_scans=self._prealign_ready_scans,
            pair_hold_scans=self._pair_missing_scans,
            realigning_after_dropout=(
                self._realigning_after_dropout
            ),
            center_observation_scans=self._center_scan_count,
            reason=reason,
        )
