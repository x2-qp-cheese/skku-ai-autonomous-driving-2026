from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..config import PaperControllerConfig
from ..perception.rear_lidar import RearLidarObservation, TangentPair
from ..types import ControlCommand, ParkingState


@dataclass(frozen=True)
class PaperParkingDebug:
    state: ParkingState = ParkingState.IDLE
    detected_vehicle_count: int = 0
    paper_steering: Optional[float] = None
    angle_term: Optional[float] = None
    distance_term: Optional[float] = None
    distance_bias_ab_mm: Optional[float] = None
    distance_bias_cd_mm: Optional[float] = None
    reason: str = "idle"


class PaperParkingController:
    """Literal implementation of Hong et al., Figure 9."""

    def __init__(self, config: PaperControllerConfig):
        self.config = config
        self.state = ParkingState.IDLE
        self.debug = PaperParkingDebug()
        self._detected_vehicle_count = 0
        self._recovery_started_at = 0.0

    def start(self) -> None:
        self._detected_vehicle_count = 0
        self._recovery_started_at = 0.0
        self.state = ParkingState.SEARCH_FIRST_CAR
        self._set_debug("paper_search_first_vehicle")

    def reset(self) -> None:
        self._detected_vehicle_count = 0
        self._recovery_started_at = 0.0
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
        if observation.near:
            self.state = ParkingState.PREALIGN_LEFT
            return self._drive(
                self.config.forward_speed,
                self._paper_to_actuator(
                    -self.config.paper_max_steering
                ),
                "figure9_is_near_left_forward",
            )

        self.state = ParkingState.REVERSE_ALIGN
        return self._reverse_align(observation)

    def _prealign_left(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        if observation.near:
            return self._drive(
                self.config.forward_speed,
                self._paper_to_actuator(
                    -self.config.paper_max_steering
                ),
                "figure9_left_forward_while_is_near",
            )
        self.state = ParkingState.REVERSE_ALIGN
        return self._reverse_align(observation)

    def _reverse_align(
        self,
        observation: RearLidarObservation,
    ) -> ControlCommand:
        if observation.near:
            self.state = ParkingState.CENTER_CHECK
            return self._stop("figure9_near_stop_and_check_dist_cd")

        pair = observation.pair
        if not pair.valid:
            return self._stop("paper_cannot_calculate_steering")

        paper_steering, angle_term, distance_term = (
            self._paper_steering(pair)
        )
        bias_ab = pair.dist_a_mm - pair.dist_b_mm
        self.debug = PaperParkingDebug(
            state=self.state,
            detected_vehicle_count=self._detected_vehicle_count,
            paper_steering=paper_steering,
            angle_term=angle_term,
            distance_term=distance_term,
            distance_bias_ab_mm=bias_ab,
            reason="equation_5_reverse",
        )
        return ControlCommand(
            speed=self.config.reverse_speed,
            steering=self._apply_steering_offset(
                self._paper_to_actuator(paper_steering)
            ),
            reason=(
                "eq5_reverse bisector=%+.1f biasAB=%+.0f "
                "paperSteer=%+.2f"
            )
            % (
                pair.angle_bisector_deg,
                bias_ab,
                paper_steering,
            ),
        )

    def _center_check(
        self,
        observation: RearLidarObservation,
        now: float,
    ) -> ControlCommand:
        dist_c = observation.dist_c_mm
        dist_d = observation.dist_d_mm

        if dist_c is None and dist_d is None:
            return self._start_recovery(
                now,
                None,
                "paper_both_dist_c_dist_d_none",
            )
        if dist_c is None or dist_d is None:
            # The paper defines neither a CD bias nor a transition for this
            # case at the center-check stage.
            return self._stop("paper_undefined_exactly_one_cd_none")

        bias_cd = dist_c - dist_d
        if abs(bias_cd) < self.config.dist_bias_cd_threshold_mm:
            self.state = ParkingState.REVERSE_STRAIGHT
            self.debug = PaperParkingDebug(
                state=self.state,
                detected_vehicle_count=self._detected_vehicle_count,
                distance_bias_cd_mm=bias_cd,
                reason="paper_dist_cd_centered",
            )
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
        if (
            observation.dist_c_mm is None
            or observation.dist_d_mm is None
        ):
            self.state = ParkingState.PARKED
            return self._stop("paper_dist_c_or_dist_d_none_finish")
        return self._drive(
            self.config.reverse_speed,
            0,
            "paper_centered_reverse_until_side_none",
        )

    def _start_recovery(
        self,
        now: float,
        bias_cd: Optional[float],
        reason: str,
    ) -> ControlCommand:
        self.state = ParkingState.RECOVERY_FORWARD
        self._recovery_started_at = now
        self.debug = PaperParkingDebug(
            state=self.state,
            detected_vehicle_count=self._detected_vehicle_count,
            distance_bias_cd_mm=bias_cd,
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
        return self._reverse_align(observation)

    def _paper_steering(
        self,
        pair: TangentPair,
    ) -> Tuple[float, float, float]:
        distance_bias_ab = pair.dist_a_mm - pair.dist_b_mm
        distance_term = self._f(distance_bias_ab)
        angle_term = self._g(-pair.angle_bisector_deg)
        steering = self._h(angle_term + distance_term)
        return steering, angle_term, distance_term

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
            reason=reason,
        )
