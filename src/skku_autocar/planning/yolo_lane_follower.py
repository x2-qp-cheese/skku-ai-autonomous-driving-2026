from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..estimation.lane_geometry import LaneGeometry
from ..types import ControlCommand


@dataclass(frozen=True)
class YoloLaneFollowerConfig:
    base_speed: int = 105
    max_speed: int = 170
    min_curve_speed: int = 60
    max_steering: int = 120
    kp_lateral: float = 190.0
    kd_lateral: float = 45.0
    kp_heading: float = 12.0
    kd_heading: float = 4.0
    min_confidence: float = 0.15
    steering_rate_limit: int = 110
    min_steering_rate_limit: int = 40
    steering_release_rate_limit: int = 22
    speed_curve_slowdown: int = 70
    lateral_priority_threshold: float = 0.10
    curve_strength_alpha: float = 0.35
    straight_steering_scale: float = 0.45
    curve_steering_scale: float = 1.45
    center_recovery_error_threshold: float = 0.14
    center_recovery_steering_boost: float = 2.0
    center_recovery_min_steering: int = 85
    center_recovery_rate_limit: int = 120
    center_recovery_max_speed: int = 50


class YoloLaneFollower:
    def __init__(self, config: YoloLaneFollowerConfig = YoloLaneFollowerConfig()):
        self.config = config
        self._last_steering = 0
        self._last_lateral_error: Optional[float] = None
        self._last_heading_error: Optional[float] = None
        self._curve_strength = 0.0

    def plan(self, lane: LaneGeometry) -> ControlCommand:
        if not lane.found or lane.confidence < self.config.min_confidence:
            self.reset()
            return ControlCommand.stop("lane_lost:%s" % lane.reason)

        raw_curve_strength = self._curve_strength_from(lane)
        curve_strength = self._smooth_curve_strength(raw_curve_strength)
        recovery_strength = self._center_recovery_strength(lane.lateral_error_norm)

        lateral_derivative = self._derivative(lane.lateral_error_norm, self._last_lateral_error)
        heading_error = self._effective_heading_error(lane.lateral_error_norm, lane.heading_error)
        heading_derivative = self._derivative(heading_error, self._last_heading_error)
        steering_scale = self._steering_scale(curve_strength)
        raw_steering = (
            self.config.kp_lateral * lane.lateral_error_norm
            + self.config.kd_lateral * lateral_derivative
            + self.config.kp_heading * heading_error
            + self.config.kd_heading * heading_derivative
        ) * steering_scale
        raw_steering = self._apply_center_recovery(raw_steering, lane.lateral_error_norm, recovery_strength)
        if self._opposes_lateral(lane.lateral_error_norm, self._last_steering):
            self._last_steering = 0
        steering = self._rate_limit(int(round(raw_steering)), curve_strength, recovery_strength)
        steering = self._clip(steering, -self.config.max_steering, self.config.max_steering)

        speed = int(round(self.config.base_speed - self.config.speed_curve_slowdown * raw_curve_strength))
        speed = self._clip(speed, self.config.min_curve_speed, self.config.max_speed)
        speed = self._apply_center_recovery_speed(speed, recovery_strength)
        speed = self._clip(speed, 0, self.config.max_speed)

        self._last_steering = steering
        self._last_lateral_error = lane.lateral_error_norm
        self._last_heading_error = heading_error
        return ControlCommand(speed=speed, steering=steering, brake=False, reason="yolo_lane_follow")

    def reset(self) -> None:
        self._last_steering = 0
        self._last_lateral_error = None
        self._last_heading_error = None
        self._curve_strength = 0.0

    @staticmethod
    def _derivative(value: float, previous: Optional[float]) -> float:
        if previous is None:
            return 0.0
        return value - previous

    def _effective_heading_error(self, lateral_error: float, heading_error: float) -> float:
        if self._opposes_lateral(lateral_error, heading_error):
            return 0.0
        return heading_error

    def _opposes_lateral(self, lateral_error: float, value: float) -> bool:
        if abs(lateral_error) < self.config.lateral_priority_threshold:
            return False
        return lateral_error * value < 0.0

    def _curve_strength_from(self, lane: LaneGeometry) -> float:
        lateral_curve = min(1.0, abs(lane.lateral_error_norm) / 0.65)
        heading_curve = min(1.0, abs(lane.heading_error) / 0.85)
        return self._clip_float(max(lateral_curve, heading_curve), 0.0, 1.0)

    def _smooth_curve_strength(self, value: float) -> float:
        alpha = self.config.curve_strength_alpha
        self._curve_strength = alpha * value + (1.0 - alpha) * self._curve_strength
        return self._curve_strength

    def _steering_scale(self, curve_strength: float) -> float:
        low = self.config.straight_steering_scale
        high = self.config.curve_steering_scale
        return low + (high - low) * curve_strength

    def _center_recovery_strength(self, lateral_error: float) -> float:
        error = abs(lateral_error)
        threshold = self.config.center_recovery_error_threshold
        if error <= threshold:
            return 0.0
        full_error = 0.65
        return self._clip_float((error - threshold) / max(1e-6, full_error - threshold), 0.0, 1.0)

    def _apply_center_recovery(self, steering: float, lateral_error: float, recovery_strength: float) -> float:
        if recovery_strength <= 0.0:
            return steering
        boost = 1.0 + (self.config.center_recovery_steering_boost - 1.0) * recovery_strength
        boosted = steering * boost
        minimum = self.config.center_recovery_min_steering * recovery_strength
        if abs(boosted) < minimum:
            direction = 1.0 if lateral_error >= 0.0 else -1.0
            boosted = direction * minimum
        return boosted

    def _apply_center_recovery_speed(self, speed: int, recovery_strength: float) -> int:
        if recovery_strength <= 0.0:
            return speed
        cap = int(round(
            self.config.base_speed
            - (self.config.base_speed - self.config.center_recovery_max_speed) * recovery_strength
        ))
        return min(speed, cap)

    def _rate_limit(self, steering: int, curve_strength: float, recovery_strength: float) -> int:
        delta = steering - self._last_steering
        min_limit = min(self.config.min_steering_rate_limit, self.config.steering_rate_limit)
        max_limit = max(self.config.min_steering_rate_limit, self.config.steering_rate_limit)
        limit = int(round(min_limit + (max_limit - min_limit) * curve_strength))
        if recovery_strength > 0.0:
            recovery_limit = int(round(
                limit + (self.config.center_recovery_rate_limit - limit) * recovery_strength
            ))
            limit = max(limit, recovery_limit)
        if self._is_releasing_steering(steering):
            limit = min(limit, self.config.steering_release_rate_limit)
        if delta > limit:
            return self._last_steering + limit
        if delta < -limit:
            return self._last_steering - limit
        return steering

    def _is_releasing_steering(self, steering: int) -> bool:
        if self._last_steering == 0:
            return False
        same_direction_or_zero = steering == 0 or steering * self._last_steering > 0
        return same_direction_or_zero and abs(steering) < abs(self._last_steering)

    @staticmethod
    def _clip(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

    @staticmethod
    def _clip_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
