from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..estimation.lane_geometry import LaneGeometry
from ..types import ControlCommand


@dataclass(frozen=True)
class YoloLaneFollowerConfig:
    base_speed: int = 105
    max_speed: int = 170
    min_curve_speed: int = 85
    max_steering: int = 120
    kp_lateral: float = 95.0
    kd_lateral: float = 28.0
    kp_heading: float = 35.0
    kd_heading: float = 10.0
    min_confidence: float = 0.15
    steering_rate_limit: int = 24
    speed_curve_slowdown: int = 35


class YoloLaneFollower:
    def __init__(self, config: YoloLaneFollowerConfig = YoloLaneFollowerConfig()):
        self.config = config
        self._last_steering = 0
        self._last_lateral_error: Optional[float] = None
        self._last_heading_error: Optional[float] = None

    def plan(self, lane: LaneGeometry) -> ControlCommand:
        if not lane.found or lane.confidence < self.config.min_confidence:
            self.reset()
            return ControlCommand.stop("lane_lost:%s" % lane.reason)

        lateral_derivative = self._derivative(lane.lateral_error_norm, self._last_lateral_error)
        heading_derivative = self._derivative(lane.heading_error, self._last_heading_error)
        raw_steering = (
            self.config.kp_lateral * lane.lateral_error_norm
            + self.config.kd_lateral * lateral_derivative
            + self.config.kp_heading * lane.heading_error
            + self.config.kd_heading * heading_derivative
        )
        steering = self._rate_limit(int(round(raw_steering)))
        steering = self._clip(steering, -self.config.max_steering, self.config.max_steering)

        curve = min(1.0, abs(lane.lateral_error_norm) + 0.65 * abs(lane.heading_error))
        speed = int(round(self.config.base_speed - self.config.speed_curve_slowdown * curve))
        speed = self._clip(speed, self.config.min_curve_speed, self.config.max_speed)

        self._last_steering = steering
        self._last_lateral_error = lane.lateral_error_norm
        self._last_heading_error = lane.heading_error
        return ControlCommand(speed=speed, steering=steering, brake=False, reason="yolo_lane_follow")

    def reset(self) -> None:
        self._last_steering = 0
        self._last_lateral_error = None
        self._last_heading_error = None

    @staticmethod
    def _derivative(value: float, previous: Optional[float]) -> float:
        if previous is None:
            return 0.0
        return value - previous

    def _rate_limit(self, steering: int) -> int:
        delta = steering - self._last_steering
        limit = self.config.steering_rate_limit
        if delta > limit:
            return self._last_steering + limit
        if delta < -limit:
            return self._last_steering - limit
        return steering

    @staticmethod
    def _clip(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))
