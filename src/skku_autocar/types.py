from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MissionMode(str, Enum):
    TIME_TRIAL = "time_trial"
    OBSTACLE_TRAFFIC_LIGHT = "obstacle_traffic_light"
    PARKING = "parking"


@dataclass(frozen=True)
class LaneEstimate:
    center_offset_norm: float = 0.0
    heading_error: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class ObstacleEstimate:
    front_mm: Optional[float] = None
    left_mm: Optional[float] = None
    right_mm: Optional[float] = None


@dataclass(frozen=True)
class TrafficLightEstimate:
    state: str = "unknown"
    confidence: float = 0.0


@dataclass(frozen=True)
class ControlCommand:
    speed: int = 0
    steering: int = 0
    brake: bool = False
    reason: str = ""

    @classmethod
    def stop(cls, reason: str = "stop") -> "ControlCommand":
        return cls(speed=0, steering=0, brake=True, reason=reason)

    def clipped(self, max_speed: int = 255, max_steering: int = 255) -> "ControlCommand":
        speed = _clip(self.speed, -max_speed, max_speed)
        steering = _clip(self.steering, -max_steering, max_steering)
        return ControlCommand(speed=speed, steering=steering, brake=self.brake, reason=self.reason)


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
