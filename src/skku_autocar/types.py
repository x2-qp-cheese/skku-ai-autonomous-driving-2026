from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParkingState(str, Enum):
    IDLE = "idle"
    SEARCH_FIRST_CAR = "search_first_car"
    SEARCH_OPEN_SPACE = "search_open_space"
    SEARCH_SECOND_CAR = "search_second_car"
    PREALIGN_LEFT = "prealign_left"
    REVERSE_ALIGN = "reverse_align"
    CENTER_CHECK = "center_check"
    PARALLEL_FORWARD = "parallel_forward"
    REVERSE_STRAIGHT = "reverse_straight"
    RECOVERY_FORWARD = "recovery_forward"
    PARKED = "parked"


@dataclass(frozen=True)
class ControlCommand:
    speed: int = 0
    steering: int = 0
    brake: bool = False
    reason: str = ""

    @classmethod
    def stop(cls, reason: str) -> "ControlCommand":
        return cls(brake=True, reason=reason)

    def clipped(
        self,
        max_speed: int = 255,
        max_steering: int = 150,
    ) -> "ControlCommand":
        return ControlCommand(
            speed=max(-max_speed, min(max_speed, int(self.speed))),
            steering=max(
                -max_steering,
                min(max_steering, int(self.steering)),
            ),
            brake=self.brake,
            reason=self.reason,
        )
