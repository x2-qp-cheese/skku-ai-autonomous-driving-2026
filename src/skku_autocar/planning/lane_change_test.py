from __future__ import annotations

from dataclasses import dataclass, replace

from ..estimation.lane_geometry import LaneGeometry
from ..types import ControlCommand


@dataclass(frozen=True)
class LaneChangeTestConfig:
    mode: str = "off"  # off, manual, timed, external
    trigger_seconds: float = 8.0
    transition_seconds: float = 2.0
    hold_seconds: float = 3.0
    max_straight_heading: float = 0.08
    speed_cap: int = 70


@dataclass(frozen=True)
class LaneChangeTestResult:
    lane: LaneGeometry
    state: str
    offset_px: float = 0.0
    active: bool = False


class LaneChangeTestController:
    """Reusable 2 -> 1 -> 2 lane-change trajectory controller.

    The normal BEV corridor produces the center of lane 2, on the right side of
    the center line. Lane 1's center is one physical BEV lane width to the left.
    This controller moves that target laterally with a smoothstep profile. The
    current keyboard/timer are test adapters; a future obstacle detector calls
    request("obstacle") without changing the trajectory or safety logic.
    """

    def __init__(self, config: LaneChangeTestConfig = LaneChangeTestConfig()):
        self.config = config
        self.state = "lane2"
        self._run_started_at = None
        self._phase_started_at = None
        self._request_source = "none"
        self._return_requested = False
        self._return_source = "none"

    def reset(self) -> None:
        self.state = "lane2"
        self._run_started_at = None
        self._phase_started_at = None
        self._request_source = "none"
        self._return_requested = False
        self._return_source = "none"

    def request(self, source: str = "external") -> bool:
        """Arm a lane change from any detector/trigger.

        The current keyboard and timer adapters call this method. A future
        obstacle detector should call ``request("obstacle")`` when its decision is
        confirmed; the trajectory and safety gates remain unchanged.
        """
        if self.config.mode == "off" or self.state not in ("lane2", "completed"):
            return False
        self.state = "armed"
        self._phase_started_at = None
        self._request_source = source
        self._return_requested = False
        self._return_source = "none"
        return True

    @property
    def request_source(self) -> str:
        return self._request_source

    def request_return(self, source: str = "external") -> bool:
        """Request lane 1 -> lane 2 after an obstacle-clear decision."""
        if self.config.mode == "off" or self.state not in ("changing_to_lane1", "lane1"):
            return False
        self._return_requested = True
        self._return_source = source
        return True

    @property
    def return_source(self) -> str:
        return self._return_source

    def update(
        self,
        lane: LaneGeometry,
        lane_width_px: float,
        bev_width_px: float,
        now: float,
        running: bool,
    ) -> LaneChangeTestResult:
        if self.config.mode == "off":
            return LaneChangeTestResult(lane=lane, state="off")
        if not running:
            self.reset()
            return LaneChangeTestResult(lane=lane, state=self.state)

        if self._run_started_at is None:
            self._run_started_at = now

        if (
            self.config.mode == "timed"
            and self.state == "lane2"
            and now - self._run_started_at >= max(0.0, self.config.trigger_seconds)
        ):
            self.request("timer")

        straight = lane.found and abs(lane.heading_error) <= max(0.0, self.config.max_straight_heading)
        if self.state == "armed" and straight:
            self.state = "changing_to_lane1"
            self._phase_started_at = now

        offset_ratio = 0.0
        if self.state == "changing_to_lane1":
            progress = self._progress(now)
            offset_ratio = -self._smoothstep(progress)
            if progress >= 1.0:
                self.state = "lane1"
                self._phase_started_at = now
                offset_ratio = -1.0
        elif self.state == "lane1":
            offset_ratio = -1.0
            timed_return = (
                self.config.mode != "external"
                and self._phase_started_at is not None
                and now - self._phase_started_at >= max(0.0, self.config.hold_seconds)
            )
            if straight and (timed_return or self._return_requested):
                self.state = "changing_to_lane2"
                self._phase_started_at = now
        elif self.state == "changing_to_lane2":
            progress = self._progress(now)
            offset_ratio = -(1.0 - self._smoothstep(progress))
            if progress >= 1.0:
                self.state = "completed"
                self._phase_started_at = None
                offset_ratio = 0.0

        offset_px = offset_ratio * max(0.0, lane_width_px)
        shifted = self._shift_lane(lane, offset_px, bev_width_px) if lane.found else lane
        applied_offset_px = shifted.center_x - lane.center_x if lane.found else 0.0
        active = self.state in ("changing_to_lane1", "lane1", "changing_to_lane2")
        return LaneChangeTestResult(
            lane=shifted,
            state=self.state,
            offset_px=applied_offset_px,
            active=active,
        )

    def apply_speed_cap(self, command: ControlCommand, active: bool) -> ControlCommand:
        if not active or command.brake:
            return command
        cap = max(0, int(self.config.speed_cap))
        speed = max(-cap, min(cap, command.speed))
        reason = "%s:lane_change_test" % command.reason if command.reason else "lane_change_test"
        return ControlCommand(speed=speed, steering=command.steering, brake=False, reason=reason)

    def _progress(self, now: float) -> float:
        duration = max(0.05, self.config.transition_seconds)
        if self._phase_started_at is None:
            return 0.0
        return min(1.0, max(0.0, (now - self._phase_started_at) / duration))

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _shift_lane(lane: LaneGeometry, offset_px: float, bev_width_px: float) -> LaneGeometry:
        half_width = max(1.0, bev_width_px / 2.0)
        center_x = max(0.0, min(max(0.0, bev_width_px - 1.0), lane.center_x + offset_px))
        lateral_error_px = center_x - lane.vehicle_center_x
        lateral_error_norm = max(-1.0, min(1.0, lateral_error_px / half_width))
        return replace(
            lane,
            center_x=center_x,
            lateral_error_px=lateral_error_px,
            lateral_error_norm=lateral_error_norm,
            reason="%s:lane_change_test" % lane.reason,
        )
