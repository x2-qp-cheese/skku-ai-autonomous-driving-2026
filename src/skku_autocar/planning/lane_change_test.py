from __future__ import annotations

from dataclasses import dataclass, replace

from ..estimation.lane_geometry import LaneGeometry
from ..types import ControlCommand


@dataclass(frozen=True)
class LaneChangeTestConfig:
    mode: str = "off"  # off, keyboard, manual, timed, external
    trigger_seconds: float = 8.0
    transition_seconds: float = 1.2
    hold_seconds: float = 3.0
    max_straight_heading: float = 0.08
    speed_cap: int = 85
    steering_min: int = 100
    steering_boost: int = 25
    steering_cap: int = 120
    settle_seconds: float = 0.0
    steering_override: bool = False
    stable_lateral_error: float = 0.12
    stable_heading_error: float = 0.18
    stable_required_frames: int = 5
    stabilize_timeout_seconds: float = 0.0
    allow_virtual_stabilize: bool = False


@dataclass(frozen=True)
class LaneChangeTestResult:
    lane: LaneGeometry
    state: str
    offset_px: float = 0.0
    active: bool = False
    direction: int = 0
    progress: float = 0.0
    stable_frames: int = 0


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
        self._settle_until = None
        self._settle_direction = 0
        self._stable_frames = 0
        self._stabilize_started_at = None

    def reset(self) -> None:
        self.state = "lane2"
        self._run_started_at = None
        self._phase_started_at = None
        self._request_source = "none"
        self._return_requested = False
        self._return_source = "none"
        self._settle_until = None
        self._settle_direction = 0
        self._stable_frames = 0
        self._stabilize_started_at = None

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
        self._clear_settle()
        self._clear_stability()
        return True

    @property
    def request_source(self) -> str:
        return self._request_source

    def request_return(self, source: str = "external") -> bool:
        """Request lane 1 -> lane 2 after an obstacle-clear decision."""
        if self.config.mode == "off" or self.state != "lane1":
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
        lane_reliable: bool = True,
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
            self._clear_settle()
            self._clear_stability()

        offset_ratio = 0.0
        direction = 0
        progress = 0.0
        if self.state == "changing_to_lane1":
            progress = self._progress(now)
            direction = -1
            offset_ratio = -self._smoothstep(progress)
            if progress >= 1.0:
                offset_ratio = -1.0
                direction = 0
                self._finish_transition_or_stabilize("stabilizing_lane1", "lane1", now)
        elif self.state == "stabilizing_lane1":
            offset_ratio = -1.0
        elif self.state == "lane1":
            offset_ratio = -1.0
            timed_return = (
                self.config.mode not in ("external", "keyboard")
                and self._phase_started_at is not None
                and now - self._phase_started_at >= max(0.0, self.config.hold_seconds)
            )
            if straight and (timed_return or self._return_requested):
                self.state = "changing_to_lane2"
                self._phase_started_at = now
                self._clear_settle()
                self._clear_stability()
        elif self.state == "changing_to_lane2":
            progress = self._progress(now)
            direction = 1
            offset_ratio = -(1.0 - self._smoothstep(progress))
            if progress >= 1.0:
                offset_ratio = 0.0
                direction = 0
                self._finish_transition_or_stabilize("stabilizing_lane2", "completed", now)
        elif self.state == "stabilizing_lane2":
            offset_ratio = 0.0

        if direction == 0:
            direction = self._settle_direction_at(now)

        offset_px = offset_ratio * max(0.0, lane_width_px)
        shifted = self._shift_lane(lane, offset_px, bev_width_px) if lane.found else lane
        if self.state == "stabilizing_lane1":
            if self._update_stability(shifted, lane_reliable, now):
                self.state = "lane1"
                self._phase_started_at = now
                self._clear_settle()
        elif self.state == "stabilizing_lane2":
            if self._update_stability(shifted, lane_reliable, now):
                self.state = "completed"
                self._phase_started_at = None
                self._clear_settle()
        applied_offset_px = shifted.center_x - lane.center_x if lane.found else 0.0
        active = self.state in (
            "changing_to_lane1",
            "stabilizing_lane1",
            "lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        )
        return LaneChangeTestResult(
            lane=shifted,
            state=self.state,
            offset_px=applied_offset_px,
            active=active,
            direction=direction,
            progress=progress,
            stable_frames=self._stable_frames,
        )

    def apply_control_adjustments(
        self,
        command: ControlCommand,
        result: LaneChangeTestResult,
    ) -> ControlCommand:
        command = self.apply_speed_cap(command, self.speed_cap_active(result))
        return self.apply_steering_assist(command, result)

    @staticmethod
    def speed_cap_active(result: LaneChangeTestResult) -> bool:
        return result.state in (
            "armed",
            "changing_to_lane1",
            "stabilizing_lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        ) or result.direction != 0

    def apply_speed_cap(self, command: ControlCommand, active: bool) -> ControlCommand:
        if not active or command.brake:
            return command
        cap = max(0, int(self.config.speed_cap))
        speed = max(-cap, min(cap, command.speed))
        reason = "%s:lane_change_test" % command.reason if command.reason else "lane_change_test"
        return ControlCommand(speed=speed, steering=command.steering, brake=False, reason=reason)

    def apply_steering_assist(
        self,
        command: ControlCommand,
        result: LaneChangeTestResult,
    ) -> ControlCommand:
        if command.brake or result.direction == 0:
            return command
        if self.config.steering_min <= 0 and self.config.steering_boost <= 0:
            return command

        direction = -1 if result.direction < 0 else 1
        minimum = max(0, int(self.config.steering_min))
        cap = max(0, int(self.config.steering_cap))
        if self.config.steering_override:
            steering = direction * (cap if cap > 0 else minimum)
        else:
            steering = command.steering
            if steering * direction < 0:
                steering = 0
            steering += direction * max(0, int(self.config.steering_boost))

        if abs(steering) < minimum:
            steering = direction * minimum

        if cap > 0:
            steering = self._clip(steering, -cap, cap)
        reason = "%s:lane_change_steer" % command.reason if command.reason else "lane_change_steer"
        return ControlCommand(speed=command.speed, steering=steering, brake=False, reason=reason)

    def _progress(self, now: float) -> float:
        duration = max(0.05, self.config.transition_seconds)
        if self._phase_started_at is None:
            return 0.0
        return min(1.0, max(0.0, (now - self._phase_started_at) / duration))

    def _begin_settle(self, now: float, direction: int) -> None:
        duration = max(0.0, float(self.config.settle_seconds))
        if duration <= 0.0:
            self._clear_settle()
            return
        self._settle_until = now + duration
        self._settle_direction = -1 if direction < 0 else 1

    def _clear_settle(self) -> None:
        self._settle_until = None
        self._settle_direction = 0

    def _settle_direction_at(self, now: float) -> int:
        if self._settle_until is None or now > self._settle_until:
            self._clear_settle()
            return 0
        return self._settle_direction

    def _finish_transition_or_stabilize(self, stabilizing_state: str, final_state: str, now: float) -> None:
        if max(0, int(self.config.stable_required_frames)) <= 0:
            self.state = final_state
            self._phase_started_at = None if final_state == "completed" else now
            self._begin_settle(now, -1 if stabilizing_state.endswith("lane1") else 1)
            return
        self.state = stabilizing_state
        self._phase_started_at = now
        self._stabilize_started_at = now
        self._stable_frames = 0
        self._clear_settle()

    def _update_stability(self, lane: LaneGeometry, lane_reliable: bool, now: float) -> bool:
        if self._stable_now(lane, lane_reliable):
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        if self._stable_frames >= max(1, int(self.config.stable_required_frames)):
            return True
        if self._stabilize_started_at is None:
            self._stabilize_started_at = now
        timeout = max(0.0, float(self.config.stabilize_timeout_seconds))
        return timeout > 0.0 and now - self._stabilize_started_at >= timeout

    def _stable_now(self, lane: LaneGeometry, lane_reliable: bool) -> bool:
        if not lane.found:
            return False
        if not self.config.allow_virtual_stabilize and not lane_reliable:
            return False
        return (
            abs(lane.lateral_error_norm) <= max(0.0, float(self.config.stable_lateral_error))
            and abs(lane.heading_error) <= max(0.0, float(self.config.stable_heading_error))
        )

    def _clear_stability(self) -> None:
        self._stable_frames = 0
        self._stabilize_started_at = None

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _clip(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

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
