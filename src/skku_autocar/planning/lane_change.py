from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..estimation.lane_geometry import LaneGeometry
from ..types import ControlCommand


@dataclass(frozen=True)
class LaneChangeConfig:
    mode: str = "off"  # off, external, timed
    trigger_seconds: float = 8.0
    transition_seconds: float = 1.2
    hold_seconds: float = 3.0
    max_straight_heading: float = 0.08
    speed_cap: int = 85
    steering_min: int = 100
    steering_boost: int = 25
    steering_cap: int = 120
    steering_override: bool = False
    unreliable_speed_cap: int = 70
    unreliable_steering_cap: int = 90
    stabilizing_steering_min: int = 70
    stable_lateral_error: float = 0.12
    stable_near_lateral_error: float = 0.18
    stable_heading_error: float = 0.18
    stable_required_frames: int = 5
    target_lane_width_px: float = 0.0
    target_approach_error: float = 0.32
    target_capture_error: float = 0.20
    target_capture_frames: int = 2
    allow_virtual_stabilize: bool = False


@dataclass(frozen=True)
class LaneChangeResult:
    lane: LaneGeometry
    state: str
    offset_px: float = 0.0
    active: bool = False
    direction: int = 0
    progress: float = 0.0
    stable_frames: int = 0
    lane_reliable: bool = True


class LaneChangeController:
    """Reusable 2 -> 1 -> 2 lane-change trajectory controller.

    The normal BEV corridor produces the lane-2 driving target. Lane changes
    shift that target by the effective adjacent-lane offset supplied by the
    runtime. Keyboard/timer changes move with a smoothstep profile. Obstacle
    avoidance selects the complete adjacent-lane target immediately, preserves
    direction-priority steering until the vehicle reaches that target, and then
    releases steering to the normal controller for parallel stabilization.
    """

    def __init__(self, config: LaneChangeConfig = LaneChangeConfig()):
        self.config = config
        self.state = "lane2"
        self._run_started_at = None
        self._phase_started_at = None
        self._request_source = "none"
        self._request_profile = "normal"
        self._return_requested = False
        self._return_source = "none"
        self._return_profile = "normal"
        self._stable_frames = 0
        self._target_capture_frames = 0
        self._locked_lane_width_px = None
        self._last_reliable_shifted_lane: Optional[LaneGeometry] = None

    def reset(self) -> None:
        self.state = "lane2"
        self._run_started_at = None
        self._phase_started_at = None
        self._request_source = "none"
        self._request_profile = "normal"
        self._return_requested = False
        self._return_source = "none"
        self._return_profile = "normal"
        self._stable_frames = 0
        self._target_capture_frames = 0
        self._locked_lane_width_px = None
        self._last_reliable_shifted_lane = None

    def request(self, source: str = "external") -> bool:
        """Arm a lane change from any detector/trigger.

        Keyboard/timer adapters and the obstacle-fusion planner call this method
        after their trigger is confirmed; trajectory and safety gates remain
        unchanged.
        """
        return self._arm_request(source, "normal")

    def request_avoidance(self, source: str = "obstacle") -> bool:
        """Arm a safety-priority change with target-arrival verification."""
        return self._arm_request(source, "avoidance")

    def _arm_request(self, source: str, profile: str) -> bool:
        if self.config.mode == "off" or self.state not in ("lane2", "completed"):
            return False
        self.state = "armed"
        self._phase_started_at = None
        self._request_source = source
        self._request_profile = profile
        self._return_requested = False
        self._return_source = "none"
        self._return_profile = "normal"
        self._clear_stability()
        self._locked_lane_width_px = None
        return True

    @property
    def request_source(self) -> str:
        return self._request_source

    def request_return(self, source: str = "external") -> bool:
        """Request lane 1 -> lane 2 after an obstacle-clear decision."""
        return self._arm_return(source, "normal")

    def request_avoidance_return(self, source: str = "obstacle") -> bool:
        """Request a safety-priority return after lane 1 is stable."""
        return self._arm_return(source, "avoidance")

    def _arm_return(self, source: str, profile: str) -> bool:
        if self.config.mode == "off" or self.state != "lane1":
            return False
        self._return_requested = True
        self._return_source = source
        self._return_profile = profile
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
    ) -> LaneChangeResult:
        if self.config.mode == "off":
            return LaneChangeResult(lane=lane, state="off")
        if not running:
            self.reset()
            return LaneChangeResult(lane=lane, state=self.state)

        if self._run_started_at is None:
            self._run_started_at = now

        if (
            self.config.mode == "timed"
            and self.state == "lane2"
            and now - self._run_started_at >= max(0.0, self.config.trigger_seconds)
        ):
            self.request("timer")

        straight = (
            lane.found
            and lane_reliable
            and abs(lane.heading_error)
            <= max(0.0, self.config.max_straight_heading)
        )
        priority_request = self._request_profile == "avoidance"
        if self.state == "armed" and (
            straight or (priority_request and lane.found and lane_reliable)
        ):
            self.state = "changing_to_lane1"
            self._phase_started_at = now
            self._lock_lane_width(lane_width_px, self._request_profile)
            self._clear_stability()

        offset_ratio = 0.0
        direction = 0
        progress = 0.0
        if self.state == "changing_to_lane1":
            direction = -1
            if self._uses_target_arrival(self.state):
                # Avoidance targets the complete adjacent-lane geometry from
                # the first control frame. Arrival is decided from measured
                # target error, not elapsed transition time.
                progress = 1.0
                offset_ratio = -1.0
            else:
                progress = self._progress(now)
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
                self.config.mode == "timed"
                and self._phase_started_at is not None
                and now - self._phase_started_at >= max(0.0, self.config.hold_seconds)
            )
            priority_return = self._return_profile == "avoidance"
            return_path_ready = straight or (
                priority_return and lane.found and lane_reliable
            )
            if return_path_ready and (timed_return or self._return_requested):
                self.state = "changing_to_lane2"
                self._phase_started_at = now
                self._lock_lane_width(lane_width_px, self._return_profile)
                self._clear_stability()
        elif self.state == "changing_to_lane2":
            direction = 1
            if self._uses_target_arrival(self.state):
                progress = 1.0
                offset_ratio = 0.0
            else:
                progress = self._progress(now)
                offset_ratio = -(1.0 - self._smoothstep(progress))
                if progress >= 1.0:
                    offset_ratio = 0.0
                    direction = 0
                    self._finish_transition_or_stabilize("stabilizing_lane2", "completed", now)
        elif self.state == "stabilizing_lane2":
            offset_ratio = 0.0

        offset_px = offset_ratio * self._effective_lane_width(lane_width_px)
        shifted, applied_offset_px = self._lane_target(
            lane,
            offset_px,
            bev_width_px,
            lane_reliable,
        )
        if not lane_reliable and self._uses_target_arrival(self.state):
            self._target_capture_frames = 0
        if (
            self.state == "changing_to_lane1"
            and self._uses_target_arrival(self.state)
            and lane_reliable
            and self._target_captured(shifted, -1)
        ):
            direction = 0
            self._finish_transition_or_stabilize("stabilizing_lane1", "lane1", now)
        elif (
            self.state == "changing_to_lane2"
            and self._uses_target_arrival(self.state)
            and lane_reliable
            and self._target_captured(shifted, 1)
        ):
            direction = 0
            self._finish_transition_or_stabilize("stabilizing_lane2", "completed", now)
        if self.state == "stabilizing_lane1":
            if self._update_stability(shifted, lane_reliable):
                self.state = "lane1"
                self._phase_started_at = now
        elif self.state == "stabilizing_lane2":
            if self._update_stability(shifted, lane_reliable):
                self.state = "completed"
                self._phase_started_at = None
        active = self.state in (
            "changing_to_lane1",
            "stabilizing_lane1",
            "lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        )
        return LaneChangeResult(
            lane=shifted,
            state=self.state,
            offset_px=applied_offset_px,
            active=active,
            direction=direction,
            progress=progress,
            stable_frames=self._stable_frames,
            lane_reliable=lane_reliable,
        )

    def apply_control_adjustments(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        command = self.apply_speed_cap(
            command,
            self.speed_cap_active(result),
            lane_reliable=result.lane_reliable,
        )
        return self.apply_steering_assist(command, result)

    @staticmethod
    def speed_cap_active(result: LaneChangeResult) -> bool:
        return result.state in (
            "armed",
            "changing_to_lane1",
            "stabilizing_lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        ) or result.direction != 0

    def apply_speed_cap(
        self,
        command: ControlCommand,
        active: bool,
        lane_reliable: bool = True,
    ) -> ControlCommand:
        if not active or command.brake:
            return command
        cap = max(0, int(self.config.speed_cap))
        if not lane_reliable:
            cap = min(cap, max(0, int(self.config.unreliable_speed_cap)))
        speed = max(-cap, min(cap, command.speed))
        suffix = "lane_change" if lane_reliable else "lane_change_unreliable"
        reason = "%s:%s" % (command.reason, suffix) if command.reason else suffix
        return ControlCommand(speed=speed, steering=command.steering, brake=False, reason=reason)

    def apply_steering_assist(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        if command.brake:
            return command
        if self._uses_avoidance_profile(result.state):
            return self._apply_stabilizing_steering(command, result)
        if (
            not result.lane_reliable
            and self._uses_target_arrival(result.state)
            and result.direction != 0
        ):
            return self._apply_unreliable_directional_steering(command, result)
        if not result.lane_reliable and self.speed_cap_active(result):
            cap = max(0, int(self.config.unreliable_steering_cap))
            steering = self._clip(command.steering, -cap, cap)
            reason = (
                "%s:lane_change_unreliable" % command.reason
                if command.reason
                else "lane_change_unreliable"
            )
            return ControlCommand(
                speed=command.speed,
                steering=steering,
                brake=False,
                reason=reason,
            )
        if result.direction == 0:
            return command
        if self.config.steering_min <= 0 and self.config.steering_boost <= 0:
            return command

        direction = -1 if result.direction < 0 else 1
        minimum = max(0, int(self.config.steering_min))
        cap = max(0, int(self.config.steering_cap))
        if (
            result.lane_reliable
            and self._uses_target_arrival(result.state)
            and self._target_approach_reached(result.lane, direction)
        ):
            steering = command.steering
            if cap > 0:
                steering = self._clip(steering, -cap, cap)
            reason = (
                "%s:lane_change_capture_feedback" % command.reason
                if command.reason
                else "lane_change_capture_feedback"
            )
            return ControlCommand(
                speed=command.speed,
                steering=steering,
                brake=False,
                reason=reason,
            )
        if self.config.steering_override:
            steering = direction * (cap if cap > 0 else minimum)
        else:
            # Heading feedback may not countersteer until target capture changes
            # the result direction to zero and starts parallel stabilization.
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

    def _apply_stabilizing_steering(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        """Amplify lane feedback until both target-center errors are settled."""
        if self._stable_now(result.lane, result.lane_reliable):
            return command

        lane = result.lane
        lateral = lane.lateral_error_norm
        near = (
            lane.near_lateral_error_norm
            if lane.near_lateral_error_norm is not None
            else lateral
        )
        minimum = max(0, int(self.config.stabilizing_steering_min))
        steering = int(command.steering)
        if steering != 0:
            direction = -1 if steering < 0 else 1
        else:
            # Let the normal lane follower choose the direction whenever it has
            # a signal. The position errors are only a zero-command fallback.
            correction = near if abs(near) > abs(lateral) else lateral
            direction = -1 if correction < 0.0 else 1
        if abs(steering) < minimum:
            steering = direction * minimum
        cap = max(0, int(self.config.steering_cap))
        if cap > 0:
            steering = self._clip(steering, -cap, cap)
        reason = (
            "%s:lane_change_stabilize" % command.reason
            if command.reason
            else "lane_change_stabilize"
        )
        return ControlCommand(
            speed=command.speed,
            steering=steering,
            brake=False,
            reason=reason,
        )

    def _apply_unreliable_directional_steering(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        direction = -1 if result.direction < 0 else 1
        cap = max(0, int(self.config.unreliable_steering_cap))
        if self.config.steering_cap > 0:
            cap = min(cap, max(0, int(self.config.steering_cap)))
        minimum = min(max(0, int(self.config.steering_min)), cap) if cap > 0 else 0
        steering = int(command.steering)
        if steering * direction <= 0 or abs(steering) < minimum:
            steering = direction * minimum
        if cap > 0:
            steering = self._clip(steering, -cap, cap)
        reason = (
            "%s:lane_change_unreliable_steer" % command.reason
            if command.reason
            else "lane_change_unreliable_steer"
        )
        return ControlCommand(
            speed=command.speed,
            steering=steering,
            brake=False,
            reason=reason,
        )

    def _progress(self, now: float) -> float:
        duration = max(0.05, self.config.transition_seconds)
        if self._phase_started_at is None:
            return 0.0
        return min(1.0, max(0.0, (now - self._phase_started_at) / duration))

    def _finish_transition_or_stabilize(self, stabilizing_state: str, final_state: str, now: float) -> None:
        if max(0, int(self.config.stable_required_frames)) <= 0:
            self.state = final_state
            self._phase_started_at = None if final_state == "completed" else now
            return
        self.state = stabilizing_state
        self._phase_started_at = now
        self._stable_frames = 0
        self._target_capture_frames = 0

    def _update_stability(
        self,
        lane: LaneGeometry,
        lane_reliable: bool,
    ) -> bool:
        if self._stable_now(lane, lane_reliable):
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        return self._stable_frames >= max(1, int(self.config.stable_required_frames))

    def _stable_now(self, lane: LaneGeometry, lane_reliable: bool) -> bool:
        if not lane.found:
            return False
        if not self.config.allow_virtual_stabilize and not lane_reliable:
            return False
        if self._uses_avoidance_profile(self.state):
            near_error = (
                lane.near_lateral_error_norm
                if lane.near_lateral_error_norm is not None
                else lane.lateral_error_norm
            )
            return (
                abs(lane.lateral_error_norm)
                <= max(0.0, float(self.config.stable_lateral_error))
                and abs(near_error)
                <= max(0.0, float(self.config.stable_near_lateral_error))
            )
        return (
            abs(lane.lateral_error_norm) <= max(0.0, float(self.config.stable_lateral_error))
            and abs(lane.heading_error) <= max(0.0, float(self.config.stable_heading_error))
        )

    def _uses_target_arrival(self, state: str) -> bool:
        if state == "changing_to_lane1":
            return self._request_profile == "avoidance"
        if state == "changing_to_lane2":
            return self._return_profile == "avoidance"
        return False

    def _target_captured(self, lane: LaneGeometry, direction: int) -> bool:
        if not lane.found or direction == 0:
            self._target_capture_frames = 0
            return False
        lateral_error = (
            lane.near_lateral_error_norm
            if lane.near_lateral_error_norm is not None
            else lane.lateral_error_norm
        )
        remaining_error = lateral_error * direction
        # The high-priority shift ends from target-lane feedback. The normal
        # lane controller then owns both steering direction and magnitude.
        capture_error = max(0.0, float(self.config.target_capture_error))
        if remaining_error <= capture_error:
            self._target_capture_frames += 1
        else:
            self._target_capture_frames = 0
        required = max(1, int(self.config.target_capture_frames))
        return self._target_capture_frames >= required

    def _target_approach_reached(
        self,
        lane: LaneGeometry,
        direction: int,
    ) -> bool:
        if not lane.found or direction == 0:
            return False
        lateral_error = (
            lane.near_lateral_error_norm
            if lane.near_lateral_error_norm is not None
            else lane.lateral_error_norm
        )
        remaining_error = lateral_error * direction
        threshold = max(
            max(0.0, float(self.config.target_capture_error)),
            max(0.0, float(self.config.target_approach_error)),
        )
        return remaining_error <= threshold

    def _uses_avoidance_profile(self, state: str) -> bool:
        if state == "stabilizing_lane1":
            return self._request_profile == "avoidance"
        if state == "stabilizing_lane2":
            return self._return_profile == "avoidance"
        return False

    def _clear_stability(self) -> None:
        self._stable_frames = 0
        self._target_capture_frames = 0

    def _lock_lane_width(self, lane_width_px: float, profile: str) -> None:
        if self._locked_lane_width_px is not None:
            return
        configured = max(0.0, float(self.config.target_lane_width_px))
        if profile == "avoidance" and configured > 0.0:
            self._locked_lane_width_px = configured
        else:
            self._locked_lane_width_px = max(0.0, float(lane_width_px))

    def _effective_lane_width(self, lane_width_px: float) -> float:
        if self._locked_lane_width_px is not None:
            return self._locked_lane_width_px
        return max(0.0, float(lane_width_px))

    def _lane_target(
        self,
        lane: LaneGeometry,
        offset_px: float,
        bev_width_px: float,
        lane_reliable: bool,
    ) -> tuple:
        if not lane.found:
            return lane, 0.0
        if lane_reliable:
            shifted = self._shift_lane_if_needed(lane, offset_px, bev_width_px)
            self._last_reliable_shifted_lane = shifted
            applied = shifted.center_x - lane.center_x
            return shifted, applied
        if self._hold_unreliable_target_active() and self._last_reliable_shifted_lane is not None:
            held = self._held_unreliable_lane(lane)
            return held, 0.0
        return lane, 0.0

    def _hold_unreliable_target_active(self) -> bool:
        return self.state in (
            "changing_to_lane1",
            "stabilizing_lane1",
            "lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        )

    def _held_unreliable_lane(self, lane: LaneGeometry) -> LaneGeometry:
        prev = self._last_reliable_shifted_lane
        assert prev is not None
        reason = prev.reason.split(":lane_change_hold_unreliable", 1)[0]
        return replace(
            prev,
            confidence=min(prev.confidence, lane.confidence),
            reason="%s:lane_change_hold_unreliable:%s" % (reason, lane.reason),
        )

    def _shift_lane_if_needed(
        self,
        lane: LaneGeometry,
        offset_px: float,
        bev_width_px: float,
    ) -> LaneGeometry:
        if abs(offset_px) <= 1e-6:
            return lane
        return self._shift_lane(lane, offset_px, bev_width_px)

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
        near_center_x = None
        near_lateral_error_px = None
        near_lateral_error_norm = None
        if lane.near_center_x is not None:
            near_center_x = max(
                0.0,
                min(max(0.0, bev_width_px - 1.0), lane.near_center_x + offset_px),
            )
            near_lateral_error_px = near_center_x - lane.vehicle_center_x
            near_lateral_error_norm = max(
                -1.0,
                min(1.0, near_lateral_error_px / half_width),
            )
        return replace(
            lane,
            center_x=center_x,
            lateral_error_px=lateral_error_px,
            lateral_error_norm=lateral_error_norm,
            near_center_x=near_center_x,
            near_lateral_error_px=near_lateral_error_px,
            near_lateral_error_norm=near_lateral_error_norm,
            reason="%s:lane_change" % lane.reason,
        )
