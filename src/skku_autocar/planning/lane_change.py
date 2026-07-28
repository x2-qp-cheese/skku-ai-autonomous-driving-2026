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
    steering_slew_limit: int = 0
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
    smooth_avoidance: bool = False
    spatial_transition_lead: float = 0.10
    trajectory_heading_gain: float = 1.6
    unreliable_hold_seconds: float = 0.25
    max_transition_seconds: float = 4.0
    return_duration_scale: float = 1.0
    return_steering_cap: int = 0
    return_stabilizing_steering_cap: int = 0


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
    unreliable_age_seconds: float = 0.0
    neutral_steering_reason: str = ""
    directional_assist_released: bool = False


class LaneChangeController:
    """Reusable 2 -> 1 -> 2 lane-change trajectory controller.

    The normal BEV corridor produces the lane-2 driving target. Lane changes
    shift that full target path by the measured adjacent-lane width. Every change,
    including obstacle avoidance, follows one smoothstep trajectory so no state
    can jump the target by a complete lane in one frame.
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
        self._last_output_steering: Optional[int] = None
        self._steering_slew_releasing = False
        self._paused_at: Optional[float] = None
        self._unreliable_started_at: Optional[float] = None
        self._last_reliable_at: Optional[float] = None

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
        self._last_output_steering = None
        self._steering_slew_releasing = False
        self._paused_at = None
        self._unreliable_started_at = None
        self._last_reliable_at = None

    def pause(self, now: float) -> None:
        """Freeze transition timers while another mission owns path priority."""
        if self._paused_at is None:
            self._paused_at = float(now)
        # Crosswalk priority is an intentional mission pause, not a failed
        # geometry sample. Start a fresh unreliable grace period after resume.
        self._unreliable_started_at = None
        self._last_reliable_at = None

    def resume(self, now: float) -> None:
        if self._paused_at is None:
            return
        paused_for = max(0.0, float(now) - self._paused_at)
        if self._run_started_at is not None:
            self._run_started_at += paused_for
        if self._phase_started_at is not None:
            self._phase_started_at += paused_for
        self._paused_at = None

    def apply_fixed_offset(
        self,
        lane: LaneGeometry,
        offset_px: float,
        bev_width_px: float,
    ) -> LaneGeometry:
        """Translate the current path without advancing lane-change state."""
        return self._shift_lane_if_needed(
            lane,
            float(offset_px),
            float(bev_width_px),
        )

    def apply_frozen_trajectory(
        self,
        lane: LaneGeometry,
        bev_width_px: float,
    ) -> LaneGeometry:
        """Rebuild the currently paused target from fresh base-lane geometry."""
        width = self._effective_lane_width(0.0)
        if width <= 0.0:
            return lane
        state = self.state
        now = (
            float(self._paused_at)
            if self._paused_at is not None
            else float(self._phase_started_at or 0.0)
        )
        if state == "changing_to_lane1":
            progress = self._progress(now)
            return self._transition_lane_target(
                lane,
                0.0,
                -width,
                progress,
                bev_width_px,
                (
                    self.config.smooth_avoidance
                    and self._request_profile == "avoidance"
                ),
            )
        if state in ("stabilizing_lane1", "lane1"):
            return self._shift_lane_if_needed(lane, -width, bev_width_px)
        if state == "changing_to_lane2":
            progress = self._progress(now)
            return self._transition_lane_target(
                lane,
                -width,
                0.0,
                progress,
                bev_width_px,
                (
                    self.config.smooth_avoidance
                    and self._return_profile == "avoidance"
                ),
            )
        return lane

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
        self._last_reliable_shifted_lane = None
        self._unreliable_started_at = None
        return True

    @property
    def request_source(self) -> str:
        return self._request_source

    def request_return(self, source: str = "external") -> bool:
        """Request lane 1 -> lane 2 after an obstacle-clear decision."""
        return self._arm_return(source, "normal")

    def request_avoidance_return(self, source: str = "obstacle") -> bool:
        """Queue a safety-priority return, executing after lane 1 is stable."""
        return self._arm_return(source, "avoidance")

    def _arm_return(self, source: str, profile: str) -> bool:
        if self.config.mode == "off":
            return False
        if self.state != "lane1":
            if profile != "avoidance" or self.state != "stabilizing_lane1":
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
        self._update_reliability_clock(now, lane_reliable)

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

        priority_return = self._return_profile == "avoidance"

        offset_ratio = 0.0
        direction = 0
        progress = 0.0
        transition_start_px = 0.0
        transition_end_px = 0.0
        spatial_transition = False
        if self.state == "changing_to_lane1":
            direction = -1
            if self._uses_target_arrival(self.state):
                progress = self._progress(now) if self.config.smooth_avoidance else 1.0
                offset_ratio = -self._smoothstep(progress)
                transition_start_px = 0.0
                transition_end_px = -self._effective_lane_width(lane_width_px)
                spatial_transition = self.config.smooth_avoidance
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
                progress = self._progress(now) if self.config.smooth_avoidance else 1.0
                offset_ratio = -(1.0 - self._smoothstep(progress))
                transition_start_px = -self._effective_lane_width(lane_width_px)
                transition_end_px = 0.0
                spatial_transition = self.config.smooth_avoidance
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
            transition_start_px=transition_start_px,
            transition_end_px=transition_end_px,
            progress=progress,
            spatial_transition=spatial_transition,
        )
        if not lane_reliable and self._uses_target_arrival(self.state):
            self._target_capture_frames = 0
        if (
            self.state == "changing_to_lane1"
            and self._uses_target_arrival(self.state)
            and lane_reliable
            and progress >= 1.0
            and self._target_captured(shifted, -1)
        ):
            direction = 0
            self._finish_transition_or_stabilize("stabilizing_lane1", "lane1", now)
        elif (
            self.state == "changing_to_lane2"
            and self._uses_target_arrival(self.state)
            and lane_reliable
            and progress >= 1.0
            and self._target_captured(shifted, 1)
        ):
            direction = 0
            self._finish_transition_or_stabilize("stabilizing_lane2", "completed", now)
        if self.state == "stabilizing_lane1":
            if self._update_stability(shifted, lane_reliable):
                # A second obstacle may be detected while the vehicle is still
                # settling into lane 1. Keep that request queued, but never
                # reverse steering until the measured lane target is stable.
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
        unreliable_age = self._unreliable_age(now, lane_reliable)
        neutral_steering_reason = self._neutral_steering_reason(
            lane_reliable,
            unreliable_age,
        )
        assist_released = self._transition_timed_out(now)
        if not lane_reliable or assist_released:
            # Never force the old transition direction from a cached path. The
            # bounded lane-follower command may bridge a short dropout. A long
            # but still reliable transition also falls back to path feedback
            # instead of holding a directional steering minimum forever.
            direction = 0
        return LaneChangeResult(
            lane=shifted,
            state=self.state,
            offset_px=applied_offset_px,
            active=active,
            direction=direction,
            progress=progress,
            stable_frames=self._stable_frames,
            lane_reliable=lane_reliable,
            unreliable_age_seconds=unreliable_age,
            neutral_steering_reason=neutral_steering_reason,
            directional_assist_released=assist_released,
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
        if result.neutral_steering_reason and not command.brake:
            self._last_output_steering = 0
            self._steering_slew_releasing = False
            reason = (
                "%s:%s" % (command.reason, result.neutral_steering_reason)
                if command.reason
                else result.neutral_steering_reason
            )
            return ControlCommand(
                speed=command.speed,
                steering=0,
                brake=False,
                reason=reason,
            )
        adjusted = self._apply_steering_assist_base(command, result)
        if adjusted.brake:
            self._last_output_steering = None
            self._steering_slew_releasing = False
            return adjusted
        if self._return_profile != "avoidance":
            if result.state == "changing_to_lane2":
                cap = max(0, int(self.config.return_steering_cap))
            elif result.state == "stabilizing_lane2":
                cap = max(
                    0,
                    int(self.config.return_stabilizing_steering_cap),
                )
            else:
                cap = 0
            if cap > 0:
                steering = self._clip(adjusted.steering, -cap, cap)
                if steering != adjusted.steering:
                    reason = (
                        "%s:lane2_return_smooth" % adjusted.reason
                        if adjusted.reason
                        else "lane2_return_smooth"
                    )
                    adjusted = ControlCommand(
                        speed=adjusted.speed,
                        steering=steering,
                        brake=False,
                        reason=reason,
                    )
        return self._apply_steering_slew(adjusted, result)

    def _apply_steering_slew(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        limit = max(0, int(self.config.steering_slew_limit))
        active = result.state in (
            "changing_to_lane1",
            "stabilizing_lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        )
        if limit <= 0:
            self._last_output_steering = int(command.steering)
            self._steering_slew_releasing = active
            return command
        if self._last_output_steering is None:
            self._last_output_steering = int(command.steering)
            self._steering_slew_releasing = active
            return command
        if not active and not self._steering_slew_releasing:
            self._last_output_steering = int(command.steering)
            return command

        previous = self._last_output_steering
        target = int(command.steering)
        steering = self._clip(target, previous - limit, previous + limit)
        self._last_output_steering = steering
        self._steering_slew_releasing = active or steering != target
        if steering == target:
            return command
        reason = (
            "%s:lane_change_slew" % command.reason
            if command.reason
            else "lane_change_slew"
        )
        return ControlCommand(
            speed=command.speed,
            steering=steering,
            brake=False,
            reason=reason,
        )

    def _apply_steering_assist_base(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        if command.brake:
            return command
        if (
            not result.lane_reliable
            and self._hold_unreliable_target_active(result.state)
        ):
            adjusted = command
            if (
                (
                    self.config.smooth_avoidance
                    or self._uses_target_arrival(result.state)
                )
                and result.direction != 0
            ):
                adjusted = self._apply_unreliable_directional_steering(
                    adjusted,
                    result,
                )
            adjusted = self._release_unreliable_steering(adjusted, result)
            return self._cap_unreliable_steering(adjusted)
        if self._uses_avoidance_profile(result.state):
            return self._apply_stabilizing_steering(command, result)
        if (
            not result.lane_reliable
            and (
                self.config.smooth_avoidance
                or self._uses_target_arrival(result.state)
            )
            and result.direction != 0
        ):
            return self._apply_unreliable_directional_steering(command, result)
        if not result.lane_reliable and self.speed_cap_active(result):
            return self._cap_unreliable_steering(command)
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
            and result.progress >= 1.0
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

    def _cap_unreliable_steering(
        self,
        command: ControlCommand,
    ) -> ControlCommand:
        cap = max(0, int(self.config.unreliable_steering_cap))
        steering = self._clip(command.steering, -cap, cap)
        reason = command.reason
        if "lane_change_unreliable" not in reason:
            reason = (
                "%s:lane_change_unreliable" % reason
                if reason
                else "lane_change_unreliable"
            )
        return ControlCommand(
            speed=command.speed,
            steering=steering,
            brake=False,
            reason=reason,
        )

    def _release_unreliable_steering(
        self,
        command: ControlCommand,
        result: LaneChangeResult,
    ) -> ControlCommand:
        """Continuously unwind cached-path steering during the short grace."""
        hold = max(0.0, float(self.config.unreliable_hold_seconds))
        if hold <= 1e-6:
            scale = 0.0
        else:
            scale = self._clip_float(
                1.0 - float(result.unreliable_age_seconds) / hold,
                0.0,
                1.0,
            )
        steering = int(round(float(command.steering) * scale))
        reason = (
            "%s:lane_change_unreliable_release" % command.reason
            if command.reason
            else "lane_change_unreliable_release"
        )
        return ControlCommand(
            speed=command.speed,
            steering=steering,
            brake=False,
            reason=reason,
        )

    def _progress(self, now: float) -> float:
        duration = max(0.05, self.config.transition_seconds)
        if (
            self.state == "changing_to_lane2"
            and self._return_profile != "avoidance"
        ):
            duration *= max(1.0, float(self.config.return_duration_scale))
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
        transition_start_px: float = 0.0,
        transition_end_px: float = 0.0,
        progress: float = 0.0,
        spatial_transition: bool = False,
    ) -> tuple:
        if not lane.found:
            return lane, 0.0
        if lane_reliable:
            if spatial_transition:
                shifted = self._transition_lane_target(
                    lane,
                    transition_start_px,
                    transition_end_px,
                    progress,
                    bev_width_px,
                    True,
                )
            else:
                shifted = self._shift_lane_if_needed(lane, offset_px, bev_width_px)
            self._last_reliable_shifted_lane = shifted
            applied = shifted.center_x - lane.center_x
            return shifted, applied
        if self._hold_unreliable_target_active() and self._last_reliable_shifted_lane is not None:
            held = self._held_unreliable_lane(lane)
            return held, 0.0
        return lane, 0.0

    def _hold_unreliable_target_active(
        self,
        state: Optional[str] = None,
    ) -> bool:
        return (self.state if state is None else state) in (
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

    def _transition_lane_target(
        self,
        lane: LaneGeometry,
        start_offset_px: float,
        end_offset_px: float,
        progress: float,
        bev_width_px: float,
        spatial: bool,
    ) -> LaneGeometry:
        if not spatial or len(lane.path_points) < 3:
            ratio = self._smoothstep(
                self._clip_float(float(progress), 0.0, 1.0)
            )
            offset = start_offset_px + (
                end_offset_px - start_offset_px
            ) * ratio
            return self._shift_lane_if_needed(lane, offset, bev_width_px)
        return self._shift_lane_spatial(
            lane,
            start_offset_px,
            end_offset_px,
            progress,
            bev_width_px,
        )

    def _shift_lane_spatial(
        self,
        lane: LaneGeometry,
        start_offset_px: float,
        end_offset_px: float,
        progress: float,
        bev_width_px: float,
    ) -> LaneGeometry:
        """Apply a near-anchored S trajectory instead of teleporting the path."""
        target_y = float(lane.target_y)
        near_y = (
            float(lane.near_target_y)
            if lane.near_target_y is not None
            else (
                float(lane.height) * 0.88
                if float(lane.height) > 0.0
                else max(float(y) for _, y in lane.path_points)
            )
        )
        span = max(1.0, near_y - target_y)
        lead = self._clip_float(
            float(self.config.spatial_transition_lead),
            0.0,
            1.0,
        )
        temporal = self._clip_float(float(progress), 0.0, 1.0)
        transformed = []
        for raw_x, raw_y in lane.path_points:
            x = float(raw_x)
            y = float(raw_y)
            forward = self._clip_float((near_y - y) / span, 0.0, 1.0)
            phase = self._clip_float(
                temporal + lead * self._smoothstep(forward),
                0.0,
                1.0,
            )
            blend = self._smoothstep(phase)
            offset = start_offset_px + (
                end_offset_px - start_offset_px
            ) * blend
            transformed.append((x + offset, y))

        center_x = self._path_x_at(
            transformed,
            target_y,
            lane.center_x,
        )
        near_center_x = self._path_x_at(
            transformed,
            near_y,
            (
                lane.near_center_x
                if lane.near_center_x is not None
                else lane.center_x
            ),
        )
        half_width = max(1.0, float(bev_width_px) / 2.0)
        lateral_error_px = center_x - lane.vehicle_center_x
        near_lateral_error_px = near_center_x - lane.vehicle_center_x
        return replace(
            lane,
            center_x=center_x,
            lateral_error_px=lateral_error_px,
            lateral_error_norm=self._clip_float(
                lateral_error_px / half_width,
                -1.0,
                1.0,
            ),
            heading_error=self._trajectory_heading(
                lane,
                transformed,
                target_y,
                near_y,
            ),
            near_center_x=near_center_x,
            near_target_y=near_y,
            near_lateral_error_px=near_lateral_error_px,
            near_lateral_error_norm=self._clip_float(
                near_lateral_error_px / half_width,
                -1.0,
                1.0,
            ),
            path_points=tuple(transformed),
            reason="%s:lane_change_scurve" % lane.reason,
        )

    def _trajectory_heading(
        self,
        lane: LaneGeometry,
        transformed: list,
        target_y: float,
        near_y: float,
    ) -> float:
        base_slope = self._path_slope(
            list(lane.path_points),
            target_y,
            near_y,
        )
        transformed_slope = self._path_slope(
            transformed,
            target_y,
            near_y,
        )
        if base_slope is None or transformed_slope is None:
            return float(lane.heading_error)
        heading = float(lane.heading_error) - float(
            self.config.trajectory_heading_gain
        ) * (transformed_slope - base_slope)
        return self._clip_float(heading, -1.0, 1.0)

    @staticmethod
    def _path_slope(
        points: list,
        target_y: float,
        near_y: float,
    ) -> Optional[float]:
        selected = [
            (float(x), float(y))
            for x, y in points
            if target_y <= float(y) <= near_y
        ]
        if len(selected) < 3:
            selected = [(float(x), float(y)) for x, y in points]
        if len(selected) < 2:
            return None
        mean_y = sum(y for _, y in selected) / len(selected)
        mean_x = sum(x for x, _ in selected) / len(selected)
        variance = sum((y - mean_y) ** 2 for _, y in selected)
        if variance <= 1e-6:
            return None
        covariance = sum(
            (y - mean_y) * (x - mean_x)
            for x, y in selected
        )
        return covariance / variance

    @staticmethod
    def _path_x_at(
        points: list,
        target_y: float,
        fallback: float,
    ) -> float:
        ordered = sorted(
            ((float(x), float(y)) for x, y in points),
            key=lambda point: point[1],
        )
        if not ordered:
            return float(fallback)
        if target_y <= ordered[0][1]:
            return ordered[0][0]
        if target_y >= ordered[-1][1]:
            return ordered[-1][0]
        for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
            if y0 <= target_y <= y1:
                if abs(y1 - y0) <= 1e-6:
                    return x1
                ratio = (target_y - y0) / (y1 - y0)
                return x0 + ratio * (x1 - x0)
        return float(fallback)

    def _update_reliability_clock(
        self,
        now: float,
        lane_reliable: bool,
    ) -> None:
        if lane_reliable:
            if self._unreliable_started_at is not None:
                dropout = max(
                    0.0,
                    float(now) - self._unreliable_started_at,
                )
                if (
                    self._phase_started_at is not None
                    and self.state in (
                        "changing_to_lane1",
                        "changing_to_lane2",
                    )
                ):
                    self._phase_started_at += dropout
            self._unreliable_started_at = None
            self._last_reliable_at = float(now)
            return
        if self._unreliable_started_at is None:
            self._unreliable_started_at = (
                float(self._last_reliable_at)
                if self._last_reliable_at is not None
                else float(now)
            )

    def _unreliable_age(
        self,
        now: float,
        lane_reliable: bool,
    ) -> float:
        if lane_reliable or self._unreliable_started_at is None:
            return 0.0
        return max(0.0, float(now) - self._unreliable_started_at)

    def _neutral_steering_reason(
        self,
        lane_reliable: bool,
        unreliable_age: float,
    ) -> str:
        safety_state = self.state in (
            "armed",
            "changing_to_lane1",
            "stabilizing_lane1",
            "lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        )
        hold = max(0.0, float(self.config.unreliable_hold_seconds))
        if safety_state and not lane_reliable and unreliable_age >= hold:
            return "lane_change_stale_geometry_neutral"
        return ""

    def _transition_timed_out(self, now: float) -> bool:
        maximum = max(0.0, float(self.config.max_transition_seconds))
        return (
            maximum > 0.0
            and self.state in ("changing_to_lane1", "changing_to_lane2")
            and self._phase_started_at is not None
            and float(now) - self._phase_started_at >= maximum
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _clip_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _clip(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

    @staticmethod
    def _shift_lane(lane: LaneGeometry, offset_px: float, bev_width_px: float) -> LaneGeometry:
        half_width = max(1.0, bev_width_px / 2.0)
        # Preserve a parallel translated path even when its far preview leaves
        # the BEV canvas. Clipping every point to an image edge creates a false
        # kink while heading still describes the unclipped curve. Control error
        # is bounded below, so geometry itself does not need destructive clipping.
        center_x = lane.center_x + offset_px
        lateral_error_px = center_x - lane.vehicle_center_x
        lateral_error_norm = max(-1.0, min(1.0, lateral_error_px / half_width))
        near_center_x = None
        near_lateral_error_px = None
        near_lateral_error_norm = None
        if lane.near_center_x is not None:
            near_center_x = lane.near_center_x + offset_px
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
            path_points=tuple(
                (float(x) + offset_px, float(y))
                for x, y in lane.path_points
            ),
            reason="%s:lane_change" % lane.reason,
        )
