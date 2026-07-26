from __future__ import annotations

import math
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
    curve_strength_release_alpha: float = 0.18
    straight_steering_scale: float = 0.45
    curve_steering_scale: float = 1.45
    center_recovery_error_threshold: float = 0.14
    center_recovery_steering_boost: float = 2.0
    center_recovery_min_steering: int = 85
    center_recovery_rate_limit: int = 120
    center_recovery_max_speed: int = 50
    center_lock_enabled: bool = False
    center_lock_error_threshold: float = 0.05
    center_lock_min_steering: int = 90
    lane_lost_hold_frames: int = 20
    lane_lost_steering_release_rate_limit: Optional[int] = None

    # Full-path tracking uses every fitted BEV centerline point between the near
    # and far control rows. It is less sensitive than steering at one dot when a
    # dashed line fit or an S-curve moves that dot laterally between frames.
    path_tracking: bool = False
    path_lateral_gain: float = 225.0
    path_heading_gain: float = 70.0
    path_derivative_gain: float = 18.0
    path_near_weight: float = 1.25
    path_far_weight: float = 0.70
    path_steering_rise_alpha: float = 0.55
    path_steering_release_alpha: float = 0.28

    # Pure-pursuit steering: instead of summing a lateral-error PID and a heading
    # PID, steer the wheel directly toward the BEV centerline's lookahead point
    # (center_x, target_y) from the vehicle origin (vehicle_center_x, height). The
    # angle to that point folds lateral offset and forward distance into one signal
    # that grows naturally on curves, so S-curves and hairpins get stronger, better
    # timed steering. Falls back to a normalized approximation if lane.height is 0.
    pure_pursuit: bool = False
    # Steering units per radian of the lookahead angle (tune: larger = sharper).
    pure_pursuit_gain: float = 150.0
    # Lookahead angle (rad) at which curve-speed slowdown saturates.
    pure_pursuit_full_angle: float = 0.6


class YoloLaneFollower:
    def __init__(self, config: YoloLaneFollowerConfig = YoloLaneFollowerConfig()):
        self.config = config
        self._last_steering = 0
        self._last_lateral_error: Optional[float] = None
        self._last_heading_error: Optional[float] = None
        self._last_path_error: Optional[float] = None
        self._curve_strength = 0.0
        self._last_command: Optional[ControlCommand] = None
        self._lane_lost_frames = 0

    def plan(self, lane: LaneGeometry) -> ControlCommand:
        if not lane.found or lane.confidence < self.config.min_confidence:
            return self._hold_last_direction(lane)

        if self.config.path_tracking:
            return self._plan_path_tracking(lane)

        if self.config.pure_pursuit:
            return self._plan_pure_pursuit(lane)

        raw_curve_strength = self._curve_strength_from(lane)
        curve_strength = self._smooth_curve_strength(raw_curve_strength)
        centering_error = self._centering_error(lane)
        recovery_strength = self._center_recovery_strength(centering_error)
        center_lock_active = self._center_lock_active(centering_error)

        lateral_derivative = self._derivative(centering_error, self._last_lateral_error)
        heading_error = self._effective_heading_error(
            centering_error,
            lane.heading_error,
            center_lock_active,
        )
        heading_derivative = self._derivative(heading_error, self._last_heading_error)
        steering_scale = self._steering_scale(curve_strength)
        raw_steering = (
            self.config.kp_lateral * centering_error
            + self.config.kd_lateral * lateral_derivative
            + self.config.kp_heading * heading_error
            + self.config.kd_heading * heading_derivative
        ) * steering_scale
        raw_steering = self._apply_center_recovery(raw_steering, centering_error, recovery_strength)
        raw_steering = self._apply_center_lock(raw_steering, centering_error, center_lock_active)
        if self._opposes_lateral(centering_error, self._last_steering, center_lock_active):
            self._last_steering = 0
        rate_limit_strength = max(recovery_strength, 1.0 if center_lock_active else 0.0)
        steering = self._rate_limit(int(round(raw_steering)), curve_strength, rate_limit_strength)
        steering = self._clip(steering, -self.config.max_steering, self.config.max_steering)

        speed = int(round(self.config.base_speed - self.config.speed_curve_slowdown * raw_curve_strength))
        speed = self._clip(speed, self.config.min_curve_speed, self.config.max_speed)
        speed = self._apply_center_recovery_speed(speed, recovery_strength)
        speed = self._clip(speed, 0, self.config.max_speed)

        self._last_steering = steering
        self._last_lateral_error = centering_error
        self._last_heading_error = heading_error
        reason = "yolo_lane_follow:center_lock" if center_lock_active else "yolo_lane_follow"
        command = ControlCommand(speed=speed, steering=steering, brake=False, reason=reason)
        self._last_command = command
        self._lane_lost_frames = 0
        return command

    def _plan_path_tracking(self, lane: LaneGeometry) -> ControlCommand:
        """Track the complete BEV centerline with bounded temporal response."""
        path_error = self._weighted_path_error(lane)
        derivative = self._derivative(path_error, self._last_path_error)
        raw_steering = (
            self.config.path_lateral_gain * path_error
            + self.config.path_heading_gain * lane.heading_error
            + self.config.path_derivative_gain * derivative
        )

        alpha = self._path_steering_alpha(raw_steering, lane.reason)
        filtered = (
            float(self._last_steering)
            + alpha * (raw_steering - float(self._last_steering))
        )
        raw_curve = min(
            1.0,
            max(abs(path_error) / 0.55, abs(lane.heading_error) / 0.85),
        )
        curve_strength = self._smooth_curve_strength(raw_curve)
        steering = self._rate_limit(
            int(round(filtered)),
            curve_strength,
            0.0,
        )
        steering = self._clip(
            steering,
            -self.config.max_steering,
            self.config.max_steering,
        )

        speed = int(
            round(
                self.config.base_speed
                - self.config.speed_curve_slowdown * raw_curve
            )
        )
        speed = self._clip(
            speed,
            self.config.min_curve_speed,
            self.config.max_speed,
        )

        self._last_steering = steering
        self._last_lateral_error = lane.lateral_error_norm
        self._last_heading_error = lane.heading_error
        self._last_path_error = path_error
        command = ControlCommand(
            speed=speed,
            steering=steering,
            brake=False,
            reason="path_tracking:whole_centerline",
        )
        self._last_command = command
        self._lane_lost_frames = 0
        return command

    def _weighted_path_error(self, lane: LaneGeometry) -> float:
        points = list(lane.path_points)
        near_y = (
            float(lane.near_target_y)
            if lane.near_target_y is not None
            else float(lane.height) * 0.88
        )
        far_y = float(lane.target_y)
        if near_y <= far_y + 1.0 or len(points) < 3:
            near = lane.near_lateral_error_norm
            if near is None:
                return float(lane.lateral_error_norm)
            return self._clip_float(
                0.65 * float(near)
                + 0.35 * float(lane.lateral_error_norm),
                -1.0,
                1.0,
            )

        half_width = max(1.0, float(lane.vehicle_center_x))
        samples = []
        for x, y in points:
            y = float(y)
            if y < far_y or y > near_y:
                continue
            progress = (y - far_y) / (near_y - far_y)
            weight = (
                self.config.path_far_weight
                + (
                    self.config.path_near_weight
                    - self.config.path_far_weight
                )
                * progress
            )
            error = (float(x) - float(lane.vehicle_center_x)) / half_width
            samples.append((error, max(0.0, float(weight))))
        if not samples:
            return float(lane.lateral_error_norm)

        weight_sum = sum(weight for _, weight in samples)
        if weight_sum <= 1e-6:
            return float(lane.lateral_error_norm)
        return self._clip_float(
            sum(error * weight for error, weight in samples) / weight_sum,
            -1.0,
            1.0,
        )

    def _path_steering_alpha(self, raw_steering: float, reason: str) -> float:
        previous = float(self._last_steering)
        same_direction = raw_steering == 0.0 or raw_steering * previous >= 0.0
        increasing = same_direction and abs(raw_steering) >= abs(previous)
        alpha = (
            self.config.path_steering_rise_alpha
            if increasing
            else self.config.path_steering_release_alpha
        )
        if ":lane_change" in reason:
            alpha = max(alpha, 0.65)
        return self._clip_float(alpha, 0.0, 1.0)

    def _plan_pure_pursuit(self, lane: LaneGeometry) -> ControlCommand:
        """Geometric steering straight from the BEV lookahead point.

        alpha = angle between the vehicle's forward axis and the line to the
        lookahead point (center_x, target_y). Forward is up (toward smaller y), so
        the forward distance to the point is (height - target_y). Steering is
        proportional to alpha, which grows with both lateral offset and curvature,
        so tighter curves command more steering without a separate heading term."""
        dx = lane.center_x - lane.vehicle_center_x           # + = point is to the right
        dy = lane.height - lane.target_y                     # forward distance (px)
        if dy > 1.0:
            alpha = math.atan2(dx, dy)
        else:
            # No forward-distance info: approximate the angle from the normalized
            # lateral error (treats the lookahead as ~45 deg full-scale).
            alpha = lane.lateral_error_norm * (math.pi / 4.0)

        near_x = lane.near_center_x
        near_y = lane.near_target_y
        near_error = lane.near_lateral_error_norm
        far_error = float(lane.lateral_error_norm)
        reason = "pure_pursuit"
        if near_x is not None and near_y is not None and near_error is not None:
            near_error = float(near_error)
            sign_conflict = far_error * near_error < 0.0
            near_dominant = abs(near_error) > abs(far_error) + 0.07
            if sign_conflict or near_dominant:
                blend = 0.55 if sign_conflict else 0.38
                aim_x = (
                    (1.0 - blend) * float(lane.center_x)
                    + blend * float(near_x)
                )
                aim_y = (
                    (1.0 - blend) * float(lane.target_y)
                    + blend * float(near_y)
                )
                aim_dx = aim_x - float(lane.vehicle_center_x)
                aim_dy = float(lane.height) - aim_y
                candidate_alpha = (
                    math.atan2(aim_dx, aim_dy)
                    if aim_dy > 1.0
                    else near_error * (math.pi / 4.0)
                )
                gain = max(1e-6, float(self.config.pure_pursuit_gain))
                far_raw = gain * alpha
                candidate_raw = gain * candidate_alpha
                bounded_delta = self._clip_float(
                    candidate_raw - far_raw,
                    -35.0,
                    35.0,
                )
                alpha = (far_raw + bounded_delta) / gain
                reason = "pure_pursuit:near_guard"

        raw_curve = min(1.0, abs(alpha) / max(1e-6, self.config.pure_pursuit_full_angle))
        curve_strength = self._smooth_curve_strength(raw_curve)

        raw_steering = self.config.pure_pursuit_gain * alpha
        steering = self._rate_limit(int(round(raw_steering)), curve_strength, 0.0)
        steering = self._clip(steering, -self.config.max_steering, self.config.max_steering)

        speed = int(round(self.config.base_speed - self.config.speed_curve_slowdown * raw_curve))
        speed = self._clip(speed, self.config.min_curve_speed, self.config.max_speed)

        self._last_steering = steering
        self._last_lateral_error = lane.lateral_error_norm
        self._last_heading_error = lane.heading_error
        command = ControlCommand(speed=speed, steering=steering, brake=False, reason=reason)
        self._last_command = command
        self._lane_lost_frames = 0
        return command

    def _hold_last_direction(self, lane: LaneGeometry) -> ControlCommand:
        # Road not detected (e.g. a crosswalk covering the lane markings):
        # keep the last steering/speed so the car maintains its current
        # heading instead of jittering or stopping. Fall back to a full stop
        # once the lane has stayed lost for too long.
        if self._last_command is None or self._lane_lost_frames >= self.config.lane_lost_hold_frames:
            self.reset()
            return ControlCommand.stop("lane_lost:%s" % lane.reason)
        self._lane_lost_frames += 1
        steering = self._release_lane_lost_steering(self._last_command.steering)
        command = ControlCommand(
            speed=self._last_command.speed,
            steering=steering,
            brake=False,
            reason="lane_lost_hold:%s" % lane.reason,
        )
        self._last_command = command
        self._last_steering = steering
        return command

    def _release_lane_lost_steering(self, steering: int) -> int:
        limit = self.config.lane_lost_steering_release_rate_limit
        if limit is None:
            limit = max(self.config.min_steering_rate_limit, self.config.steering_release_rate_limit)
        if limit <= 0:
            return steering
        if abs(steering) <= limit:
            return 0
        if steering > 0:
            return steering - limit
        return steering + limit

    def reset(self) -> None:
        self._last_steering = 0
        self._last_lateral_error = None
        self._last_heading_error = None
        self._last_path_error = None
        self._curve_strength = 0.0
        self._last_command = None
        self._lane_lost_frames = 0

    @staticmethod
    def _derivative(value: float, previous: Optional[float]) -> float:
        if previous is None:
            return 0.0
        return value - previous

    def _effective_heading_error(
        self,
        lateral_error: float,
        heading_error: float,
        center_lock_active: bool = False,
    ) -> float:
        if self._opposes_lateral(lateral_error, heading_error, center_lock_active):
            return 0.0
        return heading_error

    def _opposes_lateral(self, lateral_error: float, value: float, center_lock_active: bool = False) -> bool:
        if not center_lock_active and abs(lateral_error) < self.config.lateral_priority_threshold:
            return False
        return lateral_error * value < 0.0

    def _curve_strength_from(self, lane: LaneGeometry) -> float:
        lateral_curve = min(1.0, abs(lane.lateral_error_norm) / 0.65)
        heading_curve = min(1.0, abs(lane.heading_error) / 0.85)
        return self._clip_float(max(lateral_curve, heading_curve), 0.0, 1.0)

    def _smooth_curve_strength(self, value: float) -> float:
        if value >= self._curve_strength:
            alpha = self.config.curve_strength_alpha
        else:
            alpha = self.config.curve_strength_release_alpha
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
        if boosted * lateral_error <= 0.0 or abs(boosted) < minimum:
            direction = 1.0 if lateral_error >= 0.0 else -1.0
            boosted = direction * minimum
        return boosted

    def _center_lock_active(self, lateral_error: float) -> bool:
        if not self.config.center_lock_enabled:
            return False
        return abs(lateral_error) >= self.config.center_lock_error_threshold

    @staticmethod
    def _centering_error(lane: LaneGeometry) -> float:
        near = lane.near_lateral_error_norm
        if near is not None and abs(near) > abs(lane.lateral_error_norm):
            return near
        return lane.lateral_error_norm

    def _apply_center_lock(self, steering: float, lateral_error: float, active: bool) -> float:
        if not active:
            return steering
        minimum = abs(self.config.center_lock_min_steering)
        if steering * lateral_error > 0.0 and abs(steering) >= minimum:
            return steering
        direction = 1.0 if lateral_error >= 0.0 else -1.0
        return direction * minimum

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
