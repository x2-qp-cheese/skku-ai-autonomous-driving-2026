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
    # Recover a car that starts measurably off-center before full-speed travel
    # turns a correct but small path command into a boundary approach.
    path_center_recovery_error_threshold: float = 0.07
    path_center_recovery_heading_limit: float = 0.12
    path_center_recovery_min_steering: float = 60.0
    path_center_recovery_alpha: float = 0.90
    path_center_recovery_rate_limit: int = 120
    # S-curves need the old steering direction removed promptly, while every
    # reversal remains bounded for full-speed driving and small sign noise
    # continues through the ordinary path filter.
    path_reversal_alpha: float = 0.90
    path_reversal_min_steering: float = 25.0
    path_reversal_min_geometry: float = 0.08
    path_reversal_output_min_steering: float = 60.0
    path_reversal_rate_limit: int = 80
    # When an S-curve preview changes sign before the near-field centerline has
    # crossed the vehicle, unwind the old turn first. The new turn is blended in
    # as the near error approaches the center instead of being forced at once.
    path_reversal_near_guard_error: float = 0.025
    path_reversal_near_full_error: float = 0.12
    # Near-field disagreement is blended into the whole-path command. On a
    # curve, require a larger local displacement before overriding heading
    # preview so entry timing is preserved while boundary drift is corrected.
    path_near_conflict_error_threshold: float = 0.01
    path_near_conflict_release_alpha: float = 0.90
    path_near_conflict_heading_limit: float = 0.18
    # A large far-path heading with an already centered near field is curve
    # preview, not permission to keep increasing steering into the inner line.
    path_curve_guard_heading_threshold: float = 0.25
    path_curve_guard_near_error: float = 0.13
    path_curve_guard_release_error: float = 0.25
    path_curve_guard_steering_limit: float = 115.0
    # A curve first appears as a change in path heading while the near-field
    # lateral error is still small. Convert that geometric lead into continuous
    # steering feed-forward instead of waiting until the car has drifted.
    path_heading_lead_gain: float = 180.0
    path_heading_lead_coherent_gain: float = 195.0
    path_heading_lead_span: float = 0.15
    path_heading_lead_max_steering: float = 36.0
    # A bounded integral term removes persistent mechanical/camera centering
    # bias, but only while two physical boundaries describe a stable straight.
    path_integral_gain: float = 45.0
    path_integral_limit: float = 0.25
    path_integral_decay: float = 0.65

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
        self._path_state = "straight"
        self._path_heading_lead = 0.0
        self._path_error_integral = 0.0

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
        heading_lead = self._path_heading_lead_strength(lane, path_error)
        near_error = (
            float(lane.near_lateral_error_norm)
            if lane.near_lateral_error_norm is not None
            else float(path_error)
        )
        heading_lead_gain = self._path_heading_lead_gain(
            lane,
            path_error,
            near_error,
            heading_lead,
        )
        heading_preview_permission = self._path_heading_preview_permission(
            lane,
            near_error,
        )
        heading_feedforward = (
            heading_lead_gain
            * float(lane.heading_error)
            * heading_lead
            * heading_preview_permission
        )
        heading_lead_limit = max(
            0.0,
            float(self.config.path_heading_lead_max_steering),
        )
        heading_feedforward = self._clip_float(
            heading_feedforward,
            -heading_lead_limit,
            heading_lead_limit,
        )
        integral_steering = self._path_integral_correction(
            lane,
            path_error,
            heading_lead,
        )
        raw_steering = (
            self.config.path_lateral_gain * path_error
            + (
                self.config.path_heading_gain
                * lane.heading_error
                * heading_preview_permission
            )
            + self.config.path_derivative_gain * derivative
            + heading_feedforward
            + integral_steering
        )
        center_recovery_strength = self._path_center_recovery_strength(
            lane,
            near_error,
            heading_lead,
        )
        center_recovery = center_recovery_strength > 0.0
        if center_recovery:
            raw_steering = self._apply_path_center_recovery(
                raw_steering,
                near_error,
                center_recovery_strength,
            )
        near_conflict_strength = self._path_near_conflict_strength(
            lane,
            near_error,
            raw_steering,
        )
        near_conflict = near_conflict_strength > 0.0
        if near_conflict:
            near_steering = self._clip_float(
                float(self.config.path_lateral_gain) * float(near_error),
                -float(self.config.max_steering),
                float(self.config.max_steering),
            )
            raw_steering += near_conflict_strength * (
                near_steering - raw_steering
            )
        reversal_near_guard = self._path_reversal_near_guard_strength(
            lane,
            near_error,
            raw_steering,
        )
        if reversal_near_guard > 0.0:
            near_steering = self._clip_float(
                float(self.config.path_lateral_gain) * float(near_error),
                -float(self.config.max_steering),
                float(self.config.max_steering),
            )
            raw_steering += reversal_near_guard * (
                near_steering - raw_steering
            )
            # A fractional blend can still cross zero while the measured
            # near-field path remains on the old side (the 115931 S-curve did
            # so at near=-0.052, producing +18 steering). Preserve the near
            # sign until the configured zero band is reached.  The magnitude
            # remains proportional to near_error, so this approaches zero
            # continuously instead of holding a fixed old steering command.
            if raw_steering * float(near_error) < 0.0:
                raw_steering = near_steering
        curve_guard_limit = self._path_curve_guard_limit(
            lane,
            path_error,
            near_error,
            raw_steering,
        )
        curve_guard = curve_guard_limit is not None
        if curve_guard_limit is not None:
            raw_steering = self._clip_float(
                raw_steering,
                -curve_guard_limit,
                curve_guard_limit,
            )
        direction_reversal = self._path_direction_reversal_active(
            lane,
            path_error,
            raw_steering,
        )
        # The far path and heading can reverse one frame before the near path
        # reaches the vehicle center in an S. Let the ordinary rate limiter
        # unwind that final old-sign error; the +/−minimum is only safe once the
        # near field supports the requested turn beyond the noise guard.
        reversal_near_threshold = max(
            0.0,
            float(self.config.path_reversal_near_guard_error),
        )
        near_supports_reversal = (
            raw_steering * float(near_error) >= 0.0
            and abs(float(near_error)) >= reversal_near_threshold
        )
        coherent_reversal = (
            direction_reversal
            and not near_conflict
            and near_supports_reversal
        )

        alpha = self._path_steering_alpha(
            raw_steering,
            lane.reason,
            heading_lead,
        )
        if center_recovery:
            recovery_alpha = self._clip_float(
                float(self.config.path_center_recovery_alpha),
                0.0,
                1.0,
            )
            alpha = max(
                alpha,
                alpha
                + center_recovery_strength * (recovery_alpha - alpha),
            )
        if direction_reversal:
            alpha = max(
                alpha,
                self._clip_float(
                    float(self.config.path_reversal_alpha),
                    0.0,
                    1.0,
                ),
            )
        if near_conflict:
            conflict_alpha = self._clip_float(
                float(self.config.path_near_conflict_release_alpha),
                0.0,
                1.0,
            )
            alpha = max(
                alpha,
                alpha + near_conflict_strength * (conflict_alpha - alpha),
            )
        if reversal_near_guard > 0.0:
            alpha = max(
                alpha,
                self._clip_float(
                    float(self.config.path_reversal_alpha),
                    0.0,
                    1.0,
                ),
            )
        filtered = (
            float(self._last_steering)
            + alpha * (raw_steering - float(self._last_steering))
        )
        if coherent_reversal:
            reversal_direction = 1.0 if raw_steering >= 0.0 else -1.0
            reversal_minimum = max(
                0.0,
                float(self.config.path_reversal_output_min_steering),
            )
            if filtered * reversal_direction < reversal_minimum:
                filtered = reversal_direction * reversal_minimum
        raw_curve = min(
            1.0,
            max(abs(path_error) / 0.55, abs(lane.heading_error) / 0.85),
        )
        curve_strength = self._smooth_curve_strength(raw_curve)
        steering = self._rate_limit(
            int(round(filtered)),
            curve_strength,
            0.0,
            minimum_rate_limit=max(
                (
                    int(
                        round(
                            float(self.config.path_center_recovery_rate_limit)
                            * center_recovery_strength
                        )
                    )
                    if center_recovery
                    else 0
                ),
                (
                    int(self.config.path_reversal_rate_limit)
                    if coherent_reversal
                    else 0
                ),
                (
                    int(self.config.steering_rate_limit)
                    if reversal_near_guard > 0.0
                    else 0
                ),
            ),
            fast_release=near_conflict or reversal_near_guard > 0.0,
        )
        if curve_guard:
            curve_limit = int(round(float(curve_guard_limit)))
            steering = self._clip(steering, -curve_limit, curve_limit)
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
        self._path_heading_lead = heading_lead
        preview_transition = (
            heading_preview_permission < 1.0
            and float(lane.heading_error) * float(near_error) < 0.0
        )
        self._path_state = (
            "curve_transition"
            if reversal_near_guard > 0.0 or preview_transition
            else self._classify_path_state(
                path_error,
                lane.heading_error,
                heading_lead,
            )
        )
        command = ControlCommand(
            speed=speed,
            steering=steering,
            brake=False,
            reason="path_tracking:whole_centerline:%s" % self._path_state,
        )
        self._last_command = command
        self._lane_lost_frames = 0
        return command

    def _path_center_recovery_strength(
        self,
        lane: LaneGeometry,
        near_error: float,
        heading_lead: float,
    ) -> float:
        reason = str(lane.reason)
        if not reason.startswith("corridor_tier1"):
            return 0.0
        if any(
            token in reason
            for token in ("lane_change", "crosswalk", "coast", "virtual")
        ):
            return 0.0
        threshold = max(
            0.0,
            float(self.config.path_center_recovery_error_threshold),
        )
        if (
            float(lane.confidence) < 0.75
            or abs(float(lane.heading_error))
            > max(
                0.0,
                float(self.config.path_center_recovery_heading_limit),
            )
            or float(heading_lead) >= 0.20
        ):
            return 0.0

        # A hard minimum-steering switch at ``threshold`` made a 0.054→0.072
        # near-error change produce a 34-unit steering jump in a successful
        # replay, even though the fitted path moved only 5 px.  Blend the
        # recovery over one threshold-width so perception noise cannot toggle
        # a discontinuous command, while a vehicle that is a full 2*threshold
        # off-center still receives the configured minimum immediately.
        span = max(threshold, 0.02)
        ratio = self._clip_float(
            (abs(float(near_error)) - threshold) / span,
            0.0,
            1.0,
        )
        return ratio * ratio * (3.0 - 2.0 * ratio)

    def _apply_path_center_recovery(
        self,
        steering: float,
        near_error: float,
        strength: float,
    ) -> float:
        minimum = max(
            0.0,
            float(self.config.path_center_recovery_min_steering),
        )
        direction = 1.0 if near_error >= 0.0 else -1.0
        if steering * direction > 0.0 and abs(steering) >= minimum:
            return steering
        recovery_target = direction * minimum
        blend = self._clip_float(float(strength), 0.0, 1.0)
        return steering + blend * (recovery_target - steering)

    def _path_direction_reversal_active(
        self,
        lane: LaneGeometry,
        path_error: float,
        raw_steering: float,
    ) -> bool:
        reason = str(lane.reason)
        if not reason.startswith("corridor_tier1"):
            return False
        if any(
            token in reason
            for token in ("lane_change", "crosswalk", "coast", "virtual")
        ):
            return False
        minimum_steering = max(
            0.0,
            float(self.config.path_reversal_min_steering),
        )
        if (
            raw_steering * float(self._last_steering) >= 0.0
            or abs(raw_steering) < minimum_steering
            or abs(float(self._last_steering)) < minimum_steering
        ):
            return False
        minimum_geometry = max(
            0.0,
            float(self.config.path_reversal_min_geometry),
        )
        coherent_path = (
            abs(float(path_error)) >= minimum_geometry
            and raw_steering * float(path_error) > 0.0
        )
        coherent_heading = (
            abs(float(lane.heading_error)) >= minimum_geometry
            and raw_steering * float(lane.heading_error) > 0.0
        )
        return coherent_path or coherent_heading

    def _path_reversal_near_guard_strength(
        self,
        lane: LaneGeometry,
        near_error: float,
        raw_steering: float,
    ) -> float:
        """Phase an S-curve reversal using the near-field centerline."""
        reason = str(lane.reason)
        previous = float(self._last_steering)
        heading = float(lane.heading_error)
        far_error = float(lane.lateral_error_norm)
        minimum_geometry = max(
            0.0,
            float(self.config.path_reversal_min_geometry),
        )
        if (
            not reason.startswith("corridor_tier1")
            or any(
                token in reason
                for token in ("lane_change", "crosswalk", "coast", "virtual")
            )
            or float(lane.confidence) < 0.75
            # Do not drop the near-field veto merely because the old command
            # has almost unwound to zero.  At full speed that opened a short
            # window where the far S-curve preview could start the opposite
            # turn while the car was still displaced toward the outer line.
            # The guarded near-error band already rejects zero-crossing noise.
            or abs(previous) < 1.0
            or raw_steering * previous >= 0.0
            or raw_steering * float(near_error) >= 0.0
            or abs(heading) < minimum_geometry
            or raw_steering * heading <= 0.0
            or (
                abs(far_error) >= minimum_geometry
                and heading * far_error <= 0.0
            )
        ):
            return 0.0

        guard_error = max(
            0.0,
            float(self.config.path_reversal_near_guard_error),
        )
        full_error = max(
            guard_error + 1e-6,
            float(self.config.path_reversal_near_full_error),
        )
        magnitude = abs(float(near_error))
        if magnitude <= guard_error:
            return 0.0
        ratio = self._clip_float(
            (magnitude - guard_error) / (full_error - guard_error),
            0.0,
            1.0,
        )
        return ratio * ratio * (3.0 - 2.0 * ratio)

    def _path_near_conflict_strength(
        self,
        lane: LaneGeometry,
        near_error: float,
        raw_steering: float,
    ) -> float:
        reason = str(lane.reason)
        if not reason.startswith("corridor_tier1"):
            return 0.0
        if any(
            token in reason
            for token in ("lane_change", "crosswalk", "coast", "virtual")
        ):
            return 0.0
        threshold = max(
            0.0,
            float(self.config.path_near_conflict_error_threshold),
        )
        magnitude = abs(float(near_error))
        if (
            float(lane.confidence) < 0.75
            or magnitude < threshold
            or float(near_error) * float(raw_steering) >= 0.0
        ):
            return 0.0

        heading_limit = max(
            0.0,
            float(self.config.path_near_conflict_heading_limit),
        )
        heading = abs(float(lane.heading_error))
        if heading_limit <= 1e-6:
            heading_ratio = 0.0 if heading <= 1e-6 else 1.0
        else:
            heading_ratio = self._clip_float(
                heading / heading_limit,
                0.0,
                1.0,
            )
            heading_ratio = heading_ratio * heading_ratio * (
                3.0 - 2.0 * heading_ratio
            )

        error_span = max(threshold, 0.02)
        error_ratio = self._clip_float(
            (magnitude - threshold) / error_span,
            0.0,
            1.0,
        )
        error_strength = error_ratio * error_ratio * (
            3.0 - 2.0 * error_ratio
        )

        curve_start = max(
            threshold,
            float(self.config.path_curve_guard_near_error),
        )
        curve_release = max(
            curve_start + 1e-6,
            float(self.config.path_curve_guard_release_error),
        )
        curve_ratio = self._clip_float(
            (magnitude - curve_start) / (curve_release - curve_start),
            0.0,
            1.0,
        )
        curve_strength = curve_ratio * curve_ratio * (
            3.0 - 2.0 * curve_ratio
        )

        # Small near errors must not delay a genuine curve entry. Once the car
        # has crossed far enough toward the boundary, continuously hand control
        # back to the near field even when the far heading is still large.
        return (
            (1.0 - heading_ratio) * error_strength
            + heading_ratio * curve_strength
        )

    def _path_curve_guard_limit(
        self,
        lane: LaneGeometry,
        path_error: float,
        near_error: float,
        raw_steering: float,
    ) -> Optional[float]:
        reason = str(lane.reason)
        if not reason.startswith("corridor_tier1"):
            return None
        if any(
            token in reason
            for token in ("lane_change", "crosswalk", "coast", "virtual")
        ):
            return None
        heading = float(lane.heading_error)
        if (
            float(lane.confidence) < 0.80
            or abs(heading)
            < max(
                0.0,
                float(self.config.path_curve_guard_heading_threshold),
            )
            or heading * float(path_error) <= 0.0
            or heading * float(raw_steering) <= 0.0
        ):
            return None

        full_guard_error = max(
            0.0,
            float(self.config.path_curve_guard_near_error),
        )
        release_error = max(
            full_guard_error + 1e-6,
            float(self.config.path_curve_guard_release_error),
        )
        magnitude = abs(float(near_error))
        if magnitude >= release_error:
            return None

        blend = self._clip_float(
            (magnitude - full_guard_error)
            / (release_error - full_guard_error),
            0.0,
            1.0,
        )
        blend = blend * blend * (3.0 - 2.0 * blend)
        guarded_limit = max(
            0.0,
            float(self.config.path_curve_guard_steering_limit),
        )
        maximum = max(guarded_limit, float(self.config.max_steering))
        return guarded_limit + (maximum - guarded_limit) * blend

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
            near_weight = max(0.0, float(self.config.path_near_weight))
            far_weight = max(0.0, float(self.config.path_far_weight))
            weight_sum = near_weight + far_weight
            if weight_sum <= 1e-6:
                return float(lane.lateral_error_norm)
            return self._clip_float(
                (
                    near_weight * float(near)
                    + far_weight * float(lane.lateral_error_norm)
                )
                / weight_sum,
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

    def _path_heading_lead_strength(
        self,
        lane: LaneGeometry,
        path_error: float,
    ) -> float:
        """Measure how far path direction leads near-field displacement."""
        heading = float(lane.heading_error)
        near_error = (
            float(lane.near_lateral_error_norm)
            if lane.near_lateral_error_norm is not None
            else float(path_error)
        )
        reference = 0.70 * near_error + 0.30 * float(path_error)
        if abs(heading) <= 1e-6:
            return 0.0

        # A near/far sign split is normal at an S-curve transition. Preserve
        # heading preview when the complete path endpoint and heading agree;
        # otherwise treat the split as incoherent geometry.
        alignment = 1.0
        if abs(reference) > 0.015 and heading * reference < 0.0:
            far_error = float(lane.lateral_error_norm)
            minimum_geometry = max(
                0.015,
                float(self.config.path_reversal_min_geometry),
            )
            coherent_far_curve = (
                abs(heading)
                >= max(
                    minimum_geometry,
                    float(self.config.path_near_conflict_heading_limit),
                )
                and abs(far_error) >= minimum_geometry
                and heading * far_error > 0.0
            )
            if not coherent_far_curve:
                alignment = 0.0

        span = max(1e-6, float(self.config.path_heading_lead_span))
        lead = max(0.0, abs(heading) - abs(reference)) / span
        confidence = self._clip_float(float(lane.confidence), 0.0, 1.0)
        return self._clip_float(lead * alignment * confidence, 0.0, 1.0)

    def _path_heading_lead_gain(
        self,
        lane: LaneGeometry,
        path_error: float,
        near_error: float,
        heading_lead: float,
    ) -> float:
        """Boost only a high-confidence curve whose near/far signs agree."""
        base = float(self.config.path_heading_lead_gain)
        maximum = max(
            base,
            float(self.config.path_heading_lead_coherent_gain),
        )
        reason = str(lane.reason)
        heading = float(lane.heading_error)
        far_error = float(lane.lateral_error_norm)
        if (
            not reason.startswith("corridor_tier1")
            or any(
                token in reason
                for token in ("lane_change", "crosswalk", "coast", "virtual")
            )
            or float(lane.confidence) < 0.85
            or abs(heading) <= 0.015
            or abs(far_error) <= 0.015
            or heading * far_error <= 0.0
            or (
                abs(float(near_error)) > 0.06
                and heading * float(near_error) <= 0.0
            )
        ):
            return base
        strength = self._clip_float(
            (float(heading_lead) - 0.25) / 0.40,
            0.0,
            1.0,
        )
        strength = strength * strength * (3.0 - 2.0 * strength)
        return base + (maximum - base) * strength

    def _path_heading_preview_permission(
        self,
        lane: LaneGeometry,
        near_error: float,
    ) -> float:
        """Continuously phase an S-turn when near and far geometry disagree."""
        reason = str(lane.reason)
        heading = float(lane.heading_error)
        far_error = float(lane.lateral_error_norm)
        near_error = float(near_error)
        if (
            not reason.startswith("corridor_tier1")
            or any(
                token in reason
                for token in ("lane_change", "crosswalk", "coast", "virtual")
            )
            or float(lane.confidence) < 0.75
            or abs(near_error) <= 0.015
            or heading * near_error >= 0.0
        ):
            return 1.0
        if abs(far_error) <= 0.015 or heading * far_error <= 0.0:
            return 0.0

        # During an S transition the far path changes sign before the path
        # beside the vehicle.  Let heading preview take over only as the new
        # far displacement becomes larger than the still-opposite near
        # displacement.  A smooth 1.25x→2x dominance band avoids both the early
        # two-frame reversal seen in replay and a new binary steering jump.
        dominance = abs(far_error) / max(abs(near_error), 1e-6)
        ratio = self._clip_float(
            (dominance - 1.25) / 0.75,
            0.0,
            1.0,
        )
        return ratio * ratio * (3.0 - 2.0 * ratio)

    def _path_integral_correction(
        self,
        lane: LaneGeometry,
        path_error: float,
        heading_lead: float,
    ) -> float:
        reason = str(lane.reason)
        state = self._classify_path_state(
            path_error,
            float(lane.heading_error),
            heading_lead,
        )
        reliable_straight = (
            state == "straight"
            and reason.startswith("corridor_tier1")
            and all(
                token not in reason
                for token in ("lane_change", "crosswalk", "coast", "virtual")
            )
            and float(lane.confidence) >= 0.80
        )
        limit = max(0.0, float(self.config.path_integral_limit))
        if reliable_straight and limit > 0.0:
            if self._path_error_integral * path_error < 0.0:
                self._path_error_integral *= 0.5
            self._path_error_integral = self._clip_float(
                self._path_error_integral + float(path_error),
                -limit,
                limit,
            )
        else:
            decay = self._clip_float(
                float(self.config.path_integral_decay),
                0.0,
                1.0,
            )
            self._path_error_integral *= decay
            if abs(self._path_error_integral) < 1e-4:
                self._path_error_integral = 0.0
        return (
            float(self.config.path_integral_gain)
            * self._path_error_integral
        )

    @staticmethod
    def _classify_path_state(
        path_error: float,
        heading_error: float,
        heading_lead: float,
    ) -> str:
        if heading_lead >= 0.20:
            return "curve_entry"
        if max(abs(path_error), abs(heading_error)) >= 0.10:
            return "curve_hold"
        return "straight"

    def _path_steering_alpha(
        self,
        raw_steering: float,
        reason: str,
        heading_lead: float = 0.0,
    ) -> float:
        previous = float(self._last_steering)
        same_direction = raw_steering == 0.0 or raw_steering * previous >= 0.0
        increasing = same_direction and abs(raw_steering) >= abs(previous)
        direction_reversal = raw_steering * previous < 0.0
        if increasing or direction_reversal:
            alpha = self.config.path_steering_rise_alpha
        else:
            alpha = self.config.path_steering_release_alpha
        if heading_lead > 0.0:
            alpha = max(
                alpha,
                self.config.path_steering_release_alpha
                + (
                    self.config.path_steering_rise_alpha
                    - self.config.path_steering_release_alpha
                )
                * heading_lead,
            )
        if ":lane_change" in reason:
            alpha = max(alpha, 0.65)
        return self._clip_float(alpha, 0.0, 1.0)

    def accept_applied_command(
        self,
        command: ControlCommand,
        running: bool = True,
    ) -> None:
        """Synchronize filter state with the steering sent to the vehicle."""
        if not running:
            self.reset()
            return
        if command.brake:
            self._path_error_integral = 0.0
        self._last_steering = int(command.steering)

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
        self._path_state = "straight"
        self._path_heading_lead = 0.0
        self._path_error_integral = 0.0

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

    def _rate_limit(
        self,
        steering: int,
        curve_strength: float,
        recovery_strength: float,
        minimum_rate_limit: int = 0,
        fast_release: bool = False,
    ) -> int:
        delta = steering - self._last_steering
        min_limit = min(self.config.min_steering_rate_limit, self.config.steering_rate_limit)
        max_limit = max(self.config.min_steering_rate_limit, self.config.steering_rate_limit)
        limit = int(round(min_limit + (max_limit - min_limit) * curve_strength))
        limit = max(limit, max(0, int(minimum_rate_limit)))
        if recovery_strength > 0.0:
            recovery_limit = int(round(
                limit + (self.config.center_recovery_rate_limit - limit) * recovery_strength
            ))
            limit = max(limit, recovery_limit)
        if self._is_releasing_steering(steering) and not fast_release:
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
