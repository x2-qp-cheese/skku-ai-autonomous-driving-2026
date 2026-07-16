from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from ..estimation.lane_geometry import LaneGeometry
from ..sensors.ultrasonic import UltrasonicSnapshot
from ..types import ControlCommand
from .lane_change import LaneChangeController


@dataclass(frozen=True)
class ObstacleFusionConfig:
    enabled: bool = True
    fusion_mode: str = "fused"  # fused=YOLO+ultrasonic, yolo=video replay
    lane_width_px: float = 150.0

    # BEV is reliable near the car, while frame-space masks see obstacles before
    # they enter the ground-plane homography. Both assessments are used.
    visual_trigger_y_ratio: float = 0.05
    target_block_y_ratio: float = 0.20
    frame_visual_trigger_y_ratio: float = 0.18
    frame_target_block_y_ratio: float = 0.20
    visual_emergency_y_ratio: float = 0.88
    frame_visual_emergency_y_ratio: float = 0.72
    path_half_width_px: float = 65.0
    frame_path_half_width_scale: float = 0.42
    frame_min_path_half_width_px: float = 12.0
    min_path_overlap_ratio: float = 0.15
    contact_band_ratio: float = 0.25
    visual_action_confidence: float = 0.75
    visual_confirm_frames: int = 2
    visual_clear_frames: int = 2

    # The firmware sees roughly 2 m. Start the maneuver on the first stable
    # long-range echo rather than waiting until the old 1 m threshold.
    ultrasonic_trigger_mm: float = 2000.0
    ultrasonic_clear_mm: float = 2300.0
    ultrasonic_stop_mm: float = 300.0
    blocked_stop_mm: float = 650.0
    min_front_sensors: int = 2
    range_confirm_frames: int = 1
    range_clear_frames: int = 2
    rearm_clear_frames: int = 3
    ttc_trigger_seconds: float = 1.8
    min_closing_rate_mm_s: float = 120.0
    side_clearance_mm: float = 300.0

    # lane-side normally represents the outer solid boundaries. Only a solid
    # instance lying between the two lane-center trajectories blocks a change;
    # the valid outer boundaries of the current and destination lanes do not.
    solid_crossing_margin_px: float = 8.0
    solid_check_min_y_ratio: float = 0.20
    solid_min_overlap_ratio: float = 0.05

    approach_speed_cap: int = 120
    speed_cap: int = 120
    cooldown_seconds: float = 0.4


@dataclass(frozen=True)
class FramePathGeometry:
    """Lane-center trajectories projected back into camera-frame pixels."""

    lane1: Tuple[Tuple[float, float], ...] = ()
    lane2: Tuple[Tuple[float, float], ...] = ()

    def line(self, lane_index: int) -> Tuple[Tuple[float, float], ...]:
        return self.lane1 if lane_index == 1 else self.lane2


@dataclass(frozen=True)
class ObstacleFusionObservation:
    visual_detected: bool = False
    visual_actionable: bool = False
    visual_confidence: float = 0.0
    visual_confirmed: bool = False
    range_confirmed: bool = False
    fused_hazard: bool = False
    target_blocked: bool = False
    solid_blocked: bool = False
    side_clear: bool = False
    emergency: bool = False
    path_lane: int = 2
    closest_y_ratio: float = 0.0
    frame_y_ratio: float = 0.0
    obstacle_count: int = 0
    visual_frames: int = 0
    front_mm: Optional[int] = None
    front_sensor_count: int = 0
    range_frames: int = 0
    closing_rate_mm_s: float = 0.0
    ttc_seconds: Optional[float] = None


@dataclass(frozen=True)
class PathOccupancy:
    bottom_y_ratio: float
    current_overlap: float
    target_overlap: float


@dataclass(frozen=True)
class PathAssessment:
    current_detected: bool = False
    target_blocked: bool = False
    closest_y_ratio: float = 0.0
    obstacle_count: int = 0


class ObstacleFusionPlanner:
    """Early YOLO tracking followed by range-confirmed lane-change commitment.

    Frame-space YOLO masks provide lookahead before an obstacle enters the BEV
    source trapezoid. BEV masks still provide the near-field path association.
    Ultrasonic range/TTC confirms that the visual object is physically close,
    while side sonar, target-lane occupancy and solid boundaries veto unsafe
    changes. The lane-change controller owns full-lane arrival and alignment.
    """

    def __init__(self, config: ObstacleFusionConfig = ObstacleFusionConfig()):
        self.config = config
        self.observation = ObstacleFusionObservation()
        self._path_lane = 2
        self._visual_frames = 0
        self._clear_frames = 0
        self._visual_confirmed = False
        self._range_hazard = False
        self._range_frames = 0
        self._range_clear_frames = 0
        self._rearm_frames = 0
        self._consumed = False
        self._last_trigger_at = -1e9
        self._last_front_mm: Optional[int] = None
        self._last_front_at: Optional[float] = None
        self._closing_rate_mm_s = 0.0
        self._ttc_seconds: Optional[float] = None

    def reset(self) -> None:
        self.observation = ObstacleFusionObservation(path_lane=self._path_lane)
        self._visual_frames = 0
        self._clear_frames = 0
        self._visual_confirmed = False
        self._range_hazard = False
        self._range_frames = 0
        self._range_clear_frames = 0
        self._rearm_frames = 0
        self._consumed = False
        self._last_front_mm = None
        self._last_front_at = None
        self._closing_rate_mm_s = 0.0
        self._ttc_seconds = None

    def update(
        self,
        obstacle_masks: Sequence[Any],
        bev_shape: Tuple[int, int],
        base_centerline: Sequence[Tuple[float, float]],
        lane: LaneGeometry,
        lane_change: LaneChangeController,
        ultrasonic: UltrasonicSnapshot,
        now: float,
        running: bool,
        frame_obstacle_masks: Sequence[Any] = (),
        frame_paths: Optional[FramePathGeometry] = None,
        solid_masks: Sequence[Any] = (),
        obstacle_confidence: float = 1.0,
    ) -> Optional[str]:
        path_lane = self._desired_lane(lane_change.state)
        if not self.config.enabled:
            self.observation = ObstacleFusionObservation(path_lane=path_lane)
            return None

        if path_lane != self._path_lane:
            consumed = self._consumed
            self._path_lane = path_lane
            self.reset()
            # A path switch is part of the same avoidance event. Keep it
            # consumed until the new lane is stable and both sensors are clear,
            # otherwise the old obstacle can be interpreted as an immediate
            # request to return to the lane that was just vacated.
            self._consumed = consumed

        bev_assessment = self._measure_bev_paths(
            obstacle_masks,
            bev_shape,
            base_centerline,
            lane,
            path_lane,
        )
        frame_assessment = self._measure_frame_paths(
            frame_obstacle_masks,
            frame_paths,
            path_lane,
        )
        raw_visual = (
            bev_assessment.current_detected
            or frame_assessment.current_detected
        )
        visual_confidence = max(0.0, min(1.0, float(obstacle_confidence)))
        visual = (
            raw_visual
            and visual_confidence
            >= max(0.0, float(self.config.visual_action_confidence))
        )
        target_blocked = (
            bev_assessment.target_blocked
            or frame_assessment.target_blocked
        )
        solid_blocked = self._solid_boundary_blocked(
            solid_masks,
            bev_shape,
            base_centerline,
            lane,
            path_lane,
        )
        maneuver_active = self._lane_change_active(lane_change.state)
        side_direction = self._destination_side_direction(
            lane_change.state,
            path_lane,
        )
        side_clear = (
            self.config.fusion_mode == "yolo"
            or ultrasonic.side_clear(side_direction, self.config.side_clearance_mm)
        )
        front_mm = ultrasonic.front_min_mm
        front_count = ultrasonic.front_fresh_count

        if not running:
            self.reset()
            self.observation = ObstacleFusionObservation(
                visual_detected=raw_visual,
                visual_actionable=visual,
                visual_confidence=visual_confidence,
                target_blocked=target_blocked,
                solid_blocked=solid_blocked,
                side_clear=side_clear,
                path_lane=path_lane,
                closest_y_ratio=bev_assessment.closest_y_ratio,
                frame_y_ratio=frame_assessment.closest_y_ratio,
                obstacle_count=max(
                    bev_assessment.obstacle_count,
                    frame_assessment.obstacle_count,
                ),
                front_mm=front_mm,
                front_sensor_count=front_count,
            )
            return None

        stable_lane = lane_change.state in ("lane2", "completed", "lane1")
        self._update_visual_state(visual)
        self._update_range_state(ultrasonic, now)
        self._update_rearm_state(raw_visual, stable_lane)
        range_confirmed = (
            self.config.fusion_mode == "yolo" or self._range_hazard
        )
        fused_hazard = self._visual_confirmed and range_confirmed
        blocked = target_blocked or solid_blocked or not side_clear
        emergency = self._emergency_present(
            raw_visual,
            bev_assessment.closest_y_ratio,
            frame_assessment.closest_y_ratio,
            ultrasonic,
            fused_hazard,
            blocked,
            maneuver_active,
        )
        self.observation = ObstacleFusionObservation(
            visual_detected=raw_visual,
            visual_actionable=visual,
            visual_confidence=visual_confidence,
            visual_confirmed=self._visual_confirmed,
            range_confirmed=range_confirmed,
            fused_hazard=fused_hazard,
            target_blocked=target_blocked,
            solid_blocked=solid_blocked,
            side_clear=side_clear,
            emergency=emergency,
            path_lane=path_lane,
            closest_y_ratio=bev_assessment.closest_y_ratio,
            frame_y_ratio=frame_assessment.closest_y_ratio,
            obstacle_count=max(
                bev_assessment.obstacle_count,
                frame_assessment.obstacle_count,
            ),
            visual_frames=self._visual_frames,
            front_mm=front_mm,
            front_sensor_count=front_count,
            range_frames=self._range_frames,
            closing_rate_mm_s=self._closing_rate_mm_s,
            ttc_seconds=self._ttc_seconds,
        )

        if (
            fused_hazard
            and stable_lane
            and not self._consumed
            and not blocked
            and now - self._last_trigger_at >= max(0.0, self.config.cooldown_seconds)
        ):
            event = self._request_lane_change(lane_change, now)
            if event is not None:
                self._consumed = True
                return event
        return None

    def apply_safety(
        self,
        command: ControlCommand,
        lane_change_state: str,
        running: bool,
    ) -> ControlCommand:
        if not self.config.enabled or not running or command.brake:
            return command

        if self.observation.emergency:
            return ControlCommand.stop(
                self._append_reason(command.reason, "obstacle_fusion_stop")
            )

        active = self._lane_change_active(lane_change_state)
        if active or self.observation.fused_hazard:
            cap = max(0, int(self.config.speed_cap))
        elif self.observation.visual_detected:
            cap = max(0, int(self.config.approach_speed_cap))
        else:
            return command
        speed = max(-cap, min(cap, command.speed))
        if speed == command.speed:
            return command
        return ControlCommand(
            speed=speed,
            steering=command.steering,
            brake=False,
            reason=self._append_reason(command.reason, "obstacle_fusion_cap"),
        )

    def status_text(self) -> str:
        if not self.config.enabled:
            return "off"
        obs = self.observation
        front = "n/a" if obs.front_mm is None else str(obs.front_mm)
        ttc = "n/a" if obs.ttc_seconds is None else "%.1f" % obs.ttc_seconds
        if obs.emergency:
            state = "STOP"
        elif obs.fused_hazard and obs.solid_blocked:
            state = "SOLID_BLOCKED"
        elif obs.fused_hazard and obs.target_blocked:
            state = "TARGET_BLOCKED"
        elif obs.fused_hazard and not obs.side_clear:
            state = "SIDE_BLOCKED"
        elif obs.fused_hazard:
            state = "READY"
        elif obs.visual_confirmed:
            state = "TRACKING"
        elif obs.visual_detected and not obs.visual_actionable:
            state = "LOW_CONF"
        elif obs.visual_detected:
            state = "VISION_%d/%d" % (
                obs.visual_frames,
                max(1, int(self.config.visual_confirm_frames)),
            )
        elif self._consumed:
            state = "WAIT_NEW"
        else:
            state = "clear"
        return "L%d %s by=%.2f fy=%.2f conf=%.2f front=%s q=%d r=%d ttc=%s side=%s" % (
            obs.path_lane,
            state,
            obs.closest_y_ratio,
            obs.frame_y_ratio,
            obs.visual_confidence,
            front,
            obs.front_sensor_count,
            obs.range_frames,
            ttc,
            "clear" if obs.side_clear else "blocked/unknown",
        )

    def _update_visual_state(self, visual: bool) -> None:
        if visual:
            self._visual_frames += 1
            self._clear_frames = 0
        else:
            self._visual_frames = 0
            self._clear_frames += 1
        if self._visual_frames >= max(1, int(self.config.visual_confirm_frames)):
            self._visual_confirmed = True
        if self._clear_frames >= max(1, int(self.config.visual_clear_frames)):
            self._visual_confirmed = False

    def _update_rearm_state(self, raw_visual: bool, stable_lane: bool) -> None:
        if not self._consumed:
            self._rearm_frames = 0
            return
        if stable_lane and not raw_visual and not self._range_hazard:
            self._rearm_frames += 1
        else:
            self._rearm_frames = 0
        if self._rearm_frames >= max(1, int(self.config.rearm_clear_frames)):
            self._consumed = False
            self._rearm_frames = 0

    def _update_range_state(
        self,
        ultrasonic: UltrasonicSnapshot,
        now: float,
    ) -> None:
        required = max(1, int(self.config.min_front_sensors))
        distance = ultrasonic.front_min_mm
        if not ultrasonic.front_ready(required) or distance is None:
            self._range_frames = 0
            self._range_clear_frames += 1
            if self._range_clear_frames >= max(
                1, int(self.config.range_clear_frames)
            ):
                self._range_hazard = False
            self._ttc_seconds = None
            return

        if self._last_front_mm is not None and self._last_front_at is not None:
            dt = now - self._last_front_at
            if 0.02 <= dt <= 1.0:
                raw_rate = (float(self._last_front_mm) - float(distance)) / dt
                raw_rate = max(0.0, raw_rate)
                self._closing_rate_mm_s = (
                    0.5 * raw_rate + 0.5 * self._closing_rate_mm_s
                )
        self._last_front_mm = int(distance)
        self._last_front_at = float(now)

        minimum_rate = max(1.0, float(self.config.min_closing_rate_mm_s))
        if self._closing_rate_mm_s >= minimum_rate:
            self._ttc_seconds = float(distance) / self._closing_rate_mm_s
        else:
            self._ttc_seconds = None

        ttc_hazard = (
            self._ttc_seconds is not None
            and self._ttc_seconds <= max(0.0, self.config.ttc_trigger_seconds)
        )
        range_candidate = (
            distance <= max(0.0, self.config.ultrasonic_trigger_mm)
            or ttc_hazard
        )
        range_clear = (
            distance >= max(
                self.config.ultrasonic_trigger_mm,
                self.config.ultrasonic_clear_mm,
            )
            and not ttc_hazard
        )
        if range_candidate:
            self._range_frames += 1
            self._range_clear_frames = 0
        elif range_clear:
            self._range_frames = 0
            self._range_clear_frames += 1
        else:
            self._range_frames = 0
            self._range_clear_frames = 0

        if not self._range_hazard and self._range_frames >= max(
            1, int(self.config.range_confirm_frames)
        ):
            self._range_hazard = True
        if self._range_hazard and self._range_clear_frames >= max(
            1, int(self.config.range_clear_frames)
        ):
            self._range_hazard = False

    def _emergency_present(
        self,
        visual: bool,
        closest_bev_y: float,
        closest_frame_y: float,
        ultrasonic: UltrasonicSnapshot,
        fused_hazard: bool,
        blocked: bool,
        maneuver_active: bool,
    ) -> bool:
        visual_close = visual and (
            closest_bev_y >= self.config.visual_emergency_y_ratio
            or closest_frame_y >= self.config.frame_visual_emergency_y_ratio
        )
        required = max(1, int(self.config.min_front_sensors))
        if not ultrasonic.front_ready(required):
            return visual_close

        close_count = ultrasonic.front_close_count(self.config.ultrasonic_stop_mm)
        # Once committed, front sonar may keep seeing the obstacle in the lane
        # being vacated. Do not stop halfway across a clear destination path for
        # that unassociated echo; a current-path YOLO mask still stops the car.
        if maneuver_active and not visual:
            range_close = False
        else:
            range_close = close_count >= 2 or (visual and close_count >= 1)
        distance = ultrasonic.front_min_mm
        blocked_close = (
            fused_hazard
            and blocked
            and distance is not None
            and distance <= max(0.0, self.config.blocked_stop_mm)
        )
        return visual_close or range_close or blocked_close

    def _measure_bev_paths(
        self,
        obstacle_masks: Sequence[Any],
        bev_shape: Tuple[int, int],
        base_centerline: Sequence[Tuple[float, float]],
        lane: LaneGeometry,
        path_lane: int,
    ) -> PathAssessment:
        import numpy as np

        height, width = bev_shape
        if height <= 0 or width <= 0:
            return PathAssessment()

        current_offset = self._lane_offset(path_lane)
        target_offset = self._lane_offset(1 if path_lane == 2 else 2)
        measurements = []
        for mask in obstacle_masks:
            binary = np.asarray(mask) > 0
            if binary.shape[:2] != (height, width):
                continue
            measurement = self._measure_mask(
                binary,
                base_centerline,
                lane.center_x,
                current_offset,
                target_offset,
                max(1.0, float(self.config.path_half_width_px)),
            )
            if measurement is not None:
                measurements.append(measurement)

        return self._assess_measurements(
            measurements,
            self.config.visual_trigger_y_ratio,
            self.config.target_block_y_ratio,
        )

    def _measure_frame_paths(
        self,
        obstacle_masks: Sequence[Any],
        frame_paths: Optional[FramePathGeometry],
        path_lane: int,
    ) -> PathAssessment:
        import numpy as np

        if frame_paths is None:
            return PathAssessment()
        current_line = frame_paths.line(path_lane)
        target_line = frame_paths.line(1 if path_lane == 2 else 2)
        if len(current_line) < 2 or len(target_line) < 2:
            return PathAssessment()

        measurements = []
        for mask in obstacle_masks:
            binary = np.asarray(mask) > 0
            if binary.ndim != 2 or not binary.any():
                continue
            ys, xs = np.nonzero(binary)
            min_y = int(ys.min())
            max_y = int(ys.max())
            span = max(1, max_y - min_y + 1)
            contact_start = max(
                min_y,
                int(round(max_y - span * self.config.contact_band_ratio)),
            )
            contact = ys >= contact_start
            contact_ys = ys[contact].astype(float)
            contact_xs = xs[contact].astype(float)
            if len(contact_ys) == 0:
                continue
            current_x = self._center_x_at(
                contact_ys,
                current_line,
                float(binary.shape[1]) / 2.0,
            )
            target_x = self._center_x_at(
                contact_ys,
                target_line,
                float(binary.shape[1]) / 2.0,
            )
            lane_spacing = np.abs(current_x - target_x)
            half_width = np.maximum(
                max(1.0, self.config.frame_min_path_half_width_px),
                lane_spacing * max(0.0, self.config.frame_path_half_width_scale),
            )
            measurements.append(
                PathOccupancy(
                    bottom_y_ratio=max_y / float(max(1, binary.shape[0] - 1)),
                    current_overlap=float(
                        np.mean(np.abs(contact_xs - current_x) <= half_width)
                    ),
                    target_overlap=float(
                        np.mean(np.abs(contact_xs - target_x) <= half_width)
                    ),
                )
            )

        return self._assess_measurements(
            measurements,
            self.config.frame_visual_trigger_y_ratio,
            self.config.frame_target_block_y_ratio,
        )

    def _solid_boundary_blocked(
        self,
        solid_masks: Sequence[Any],
        bev_shape: Tuple[int, int],
        base_centerline: Sequence[Tuple[float, float]],
        lane: LaneGeometry,
        path_lane: int,
    ) -> bool:
        import numpy as np

        height, width = bev_shape
        if height <= 0 or width <= 0 or not solid_masks:
            return False
        current_offset = self._lane_offset(path_lane)
        target_offset = self._lane_offset(1 if path_lane == 2 else 2)
        crossing_margin = max(0.0, float(self.config.solid_crossing_margin_px))
        min_y = height * max(0.0, self.config.solid_check_min_y_ratio)
        minimum_overlap = max(
            0.0,
            min(1.0, self.config.solid_min_overlap_ratio),
        )

        for mask in solid_masks:
            binary = np.asarray(mask) > 0
            if binary.shape[:2] != (height, width):
                continue
            ys, xs = np.nonzero(binary)
            keep = ys >= min_y
            ys = ys[keep].astype(float)
            xs = xs[keep].astype(float)
            if len(ys) < 20:
                continue
            base_x = self._center_x_at(ys, base_centerline, lane.center_x)
            current_x = base_x + current_offset
            target_x = base_x + target_offset
            crossing_left = np.minimum(current_x, target_x) - crossing_margin
            crossing_right = np.maximum(current_x, target_x) + crossing_margin
            overlap = float(
                np.mean((xs >= crossing_left) & (xs <= crossing_right))
            )
            if overlap >= minimum_overlap:
                return True
        return False

    def _measure_mask(
        self,
        binary: Any,
        base_centerline: Sequence[Tuple[float, float]],
        fallback_x: float,
        current_offset: float,
        target_offset: float,
        half_width: float,
    ) -> Optional[PathOccupancy]:
        import numpy as np

        ys, xs = np.nonzero(binary)
        if len(ys) == 0:
            return None
        min_y = int(ys.min())
        max_y = int(ys.max())
        span = max(1, max_y - min_y + 1)
        contact_start = max(
            min_y,
            int(round(max_y - span * self.config.contact_band_ratio)),
        )
        contact = ys >= contact_start
        contact_ys = ys[contact].astype(float)
        contact_xs = xs[contact].astype(float)
        if len(contact_ys) == 0:
            return None
        base_x = self._center_x_at(contact_ys, base_centerline, fallback_x)
        return PathOccupancy(
            bottom_y_ratio=max_y / float(max(1, binary.shape[0] - 1)),
            current_overlap=float(
                np.mean(
                    np.abs(contact_xs - (base_x + current_offset))
                    <= half_width
                )
            ),
            target_overlap=float(
                np.mean(
                    np.abs(contact_xs - (base_x + target_offset))
                    <= half_width
                )
            ),
        )

    def _assess_measurements(
        self,
        measurements: Sequence[PathOccupancy],
        current_y_threshold: float,
        target_y_threshold: float,
    ) -> PathAssessment:
        overlap_min = max(0.0, min(1.0, self.config.min_path_overlap_ratio))
        current = [
            item
            for item in measurements
            if item.bottom_y_ratio >= current_y_threshold
            and item.current_overlap >= overlap_min
        ]
        target_blocked = any(
            item.bottom_y_ratio >= target_y_threshold
            and item.target_overlap >= overlap_min
            for item in measurements
        )
        return PathAssessment(
            current_detected=bool(current),
            target_blocked=target_blocked,
            closest_y_ratio=max(
                (item.bottom_y_ratio for item in current),
                default=0.0,
            ),
            obstacle_count=len(measurements),
        )

    def _request_lane_change(
        self,
        lane_change: LaneChangeController,
        now: float,
    ) -> Optional[str]:
        source = "obstacle_fusion"
        if lane_change.state in ("lane2", "completed"):
            accepted = lane_change.request_avoidance(source)
            direction = "lane2 -> lane1"
        elif lane_change.state == "lane1":
            accepted = lane_change.request_avoidance_return(source)
            direction = "lane1 -> lane2"
        else:
            return None
        if not accepted:
            return None
        self._last_trigger_at = now
        return "obstacle fusion: %s requested (%s)" % (
            direction,
            self.status_text(),
        )

    def _lane_offset(self, lane_index: int) -> float:
        return (
            -max(0.0, float(self.config.lane_width_px))
            if lane_index == 1
            else 0.0
        )

    @staticmethod
    def _desired_lane(state: str) -> int:
        if state in ("changing_to_lane1", "stabilizing_lane1", "lane1"):
            return 1
        return 2

    @staticmethod
    def _center_x_at(
        ys: Any,
        centerline: Sequence[Tuple[float, float]],
        fallback_x: float,
    ) -> Any:
        import numpy as np

        if len(centerline) < 2:
            return np.full_like(ys, float(fallback_x), dtype=float)
        points = sorted(centerline, key=lambda point: point[1])
        line_y = np.asarray([point[1] for point in points], dtype=float)
        line_x = np.asarray([point[0] for point in points], dtype=float)
        return np.interp(ys, line_y, line_x)

    @staticmethod
    def _lane_change_active(state: str) -> bool:
        return state in (
            "armed",
            "changing_to_lane1",
            "stabilizing_lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        )

    @staticmethod
    def _destination_side_direction(state: str, path_lane: int) -> int:
        if state in ("armed", "changing_to_lane1", "stabilizing_lane1"):
            return -1
        if state in ("changing_to_lane2", "stabilizing_lane2"):
            return 1
        return -1 if path_lane == 2 else 1

    @staticmethod
    def _append_reason(reason: str, suffix: str) -> str:
        return "%s:%s" % (reason, suffix) if reason else suffix
