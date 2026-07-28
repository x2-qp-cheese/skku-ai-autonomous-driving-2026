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
    frame_visual_trigger_y_ratio: float = 0.18
    visual_emergency_y_ratio: float = 0.88
    frame_visual_emergency_y_ratio: float = 0.72
    path_half_width_px: float = 65.0
    frame_path_half_width_scale: float = 0.42
    frame_min_path_half_width_px: float = 12.0
    min_path_overlap_ratio: float = 0.15
    min_current_path_overlap_ratio: float = 0.40
    min_physical_lane_overlap_ratio: float = 0.0
    max_current_path_distance_ratio: float = 0.48
    frame_boundary_margin_px: float = 4.0
    contact_band_ratio: float = 0.25
    visual_action_confidence: float = 0.75
    visual_commit_enabled: bool = False
    visual_commit_confidence: float = 0.90
    visual_commit_frame_y_ratio: float = 0.40
    range_visual_fallback_enabled: bool = True
    range_visual_fallback_confidence: float = 0.90
    visual_confirm_frames: int = 2
    visual_clear_frames: int = 2

    # YOLO tracks first, but range confirmation owns maneuver timing. At full
    # competition speed the front-sonar estimate can jump by more than 1 m as
    # adjacent beams acquire the obstacle, so commit while it is still within
    # the reliable 3.2 m firmware range instead of waiting for a 2 m sample.
    ultrasonic_trigger_mm: float = 2600.0
    ultrasonic_clear_mm: float = 2900.0
    ultrasonic_stop_mm: float = 300.0
    emergency_stop_enabled: bool = False
    min_front_sensors: int = 2
    range_confirm_frames: int = 1
    range_clear_frames: int = 2
    rearm_clear_frames: int = 3
    ttc_trigger_seconds: float = 0.0
    min_closing_rate_mm_s: float = 120.0

    visual_slowdown_enabled: bool = False
    approach_speed_cap: int = 120
    speed_cap: int = 120
    cooldown_seconds: float = 0.4


@dataclass(frozen=True)
class FramePathGeometry:
    """Lane-center trajectories projected back into camera-frame pixels."""

    lane1: Tuple[Tuple[float, float], ...] = ()
    lane2: Tuple[Tuple[float, float], ...] = ()
    lane2_left_boundary: Tuple[Tuple[float, float], ...] = ()
    lane2_right_boundary: Tuple[Tuple[float, float], ...] = ()

    def line(self, lane_index: int) -> Tuple[Tuple[float, float], ...]:
        return self.lane1 if lane_index == 1 else self.lane2

    def physical_bounds(
        self,
        lane_index: int,
    ) -> Optional[
        Tuple[
            Tuple[Tuple[float, float], ...],
            Tuple[Tuple[float, float], ...],
        ]
    ]:
        if (
            lane_index == 2
            and len(self.lane2_left_boundary) >= 2
            and len(self.lane2_right_boundary) >= 2
        ):
            return self.lane2_left_boundary, self.lane2_right_boundary
        return None


@dataclass(frozen=True)
class ObstacleFusionObservation:
    visual_detected: bool = False
    visual_actionable: bool = False
    visual_confidence: float = 0.0
    visual_confirmed: bool = False
    range_confirmed: bool = False
    visual_commit_confirmed: bool = False
    fused_hazard: bool = False
    emergency: bool = False
    maneuver_active: bool = False
    path_lane: int = 2
    closest_y_ratio: float = 0.0
    frame_y_ratio: float = 0.0
    physical_lane_overlap: float = 0.0
    obstacle_count: int = 0
    visual_frames: int = 0
    front_mm: Optional[int] = None
    front_sensor_count: int = 0
    range_frames: int = 0
    closing_rate_mm_s: float = 0.0
    ttc_seconds: Optional[float] = None
    plan_ready: bool = False
    planned_target_lane: Optional[int] = None


@dataclass(frozen=True)
class PathOccupancy:
    bottom_y_ratio: float
    current_overlap: float
    current_distance_px: float = 0.0
    current_distance_ratio: float = 0.0
    physical_lane_overlap: float = 1.0


@dataclass(frozen=True)
class PathAssessment:
    current_detected: bool = False
    range_fallback_candidate: bool = False
    closest_y_ratio: float = 0.0
    physical_lane_overlap: float = 0.0
    physical_lane_known: bool = False
    obstacle_count: int = 0


class ObstacleFusionPlanner:
    """Early map-based path planning followed by range-confirmed execution.

    Frame-space YOLO masks provide lookahead before an obstacle enters the BEV
    source trapezoid. BEV masks still provide the near-field path association.
    Ultrasonic range/TTC confirms that the visual object is physically close.
    Contest obstacle layouts guarantee that the opposite lane is the escape
    route, so planning only asks whether the obstacle is on the current driving
    path. The lane-change controller owns full-lane arrival and alignment.
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
        self._last_trigger_path_lane: Optional[int] = None
        self._planned_from_lane: Optional[int] = None
        self._planned_target_lane: Optional[int] = None
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
        self._last_trigger_path_lane = None
        self._clear_path_plan()
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
        obstacle_confidence: float = 1.0,
    ) -> Optional[str]:
        path_lane = self._desired_lane(lane_change.state)
        if not self.config.enabled:
            self.observation = ObstacleFusionObservation(path_lane=path_lane)
            return None

        if path_lane != self._path_lane:
            consumed = self._consumed
            last_trigger_path_lane = self._last_trigger_path_lane
            self._path_lane = path_lane
            self.reset()
            # Preserve event identity across the path switch. The old source-lane
            # obstacle cannot trigger a return, while a mapped obstacle on the
            # new stable current path may create a different-path plan.
            self._consumed = consumed
            self._last_trigger_path_lane = last_trigger_path_lane

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
        planning_suppressed = lane_change.state in (
            "armed",
            "changing_to_lane1",
            "changing_to_lane2",
        )
        raw_path_visual = (
            bev_assessment.current_detected
            or frame_assessment.current_detected
        )
        if planning_suppressed:
            raw_path_visual = False
        visual_confidence = max(0.0, min(1.0, float(obstacle_confidence)))
        maneuver_active = self._lane_change_active(lane_change.state)
        front_mm = ultrasonic.front_min_mm
        front_count = ultrasonic.front_fresh_count
        obstacle_count = max(
            bev_assessment.obstacle_count,
            frame_assessment.obstacle_count,
        )

        if not running:
            self.reset()
            self.observation = ObstacleFusionObservation(
                visual_detected=raw_path_visual,
                visual_actionable=(
                    raw_path_visual
                    and visual_confidence
                    >= max(0.0, float(self.config.visual_action_confidence))
                ),
                visual_confidence=visual_confidence,
                path_lane=path_lane,
                closest_y_ratio=bev_assessment.closest_y_ratio,
                frame_y_ratio=frame_assessment.closest_y_ratio,
                physical_lane_overlap=frame_assessment.physical_lane_overlap,
                obstacle_count=obstacle_count,
                front_mm=front_mm,
                front_sensor_count=front_count,
            )
            return None

        stable_lane = lane_change.state in ("lane2", "completed", "lane1")
        request_ready_lane = stable_lane or lane_change.state == "stabilizing_lane1"
        self._update_range_state(ultrasonic, now)
        raw_visual = False
        if not planning_suppressed:
            raw_visual = raw_path_visual or self._range_visual_fallback(
                raw_path_visual,
                bev_assessment.range_fallback_candidate
                or frame_assessment.range_fallback_candidate,
                obstacle_count,
                visual_confidence,
            )
        visual = (
            raw_visual
            and visual_confidence
            >= max(0.0, float(self.config.visual_action_confidence))
        )
        self._update_visual_state(visual)
        self._update_rearm_state(
            raw_visual,
            stable_lane,
        )
        self._update_path_plan(path_lane, request_ready_lane)
        different_path_event = (
            self._consumed
            and self._last_trigger_path_lane is not None
            and path_lane != self._last_trigger_path_lane
        )
        current_path_event = different_path_event and raw_path_visual
        visual_commit_confirmed = self._visual_commit_confirmed(
            frame_assessment,
            frame_obstacle_masks,
            visual_confidence,
            allow_projected_current=current_path_event,
        )
        range_confirmed = (
            self.config.fusion_mode == "yolo"
            or self._range_hazard
            or visual_commit_confirmed
        )
        fused_hazard = self._visual_confirmed and range_confirmed
        emergency = self._emergency_present(
            raw_visual,
            bev_assessment.closest_y_ratio,
            frame_assessment.closest_y_ratio,
            ultrasonic,
            maneuver_active,
        )
        self.observation = ObstacleFusionObservation(
            visual_detected=raw_visual,
            visual_actionable=visual,
            visual_confidence=visual_confidence,
            visual_confirmed=self._visual_confirmed,
            range_confirmed=range_confirmed,
            visual_commit_confirmed=visual_commit_confirmed,
            fused_hazard=fused_hazard,
            emergency=emergency,
            maneuver_active=maneuver_active,
            path_lane=path_lane,
            closest_y_ratio=bev_assessment.closest_y_ratio,
            frame_y_ratio=frame_assessment.closest_y_ratio,
            physical_lane_overlap=frame_assessment.physical_lane_overlap,
            obstacle_count=obstacle_count,
            visual_frames=self._visual_frames,
            front_mm=front_mm,
            front_sensor_count=front_count,
            range_frames=self._range_frames,
            closing_rate_mm_s=self._closing_rate_mm_s,
            ttc_seconds=self._ttc_seconds,
            plan_ready=self._path_plan_ready(path_lane),
            planned_target_lane=self._planned_target_lane,
        )

        if (
            fused_hazard
            and request_ready_lane
            and self._path_plan_ready(path_lane)
            and (not self._consumed or current_path_event)
            and now - self._last_trigger_at >= max(0.0, self.config.cooldown_seconds)
        ):
            event = self._request_lane_change(lane_change, now)
            if event is not None:
                self._consumed = True
                self._last_trigger_path_lane = path_lane
                self._clear_path_plan()
                return event
        return None

    def _range_visual_fallback(
        self,
        raw_path_visual: bool,
        range_fallback_candidate: bool,
        obstacle_count: int,
        visual_confidence: float,
    ) -> bool:
        if not self.config.range_visual_fallback_enabled:
            return False
        if raw_path_visual or obstacle_count <= 0:
            return False
        if not range_fallback_candidate:
            return False
        if not self._range_hazard:
            return False
        required_confidence = max(
            max(0.0, float(self.config.visual_action_confidence)),
            max(0.0, float(self.config.range_visual_fallback_confidence)),
        )
        return visual_confidence >= required_confidence

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
        elif (
            self.config.visual_slowdown_enabled
            and self.observation.visual_detected
        ):
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
        elif obs.maneuver_active:
            state = "COMMITTED"
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
        plan = (
            "-"
            if obs.planned_target_lane is None
            else "L%d" % obs.planned_target_lane
        )
        return "L%d %s plan=%s by=%.2f fy=%.2f in=%.2f vc=%s conf=%.2f front=%s q=%d r=%d ttc=%s" % (
            obs.path_lane,
            state,
            plan,
            obs.closest_y_ratio,
            obs.frame_y_ratio,
            obs.physical_lane_overlap,
            "Y" if obs.visual_commit_confirmed else "N",
            obs.visual_confidence,
            front,
            obs.front_sensor_count,
            obs.range_frames,
            ttc,
        )

    def _visual_commit_confirmed(
        self,
        frame_assessment: PathAssessment,
        frame_obstacle_masks: Sequence[Any],
        visual_confidence: float,
        allow_projected_current: bool = False,
    ) -> bool:
        if not self.config.visual_commit_enabled:
            return False
        if not self._visual_confirmed:
            return False
        if not frame_obstacle_masks:
            return False
        if not frame_assessment.current_detected:
            return False
        if not frame_assessment.physical_lane_known and not allow_projected_current:
            return False
        if visual_confidence < max(
            float(self.config.visual_action_confidence),
            float(self.config.visual_commit_confidence),
        ):
            return False
        return frame_assessment.closest_y_ratio >= max(
            float(self.config.frame_visual_trigger_y_ratio),
            float(self.config.visual_commit_frame_y_ratio),
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

    def _update_rearm_state(
        self,
        raw_visual: bool,
        stable_lane: bool,
    ) -> Optional[int]:
        if not self._consumed:
            self._rearm_frames = 0
            return None
        if stable_lane and not raw_visual and not self._range_hazard:
            self._rearm_frames += 1
        else:
            self._rearm_frames = 0
        if self._rearm_frames >= max(1, int(self.config.rearm_clear_frames)):
            cleared_source_lane = self._last_trigger_path_lane
            self._consumed = False
            self._last_trigger_path_lane = None
            self._rearm_frames = 0
            return cleared_source_lane
        return None

    def _update_path_plan(
        self,
        path_lane: int,
        stable_lane: bool,
    ) -> None:
        if not stable_lane or not self._visual_confirmed:
            self._clear_path_plan()
            return
        self._planned_from_lane = path_lane
        self._planned_target_lane = 1 if path_lane == 2 else 2

    def _path_plan_ready(self, path_lane: int) -> bool:
        return (
            self._planned_from_lane == path_lane
            and self._planned_target_lane in (1, 2)
            and self._planned_target_lane != path_lane
        )

    def _clear_path_plan(self) -> None:
        self._planned_from_lane = None
        self._planned_target_lane = None

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
        avoidance_committed: bool,
    ) -> bool:
        if not self.config.emergency_stop_enabled:
            return False

        visual_close = visual and (
            closest_bev_y >= self.config.visual_emergency_y_ratio
            or closest_frame_y >= self.config.frame_visual_emergency_y_ratio
        )
        if avoidance_committed:
            # Neither front sonar nor frame-path overlap is spatially reliable
            # while crossing lanes or clearing the source obstacle immediately
            # afterward. Braking here only prevents the lateral escape from
            # completing.
            return False

        required = max(1, int(self.config.min_front_sensors))
        if not ultrasonic.front_ready(required):
            return visual_close

        close_count = ultrasonic.front_close_count(self.config.ultrasonic_stop_mm)
        range_close = close_count >= 2 or (visual and close_count >= 1)
        return visual_close or range_close

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
                max(1.0, float(self.config.path_half_width_px)),
            )
            if measurement is not None:
                measurements.append(measurement)

        return self._assess_measurements(
            measurements,
            self.config.visual_trigger_y_ratio,
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
        if len(current_line) < 2:
            return PathAssessment()
        physical_bounds = frame_paths.physical_bounds(path_lane)

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
            current_distance = np.abs(contact_xs - current_x)
            half_width = np.maximum(
                max(1.0, self.config.frame_min_path_half_width_px),
                max(1.0, self.config.lane_width_px)
                * max(0.0, self.config.frame_path_half_width_scale),
            )
            inside_physical_lane = np.ones(len(contact_xs), dtype=bool)
            if physical_bounds is not None:
                left_boundary, right_boundary = physical_bounds
                left_x = self._center_x_at(
                    contact_ys,
                    left_boundary,
                    float(binary.shape[1]) / 2.0,
                )
                right_x = self._center_x_at(
                    contact_ys,
                    right_boundary,
                    float(binary.shape[1]) / 2.0,
                )
                margin = max(
                    0.0,
                    float(self.config.frame_boundary_margin_px),
                )
                lower = np.minimum(left_x, right_x) - margin
                upper = np.maximum(left_x, right_x) + margin
                inside_physical_lane = (
                    (contact_xs >= lower)
                    & (contact_xs <= upper)
                )
            measurements.append(
                PathOccupancy(
                    bottom_y_ratio=max_y / float(max(1, binary.shape[0] - 1)),
                    current_overlap=float(
                        np.mean(
                            np.abs(contact_xs - current_x) <= half_width
                        )
                    ),
                    current_distance_px=float(np.mean(current_distance)),
                    current_distance_ratio=float(
                        np.mean(
                            current_distance / max(1.0, self.config.lane_width_px)
                        )
                    ),
                    physical_lane_overlap=float(
                        np.mean(inside_physical_lane)
                    ),
                )
            )

        return self._assess_measurements(
            measurements,
            self.config.frame_visual_trigger_y_ratio,
            physical_lane_known=physical_bounds is not None,
        )

    def _measure_mask(
        self,
        binary: Any,
        base_centerline: Sequence[Tuple[float, float]],
        fallback_x: float,
        current_offset: float,
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
        current_x = base_x + current_offset
        current_distance = np.abs(contact_xs - current_x)
        lane_width = max(1.0, float(self.config.lane_width_px))
        return PathOccupancy(
            bottom_y_ratio=max_y / float(max(1, binary.shape[0] - 1)),
            current_overlap=float(
                np.mean(
                    current_distance <= half_width
                )
            ),
            current_distance_px=float(np.mean(current_distance)),
            current_distance_ratio=float(np.mean(current_distance / lane_width)),
        )

    def _assess_measurements(
        self,
        measurements: Sequence[PathOccupancy],
        current_y_threshold: float,
        physical_lane_known: bool = False,
    ) -> PathAssessment:
        overlap_min = max(0.0, min(1.0, self.config.min_path_overlap_ratio))
        current_overlap_min = max(
            overlap_min,
            max(
                0.0,
                min(1.0, self.config.min_current_path_overlap_ratio),
            ),
        )
        max_current_distance = max(
            0.0,
            float(self.config.max_current_path_distance_ratio),
        )
        min_physical_overlap = max(
            0.0,
            min(
                1.0,
                float(self.config.min_physical_lane_overlap_ratio),
            ),
        )
        def inside_current_path(item: PathOccupancy) -> bool:
            return item.current_distance_ratio <= max_current_distance

        def inside_physical_lane(item: PathOccupancy) -> bool:
            return item.physical_lane_overlap >= min_physical_overlap

        def current_preferred(item: PathOccupancy) -> bool:
            return (
                item.current_overlap >= current_overlap_min
                and inside_current_path(item)
                and inside_physical_lane(item)
            )

        current = [
            item
            for item in measurements
            if item.bottom_y_ratio >= current_y_threshold
            and current_preferred(item)
        ]
        range_fallback = any(
            item.bottom_y_ratio >= current_y_threshold
            and inside_current_path(item)
            and inside_physical_lane(item)
            for item in measurements
        )
        return PathAssessment(
            current_detected=bool(current),
            range_fallback_candidate=range_fallback,
            closest_y_ratio=max(
                (item.bottom_y_ratio for item in current),
                default=0.0,
            ),
            physical_lane_overlap=max(
                (
                    item.physical_lane_overlap
                    for item in measurements
                    if item.bottom_y_ratio >= current_y_threshold
                ),
                default=0.0,
            ),
            physical_lane_known=physical_lane_known,
            obstacle_count=len(measurements),
        )

    def _request_lane_change(
        self,
        lane_change: LaneChangeController,
        now: float,
        avoidance: bool = True,
    ) -> Optional[str]:
        source = "obstacle_fusion"
        if lane_change.state in ("lane2", "completed"):
            accepted = lane_change.request_avoidance(source)
            direction = "lane2 -> lane1"
        elif lane_change.state in ("stabilizing_lane1", "lane1"):
            if avoidance:
                accepted = lane_change.request_avoidance_return(source)
            else:
                accepted = lane_change.request_return("obstacle_clear")
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
        if state in ("stabilizing_lane1", "lane1"):
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
    def _append_reason(reason: str, suffix: str) -> str:
        return "%s:%s" % (reason, suffix) if reason else suffix
