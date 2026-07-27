from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

from ..estimation.bev_corridor import BevClassMasks, BevCorridorLaneEstimator
from ..estimation.lane_geometry import LaneGeometry
from ..perception.bev import BevTransformer
from ..perception.yolo_lane import YoloClassMasks, YoloLaneMask
from ..planning.lane_change import LaneChangeConfig, LaneChangeController, LaneChangeResult
from ..planning.local_occupancy import (
    LocalOccupancyConfig,
    LocalOccupancyGrid,
    LocalOccupancySnapshot,
)
from ..planning.obstacle_fusion import (
    FramePathGeometry,
    ObstacleFusionConfig,
    ObstacleFusionPlanner,
)
from ..sensors.ultrasonic import UltrasonicConfig, UltrasonicFilter
from ..types import ControlCommand


LOG = logging.getLogger("skku_autocar.obstacle_mode")


@dataclass(frozen=True)
class ObstacleFrameResult:
    """Obstacle-mode additions to one otherwise normal BEV driving frame."""

    lane: LaneGeometry
    lane_change_state: str = "off"
    status: str = "off"
    frame_obstacle_masks: Tuple[Any, ...] = ()
    bev_obstacle_masks: Tuple[Any, ...] = ()
    blocks_light_stop: bool = False


class ObstacleDriveMode:
    """Optional YOLO/ultrasonic obstacle avoidance extension.

    The normal drive loop owns perception, BEV lane estimation, lane following,
    traffic lights and command output. This object only alters that path while
    ``--obstacle-avoidance on`` is selected.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        transformer: BevTransformer,
        corridor_estimator: BevCorridorLaneEstimator,
    ) -> None:
        self.enabled = args.obstacle_avoidance == "on"
        self._allow_light_stop = args.light_stop_during_lane_change == "on"
        self._transformer = transformer
        self._corridor_estimator = corridor_estimator
        self._lane_change = LaneChangeController(build_lane_change_config(args))
        self._planner = ObstacleFusionPlanner(build_obstacle_fusion_config(args))
        self._local_map = LocalOccupancyGrid(build_local_occupancy_config(args))
        self._ultrasonic = UltrasonicFilter(build_ultrasonic_config(args))
        self._result: Optional[LaneChangeResult] = None
        self._last_output_lane: Optional[LaneGeometry] = None
        self._last_reliable_output_lane: Optional[LaneGeometry] = None
        self._crosswalk_offset_px: Optional[float] = None
        self._frame = ObstacleFrameResult(lane=_empty_lane())

    @property
    def frame(self) -> ObstacleFrameResult:
        return self._frame

    @property
    def lane_change_state(self) -> str:
        return self._frame.lane_change_state

    @property
    def status_text(self) -> str:
        return self._frame.status

    @property
    def frame_obstacle_masks(self) -> Tuple[Any, ...]:
        return self._frame.frame_obstacle_masks

    @property
    def bev_obstacle_masks(self) -> Tuple[Any, ...]:
        return self._frame.bev_obstacle_masks

    @property
    def blocks_light_stop(self) -> bool:
        return self._frame.blocks_light_stop

    def validate_runtime(self, segmenter: Any, no_serial: bool) -> None:
        if not self.enabled:
            return
        if not segmenter.has_obstacle_class:
            LOG.warning(
                "model has no 'obstacle' class; obstacle fusion cannot request a lane change"
            )
        if no_serial and self._planner.config.fusion_mode == "fused":
            LOG.warning(
                "fused obstacle mode requires Arduino ultrasonic data; "
                "use --obstacle-fusion-mode yolo for video replay"
            )

    def start_serial(self, vehicle: Any) -> None:
        if self.enabled:
            vehicle.write_line("USON")

    def update_serial(self, vehicle: Any, now: float) -> None:
        if self.enabled and vehicle is not None:
            self._ultrasonic.update_lines(vehicle.read_lines(), now)

    def accept_serial_lines(self, lines: Sequence[str], now: float) -> None:
        if self.enabled:
            self._ultrasonic.update_lines(lines, now)

    def stop_serial(self, vehicle: Any) -> None:
        if self.enabled:
            vehicle.write_line("USOFF")

    def update(
        self,
        class_masks: YoloClassMasks,
        bev: BevClassMasks,
        lane: LaneGeometry,
        mask_result: Optional[YoloLaneMask],
        frame_shape: tuple,
        now: float,
        running: bool,
    ) -> LaneGeometry:
        if not self.enabled:
            self._result = None
            self._frame = ObstacleFrameResult(lane=lane)
            return lane
        if not running:
            self._last_output_lane = None
            self._last_reliable_output_lane = None
            self._crosswalk_offset_px = None

        lane_reliable = lane_change_geometry_reliable(mask_result, lane)
        map_snapshot = self._local_map.update(
            bev.obstacle,
            bev.shape,
            class_masks.obstacle_conf,
            now,
            running,
        )
        if self._local_map.config.enabled:
            planning_masks = map_snapshot.instances
            planning_confidence = max(
                float(class_masks.obstacle_conf),
                map_snapshot.confidence,
            )
            debug_masks = planning_masks
        else:
            planning_masks = tuple(bev.obstacle)
            planning_confidence = float(class_masks.obstacle_conf)
            debug_masks = planning_masks

        crosswalk_priority = running and (
            self._corridor_estimator.last_crosswalk_visible
            or "crosswalk_transit_hold:" in lane.reason
        )
        if crosswalk_priority:
            self._lane_change.pause(now)
            return self._hold_crosswalk_path(
                lane,
                bev.shape,
                class_masks,
                debug_masks,
                map_snapshot,
            )
        self._lane_change.resume(now)
        self._crosswalk_offset_px = None

        frame_paths = build_obstacle_frame_paths(
            self._transformer,
            self._corridor_estimator.last_centerline_bev,
            self._planner.config.lane_width_px,
            frame_shape[:2],
            lane2_left_boundary=getattr(
                self._corridor_estimator,
                "last_center_line_bev",
                (),
            ),
            lane2_right_boundary=getattr(
                self._corridor_estimator,
                "last_right_line_bev",
                (),
            ),
            frame_center_masks=class_masks.center,
            frame_side_masks=class_masks.side,
        )
        event = self._planner.update(
            planning_masks,
            bev.shape,
            self._corridor_estimator.last_centerline_bev,
            lane,
            self._lane_change,
            self._ultrasonic.snapshot(now),
            now,
            running,
            frame_obstacle_masks=class_masks.obstacle,
            frame_paths=frame_paths,
            solid_masks=bev.side,
            obstacle_confidence=planning_confidence,
        )
        if event:
            LOG.info("%s", event)

        result = self._lane_change.update(
            lane,
            self._lane_change_width_px(self._corridor_estimator.last_lane_width_px),
            bev.shape[1],
            now,
            running,
            lane_reliable=lane_reliable,
        )
        if result.offset_px and self._corridor_estimator.last_centerline_bev:
            self._corridor_estimator.last_centerline_bev = [
                (x + result.offset_px, y)
                for x, y in self._corridor_estimator.last_centerline_bev
            ]

        active_transition = result.state not in ("off", "lane2", "completed")
        self._result = result
        self._frame = ObstacleFrameResult(
            lane=result.lane,
            lane_change_state=result.state,
            status=self._status_text(map_snapshot),
            frame_obstacle_masks=tuple(class_masks.obstacle),
            bev_obstacle_masks=tuple(debug_masks),
            blocks_light_stop=active_transition and not self._allow_light_stop,
        )
        self._last_output_lane = result.lane if running else None
        if (
            running
            and lane_reliable
            and result.lane.found
            and float(result.lane.confidence) >= 0.45
        ):
            self._last_reliable_output_lane = result.lane
        return result.lane

    def _hold_crosswalk_path(
        self,
        lane: LaneGeometry,
        bev_shape: tuple,
        class_masks: YoloClassMasks,
        debug_masks: Sequence[Any],
        map_snapshot: LocalOccupancySnapshot,
    ) -> LaneGeometry:
        """Pause lane-change state while preserving the current path shape."""
        if self._crosswalk_offset_px is None:
            self._crosswalk_offset_px = (
                float(self._result.offset_px)
                if self._result is not None
                else 0.0
            )
        bev_width = float(bev_shape[1]) if len(bev_shape) > 1 else 1.0
        current = self._lane_change.apply_fixed_offset(
            lane,
            self._crosswalk_offset_px,
            bev_width,
        )
        reason = current.reason.split(":crosswalk_priority_hold", 1)[0]
        held = replace(
            current,
            reason="%s:crosswalk_priority_hold" % reason,
        )
        self._corridor_estimator.last_centerline_bev = list(held.path_points)
        self._result = LaneChangeResult(
            lane=held,
            state="crosswalk_hold",
            offset_px=self._crosswalk_offset_px,
            active=False,
            lane_reliable=False,
        )
        self._frame = ObstacleFrameResult(
            lane=held,
            lane_change_state="crosswalk_hold",
            status="%s crosswalk-priority" % self._status_text(map_snapshot),
            frame_obstacle_masks=tuple(class_masks.obstacle),
            bev_obstacle_masks=tuple(debug_masks),
            blocks_light_stop=False,
        )
        self._last_output_lane = held
        if lane.found and float(lane.confidence) >= 0.45:
            self._last_reliable_output_lane = held
        return held

    def apply_steering(self, command: ControlCommand) -> ControlCommand:
        if not self.enabled or self._result is None:
            return command
        return self._lane_change.apply_steering_assist(command, self._result)

    def apply_speed_cap(self, command: ControlCommand) -> ControlCommand:
        if not self.enabled or self._result is None:
            return command
        active = self._lane_change.speed_cap_active(self._result)
        return self._lane_change.apply_speed_cap(
            command,
            active,
            lane_reliable=self._result.lane_reliable,
        )

    def apply_safety(self, command: ControlCommand, running: bool) -> ControlCommand:
        if not self.enabled:
            return command
        return self._planner.apply_safety(command, self.lane_change_state, running)

    def _lane_change_width_px(self, lane_width_px: float) -> float:
        configured = max(0.0, float(self._lane_change.config.target_lane_width_px))
        if configured > 0.0:
            return configured
        return max(0.0, float(lane_width_px))

    def handle_key(self, running: bool) -> tuple:
        if not self.enabled:
            return "ignored_disabled", "lane-change key ignored: obstacle mode is off"
        return handle_lane_change_key(self._lane_change, running)

    def log_configuration(self, args: argparse.Namespace) -> None:
        if not self.enabled:
            LOG.info("obstacle_avoidance=off (normal BEV2 driving path)")
            return
        LOG.info(
            "obstacle_avoidance=on fusion=%s local_map=%s lane_change=%s target_width=%.0fpx "
            "range=%.0f/%.0fmm emergency_stop=%s visual_slowdown=%s speed_cap=%d/%d",
            args.obstacle_fusion_mode,
            args.obstacle_local_map,
            args.lane_change_mode,
            self._lane_change_width_px(self._corridor_estimator.last_lane_width_px),
            args.obstacle_trigger_mm,
            args.obstacle_clear_mm,
            args.obstacle_emergency_stop,
            args.obstacle_visual_slowdown,
            args.obstacle_approach_speed_cap,
            args.obstacle_speed_cap,
        )
        LOG.info(
            "obstacle vision bev_y=%.2f/%.2f frame_y=%.2f/%.2f action_conf=%.2f "
            "overlap=%.2f physical=%.2f visual_commit=%s/%.2f@%.2f "
            "confirm=%d front_quorum=%d side=%.0fmm",
            args.obstacle_visual_trigger_y,
            args.obstacle_target_block_y,
            args.obstacle_frame_visual_trigger_y,
            args.obstacle_frame_target_block_y,
            args.obstacle_action_confidence,
            args.obstacle_current_path_min_overlap,
            args.obstacle_physical_lane_min_overlap,
            args.obstacle_visual_commit,
            args.obstacle_visual_commit_confidence,
            args.obstacle_visual_commit_frame_y,
            args.obstacle_visual_confirm_frames,
            args.obstacle_min_front_sensors,
            args.obstacle_side_clearance_mm,
        )

    def _status_text(self, snapshot: LocalOccupancySnapshot) -> str:
        status = self._planner.status_text()
        if not self._local_map.config.enabled:
            return status
        age = (
            "n/a"
            if snapshot.age_seconds == float("inf")
            else "%.2f" % snapshot.age_seconds
        )
        return "%s map=%.2f%% p=%.2f age=%s" % (
            status,
            snapshot.occupied_ratio * 100.0,
            snapshot.confidence,
            age,
        )


def lane_change_geometry_reliable(
    mask_result: Optional[YoloLaneMask],
    lane: LaneGeometry,
) -> bool:
    if mask_result is None or not lane.found:
        return False
    name = mask_result.class_name.lower()
    if any(token in name for token in ("coast", "no_corridor", "lane_lost")):
        return False
    if "virtual" not in name:
        return True
    return "center+virtual-right-side" in name


def build_lane_change_config(args: argparse.Namespace) -> LaneChangeConfig:
    steering_cap = args.lane_change_steering_cap
    if steering_cap is None:
        steering_cap = args.max_steering
    return LaneChangeConfig(
        mode=args.lane_change_mode,
        trigger_seconds=args.lane_change_trigger_seconds,
        transition_seconds=args.lane_change_transition_seconds,
        hold_seconds=args.lane_change_hold_seconds,
        max_straight_heading=args.lane_change_max_heading,
        speed_cap=args.lane_change_speed_cap,
        steering_min=args.lane_change_steering_min,
        steering_boost=args.lane_change_steering_boost,
        steering_cap=steering_cap,
        steering_slew_limit=args.lane_change_steering_slew_limit,
        steering_override=args.lane_change_steering_override == "on",
        unreliable_speed_cap=args.lane_change_unreliable_speed_cap,
        unreliable_steering_cap=args.lane_change_unreliable_steering_cap,
        stabilizing_steering_min=args.lane_change_stabilizing_steering_min,
        stable_lateral_error=args.lane_change_stable_lateral_error,
        stable_near_lateral_error=args.lane_change_stable_near_error,
        stable_heading_error=args.lane_change_stable_heading_error,
        stable_required_frames=args.lane_change_stable_frames,
        target_lane_width_px=args.lane_change_target_width_px,
        target_approach_error=args.lane_change_target_approach_error,
        target_capture_error=args.lane_change_target_capture_error,
        target_capture_frames=args.lane_change_target_capture_frames,
        allow_virtual_stabilize=args.lane_change_allow_virtual_stabilize == "on",
        smooth_avoidance=args.lane_change_smooth_avoidance == "on",
        return_duration_scale=args.lane_change_return_duration_scale,
        return_steering_cap=args.lane_change_return_steering_cap,
        return_stabilizing_steering_cap=(
            args.lane_change_return_stabilizing_steering_cap
        ),
    )


def resolve_lane_change_target_width_px(args: argparse.Namespace) -> float:
    configured = max(0.0, float(args.lane_change_target_width_px))
    if configured > 0.0:
        return configured
    return max(0.0, float(args.corridor_lane_width_px))


def build_obstacle_frame_paths(
    transformer: BevTransformer,
    base_centerline: list,
    lane_width_px: float,
    frame_hw: tuple,
    lane2_left_boundary: tuple = (),
    lane2_right_boundary: tuple = (),
    frame_center_masks: tuple = (),
    frame_side_masks: tuple = (),
) -> Optional[FramePathGeometry]:
    if len(base_centerline) < 2:
        return None
    width = max(0.0, float(lane_width_px))
    lane2_bev = [(float(x), float(y)) for x, y in base_centerline]
    lane1_bev = [(float(x) - width, float(y)) for x, y in base_centerline]
    lane1_frame = transformer.bev_to_frame(lane1_bev, frame_hw)
    lane2_frame = transformer.bev_to_frame(lane2_bev, frame_hw)
    left_frame = (
        transformer.bev_to_frame(lane2_left_boundary, frame_hw)
        if len(lane2_left_boundary) >= 2
        else ()
    )
    right_frame = (
        transformer.bev_to_frame(lane2_right_boundary, frame_hw)
        if len(lane2_right_boundary) >= 2
        else ()
    )
    # Raw lane masks extend above the BEV source trapezoid. They prevent a far,
    # off-track object from being classified against a projected boundary that
    # has clamped to its first valid BEV point.
    direct_left = _frame_boundary_from_masks(
        frame_center_masks,
        left_frame,
        merge_instances=True,
    )
    direct_right = _frame_boundary_from_masks(
        frame_side_masks,
        right_frame,
        merge_instances=False,
    )
    if len(direct_left) >= 2:
        left_frame = direct_left
    if len(direct_right) >= 2:
        right_frame = direct_right
    return FramePathGeometry(
        lane1=tuple((float(x), float(y)) for x, y in lane1_frame),
        lane2=tuple((float(x), float(y)) for x, y in lane2_frame),
        lane2_left_boundary=tuple(
            (float(x), float(y))
            for x, y in left_frame
        ),
        lane2_right_boundary=tuple(
            (float(x), float(y))
            for x, y in right_frame
        ),
    )


def _frame_boundary_from_masks(
    masks: tuple,
    reference_line: Sequence[Tuple[float, float]],
    merge_instances: bool,
) -> Tuple[Tuple[float, float], ...]:
    import numpy as np

    candidates = []
    for mask in masks:
        binary = np.asarray(mask) > 0
        if binary.ndim != 2 or not binary.any():
            continue
        ys = np.nonzero(binary.any(axis=1))[0]
        points = []
        for y in ys[::3]:
            xs = np.nonzero(binary[int(y)])[0]
            if len(xs):
                points.append((float(np.median(xs)), float(y)))
        if len(points) >= 2:
            candidates.append(tuple(points))
    if not candidates:
        return ()
    if merge_instances:
        return tuple(
            sorted(
                (point for line in candidates for point in line),
                key=lambda point: point[1],
            )
        )
    if len(candidates) == 1 or len(reference_line) < 2:
        return max(candidates, key=len)

    reference = sorted(reference_line, key=lambda point: point[1])
    reference_y = np.asarray([point[1] for point in reference], dtype=float)
    reference_x = np.asarray([point[0] for point in reference], dtype=float)

    def distance(line: Tuple[Tuple[float, float], ...]) -> float:
        points = np.asarray(line, dtype=float)
        overlap = (
            (points[:, 1] >= reference_y[0])
            & (points[:, 1] <= reference_y[-1])
        )
        if overlap.any():
            points = points[overlap]
        expected_x = np.interp(points[:, 1], reference_y, reference_x)
        return float(np.median(np.abs(points[:, 0] - expected_x)))

    return min(candidates, key=distance)


def build_obstacle_fusion_config(args: argparse.Namespace) -> ObstacleFusionConfig:
    return ObstacleFusionConfig(
        enabled=args.obstacle_avoidance == "on",
        fusion_mode=args.obstacle_fusion_mode,
        lane_width_px=resolve_lane_change_target_width_px(args),
        visual_trigger_y_ratio=args.obstacle_visual_trigger_y,
        target_block_y_ratio=args.obstacle_target_block_y,
        frame_visual_trigger_y_ratio=args.obstacle_frame_visual_trigger_y,
        frame_target_block_y_ratio=args.obstacle_frame_target_block_y,
        visual_emergency_y_ratio=args.obstacle_visual_emergency_y,
        frame_visual_emergency_y_ratio=args.obstacle_frame_visual_emergency_y,
        path_half_width_px=args.obstacle_path_half_width_px,
        frame_path_half_width_scale=args.obstacle_frame_path_width_scale,
        min_path_overlap_ratio=args.obstacle_min_overlap,
        min_current_path_overlap_ratio=args.obstacle_current_path_min_overlap,
        min_physical_lane_overlap_ratio=args.obstacle_physical_lane_min_overlap,
        max_current_path_distance_ratio=args.obstacle_current_path_max_distance_ratio,
        frame_boundary_margin_px=args.obstacle_frame_boundary_margin_px,
        contact_band_ratio=args.obstacle_contact_band,
        visual_action_confidence=args.obstacle_action_confidence,
        visual_commit_enabled=args.obstacle_visual_commit == "on",
        visual_commit_confidence=args.obstacle_visual_commit_confidence,
        visual_commit_frame_y_ratio=args.obstacle_visual_commit_frame_y,
        range_visual_fallback_enabled=args.obstacle_range_visual_fallback == "on",
        range_visual_fallback_confidence=args.obstacle_range_visual_confidence,
        visual_confirm_frames=args.obstacle_visual_confirm_frames,
        visual_clear_frames=args.obstacle_visual_clear_frames,
        ultrasonic_trigger_mm=args.obstacle_trigger_mm,
        ultrasonic_clear_mm=args.obstacle_clear_mm,
        ultrasonic_stop_mm=args.obstacle_stop_mm,
        blocked_stop_mm=args.obstacle_blocked_stop_mm,
        emergency_stop_enabled=args.obstacle_emergency_stop == "on",
        min_front_sensors=args.obstacle_min_front_sensors,
        range_confirm_frames=args.obstacle_range_confirm_frames,
        range_clear_frames=args.obstacle_range_clear_frames,
        rearm_clear_frames=args.obstacle_rearm_clear_frames,
        source_clear_confirm_frames=args.obstacle_source_clear_frames,
        ttc_trigger_seconds=args.obstacle_ttc_seconds,
        min_closing_rate_mm_s=args.obstacle_min_closing_rate,
        side_clearance_mm=args.obstacle_side_clearance_mm,
        solid_crossing_margin_px=args.obstacle_solid_crossing_margin_px,
        solid_min_overlap_ratio=args.obstacle_solid_min_overlap,
        visual_slowdown_enabled=args.obstacle_visual_slowdown == "on",
        approach_speed_cap=args.obstacle_approach_speed_cap,
        speed_cap=args.obstacle_speed_cap,
        cooldown_seconds=args.obstacle_cooldown_seconds,
    )


def build_local_occupancy_config(args: argparse.Namespace) -> LocalOccupancyConfig:
    return LocalOccupancyConfig(
        enabled=args.obstacle_local_map == "on",
        decay_seconds=args.obstacle_map_decay_seconds,
        hit_probability=args.obstacle_map_hit_probability,
        occupied_probability=args.obstacle_map_occupied_probability,
        inflation_radius_px=args.obstacle_map_inflation_px,
    )


def build_ultrasonic_config(args: argparse.Namespace) -> UltrasonicConfig:
    return UltrasonicConfig(
        min_valid_mm=args.ultrasonic_min_valid_mm,
        max_valid_mm=args.ultrasonic_max_valid_mm,
        median_window=args.ultrasonic_median_window,
        max_age_seconds=args.ultrasonic_max_age,
    )


def handle_lane_change_key(
    lane_change_controller: LaneChangeController,
    running: bool,
) -> tuple:
    if not running:
        return "ignored_not_running", "lane-change key ignored while paused"
    if lane_change_controller.state in ("lane2", "completed"):
        if lane_change_controller.request("operator"):
            return "request", "lane-change request armed: waiting for a straight frame"
        return "ignored_request", "lane-change request ignored"
    if lane_change_controller.state in ("changing_to_lane1", "lane1"):
        if lane_change_controller.request_return("operator_return"):
            return "return", "lane-change return requested: waiting for a straight frame"
        return "ignored_return", "lane-change return ignored"
    return "ignored_busy", "lane-change key ignored while transition is busy"


def add_obstacle_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("optional obstacle avoidance")
    group.add_argument(
        "--obstacle-avoidance", choices=("on", "off"), default="on",
        help="enable the optional YOLO and ultrasonic obstacle-avoidance module",
    )
    group.add_argument(
        "--obstacle-fusion-mode", choices=("fused", "yolo"), default="fused",
        help="fused uses YOLO and ultrasonic confirmation; yolo is for video replay",
    )
    group.add_argument(
        "--obstacle-local-map",
        choices=("on", "off"),
        default="on",
        help="accumulate obstacle masks in a short-horizon BEV occupancy grid",
    )
    group.add_argument(
        "--obstacle-visual-slowdown",
        choices=("on", "off"),
        default="off",
        help="slow on YOLO-only tracking before ultrasonic range confirmation",
    )
    group.add_argument(
        "--lane-change-mode", choices=("off", "external", "timed"), default="external",
        help="lane-change trigger source used only while obstacle mode is enabled",
    )
    group.add_argument(
        "--light-stop-during-lane-change",
        choices=("on", "off"),
        default="off",
        help="allow traffic-light braking while an obstacle lane change is active",
    )

    lane_specs = (
        ("--lane-change-trigger-seconds", float, LaneChangeConfig.trigger_seconds),
        ("--lane-change-transition-seconds", float, 1.0),
        ("--lane-change-hold-seconds", float, LaneChangeConfig.hold_seconds),
        ("--lane-change-max-heading", float, LaneChangeConfig.max_straight_heading),
        ("--lane-change-speed-cap", int, 255),
        ("--lane-change-steering-min", int, 150),
        ("--lane-change-steering-boost", int, 35),
        ("--lane-change-steering-slew-limit", int, 0),
        ("--lane-change-unreliable-speed-cap", int, 255),
        ("--lane-change-unreliable-steering-cap", int, 150),
        ("--lane-change-stabilizing-steering-min", int, 90),
        ("--lane-change-stable-lateral-error", float, LaneChangeConfig.stable_lateral_error),
        ("--lane-change-stable-heading-error", float, LaneChangeConfig.stable_heading_error),
        ("--lane-change-stable-near-error", float, LaneChangeConfig.stable_near_lateral_error),
        ("--lane-change-stable-frames", int, LaneChangeConfig.stable_required_frames),
        ("--lane-change-target-width-px", float, 120.0),
        ("--lane-change-target-approach-error", float, 0.20),
        ("--lane-change-target-capture-frames", int, LaneChangeConfig.target_capture_frames),
        ("--lane-change-return-duration-scale", float, 1.35),
        ("--lane-change-return-steering-cap", int, 115),
        ("--lane-change-return-stabilizing-steering-cap", int, 90),
    )
    for name, value_type, default in lane_specs:
        group.add_argument(name, type=value_type, default=default)
    group.add_argument("--lane-change-steering-cap", type=int, default=150)
    group.add_argument(
        "--lane-change-steering-override", choices=("on", "off"), default="on"
    )
    group.add_argument(
        "--lane-change-target-capture-error",
        "--lane-change-target-clearance-margin",
        dest="lane_change_target_capture_error",
        type=float,
        default=LaneChangeConfig.target_capture_error,
    )
    group.add_argument(
        "--lane-change-allow-virtual-stabilize",
        choices=("on", "off"),
        default="off",
    )
    group.add_argument(
        "--lane-change-smooth-avoidance",
        choices=("on", "off"),
        default="on",
        help="use timed smoothstep obstacle lane changes; off uses immediate target-lane capture",
    )

    obstacle_specs = (
        ("--obstacle-visual-trigger-y", float, ObstacleFusionConfig.visual_trigger_y_ratio),
        ("--obstacle-target-block-y", float, ObstacleFusionConfig.target_block_y_ratio),
        ("--obstacle-frame-visual-trigger-y", float, 0.10),
        ("--obstacle-frame-target-block-y", float, ObstacleFusionConfig.frame_target_block_y_ratio),
        ("--obstacle-visual-emergency-y", float, ObstacleFusionConfig.visual_emergency_y_ratio),
        ("--obstacle-frame-visual-emergency-y", float, ObstacleFusionConfig.frame_visual_emergency_y_ratio),
        ("--obstacle-path-half-width-px", float, ObstacleFusionConfig.path_half_width_px),
        ("--obstacle-frame-path-width-scale", float, ObstacleFusionConfig.frame_path_half_width_scale),
        ("--obstacle-min-overlap", float, ObstacleFusionConfig.min_path_overlap_ratio),
        (
            "--obstacle-current-path-min-overlap",
            float,
            ObstacleFusionConfig.min_current_path_overlap_ratio,
        ),
        (
            "--obstacle-physical-lane-min-overlap",
            float,
            ObstacleFusionConfig.min_physical_lane_overlap_ratio,
        ),
        ("--obstacle-current-path-max-distance-ratio", float, ObstacleFusionConfig.max_current_path_distance_ratio),
        (
            "--obstacle-frame-boundary-margin-px",
            float,
            ObstacleFusionConfig.frame_boundary_margin_px,
        ),
        ("--obstacle-contact-band", float, ObstacleFusionConfig.contact_band_ratio),
        ("--obstacle-action-confidence", float, ObstacleFusionConfig.visual_action_confidence),
        (
            "--obstacle-visual-commit-confidence",
            float,
            ObstacleFusionConfig.visual_commit_confidence,
        ),
        (
            "--obstacle-visual-commit-frame-y",
            float,
            ObstacleFusionConfig.visual_commit_frame_y_ratio,
        ),
        ("--obstacle-range-visual-confidence", float, ObstacleFusionConfig.range_visual_fallback_confidence),
        ("--obstacle-visual-confirm-frames", int, 2),
        ("--obstacle-visual-clear-frames", int, ObstacleFusionConfig.visual_clear_frames),
        ("--obstacle-trigger-mm", float, 2600.0),
        ("--obstacle-clear-mm", float, 2900.0),
        ("--obstacle-stop-mm", float, ObstacleFusionConfig.ultrasonic_stop_mm),
        ("--obstacle-blocked-stop-mm", float, ObstacleFusionConfig.blocked_stop_mm),
        ("--obstacle-min-front-sensors", int, ObstacleFusionConfig.min_front_sensors),
        ("--obstacle-range-confirm-frames", int, ObstacleFusionConfig.range_confirm_frames),
        ("--obstacle-range-clear-frames", int, ObstacleFusionConfig.range_clear_frames),
        ("--obstacle-rearm-clear-frames", int, ObstacleFusionConfig.rearm_clear_frames),
        (
            "--obstacle-source-clear-frames",
            int,
            ObstacleFusionConfig.source_clear_confirm_frames,
        ),
        ("--obstacle-ttc-seconds", float, ObstacleFusionConfig.ttc_trigger_seconds),
        ("--obstacle-min-closing-rate", float, ObstacleFusionConfig.min_closing_rate_mm_s),
        ("--obstacle-side-clearance-mm", float, ObstacleFusionConfig.side_clearance_mm),
        ("--obstacle-solid-crossing-margin-px", float, ObstacleFusionConfig.solid_crossing_margin_px),
        ("--obstacle-solid-min-overlap", float, ObstacleFusionConfig.solid_min_overlap_ratio),
        ("--obstacle-approach-speed-cap", int, 255),
        ("--obstacle-speed-cap", int, 255),
        ("--obstacle-cooldown-seconds", float, ObstacleFusionConfig.cooldown_seconds),
    )
    for name, value_type, default in obstacle_specs:
        group.add_argument(name, type=value_type, default=default)
    group.add_argument(
        "--obstacle-visual-commit",
        choices=("on", "off"),
        default="off",
        help="allow a confirmed high-confidence in-lane mask to commit before front range acquisition",
    )
    group.add_argument(
        "--obstacle-range-visual-fallback",
        choices=("on", "off"),
        default="on",
        help="allow high-confidence YOLO plus confirmed front range to trigger when path overlap is ambiguous",
    )
    group.add_argument(
        "--obstacle-emergency-stop",
        choices=("on", "off"),
        default="off",
        help="stop on near obstacle emergency conditions instead of forcing avoidance",
    )

    map_specs = (
        ("--obstacle-map-decay-seconds", float, LocalOccupancyConfig.decay_seconds),
        ("--obstacle-map-hit-probability", float, LocalOccupancyConfig.hit_probability),
        (
            "--obstacle-map-occupied-probability",
            float,
            LocalOccupancyConfig.occupied_probability,
        ),
        ("--obstacle-map-inflation-px", int, LocalOccupancyConfig.inflation_radius_px),
    )
    for name, value_type, default in map_specs:
        group.add_argument(name, type=value_type, default=default)

    ultrasonic_specs = (
        ("--ultrasonic-min-valid-mm", int, UltrasonicConfig.min_valid_mm),
        ("--ultrasonic-max-valid-mm", int, UltrasonicConfig.max_valid_mm),
        ("--ultrasonic-median-window", int, UltrasonicConfig.median_window),
        ("--ultrasonic-max-age", float, UltrasonicConfig.max_age_seconds),
    )
    for name, value_type, default in ultrasonic_specs:
        group.add_argument(name, type=value_type, default=default)


def _empty_lane() -> LaneGeometry:
    return LaneGeometry(
        found=False,
        center_x=0.0,
        vehicle_center_x=0.0,
        target_y=0.0,
        lateral_error_px=0.0,
        lateral_error_norm=0.0,
        heading_error=0.0,
        confidence=0.0,
        reason="obstacle_mode_not_initialized",
    )
