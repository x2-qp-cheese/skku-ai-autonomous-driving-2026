#!/usr/bin/env python3
"""Deterministic full-speed feasibility checks for the obstacle lane trajectory.

This is not a vehicle dynamics model: the competition car has no calibrated
wheelbase, steering angle, or wheel-speed feedback. It verifies the quantities
the software can establish soundly before a track run:

* target motion is continuous and much smaller than the former 160 px teleport;
* spatial slope/curvature remain inside the BEV path geometry limits;
* requested steering respects the configured actuator cap and slew limit;
* outbound and return trajectories are symmetric;
* every non-brake command remains at speed 255.

Vehicle-body clearance is reported only when a measured physical width is
provided. A BEV pixel width is not a trustworthy substitute for that
measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from skku_autocar.estimation.lane_geometry import LaneGeometry
from skku_autocar.planning.lane_change import (
    LaneChangeConfig,
    LaneChangeController,
)
from skku_autocar.planning.yolo_lane_follower import (
    YoloLaneFollower,
    YoloLaneFollowerConfig,
)
from skku_autocar.types import ControlCommand


@dataclass(frozen=True)
class DirectionMetrics:
    first_lookahead_shift_px: float
    first_near_shift_px: float
    max_lookahead_step_px: float
    max_near_step_px: float
    max_abs_path_slope: float
    max_path_slope_delta: float
    max_abs_steering: int
    max_steering_step: int
    wrong_direction_commands: int
    minimum_speed: int
    maximum_speed: int
    brake_commands: int


def _base_lane(
    center_x: float,
    bev_size: int,
    lookahead_ratio: float,
    near_ratio: float,
) -> LaneGeometry:
    target_y = bev_size * lookahead_ratio
    near_y = bev_size * near_ratio
    vehicle_x = bev_size * (0.5 + 0.045)
    path = tuple(
        (
            float(center_x),
            bev_size * (0.02 + (0.98 - 0.02) * index / 23.0),
        )
        for index in range(24)
    )
    error_px = center_x - vehicle_x
    return LaneGeometry(
        found=True,
        center_x=center_x,
        vehicle_center_x=vehicle_x,
        target_y=target_y,
        lateral_error_px=error_px,
        lateral_error_norm=error_px / (bev_size / 2.0),
        heading_error=0.0,
        confidence=1.0,
        reason="corridor_tier1",
        height=float(bev_size),
        near_center_x=center_x,
        near_target_y=near_y,
        near_lateral_error_px=error_px,
        near_lateral_error_norm=error_px / (bev_size / 2.0),
        path_points=path,
    )


def _follower() -> YoloLaneFollower:
    return YoloLaneFollower(
        YoloLaneFollowerConfig(
            base_speed=255,
            max_speed=255,
            min_curve_speed=255,
            max_steering=150,
            steering_rate_limit=80,
            min_steering_rate_limit=35,
            steering_release_rate_limit=55,
            path_tracking=True,
            path_lateral_gain=225.0,
            path_heading_gain=65.0,
            path_derivative_gain=18.0,
            path_near_weight=1.45,
            path_far_weight=0.55,
            path_steering_rise_alpha=0.72,
            path_steering_release_alpha=0.28,
            path_heading_lead_gain=170.0,
            path_heading_lead_coherent_gain=195.0,
            path_heading_lead_span=0.16,
            path_heading_lead_max_steering=32.0,
        )
    )


def _controller(
    transition_seconds: float,
    lane_width_px: float,
    spatial_lead: float,
    smooth: bool,
) -> LaneChangeController:
    return LaneChangeController(
        LaneChangeConfig(
            mode="external",
            transition_seconds=transition_seconds,
            target_lane_width_px=lane_width_px,
            speed_cap=255,
            steering_min=80,
            steering_boost=25,
            steering_cap=150,
            steering_slew_limit=35,
            unreliable_speed_cap=255,
            unreliable_steering_cap=90,
            stabilizing_steering_min=0,
            stable_lateral_error=0.18,
            stable_near_lateral_error=0.24,
            stable_required_frames=4,
            target_capture_error=0.30,
            target_capture_frames=2,
            smooth_avoidance=smooth,
            spatial_transition_lead=spatial_lead,
            trajectory_heading_gain=1.6,
            max_transition_seconds=4.0,
        )
    )


def _path_shape_metrics(
    path: Tuple[Tuple[float, float], ...],
) -> Tuple[float, float]:
    slopes: List[float] = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if abs(y1 - y0) > 1e-9:
            slopes.append((x1 - x0) / (y1 - y0))
    max_slope = max((abs(value) for value in slopes), default=0.0)
    max_delta = max(
        (abs(a - b) for a, b in zip(slopes, slopes[1:])),
        default=0.0,
    )
    return max_slope, max_delta


def _simulate(
    direction: int,
    fps: float,
    transition_seconds: float,
    lane_width_px: float,
    spatial_lead: float,
    smooth: bool = True,
) -> Tuple[DirectionMetrics, List[dict]]:
    bev_size = 480
    vehicle_x = bev_size * (0.5 + 0.045)
    base_center = vehicle_x if direction < 0 else vehicle_x + lane_width_px
    base = _base_lane(base_center, bev_size, 0.32, 0.88)
    controller = _controller(
        transition_seconds,
        lane_width_px,
        spatial_lead,
        smooth,
    )
    follower = _follower()

    if direction > 0:
        controller.state = "lane1"
        controller._locked_lane_width_px = lane_width_px

    idle = controller.update(
        base,
        lane_width_px,
        bev_size,
        -1.0 / fps,
        True,
    )
    idle_command = follower.plan(idle.lane)
    idle_command = controller.apply_control_adjustments(idle_command, idle)
    follower.accept_applied_command(idle_command)

    if direction < 0:
        controller.request_avoidance("validation")
    else:
        controller.request_avoidance_return("validation")
        # The lane1 update arms the return; the next update at the same timestamp
        # is the first actual spatial-transition frame.
        controller.update(base, lane_width_px, bev_size, 0.0, True)

    frame_count = int(math.ceil(transition_seconds * fps)) + 3
    records: List[dict] = []
    previous_center = vehicle_x
    previous_near = vehicle_x
    previous_steering = 0
    max_slope = 0.0
    max_slope_delta = 0.0

    for frame in range(frame_count):
        now = frame / fps
        result = controller.update(
            base,
            lane_width_px,
            bev_size,
            now,
            True,
        )
        planned = follower.plan(result.lane)
        applied = controller.apply_control_adjustments(planned, result)
        follower.accept_applied_command(applied)
        path_slope, path_delta = _path_shape_metrics(
            result.lane.path_points
        )
        max_slope = max(max_slope, path_slope)
        max_slope_delta = max(max_slope_delta, path_delta)
        near = float(result.lane.near_center_x or result.lane.center_x)
        records.append(
            {
                "frame": frame,
                "time": now,
                "progress": result.progress,
                "center_x": result.lane.center_x,
                "near_center_x": near,
                "center_step": result.lane.center_x - previous_center,
                "near_step": near - previous_near,
                "heading": result.lane.heading_error,
                "steering": applied.steering,
                "steering_step": applied.steering - previous_steering,
                "speed": applied.speed,
                "brake": applied.brake,
            }
        )
        previous_center = result.lane.center_x
        previous_near = near
        previous_steering = applied.steering

    first = records[0]
    metrics = DirectionMetrics(
        first_lookahead_shift_px=abs(
            first["center_x"] - vehicle_x
        ),
        first_near_shift_px=abs(first["near_center_x"] - vehicle_x),
        max_lookahead_step_px=max(
            abs(item["center_step"]) for item in records
        ),
        max_near_step_px=max(abs(item["near_step"]) for item in records),
        max_abs_path_slope=max_slope,
        max_path_slope_delta=max_slope_delta,
        max_abs_steering=max(abs(item["steering"]) for item in records),
        max_steering_step=max(
            abs(item["steering_step"]) for item in records
        ),
        wrong_direction_commands=sum(
            1
            for item in records
            if item["steering"] * direction < 0
        ),
        minimum_speed=min(item["speed"] for item in records),
        maximum_speed=max(item["speed"] for item in records),
        brake_commands=sum(1 for item in records if item["brake"]),
    )
    return metrics, records


def _checks(
    outbound: DirectionMetrics,
    returning: DirectionMetrics,
    hard_target_first_shift: float,
    lane_width_px: float,
) -> Dict[str, bool]:
    directions = (outbound, returning)
    return {
        "first_target_is_not_full_lane_jump": all(
            item.first_lookahead_shift_px <= 0.20 * lane_width_px
            for item in directions
        )
        and hard_target_first_shift >= 0.95 * lane_width_px,
        "near_field_starts_anchored": all(
            item.first_near_shift_px <= 0.01 for item in directions
        ),
        "temporal_target_step_is_bounded": all(
            item.max_lookahead_step_px <= 35.0
            and item.max_near_step_px <= 35.0
            for item in directions
        ),
        "spatial_geometry_is_bounded": all(
            item.max_abs_path_slope < 0.50
            and item.max_path_slope_delta < 0.20
            for item in directions
        ),
        "actuator_limits_are_respected": all(
            item.max_abs_steering <= 150
            and item.max_steering_step <= 35
            and item.wrong_direction_commands == 0
            for item in directions
        ),
        "speed_is_always_255_without_brake": all(
            item.minimum_speed == 255
            and item.maximum_speed == 255
            and item.brake_commands == 0
            for item in directions
        ),
        "outbound_and_return_are_symmetric": (
            abs(
                outbound.first_lookahead_shift_px
                - returning.first_lookahead_shift_px
            )
            <= 1e-6
            and abs(
                outbound.max_lookahead_step_px
                - returning.max_lookahead_step_px
            )
            <= 1e-6
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--transition-seconds", type=float, default=0.85)
    parser.add_argument("--lane-width-px", type=float, default=160.0)
    parser.add_argument("--road-width-mm", type=float, default=850.0)
    parser.add_argument(
        "--vehicle-width-mm",
        type=float,
        default=0.0,
        help="measured widest vehicle-body width; 0 leaves body clearance unchecked",
    )
    parser.add_argument("--spatial-lead", type=float, default=0.10)
    parser.add_argument("--show-records", action="store_true")
    args = parser.parse_args()

    outbound, outbound_records = _simulate(
        -1,
        args.fps,
        args.transition_seconds,
        args.lane_width_px,
        args.spatial_lead,
    )
    returning, return_records = _simulate(
        1,
        args.fps,
        args.transition_seconds,
        args.lane_width_px,
        args.spatial_lead,
    )
    hard_target, _ = _simulate(
        -1,
        args.fps,
        args.transition_seconds,
        args.lane_width_px,
        args.spatial_lead,
        smooth=False,
    )
    checks = _checks(
        outbound,
        returning,
        hard_target.first_lookahead_shift_px,
        args.lane_width_px,
    )
    lane_width_mm = args.road_width_mm / 2.0
    measured_vehicle_width_mm = max(0.0, float(args.vehicle_width_mm))
    if measured_vehicle_width_mm > 0.0:
        nominal_line_clearance_mm = (
            lane_width_mm - measured_vehicle_width_mm
        ) / 2.0
        checks["measured_body_has_positive_centered_clearance"] = (
            nominal_line_clearance_mm > 0.0
        )
        clearance = {
            "checked": True,
            "vehicle_width_mm": measured_vehicle_width_mm,
            "nominal_clearance_each_side_mm": nominal_line_clearance_mm,
        }
    else:
        clearance = {
            "checked": False,
            "reason": "supply --vehicle-width-mm from a physical measurement",
        }
    report = {
        "assumptions": {
            "control_fps": args.fps,
            "transition_seconds": args.transition_seconds,
            "lane_width_px": args.lane_width_px,
            "lane_width_mm_from_rules": lane_width_mm,
            "spatial_lead": args.spatial_lead,
            "command_speed": 255,
            "former_hard_target_first_shift_px": (
                hard_target.first_lookahead_shift_px
            ),
        },
        "vehicle_body_clearance": clearance,
        "outbound": asdict(outbound),
        "return": asdict(returning),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if args.show_records:
        report["outbound_records"] = outbound_records
        report["return_records"] = return_records
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
