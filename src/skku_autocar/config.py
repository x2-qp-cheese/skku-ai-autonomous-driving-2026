from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SerialConfig:
    port: Optional[str] = None
    baudrate: int = 115200
    timeout_s: float = 0.05
    startup_delay_s: float = 0.0
    ready_timeout_s: float = 3.0


@dataclass(frozen=True)
class RearLidarConfig:
    """Sensor mounting plus the constants stated in the paper."""

    port: Optional[str] = None
    angle_offset_deg: float = -90.0
    clockwise_angles: bool = False
    rear_fov_deg: float = 110.0
    stale_after_s: float = 0.45
    near_distance_mm: float = 600.0
    prealign_distance_mm: float = 1250.0
    side_angle_min_deg: float = 70.0
    side_angle_max_deg: float = 100.0
    side_distance_limit_mm: float = 2000.0
    scan_type: str = "normal"
    input_buffer_limit_bytes: int = 3000
    # The paper does not define how the two parked vehicles are detected.
    # These values belong only to LiDAR object clustering and scan
    # confirmation; the paper's parking transitions and equations remain
    # unchanged.
    vehicle_min_cluster_points: int = 3
    vehicle_cluster_base_gap_mm: float = 150.0
    vehicle_cluster_max_gap_mm: float = 350.0
    vehicle_cluster_gap_factor: float = 2.5
    vehicle_cluster_min_extent_mm: float = 60.0
    vehicle_cluster_max_extent_mm: float = 1200.0
    vehicle_track_max_shift_mm: float = 500.0
    vehicle_confirm_scans: int = 2
    gap_confirm_scans: int = 3
    # A/B are estimated from several recent rotations because a single
    # rotation contains too few returns from the curved toy-car bodies.
    tangent_fusion_scans: int = 5
    tangent_cluster_min_extent_mm: float = 30.0
    tangent_distance_limit_mm: float = 2500.0
    tangent_min_angular_gap_deg: float = 15.0
    side_line_min_angle_deg: float = 25.0
    side_line_min_points: int = 4
    side_line_min_extent_mm: float = 100.0
    side_line_min_linearity: float = 5.0
    side_line_max_disagreement_deg: float = 20.0


@dataclass(frozen=True)
class PaperControllerConfig:
    """Values from Hong et al., Figure 9 and Equations (2)-(5)."""

    park_hold_s: float = 4.0
    exit_speed: int = 50
    exit_forward_s: float = 4.0
    exit_turn_steering: int = 150
    exit_turn_right_s: float = 5.0
    first_vehicle_end_trigger_y_mm: float = 30.0
    forward_speed: int = 80
    reverse_speed: int = -80
    inside_reverse_speed: int = -50
    parallel_forward_speed: int = 35
    parallel_forward_max_steering: float = 3.0
    # The paper's 600 mm setup threshold assumes its own lateral spacing.
    # Our prealign_distance_mm is the scaled equivalent.
    prealign_clear_confirm_scans: int = 3
    pair_filter_scans: int = 3
    pair_hold_scans: int = 3
    pair_max_angle_jump_deg: float = 15.0
    pair_max_distance_jump_mm: float = 700.0
    center_observation_scans: int = 4
    side_entry_confirm_scans: int = 3
    cd_direct_reverse_min_distance_mm: float = 600.0
    park_finish_distance_jump_mm: float = 100.0
    park_finish_side_missing_scans: int = 3
    cd_center_confirm_scans: int = 5
    paper_max_steering: float = 7.0
    reverse_align_max_steering: float = 6.0
    reverse_align_steering_max_step: float = 1.0
    actuator_max_steering: int = 150
    actuator_steering_offset: int = 0
    distance_bias_scale_mm: float = 600.0
    entry_start_angle_deg: float = 30.0
    entry_full_steer_angle_deg: float = 30.0
    cd_target_balance_ratio: float = -0.16
    cd_full_steer_error_ratio: float = 0.15
    cd_center_tolerance_ratio: float = 0.025
    cd_stability_span_ratio: float = 0.04
    cd_steering_max_step: float = 1.0
    cd_balance_steering_weight: float = 0.45
    parallel_heading_trigger_deg: float = 14.0
    parallel_heading_exit_deg: float = 7.0
    parallel_heading_tolerance_deg: float = 7.0
    parallel_heading_full_steer_deg: float = 25.0
    parallel_heading_confirm_scans: int = 5
    parallel_heading_missing_scans: int = 3
    parallel_heading_stability_deg: float = 6.0
    parallel_cd_missing_exit_scans: int = 5
    parallel_no_improvement_exit_scans: int = 15
    parallel_min_improvement_deg: float = 0.25
    dist_bias_cd_threshold_mm: float = 250.0
    recovery_forward_s: float = 3.0
    command_rate_hz: float = 20.0


@dataclass(frozen=True)
class RuntimeConfig:
    auto_start: bool = False
    motor_enabled: bool = True
    debug_window: bool = True
    record_debug_video: bool = True
    display_range_mm: float = 3000.0
    record_directory: str = "data/parking_v2"


@dataclass(frozen=True)
class AppConfig:
    serial: SerialConfig
    lidar: RearLidarConfig
    controller: PaperControllerConfig
    runtime: RuntimeConfig


def load_config(path: str) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = AppConfig(
        serial=SerialConfig(**_section(raw, "serial")),
        lidar=RearLidarConfig(**_section(raw, "lidar")),
        controller=PaperControllerConfig(**_section(raw, "controller")),
        runtime=RuntimeConfig(**_section(raw, "runtime")),
    )
    _validate(config)
    return config


def _section(raw: object, name: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("config root must be an object")
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("config section %r must be an object" % name)
    return value


def _validate(config: AppConfig) -> None:
    lidar = config.lidar
    controller = config.controller
    if not 0.0 < lidar.rear_fov_deg <= 180.0:
        raise ValueError("rear_fov_deg must be in (0, 180]")
    if lidar.near_distance_mm <= 0.0:
        raise ValueError("near_distance_mm must be positive")
    if lidar.prealign_distance_mm <= 0.0:
        raise ValueError("prealign_distance_mm must be positive")
    if lidar.side_distance_limit_mm <= 0.0:
        raise ValueError("side_distance_limit_mm must be positive")
    if lidar.scan_type not in ("normal", "express"):
        raise ValueError("scan_type must be 'normal' or 'express'")
    if lidar.input_buffer_limit_bytes < 500:
        raise ValueError(
            "input_buffer_limit_bytes must be at least 500"
        )
    if lidar.vehicle_min_cluster_points < 2:
        raise ValueError("vehicle_min_cluster_points must be at least 2")
    if lidar.vehicle_cluster_base_gap_mm <= 0.0:
        raise ValueError("vehicle_cluster_base_gap_mm must be positive")
    if (
        lidar.vehicle_cluster_max_gap_mm
        < lidar.vehicle_cluster_base_gap_mm
    ):
        raise ValueError(
            "vehicle_cluster_max_gap_mm must be >= base gap"
        )
    if lidar.vehicle_cluster_gap_factor <= 0.0:
        raise ValueError("vehicle_cluster_gap_factor must be positive")
    if lidar.vehicle_cluster_min_extent_mm <= 0.0:
        raise ValueError("vehicle_cluster_min_extent_mm must be positive")
    if (
        lidar.vehicle_cluster_max_extent_mm
        < lidar.vehicle_cluster_min_extent_mm
    ):
        raise ValueError(
            "vehicle_cluster_max_extent_mm must be >= min extent"
        )
    if lidar.vehicle_track_max_shift_mm <= 0.0:
        raise ValueError("vehicle_track_max_shift_mm must be positive")
    if lidar.vehicle_confirm_scans < 1:
        raise ValueError("vehicle_confirm_scans must be positive")
    if lidar.gap_confirm_scans < 1:
        raise ValueError("gap_confirm_scans must be positive")
    if lidar.tangent_fusion_scans < 1:
        raise ValueError("tangent_fusion_scans must be positive")
    if lidar.tangent_cluster_min_extent_mm <= 0.0:
        raise ValueError(
            "tangent_cluster_min_extent_mm must be positive"
        )
    if (
        lidar.tangent_cluster_min_extent_mm
        > lidar.vehicle_cluster_max_extent_mm
    ):
        raise ValueError(
            "tangent cluster extent exceeds maximum extent"
        )
    if lidar.tangent_distance_limit_mm <= 0.0:
        raise ValueError(
            "tangent_distance_limit_mm must be positive"
        )
    if not 0.0 < lidar.tangent_min_angular_gap_deg < 180.0:
        raise ValueError(
            "tangent_min_angular_gap_deg must be in (0, 180)"
        )
    if not 0.0 <= lidar.side_line_min_angle_deg < lidar.rear_fov_deg:
        raise ValueError(
            "side_line_min_angle_deg must be in [0, rear_fov_deg)"
        )
    if lidar.side_line_min_points < 2:
        raise ValueError("side_line_min_points must be at least 2")
    if lidar.side_line_min_extent_mm <= 0.0:
        raise ValueError("side_line_min_extent_mm must be positive")
    if lidar.side_line_min_linearity <= 1.0:
        raise ValueError("side_line_min_linearity must exceed 1")
    if not 0.0 < lidar.side_line_max_disagreement_deg <= 90.0:
        raise ValueError(
            "side_line_max_disagreement_deg must be in (0, 90]"
        )
    if controller.paper_max_steering <= 0.0:
        raise ValueError("paper_max_steering must be positive")
    if not (
        0.0
        < controller.reverse_align_max_steering
        <= controller.paper_max_steering
    ):
        raise ValueError(
            "reverse_align_max_steering must be in "
            "(0, paper_max_steering]"
        )
    if controller.reverse_align_steering_max_step <= 0.0:
        raise ValueError(
            "reverse_align_steering_max_step must be positive"
        )
    if controller.actuator_max_steering <= 0:
        raise ValueError("actuator_max_steering must be positive")
    if (
        abs(controller.actuator_steering_offset)
        > controller.actuator_max_steering
    ):
        raise ValueError(
            "actuator_steering_offset exceeds steering range"
        )
    if controller.forward_speed <= 0:
        raise ValueError("forward_speed must be positive")
    if controller.first_vehicle_end_trigger_y_mm < 0.0:
        raise ValueError(
            "first_vehicle_end_trigger_y_mm cannot be negative"
        )
    if controller.reverse_speed >= 0:
        raise ValueError("reverse_speed must be negative")
    if controller.inside_reverse_speed >= 0:
        raise ValueError("inside_reverse_speed must be negative")
    if controller.parallel_forward_speed <= 0:
        raise ValueError("parallel_forward_speed must be positive")
    if not (
        0.0
        < controller.parallel_forward_max_steering
        <= controller.paper_max_steering
    ):
        raise ValueError(
            "parallel_forward_max_steering must be in "
            "(0, paper_max_steering]"
        )
    if controller.prealign_clear_confirm_scans < 1:
        raise ValueError(
            "prealign_clear_confirm_scans must be positive"
        )
    if controller.pair_filter_scans < 1:
        raise ValueError("pair_filter_scans must be positive")
    if controller.pair_hold_scans < 0:
        raise ValueError("pair_hold_scans cannot be negative")
    if controller.pair_max_angle_jump_deg <= 0.0:
        raise ValueError("pair_max_angle_jump_deg must be positive")
    if controller.pair_max_distance_jump_mm <= 0.0:
        raise ValueError(
            "pair_max_distance_jump_mm must be positive"
        )
    if not 0.0 < controller.entry_full_steer_angle_deg <= 90.0:
        raise ValueError(
            "entry_full_steer_angle_deg must be in (0, 90]"
        )
    if not 0.0 < controller.entry_start_angle_deg <= 90.0:
        raise ValueError(
            "entry_start_angle_deg must be in (0, 90]"
        )
    if controller.center_observation_scans < 1:
        raise ValueError("center_observation_scans must be positive")
    if controller.side_entry_confirm_scans < 1:
        raise ValueError("side_entry_confirm_scans must be positive")
    if controller.cd_direct_reverse_min_distance_mm <= 0.0:
        raise ValueError(
            "cd_direct_reverse_min_distance_mm must be positive"
        )
    if controller.park_finish_distance_jump_mm <= 0.0:
        raise ValueError(
            "park_finish_distance_jump_mm must be positive"
        )
    if controller.park_finish_side_missing_scans < 1:
        raise ValueError(
            "park_finish_side_missing_scans must be positive"
        )
    if controller.cd_center_confirm_scans < 1:
        raise ValueError("cd_center_confirm_scans must be positive")
    if controller.cd_full_steer_error_ratio <= 0.0:
        raise ValueError(
            "cd_full_steer_error_ratio must be positive"
        )
    if controller.cd_center_tolerance_ratio <= 0.0:
        raise ValueError(
            "cd_center_tolerance_ratio must be positive"
        )
    if controller.cd_stability_span_ratio <= 0.0:
        raise ValueError(
            "cd_stability_span_ratio must be positive"
        )
    if controller.cd_steering_max_step <= 0.0:
        raise ValueError("cd_steering_max_step must be positive")
    if not 0.0 < controller.cd_balance_steering_weight <= 1.0:
        raise ValueError(
            "cd_balance_steering_weight must be in (0, 1]"
        )
    if controller.parallel_heading_trigger_deg <= 0.0:
        raise ValueError(
            "parallel_heading_trigger_deg must be positive"
        )
    if not (
        0.0
        < controller.parallel_heading_exit_deg
        < controller.parallel_heading_trigger_deg
    ):
        raise ValueError(
            "parallel_heading_exit_deg must be below trigger"
        )
    if controller.parallel_heading_tolerance_deg <= 0.0:
        raise ValueError(
            "parallel_heading_tolerance_deg must be positive"
        )
    if controller.parallel_heading_full_steer_deg <= 0.0:
        raise ValueError(
            "parallel_heading_full_steer_deg must be positive"
        )
    if controller.parallel_heading_confirm_scans < 1:
        raise ValueError(
            "parallel_heading_confirm_scans must be positive"
        )
    if controller.parallel_heading_missing_scans < 1:
        raise ValueError(
            "parallel_heading_missing_scans must be positive"
        )
    if controller.parallel_heading_stability_deg <= 0.0:
        raise ValueError(
            "parallel_heading_stability_deg must be positive"
        )
    if controller.parallel_cd_missing_exit_scans < 1:
        raise ValueError(
            "parallel_cd_missing_exit_scans must be positive"
        )
    if controller.parallel_no_improvement_exit_scans < 1:
        raise ValueError(
            "parallel_no_improvement_exit_scans must be positive"
        )
    if controller.parallel_min_improvement_deg <= 0.0:
        raise ValueError(
            "parallel_min_improvement_deg must be positive"
        )
    if not 3.0 <= controller.park_hold_s <= 5.0:
        raise ValueError("park_hold_s must be in [3, 5]")
    if controller.exit_speed <= 0:
        raise ValueError("exit_speed must be positive")
    if controller.exit_forward_s <= 0.0:
        raise ValueError("exit_forward_s must be positive")
    if not (
        0
        < controller.exit_turn_steering
        <= controller.actuator_max_steering
    ):
        raise ValueError(
            "exit_turn_steering must be in actuator range"
        )
    if controller.exit_turn_right_s <= 0.0:
        raise ValueError("exit_turn_right_s must be positive")
    if controller.command_rate_hz <= 0.0:
        raise ValueError("command_rate_hz must be positive")
