from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .config import CameraConfig, SerialConfig
from .estimation.parking_fusion import ParkingFusionConfig
from .estimation.parking_geometry import ParkingGeometryConfig
from .estimation.parking_lidar import LidarParkingConfig, RectangleRoi
from .perception.bev import BevConfig
from .planning.hybrid_parking_path import HybridPathConfig, VehicleModel
from .planning.model_based_parking import ModelBasedParkingConfig
from .planning.reverse_parking_path import ReversePathConfig
from .planning.t_parking_planner import ParkingPlannerConfig


@dataclass(frozen=True)
class ParkingYoloConfig:
    model_path: str = "trained_model/0725best.pt"
    confidence: float = 0.35
    image_size: int = 640
    device: str = "auto"
    min_mask_area_ratio: float = 0.0003


@dataclass(frozen=True)
class ParkingRuntimeConfig:
    auto_start: bool = False
    camera_enabled: bool = True
    camera_debug_only: bool = False
    front_camera_enabled: bool = False
    command_rate_hz: float = 20.0
    lidar_video_offset_s: float = 0.0
    require_lidar: bool = True
    debug_window: bool = True
    lidar_display_rotation_deg: float = 0.0
    # Legacy ``lidar_debug_*`` names are retained for config compatibility, but
    # these dimensions now drive both visualization and full-inside control.
    lidar_debug_vehicle_width_mm: float = 600.0
    lidar_debug_vehicle_length_mm: float = 1000.0
    # Positive distance means the LiDAR origin is behind the rear bumper.
    lidar_debug_sensor_behind_vehicle_rear_mm: float = 100.0
    lidar_debug_rear_axle_to_rear_bumper_mm: float = 200.0
    locked_slot_tracking_enabled: bool = True
    locked_slot_min_points: int = 8
    locked_slot_max_points: int = 180
    locked_slot_min_range_mm: float = 200.0
    locked_slot_max_range_mm: float = 3500.0
    locked_slot_max_correspondence_mm: float = 320.0
    locked_slot_trim_ratio: float = 0.65
    locked_slot_iterations: int = 6
    locked_slot_max_translation_per_scan_mm: float = 300.0
    locked_slot_max_rotation_per_scan_deg: float = 15.0
    locked_slot_max_hold_scans: int = 12


@dataclass(frozen=True)
class ParkingAppConfig:
    rear_camera: CameraConfig
    front_camera: CameraConfig
    serial: SerialConfig
    yolo: ParkingYoloConfig
    bev: BevConfig
    geometry: ParkingGeometryConfig
    fusion: ParkingFusionConfig
    lidar: LidarParkingConfig
    path: ReversePathConfig
    planner: ParkingPlannerConfig
    vehicle: VehicleModel
    hybrid_path: HybridPathConfig
    model_planner: ModelBasedParkingConfig
    runtime: ParkingRuntimeConfig


def load_parking_config(path: str) -> ParkingAppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("parking config root must be an object")

    lidar_data = section(data, "lidar")
    car_roi = RectangleRoi(**section(lidar_data, "car_detection_roi"))
    safety_roi = RectangleRoi(**section(lidar_data, "safety_roi"))
    tracking_roi_data = lidar_data.get("slot_tracking_roi")
    lidar_values = {
        key: value
        for key, value in lidar_data.items()
        if key not in ("car_detection_roi", "safety_roi", "slot_tracking_roi")
    }
    lidar_values["car_detection_roi"] = car_roi
    lidar_values["safety_roi"] = safety_roi
    if tracking_roi_data is not None:
        if not isinstance(tracking_roi_data, dict):
            raise ValueError("config section 'slot_tracking_roi' must be an object")
        lidar_values["slot_tracking_roi"] = RectangleRoi(**tracking_roi_data)

    config = ParkingAppConfig(
        rear_camera=CameraConfig(**section(data, "rear_camera")),
        front_camera=CameraConfig(**section(data, "front_camera")),
        serial=SerialConfig(**section(data, "serial")),
        yolo=ParkingYoloConfig(**section(data, "yolo")),
        bev=BevConfig(**section(data, "bev")),
        geometry=ParkingGeometryConfig(**section(data, "geometry")),
        fusion=ParkingFusionConfig(**section(data, "fusion")),
        lidar=LidarParkingConfig(**lidar_values),
        path=ReversePathConfig(**section(data, "path")),
        planner=ParkingPlannerConfig(**section(data, "planner")),
        vehicle=VehicleModel(**section(data, "vehicle")),
        hybrid_path=HybridPathConfig(**section(data, "hybrid_path")),
        model_planner=ModelBasedParkingConfig(**section(data, "model_planner")),
        runtime=ParkingRuntimeConfig(**section(data, "runtime")),
    )
    validate_model_based_parking_config(config)
    return config


def section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("config section '%s' must be an object" % name)
    return value


def validate_model_based_parking_config(config: ParkingAppConfig) -> None:
    vehicle = config.vehicle
    planner = config.model_planner
    if config.lidar.max_distance_mm <= config.lidar.min_distance_mm:
        raise ValueError(
            "lidar.max_distance_mm must be larger than lidar.min_distance_mm"
        )
    if (
        config.lidar.gap_cluster_max_span_mm > 0.0
        and config.lidar.gap_cluster_max_span_mm
        <= config.lidar.gap_cluster_min_span_mm
    ):
        raise ValueError(
            "lidar.gap_cluster_max_span_mm must be larger than "
            "lidar.gap_cluster_min_span_mm"
        )
    if vehicle.wheelbase_mm <= 0.0:
        raise ValueError("vehicle.wheelbase_mm must be positive")
    if vehicle.width_mm <= 0.0 or vehicle.length_mm <= 0.0:
        raise ValueError("vehicle width and length must be positive")
    if not 0.0 < vehicle.max_steering_angle_deg < 60.0:
        raise ValueError("vehicle.max_steering_angle_deg must be between 0 and 60")
    if not 0.0 <= vehicle.rear_axle_to_rear_bumper_mm < vehicle.length_mm:
        raise ValueError(
            "vehicle.rear_axle_to_rear_bumper_mm must be within vehicle length"
        )
    if not 3.0 <= planner.park_hold_s <= 5.0:
        raise ValueError("model_planner.park_hold_s must be between 3 and 5")
    if planner.maneuver_forward_speed <= 0:
        raise ValueError("model_planner.maneuver_forward_speed must be positive")
    if planner.maneuver_reverse_speed >= 0 or planner.final_reverse_speed >= 0:
        raise ValueError("model planner reverse speeds must be negative")
    if planner.entry_setup_speed <= 0:
        raise ValueError("model_planner.entry_setup_speed must be positive")
    if (
        planner.lidar_first_car_gate_min_x_mm < 0.0
        or planner.lidar_first_car_gate_max_x_mm
        <= planner.lidar_first_car_gate_min_x_mm
    ):
        raise ValueError("model_planner LiDAR first-car gate is invalid")
    if planner.lidar_first_car_gate_max_range_mm <= 0.0:
        raise ValueError("LiDAR first-car range must be positive")
    if (
        planner.lidar_first_car_confirm_scans <= 0
        or planner.lidar_first_car_lost_scans <= 0
        or planner.pair_confirm_scans <= 0
        or planner.parking_complete_confirm_scans <= 0
    ):
        raise ValueError("LiDAR triangulation scan counts must be positive")
    if planner.reverse_lookahead_mm <= 0.0:
        raise ValueError("reverse_lookahead_mm must be positive")
    if not 0.0 <= planner.steering_filter_alpha <= 1.0:
        raise ValueError("steering_filter_alpha must be between 0 and 1")
    if planner.steering_max_delta_per_scan <= 0:
        raise ValueError("steering_max_delta_per_scan must be positive")
