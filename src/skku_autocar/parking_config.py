from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .config import CameraConfig, SerialConfig
from .estimation.parking_geometry import ParkingGeometryConfig
from .estimation.parking_lidar import LidarParkingConfig, RectangleRoi
from .perception.bev import BevConfig
from .planning.reverse_parking_path import ReversePathConfig
from .planning.t_parking_planner import ParkingPlannerConfig


@dataclass(frozen=True)
class ParkingYoloConfig:
    model_path: str = "trained_model/parking_best.pt"
    confidence: float = 0.35
    image_size: int = 640
    device: str = "auto"
    min_mask_area_ratio: float = 0.0003


@dataclass(frozen=True)
class ParkingRuntimeConfig:
    command_rate_hz: float = 20.0
    lidar_video_offset_s: float = 0.0
    require_lidar: bool = True
    debug_window: bool = True
    lidar_display_rotation_deg: float = 0.0
    lidar_debug_vehicle_width_mm: float = 550.0
    lidar_debug_vehicle_length_mm: float = 1000.0
    # Positive distance means the LiDAR origin is behind the rear bumper.
    lidar_debug_sensor_behind_vehicle_rear_mm: float = 100.0


@dataclass(frozen=True)
class ParkingAppConfig:
    rear_camera: CameraConfig
    serial: SerialConfig
    yolo: ParkingYoloConfig
    bev: BevConfig
    geometry: ParkingGeometryConfig
    lidar: LidarParkingConfig
    path: ReversePathConfig
    planner: ParkingPlannerConfig
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
    lidar_values = {
        key: value
        for key, value in lidar_data.items()
        if key not in ("car_detection_roi", "safety_roi")
    }
    lidar_values["car_detection_roi"] = car_roi
    lidar_values["safety_roi"] = safety_roi

    return ParkingAppConfig(
        rear_camera=CameraConfig(**section(data, "rear_camera")),
        serial=SerialConfig(**section(data, "serial")),
        yolo=ParkingYoloConfig(**section(data, "yolo")),
        bev=BevConfig(**section(data, "bev")),
        geometry=ParkingGeometryConfig(**section(data, "geometry")),
        lidar=LidarParkingConfig(**lidar_values),
        path=ReversePathConfig(**section(data, "path")),
        planner=ParkingPlannerConfig(**section(data, "planner")),
        runtime=ParkingRuntimeConfig(**section(data, "runtime")),
    )


def section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("config section '%s' must be an object" % name)
    return value
