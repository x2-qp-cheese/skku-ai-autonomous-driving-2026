import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"


@dataclass(frozen=True)
class LaneConfig:
    # legacy fields kept for JSON backward-compatibility
    gray_threshold: int = 175
    min_contour_area: int = 700
    roi_bottom_left_x: float = 0.18
    roi_bottom_right_x: float = 0.82
    roi_top_left_x: float = 0.35
    roi_top_right_x: float = 0.65
    roi_top_y: float = 0.55
    # white lane color filter (HLS space)
    white_l_min: int = 200
    white_s_max: int = 80
    # Canny edge detection
    canny_low: int = 50
    canny_high: int = 150
    # Bird's Eye View perspective transform – source trapezoid (ratios of frame size)
    bev_src_bottom_left_x: float = 0.15
    bev_src_bottom_right_x: float = 0.85
    bev_src_top_left_x: float = 0.42
    bev_src_top_right_x: float = 0.58
    bev_src_top_y: float = 0.62
    bev_src_bottom_y: float = 0.93
    # BEV destination rectangle (ratios of warped image width)
    bev_dst_left_x: float = 0.25
    bev_dst_right_x: float = 0.75
    # Sliding window
    n_windows: int = 9
    window_margin: int = 80
    min_pix: int = 50
    # Polynomial degree (2 or 3)
    poly_deg: int = 2


@dataclass(frozen=True)
class LidarConfig:
    port: Optional[str] = None
    rpm: int = 660
    front_angle_min: float = 330.0
    front_angle_max: float = 30.0
    stop_distance_mm: float = 300.0


@dataclass(frozen=True)
class SerialConfig:
    arduino_port: Optional[str] = None
    baudrate: int = 115200
    timeout_s: float = 0.1


@dataclass(frozen=True)
class ControlConfig:
    base_speed: int = 90
    max_speed: int = 160
    max_steering: int = 120
    lane_kp: float = 85.0
    lane_lost_brake: bool = True


@dataclass(frozen=True)
class MissionConfig:
    mode: str = "time_trial"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    lane: LaneConfig
    lidar: LidarConfig
    serial: SerialConfig
    control: ControlConfig
    mission: MissionConfig


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError("config section '%s' must be an object" % name)
    return value


def config_from_dict(data: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        camera=CameraConfig(**_section(data, "camera")),
        lane=LaneConfig(**_section(data, "lane")),
        lidar=LidarConfig(**_section(data, "lidar")),
        serial=SerialConfig(**_section(data, "serial")),
        control=ControlConfig(**_section(data, "control")),
        mission=MissionConfig(**_section(data, "mission")),
    )


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return config_from_dict(data)
