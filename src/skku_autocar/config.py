import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class CameraConfig:
    index: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fourcc: str = "MJPG"


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


@dataclass(frozen=True)
class MissionConfig:
    mode: str = "time_trial"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
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
