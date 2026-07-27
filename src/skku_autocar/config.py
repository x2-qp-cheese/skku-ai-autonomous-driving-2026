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
    side_angle_min_deg: float = 70.0
    side_angle_max_deg: float = 100.0
    side_distance_limit_mm: float = 2000.0


@dataclass(frozen=True)
class PaperControllerConfig:
    """Values from Hong et al., Figure 9 and Equations (2)-(5)."""

    forward_speed: int = 80
    reverse_speed: int = -80
    paper_max_steering: float = 7.0
    actuator_max_steering: int = 150
    actuator_steering_offset: int = 0
    distance_bias_scale_mm: float = 600.0
    dist_bias_cd_threshold_mm: float = 250.0
    recovery_forward_s: float = 3.0
    command_rate_hz: float = 20.0


@dataclass(frozen=True)
class RuntimeConfig:
    auto_start: bool = False
    motor_enabled: bool = True
    debug_window: bool = True
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
    if lidar.side_distance_limit_mm <= 0.0:
        raise ValueError("side_distance_limit_mm must be positive")
    if controller.paper_max_steering <= 0.0:
        raise ValueError("paper_max_steering must be positive")
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
    if controller.reverse_speed >= 0:
        raise ValueError("reverse_speed must be negative")
    if controller.command_rate_hz <= 0.0:
        raise ValueError("command_rate_hz must be positive")
