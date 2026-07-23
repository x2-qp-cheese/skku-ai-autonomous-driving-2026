from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .parking_geometry import ParkingGeometry


@dataclass(frozen=True)
class ParkingFusionConfig:
    """Policy for combining LiDAR-selected slots with camera line geometry."""

    camera_min_confidence: float = 0.35
    max_lateral_disagreement_norm: float = 0.65
    max_heading_disagreement_deg: float = 25.0
    max_depth_disagreement_px: float = 220.0
    prefer_camera_back_line: bool = True


def fuse_parking_geometry(
    lidar_geometry: ParkingGeometry,
    camera_geometry: Optional[ParkingGeometry],
    config: ParkingFusionConfig = ParkingFusionConfig(),
) -> ParkingGeometry:
    """Use LiDAR for slot identity and camera lines for fine reverse control.

    The competition setup makes the target bay identifiable by the two parked
    cars that border it, so LiDAR remains the authority for selecting the slot.
    Camera geometry is allowed to take over only after it agrees with that
    LiDAR-selected slot closely enough. If the camera is missing, noisy, or
    looking at a neighboring painted bay, the LiDAR box remains the fallback.
    """

    if not _full_geometry(lidar_geometry):
        return lidar_geometry
    if not _usable_camera(camera_geometry, config):
        return lidar_geometry

    assert camera_geometry is not None
    compatible, _ = camera_lidar_compatible(
        lidar_geometry,
        camera_geometry,
        config,
    )
    if not compatible:
        return lidar_geometry

    confidence = min(
        1.0,
        max(lidar_geometry.confidence, camera_geometry.confidence),
    )
    if camera_geometry.has_back_line and config.prefer_camera_back_line:
        return replace(
            camera_geometry,
            vehicle_inside_ratio=lidar_geometry.vehicle_inside_ratio,
            vehicle_fully_inside=lidar_geometry.vehicle_fully_inside,
            confidence=confidence,
            coasted=camera_geometry.coasted or lidar_geometry.coasted,
            reason="camera_lidar_fused",
        )

    return replace(
        lidar_geometry,
        left=camera_geometry.left,
        right=camera_geometry.right,
        lateral_error_px=camera_geometry.lateral_error_px,
        lateral_error_norm=camera_geometry.lateral_error_norm,
        heading_error_deg=camera_geometry.heading_error_deg,
        slot_width_px=camera_geometry.slot_width_px,
        slot_center_x_px=camera_geometry.slot_center_x_px,
        slot_center_y_px=camera_geometry.slot_center_y_px,
        slot_direction_x=camera_geometry.slot_direction_x,
        slot_direction_y=camera_geometry.slot_direction_y,
        vehicle_inside_ratio=lidar_geometry.vehicle_inside_ratio,
        vehicle_fully_inside=lidar_geometry.vehicle_fully_inside,
        confidence=confidence,
        observed_line_count=camera_geometry.observed_line_count,
        coasted=camera_geometry.coasted or lidar_geometry.coasted,
        reason="camera_side_lidar_depth",
    )


def camera_lidar_compatible(
    lidar_geometry: ParkingGeometry,
    camera_geometry: ParkingGeometry,
    config: ParkingFusionConfig = ParkingFusionConfig(),
) -> tuple[bool, str]:
    if not _full_geometry(lidar_geometry):
        return False, "lidar_geometry_unusable"
    if not _usable_camera(camera_geometry, config):
        return False, "camera_geometry_unusable"

    lateral_disagreement = abs(
        camera_geometry.lateral_error_norm - lidar_geometry.lateral_error_norm
    )
    if lateral_disagreement > config.max_lateral_disagreement_norm:
        return False, "lateral_disagreement"

    heading_disagreement = angular_difference_deg(
        camera_geometry.heading_error_deg,
        lidar_geometry.heading_error_deg,
    )
    if heading_disagreement > config.max_heading_disagreement_deg:
        return False, "heading_disagreement"

    if (
        camera_geometry.has_back_line
        and camera_geometry.depth_remaining_px is not None
        and lidar_geometry.depth_remaining_px is not None
        and abs(camera_geometry.depth_remaining_px - lidar_geometry.depth_remaining_px)
        > config.max_depth_disagreement_px
    ):
        return False, "depth_disagreement"

    return True, "compatible"


def _usable_camera(
    camera_geometry: Optional[ParkingGeometry],
    config: ParkingFusionConfig,
) -> bool:
    return (
        camera_geometry is not None
        and camera_geometry.found
        and camera_geometry.has_side_pair
        and camera_geometry.confidence >= config.camera_min_confidence
    )


def _full_geometry(geometry: ParkingGeometry) -> bool:
    return (
        geometry.found
        and geometry.has_side_pair
        and geometry.has_back_line
        and geometry.depth_remaining_px is not None
        and geometry.stop_target_x_px is not None
        and geometry.stop_target_y_px is not None
    )


def angular_difference_deg(first: float, second: float) -> float:
    return abs(((first - second + 180.0) % 360.0) - 180.0)
