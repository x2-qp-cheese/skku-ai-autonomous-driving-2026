from __future__ import annotations

from math import atan2, degrees, hypot
from typing import Optional, Tuple

from .parking_geometry import ParkingGeometry, ParkingGeometryConfig, ParkingLine
from .parking_lidar import (
    LidarParkingConfig,
    LidarParkingObservation,
    infer_dynamic_slot_polygon,
)


Point = Tuple[float, float]


class LidarSlotGeometryProjector:
    """Project the tracked LiDAR parking box into the rear-BEV path frame.

    LiDAR coordinates use x to vehicle-right and y toward the rear.  Rear-BEV
    pixels use x to vehicle-right and y toward the vehicle, so positive LiDAR
    rear distance maps to decreasing image y.  The projected rectangle becomes
    the virtual left, right, and back lines consumed by the reverse path planner.
    """

    def __init__(
        self,
        lidar_config: LidarParkingConfig,
        geometry_config: ParkingGeometryConfig,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("LiDAR slot geometry canvas must be positive")
        if lidar_config.parking_space_width_mm <= 0.0:
            raise ValueError("parking_space_width_mm must be positive")
        self.lidar_config = lidar_config
        self.geometry_config = geometry_config
        self.canvas_width = int(canvas_width)
        self.canvas_height = int(canvas_height)
        self.pixels_per_mm = (
            geometry_config.expected_slot_width_px
            / lidar_config.parking_space_width_mm
        )

    def project(self, observation: LidarParkingObservation) -> ParkingGeometry:
        polygon = infer_dynamic_slot_polygon(
            observation,
            self.lidar_config.parking_space_depth_mm,
        )
        if polygon is None:
            return ParkingGeometry(reason="lidar_slot_box_unavailable")

        points = tuple(self._world_to_bev(point) for point in polygon)
        entrance_first, entrance_second, far_second, far_first = points
        entrance_center = midpoint(entrance_first, entrance_second)
        back_center = midpoint(far_first, far_second)
        direction = normalize(subtract(back_center, entrance_center))
        if direction is None:
            return ParkingGeometry(reason="lidar_slot_box_invalid_depth")

        first_side = virtual_line(entrance_first, far_first, direction)
        second_side = virtual_line(entrance_second, far_second, direction)
        if first_side is None or second_side is None:
            return ParkingGeometry(reason="lidar_slot_box_invalid_side")

        # For a bay pointing upward in rear BEV, the left normal is (-1, 0).
        # This role assignment also works while the box rotates with the car.
        left_normal = (direction[1], -direction[0])
        first_side_mid = midpoint(entrance_first, far_first)
        second_side_mid = midpoint(entrance_second, far_second)
        if dot(first_side_mid, left_normal) >= dot(second_side_mid, left_normal):
            left, left_far = first_side, far_first
            right, right_far = second_side, far_second
        else:
            left, left_far = second_side, far_second
            right, right_far = first_side, far_first

        back = virtual_line(left_far, right_far)
        if back is None:
            return ParkingGeometry(reason="lidar_slot_box_invalid_back")

        vehicle = self._rear_axle_pixel()
        clearance = max(0.0, self.geometry_config.desired_back_clearance_px)
        stop_target = (
            back_center[0] - direction[0] * clearance,
            back_center[1] - direction[1] * clearance,
        )
        depth_to_back = dot(subtract(back_center, vehicle), direction)
        depth_remaining = dot(subtract(stop_target, vehicle), direction)
        slot_width = distance(entrance_first, entrance_second)
        slot_center = midpoint(entrance_center, back_center)

        # Positive lateral error means the bay center is to the vehicle-right
        # when the bay direction is straight up in the rear-BEV frame.
        lateral_error = dot(subtract(vehicle, entrance_center), left_normal)
        lateral_norm = lateral_error / max(1.0, slot_width / 2.0)
        heading_error = degrees(atan2(direction[0], -direction[1]))

        confidence = 0.65 if observation.coasted else 0.95
        confirmed = observation.gap_confirmed
        found = (
            confirmed
            and confidence >= self.geometry_config.min_geometry_confidence
        )
        if observation.coasted:
            reason = "lidar_slot_box_hold"
        elif confirmed:
            reason = "lidar_slot_box"
        else:
            reason = "lidar_slot_box_confirming"

        return ParkingGeometry(
            found=found,
            has_side_pair=True,
            has_back_line=True,
            left=left,
            right=right,
            back=back,
            lateral_error_px=lateral_error,
            lateral_error_norm=clip(lateral_norm, -2.0, 2.0),
            heading_error_deg=heading_error,
            depth_to_back_px=depth_to_back,
            depth_remaining_px=depth_remaining,
            slot_width_px=slot_width,
            vehicle_x_px=vehicle[0],
            vehicle_y_px=vehicle[1],
            slot_center_x_px=slot_center[0],
            slot_center_y_px=slot_center[1],
            slot_direction_x=direction[0],
            slot_direction_y=direction[1],
            back_center_x_px=back_center[0],
            back_center_y_px=back_center[1],
            stop_target_x_px=stop_target[0],
            stop_target_y_px=stop_target[1],
            confidence=confidence,
            observed_line_count=0,
            coasted=observation.coasted,
            reason=reason,
        )

    def _rear_axle_pixel(self) -> Point:
        return (
            self.canvas_width * self.geometry_config.vehicle_center_x_ratio,
            self.canvas_height * self.geometry_config.vehicle_reference_y_ratio,
        )

    def _world_to_bev(self, point: Point) -> Point:
        rear_axle = self._rear_axle_pixel()
        relative_x = point[0]
        relative_y_back = (
            point[1] - self.lidar_config.sensor_to_rear_axle_y_back_mm
        )
        return (
            rear_axle[0] + relative_x * self.pixels_per_mm,
            rear_axle[1] - relative_y_back * self.pixels_per_mm,
        )


def virtual_line(
    first: Point,
    second: Point,
    preferred_direction: Optional[Point] = None,
) -> Optional[ParkingLine]:
    vector = subtract(second, first)
    direction = normalize(vector)
    if direction is None:
        return None
    if preferred_direction is not None and dot(direction, preferred_direction) < 0.0:
        direction = (-direction[0], -direction[1])
    return ParkingLine(
        center_x=(first[0] + second[0]) / 2.0,
        center_y=(first[1] + second[1]) / 2.0,
        direction_x=direction[0],
        direction_y=direction[1],
        length_px=distance(first, second),
        residual_px=0.0,
        quality=1.0,
        point_count=2,
        mask_index=-1,
    )


def midpoint(first: Point, second: Point) -> Point:
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def subtract(first: Point, second: Point) -> Point:
    return first[0] - second[0], first[1] - second[1]


def distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def normalize(vector: Point) -> Optional[Point]:
    length = hypot(vector[0], vector[1])
    if length <= 1e-9:
        return None
    return vector[0] / length, vector[1] / length


def dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
