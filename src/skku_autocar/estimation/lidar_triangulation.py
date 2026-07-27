from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, degrees, hypot
from typing import Optional, Tuple

from .parking_lidar import LidarParkingObservation


Point = Tuple[float, float]


@dataclass(frozen=True)
class LidarDecisionTriangle:
    """LiDAR-car1-car2 triangle used by the parking paper.

    Coordinates are adapted for this vehicle's rear-mounted LiDAR:
    ``x`` is positive to vehicle-right and ``y_back`` is positive behind the
    vehicle.  Therefore the reference direction for reverse parking is
    ``+y_back`` rather than the front direction used in the paper.
    """

    valid: bool = False
    lidar: Point = (0.0, 0.0)
    car1_edge: Point = (0.0, 0.0)
    car2_edge: Point = (0.0, 0.0)
    lidar_to_car1_mm: float = 0.0
    lidar_to_car2_mm: float = 0.0
    car_gap_mm: float = 0.0
    decision_angle_deg: float = 0.0
    correction_angle_deg: float = 0.0
    reason: str = "triangle_unavailable"


def decision_triangle_from_observation(
    observation: LidarParkingObservation,
) -> LidarDecisionTriangle:
    """Build the paper's decision triangle from a fresh two-car observation."""

    values = (
        observation.gap_near_edge_x_right_mm,
        observation.gap_near_edge_y_back_mm,
        observation.gap_far_edge_x_right_mm,
        observation.gap_far_edge_y_back_mm,
    )
    if (
        not observation.valid
        or not observation.gap_pair_observed
        or not observation.second_car_seen
        or observation.coasted
        or any(value is None for value in values)
    ):
        return LidarDecisionTriangle(reason="fresh_two_car_pair_required")

    car1 = (float(values[0]), float(values[1]))
    car2 = (float(values[2]), float(values[3]))
    lidar_to_car1 = hypot(car1[0], car1[1])
    lidar_to_car2 = hypot(car2[0], car2[1])
    gap = hypot(car2[0] - car1[0], car2[1] - car1[1])
    if min(lidar_to_car1, lidar_to_car2, gap) <= 1.0:
        return LidarDecisionTriangle(
            car1_edge=car1,
            car2_edge=car2,
            reason="degenerate_decision_triangle",
        )

    # Law of cosines: angle at car 1 between the LiDAR ray and car1-car2 edge.
    cosine = (
        lidar_to_car1 * lidar_to_car1
        + gap * gap
        - lidar_to_car2 * lidar_to_car2
    ) / (2.0 * lidar_to_car1 * gap)
    decision_angle = degrees(acos(max(-1.0, min(1.0, cosine))))

    depth_x = observation.slot_depth_x_right
    depth_y = observation.slot_depth_y_back
    if depth_x is None or depth_y is None:
        # Choose the perpendicular that points away from the LiDAR and into
        # the right-side parking bay.
        axis_x = (car2[0] - car1[0]) / gap
        axis_y = (car2[1] - car1[1]) / gap
        depth_x, depth_y = -axis_y, axis_x
        center_x = (car1[0] + car2[0]) / 2.0
        center_y = (car1[1] + car2[1]) / 2.0
        if center_x * depth_x + center_y * depth_y < 0.0:
            depth_x, depth_y = -depth_x, -depth_y
    depth_length = hypot(float(depth_x), float(depth_y))
    if depth_length <= 1e-6:
        return LidarDecisionTriangle(
            car1_edge=car1,
            car2_edge=car2,
            reason="invalid_slot_depth_direction",
        )

    # atan2(x, y_back) is the signed correction from the vehicle's rear axis.
    correction_angle = degrees(
        atan2(float(depth_x) / depth_length, float(depth_y) / depth_length)
    )
    return LidarDecisionTriangle(
        valid=True,
        car1_edge=car1,
        car2_edge=car2,
        lidar_to_car1_mm=lidar_to_car1,
        lidar_to_car2_mm=lidar_to_car2,
        car_gap_mm=gap,
        decision_angle_deg=decision_angle,
        correction_angle_deg=correction_angle,
        reason="lidar_decision_triangle_ready",
    )
