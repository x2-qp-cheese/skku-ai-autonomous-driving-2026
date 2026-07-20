from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple

from ..estimation.parking_geometry import ParkingGeometry


Point = Tuple[float, float]


@dataclass(frozen=True)
class ReversePathConfig:
    """Image-plane path parameters for rear-BEV reverse parking."""

    samples: int = 41
    start_tangent_px: float = 120.0
    end_tangent_px: float = 120.0
    lookahead_px: float = 90.0
    minimum_target_distance_px: float = 45.0
    maximum_curvature_per_px: float = 0.040
    full_steering_curvature_per_px: float = 0.012


@dataclass(frozen=True)
class ReversePath:
    found: bool = False
    points: Tuple[Point, ...] = ()
    lookahead_point: Optional[Point] = None
    curvature_per_px: float = 0.0
    maximum_curvature_per_px: float = 0.0
    reason: str = "no_geometry"


class ReverseParkingPathGenerator:
    """Create a smooth path from the rear axle to the detected bay center.

    The first Bézier tangent points straight backward from the car. The final
    tangent follows the center direction of the two side lines. Regenerating it
    every frame turns the path into a lightweight visual-servo reference.
    """

    def __init__(self, config: ReversePathConfig = ReversePathConfig()):
        self.config = config

    def generate(self, geometry: ParkingGeometry) -> ReversePath:
        if not geometry.found or not geometry.has_side_pair:
            return ReversePath(reason="parking_side_lines_missing")
        if (
            not geometry.has_back_line
            or geometry.stop_target_x_px is None
            or geometry.stop_target_y_px is None
        ):
            return ReversePath(reason="parking_back_line_missing")

        start = (geometry.vehicle_x_px, geometry.vehicle_y_px)
        target = (geometry.stop_target_x_px, geometry.stop_target_y_px)
        direction = normalize((geometry.slot_direction_x, geometry.slot_direction_y))
        if direction is None:
            return ReversePath(reason="invalid_slot_direction")
        progress = dot((target[0] - start[0], target[1] - start[1]), direction)
        if progress < self.config.minimum_target_distance_px:
            return ReversePath(reason="target_not_behind_vehicle")

        first_control = (start[0], start[1] - self.config.start_tangent_px)
        second_control = (
            target[0] - direction[0] * self.config.end_tangent_px,
            target[1] - direction[1] * self.config.end_tangent_px,
        )
        count = max(5, int(self.config.samples))
        points = tuple(
            cubic_bezier(start, first_control, second_control, target, index / float(count - 1))
            for index in range(count)
        )
        maximum_curvature = sampled_maximum_curvature(points)
        if maximum_curvature > self.config.maximum_curvature_per_px:
            return ReversePath(
                points=points,
                maximum_curvature_per_px=maximum_curvature,
                reason="path_too_tight",
            )

        lookahead = point_at_distance(points, self.config.lookahead_px)
        dx = lookahead[0] - start[0]
        dy_reverse = start[1] - lookahead[1]
        distance_squared = dx * dx + dy_reverse * dy_reverse
        if distance_squared < 1.0:
            return ReversePath(points=points, reason="lookahead_too_close")
        curvature = 2.0 * dx / distance_squared
        return ReversePath(
            found=True,
            points=points,
            lookahead_point=lookahead,
            curvature_per_px=curvature,
            maximum_curvature_per_px=maximum_curvature,
            reason="reverse_path_ready",
        )


def cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    one_minus = 1.0 - t
    a = one_minus ** 3
    b = 3.0 * one_minus ** 2 * t
    c = 3.0 * one_minus * t ** 2
    d = t ** 3
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def point_at_distance(points: Tuple[Point, ...], distance: float) -> Point:
    if not points:
        raise ValueError("path must contain at least one point")
    remaining = max(0.0, distance)
    for first, second in zip(points, points[1:]):
        segment = hypot(second[0] - first[0], second[1] - first[1])
        if segment >= remaining and segment > 1e-9:
            ratio = remaining / segment
            return (
                first[0] + ratio * (second[0] - first[0]),
                first[1] + ratio * (second[1] - first[1]),
            )
        remaining -= segment
    return points[-1]


def sampled_maximum_curvature(points: Tuple[Point, ...]) -> float:
    maximum = 0.0
    for previous, current, following in zip(points, points[1:], points[2:]):
        a = hypot(current[0] - previous[0], current[1] - previous[1])
        b = hypot(following[0] - current[0], following[1] - current[1])
        c = hypot(following[0] - previous[0], following[1] - previous[1])
        denominator = a * b * c
        if denominator <= 1e-9:
            continue
        twice_area = abs(
            (current[0] - previous[0]) * (following[1] - previous[1])
            - (current[1] - previous[1]) * (following[0] - previous[0])
        )
        maximum = max(maximum, 2.0 * twice_area / denominator)
    return maximum


def normalize(vector: Point) -> Optional[Point]:
    length = hypot(vector[0], vector[1])
    if length <= 1e-9:
        return None
    return vector[0] / length, vector[1] / length


def dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]
