from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin
from typing import Optional, Tuple

from ..estimation.parking_geometry import ParkingGeometry


Point = Tuple[float, float]


@dataclass(frozen=True)
class ReversePathConfig:
    """Image-plane path parameters for rear-BEV reverse parking."""

    samples: int = 21
    start_tangent_px: float = 120.0
    end_tangent_px: float = 120.0
    lookahead_px: float = 90.0
    minimum_target_distance_px: float = 8.0
    maximum_curvature_per_px: float = 0.0054
    full_steering_curvature_per_px: float = 0.0054


@dataclass(frozen=True)
class ReversePath:
    found: bool = False
    points: Tuple[Point, ...] = ()
    lookahead_point: Optional[Point] = None
    curvature_per_px: float = 0.0
    maximum_curvature_per_px: float = 0.0
    reason: str = "no_geometry"


class ReverseParkingPathGenerator:
    """Generate one short centerline target for the next control update.

    The target is regenerated from the current locked-slot pose on every LiDAR
    update. It is intentionally not a precomputed full parking trajectory.
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
        stop_target = (geometry.stop_target_x_px, geometry.stop_target_y_px)
        direction = normalize((geometry.slot_direction_x, geometry.slot_direction_y))
        if direction is None:
            return ReversePath(reason="invalid_slot_direction")
        progress = dot(
            (stop_target[0] - start[0], stop_target[1] - start[1]),
            direction,
        )
        if progress < self.config.minimum_target_distance_px:
            return ReversePath(reason="target_not_behind_vehicle")

        # The back-clearance target lies on the locked slot centerline. Project
        # the rear axle onto that line and select only one nearby target. This is
        # intentionally a receding-horizon reference, not a full parking path.
        start_from_line = (
            start[0] - stop_target[0],
            start[1] - stop_target[1],
        )
        along = dot(start_from_line, direction)
        projection = (
            stop_target[0] + direction[0] * along,
            stop_target[1] + direction[1] * along,
        )
        lookahead_distance = min(max(1.0, self.config.lookahead_px), progress)
        target = (
            projection[0] + direction[0] * lookahead_distance,
            projection[1] + direction[1] * lookahead_distance,
        )

        dx = target[0] - start[0]
        dy_reverse = start[1] - target[1]
        distance_squared = dx * dx + dy_reverse * dy_reverse
        if distance_squared < 1.0:
            return ReversePath(reason="lookahead_too_close")
        curvature = 2.0 * dx / distance_squared
        curvature_limit = max(1e-9, abs(self.config.maximum_curvature_per_px))
        limited = abs(curvature) > curvature_limit
        curvature = max(-curvature_limit, min(curvature_limit, curvature))
        points = short_arc_points(
            start,
            curvature,
            target,
            max(5, int(self.config.samples)),
        )
        return ReversePath(
            found=True,
            points=points,
            lookahead_point=target,
            curvature_per_px=curvature,
            maximum_curvature_per_px=abs(curvature),
            reason="local_target_curvature_limited" if limited else "local_target_ready",
        )


def short_arc_points(
    start: Point,
    curvature: float,
    target: Point,
    samples: int,
) -> Tuple[Point, ...]:
    """Sample the short rear-axle arc used only until the next LiDAR update."""

    dx = target[0] - start[0]
    dy_reverse = start[1] - target[1]
    if abs(curvature) <= 1e-9:
        return tuple(
            (
                start[0] + dx * index / float(samples - 1),
                start[1] - dy_reverse * index / float(samples - 1),
            )
            for index in range(samples)
        )
    bearing = atan2(dx, dy_reverse)
    arc_length = abs((2.0 * bearing) / curvature)
    if arc_length <= 1e-6:
        arc_length = hypot(dx, dy_reverse)
    points = []
    for index in range(samples):
        distance = arc_length * index / float(samples - 1)
        angle = curvature * distance
        local_x = (1.0 - cos(angle)) / curvature
        local_y = sin(angle) / curvature
        points.append((start[0] + local_x, start[1] - local_y))
    return tuple(points)


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
