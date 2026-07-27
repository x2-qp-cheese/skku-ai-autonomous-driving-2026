from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin
from typing import Optional, Sequence, Tuple

from ..config import RearLidarConfig
from ..sensors.lidar import LidarPoint, LidarScan


@dataclass(frozen=True)
class RearPoint:
    angle_deg: float
    distance_mm: float
    x_right_mm: float
    y_back_mm: float
    quality: int


@dataclass(frozen=True)
class TangentPair:
    """Figure 7: tangent A/B and the bisector of their free sector."""

    valid: bool = False
    angle_a_deg: float = 0.0
    angle_b_deg: float = 0.0
    dist_a_mm: float = 0.0
    dist_b_mm: float = 0.0
    angle_bisector_deg: float = 0.0
    reason: str = "two_tangents_not_visible"


@dataclass(frozen=True)
class RearLidarObservation:
    timestamp: float = 0.0
    valid: bool = False
    points: Tuple[RearPoint, ...] = ()
    right_vehicle_present: bool = False
    near: bool = False
    pair: TangentPair = TangentPair()
    dist_c_mm: Optional[float] = None
    dist_d_mm: Optional[float] = None
    reason: str = "no_scan"


class RearLidarPerception:
    """Extract only the LiDAR variables explicitly used by the paper.

    Paper coordinates are rear=0 degrees, left=-90 degrees, right=+90
    degrees.  No vehicle-shape classifier, scan confirmation, tracking,
    smoothing, or held detections are applied.
    """

    def __init__(self, config: RearLidarConfig):
        self.config = config

    def reset(self) -> None:
        return None

    def observe(self, scan: Optional[LidarScan]) -> RearLidarObservation:
        if scan is None:
            return RearLidarObservation(reason="no_lidar_scan")

        points = tuple(
            sorted(
                (
                    transformed
                    for raw in scan.points
                    if (transformed := self._transform(raw)) is not None
                ),
                key=lambda point: point.angle_deg,
            )
        )
        if not points:
            return RearLidarObservation(
                timestamp=scan.timestamp,
                reason="no_points_in_paper_fov",
            )

        dist_c = self._paper_sector_min(
            points,
            -self.config.side_angle_max_deg,
            -self.config.side_angle_min_deg,
        )
        dist_d = self._paper_sector_min(
            points,
            self.config.side_angle_min_deg,
            self.config.side_angle_max_deg,
        )
        pair = self._paper_tangent_pair(points)
        near = any(
            point.distance_mm < self.config.near_distance_mm
            for point in points
        )

        return RearLidarObservation(
            timestamp=scan.timestamp,
            valid=True,
            points=points,
            # Section 3.1 does not define a detector.  The only right-hand
            # region defined by the paper is Dist_D (+70 to +100 degrees,
            # values below 2000 mm), so it is used without extra thresholds.
            right_vehicle_present=dist_d is not None,
            near=near,
            pair=pair,
            dist_c_mm=dist_c,
            dist_d_mm=dist_d,
            reason="paper_values_available",
        )

    def _transform(self, point: LidarPoint) -> Optional[RearPoint]:
        if point.distance_mm <= 0.0:
            return None
        bearing_deg = point.angle_deg + self.config.angle_offset_deg
        if self.config.clockwise_angles:
            bearing_deg = -bearing_deg
        bearing = radians(bearing_deg)
        x_right = point.distance_mm * sin(bearing)
        y_back = -point.distance_mm * cos(bearing)
        paper_angle = _wrap_degrees(degrees(atan2(x_right, y_back)))
        if abs(paper_angle) > self.config.rear_fov_deg:
            return None
        return RearPoint(
            angle_deg=paper_angle,
            distance_mm=float(point.distance_mm),
            x_right_mm=x_right,
            y_back_mm=y_back,
            quality=point.quality,
        )

    def _paper_tangent_pair(
        self,
        points: Sequence[RearPoint],
    ) -> TangentPair:
        """Use the two raw points touching the free sector around rear=0.

        The paper illustrates these tangent points but does not specify a
        clustering or vehicle-classification method.  Taking the closest
        angular point on each side of the rear axis is the direct geometric
        interpretation and introduces no extra detection constants.
        """

        left = [point for point in points if point.angle_deg < 0.0]
        right = [point for point in points if point.angle_deg > 0.0]
        if not left or not right:
            return TangentPair()
        tangent_a = max(left, key=lambda point: point.angle_deg)
        tangent_b = min(right, key=lambda point: point.angle_deg)
        return TangentPair(
            valid=True,
            angle_a_deg=tangent_a.angle_deg,
            angle_b_deg=tangent_b.angle_deg,
            dist_a_mm=tangent_a.distance_mm,
            dist_b_mm=tangent_b.distance_mm,
            angle_bisector_deg=(
                tangent_a.angle_deg + tangent_b.angle_deg
            )
            / 2.0,
            reason="figure7_tangent_pair",
        )

    def _paper_sector_min(
        self,
        points: Sequence[RearPoint],
        angle_min_deg: float,
        angle_max_deg: float,
    ) -> Optional[float]:
        distances = [
            point.distance_mm
            for point in points
            if angle_min_deg <= point.angle_deg <= angle_max_deg
            and point.distance_mm
            < self.config.side_distance_limit_mm
        ]
        return min(distances) if distances else None


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
