from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
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
class PointCluster:
    points: Tuple[RearPoint, ...]
    center_x_mm: float
    center_y_mm: float
    extent_mm: float


@dataclass(frozen=True)
class SideLineEstimate:
    valid: bool = False
    heading_deg: float = 0.0
    point_count: int = 0
    extent_mm: float = 0.0
    linearity: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class RearLidarObservation:
    timestamp: float = 0.0
    valid: bool = False
    points: Tuple[RearPoint, ...] = ()
    raw_point_count: int = 0
    left_fov_point_count: int = 0
    right_fov_point_count: int = 0
    tangent_cluster_count: int = 0
    tangent_candidate_count: int = 0
    right_vehicle_present: bool = False
    right_vehicle_raw: bool = False
    right_vehicle_cluster_points: int = 0
    right_vehicle_seen_scans: int = 0
    right_vehicle_missing_scans: int = 0
    right_vehicle_track_id: int = 0
    prealign_near: bool = False
    near: bool = False
    pair_source_scans: int = 0
    pair: TangentPair = TangentPair()
    dist_c_mm: Optional[float] = None
    dist_d_mm: Optional[float] = None
    left_side_line: SideLineEstimate = SideLineEstimate()
    right_side_line: SideLineEstimate = SideLineEstimate()
    slot_heading_deg: Optional[float] = None
    reason: str = "no_scan"


class RearLidarPerception:
    """Extract only the LiDAR variables explicitly used by the paper.

    Paper coordinates are rear=0 degrees, left=-90 degrees, right=+90
    degrees.  Recent rotations are fused only for the sparse A/B tangencies;
    current-scan points remain the source of near and C/D measurements.
    """

    def __init__(self, config: RearLidarConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._right_vehicle_present = False
        self._right_seen_scans = 0
        self._right_missing_scans = 0
        self._right_track_id = 0
        self._right_track_center: Optional[Tuple[float, float]] = None
        self._last_tracking_scan_timestamp: Optional[float] = None
        self._pair_scan_buffer = deque(
            maxlen=self.config.tangent_fusion_scans
        )
        self._last_pair_scan_timestamp: Optional[float] = None
        self._last_tangent_cluster_count = 0
        self._last_tangent_candidate_count = 0

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
        if scan.timestamp != self._last_pair_scan_timestamp:
            self._pair_scan_buffer.append(points)
            self._last_pair_scan_timestamp = scan.timestamp
        fused_pair_points = tuple(
            point
            for buffered_scan in self._pair_scan_buffer
            for point in buffered_scan
        )
        if not points:
            if scan.timestamp != self._last_tracking_scan_timestamp:
                self._update_right_vehicle_track(None)
                self._last_tracking_scan_timestamp = scan.timestamp
            return RearLidarObservation(
                timestamp=scan.timestamp,
                raw_point_count=len(scan.points),
                right_vehicle_present=self._right_vehicle_present,
                right_vehicle_seen_scans=self._right_seen_scans,
                right_vehicle_missing_scans=self._right_missing_scans,
                right_vehicle_track_id=self._right_track_id,
                pair_source_scans=len(self._pair_scan_buffer),
                pair=self._paper_tangent_pair(fused_pair_points),
                tangent_cluster_count=(
                    self._last_tangent_cluster_count
                ),
                tangent_candidate_count=(
                    self._last_tangent_candidate_count
                ),
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
        left_side_line = self._side_line_estimate(points, right=False)
        right_side_line = self._side_line_estimate(points, right=True)
        slot_heading = self._combine_side_headings(
            left_side_line,
            right_side_line,
        )
        pair = self._paper_tangent_pair(fused_pair_points)
        right_cluster = self._select_right_vehicle_cluster(points)
        right_raw = right_cluster is not None
        if scan.timestamp != self._last_tracking_scan_timestamp:
            self._update_right_vehicle_track(right_cluster)
            self._last_tracking_scan_timestamp = scan.timestamp
        near = any(
            point.distance_mm < self.config.near_distance_mm
            for point in points
        )
        prealign_near = any(
            point.distance_mm < self.config.prealign_distance_mm
            for point in points
        )

        return RearLidarObservation(
            timestamp=scan.timestamp,
            valid=True,
            points=points,
            raw_point_count=len(scan.points),
            left_fov_point_count=sum(
                point.angle_deg < 0.0 for point in points
            ),
            right_fov_point_count=sum(
                point.angle_deg > 0.0 for point in points
            ),
            tangent_cluster_count=self._last_tangent_cluster_count,
            tangent_candidate_count=(
                self._last_tangent_candidate_count
            ),
            right_vehicle_present=self._right_vehicle_present,
            right_vehicle_raw=right_raw,
            right_vehicle_cluster_points=(
                len(right_cluster.points)
                if right_cluster is not None
                else 0
            ),
            right_vehicle_seen_scans=self._right_seen_scans,
            right_vehicle_missing_scans=self._right_missing_scans,
            right_vehicle_track_id=self._right_track_id,
            prealign_near=prealign_near,
            near=near,
            pair_source_scans=len(self._pair_scan_buffer),
            pair=pair,
            dist_c_mm=dist_c,
            dist_d_mm=dist_d,
            left_side_line=left_side_line,
            right_side_line=right_side_line,
            slot_heading_deg=slot_heading,
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
        """Use the inner endpoints of two clusters bounding a right gap.

        A/B in Figure 7 are the two vehicle tangencies around the parking
        space.  They are not required to have opposite signs in LiDAR
        coordinates: before entry, both parked vehicles and their gap are on
        the vehicle's right.  Among adjacent vehicle-like clusters, select
        the widest angular free sector whose bisector still points to the
        configured right-side parking area.
        """

        close_points = [
            point
            for point in points
            if (
                point.distance_mm
                < self.config.tangent_distance_limit_mm
            )
        ]
        clusters = sorted(
            self._qualified_clusters(
                close_points,
                min_extent_mm=(
                    self.config.tangent_cluster_min_extent_mm
                ),
            ),
            key=lambda cluster: min(
                point.angle_deg for point in cluster.points
            ),
        )
        self._last_tangent_cluster_count = len(clusters)
        self._last_tangent_candidate_count = 0
        if len(clusters) < 2:
            return TangentPair(reason="two_vehicle_clusters_not_visible")

        candidates = []
        for lower, upper in zip(clusters, clusters[1:]):
            tangent_a = max(
                lower.points,
                key=lambda point: point.angle_deg,
            )
            tangent_b = min(
                upper.points,
                key=lambda point: point.angle_deg,
            )
            bisector = (
                tangent_a.angle_deg + tangent_b.angle_deg
            ) / 2.0
            angular_gap = (
                tangent_b.angle_deg - tangent_a.angle_deg
            )
            if (
                angular_gap
                < self.config.tangent_min_angular_gap_deg
            ):
                continue
            free_width = hypot(
                tangent_a.x_right_mm - tangent_b.x_right_mm,
                tangent_a.y_back_mm - tangent_b.y_back_mm,
            )
            candidates.append(
                (
                    angular_gap,
                    free_width,
                    bisector,
                    tangent_a,
                    tangent_b,
                )
            )
        self._last_tangent_candidate_count = len(candidates)
        if not candidates:
            return TangentPair(reason="parking_gap_not_visible")

        # Before entry the target gap is on the right.  While reversing,
        # however, the vehicle rotates through the entrance and the same
        # gap's bisector naturally crosses 0 degrees.  Prefer a right-side
        # candidate whenever one exists, but do not discard the only
        # geometrically valid gap merely because it has crossed the rear
        # centerline.
        right_candidates = [
            item for item in candidates if item[2] >= 0.0
        ]
        selection_pool = right_candidates or candidates
        _, _, bisector, tangent_a, tangent_b = max(
            selection_pool,
            key=lambda item: (item[0], item[1]),
        )
        return TangentPair(
            valid=True,
            angle_a_deg=tangent_a.angle_deg,
            angle_b_deg=tangent_b.angle_deg,
            dist_a_mm=tangent_a.distance_mm,
            dist_b_mm=tangent_b.distance_mm,
            angle_bisector_deg=bisector,
            reason=(
                "figure7_right_gap_cluster_tangents"
                if right_candidates
                else "figure7_gap_crossed_rear_centerline"
            ),
        )

    def _select_right_vehicle_cluster(
        self,
        points: Sequence[RearPoint],
    ) -> Optional[PointCluster]:
        """Track a vehicle cluster across the complete right paper FOV.

        Dist_D ends at +100 degrees, but the paper sensor view extends to
        +110 degrees.  A vehicle is therefore not allowed to disappear merely
        because its points cross the +100 degree Dist_D boundary.
        """

        candidates = self._qualified_clusters(
            point
            for point in points
            if (
                self.config.side_angle_min_deg
                <= point.angle_deg
                <= self.config.rear_fov_deg
                and point.distance_mm
                < self.config.side_distance_limit_mm
            )
        )
        if not candidates:
            return None

        if self._right_track_center is not None:
            compatible = [
                cluster
                for cluster in candidates
                if self._cluster_shift_mm(
                    cluster,
                    self._right_track_center,
                )
                <= self.config.vehicle_track_max_shift_mm
            ]
            if not compatible:
                return None
            return min(
                compatible,
                key=lambda cluster: self._cluster_shift_mm(
                    cluster,
                    self._right_track_center,
                ),
            )

        return max(
            candidates,
            key=lambda cluster: (
                len(cluster.points),
                -hypot(cluster.center_x_mm, cluster.center_y_mm),
            ),
        )

    def _update_right_vehicle_track(
        self,
        cluster: Optional[PointCluster],
    ) -> None:
        if cluster is not None:
            self._right_seen_scans += 1
            self._right_missing_scans = 0
            self._right_track_center = (
                cluster.center_x_mm,
                cluster.center_y_mm,
            )
            if (
                not self._right_vehicle_present
                and self._right_seen_scans
                >= self.config.vehicle_confirm_scans
            ):
                self._right_vehicle_present = True
                self._right_track_id += 1
            return

        self._right_seen_scans = 0
        self._right_missing_scans += 1
        if not self._right_vehicle_present:
            self._right_track_center = None
        if (
            self._right_vehicle_present
            and self._right_missing_scans
            >= self.config.gap_confirm_scans
        ):
            self._right_vehicle_present = False
            self._right_track_center = None

    def _qualified_clusters(
        self,
        points: Sequence[RearPoint],
        *,
        min_extent_mm: Optional[float] = None,
    ) -> Tuple[PointCluster, ...]:
        minimum_extent = (
            self.config.vehicle_cluster_min_extent_mm
            if min_extent_mm is None
            else min_extent_mm
        )
        return tuple(
            cluster
            for cluster in self._euclidean_clusters(tuple(points))
            if (
                len(cluster.points)
                >= self.config.vehicle_min_cluster_points
                and minimum_extent
                <= cluster.extent_mm
                <= self.config.vehicle_cluster_max_extent_mm
            )
        )

    def _euclidean_clusters(
        self,
        points: Sequence[RearPoint],
    ) -> Tuple[PointCluster, ...]:
        ordered = sorted(points, key=lambda point: point.angle_deg)
        if not ordered:
            return ()

        grouped = []
        current = [ordered[0]]
        for point in ordered[1:]:
            previous = current[-1]
            angular_gap_mm = min(
                previous.distance_mm,
                point.distance_mm,
            ) * radians(abs(point.angle_deg - previous.angle_deg))
            allowed_gap = min(
                self.config.vehicle_cluster_max_gap_mm,
                max(
                    self.config.vehicle_cluster_base_gap_mm,
                    angular_gap_mm
                    * self.config.vehicle_cluster_gap_factor,
                ),
            )
            physical_gap = hypot(
                point.x_right_mm - previous.x_right_mm,
                point.y_back_mm - previous.y_back_mm,
            )
            if physical_gap <= allowed_gap:
                current.append(point)
            else:
                grouped.append(self._make_cluster(current))
                current = [point]
        grouped.append(self._make_cluster(current))
        return tuple(grouped)

    @staticmethod
    def _make_cluster(points: Sequence[RearPoint]) -> PointCluster:
        cluster_points = tuple(points)
        center_x = sum(point.x_right_mm for point in cluster_points) / len(
            cluster_points
        )
        center_y = sum(point.y_back_mm for point in cluster_points) / len(
            cluster_points
        )
        extent = max(
            hypot(
                first.x_right_mm - second.x_right_mm,
                first.y_back_mm - second.y_back_mm,
            )
            for first in cluster_points
            for second in cluster_points
        )
        return PointCluster(
            points=cluster_points,
            center_x_mm=center_x,
            center_y_mm=center_y,
            extent_mm=extent,
        )

    @staticmethod
    def _cluster_shift_mm(
        cluster: PointCluster,
        previous_center: Tuple[float, float],
    ) -> float:
        return hypot(
            cluster.center_x_mm - previous_center[0],
            cluster.center_y_mm - previous_center[1],
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

    def _side_line_estimate(
        self,
        points: Sequence[RearPoint],
        *,
        right: bool,
    ) -> SideLineEstimate:
        """Fit the visible side of a parked vehicle with a PCA line.

        C/D describe lateral position at one instant, but cannot reveal
        whether the ego vehicle is rotated inside the slot.  A line fitted
        to either adjacent parked vehicle supplies that missing heading.
        """

        side_points = tuple(
            point
            for point in points
            if (
                point.distance_mm
                < self.config.side_distance_limit_mm
                and abs(point.angle_deg)
                >= self.config.side_line_min_angle_deg
                and (
                    point.x_right_mm > 0.0
                    if right
                    else point.x_right_mm < 0.0
                )
            )
        )
        estimates = []
        for cluster in self._euclidean_clusters(side_points):
            if (
                len(cluster.points)
                < self.config.side_line_min_points
                or cluster.extent_mm
                < self.config.side_line_min_extent_mm
                or cluster.extent_mm
                > self.config.vehicle_cluster_max_extent_mm
            ):
                continue
            estimate = self._fit_cluster_line(cluster)
            if (
                estimate.linearity
                < self.config.side_line_min_linearity
            ):
                continue
            estimates.append(estimate)
        if not estimates:
            return SideLineEstimate()
        return max(estimates, key=lambda item: item.score)

    @staticmethod
    def _fit_cluster_line(
        cluster: PointCluster,
    ) -> SideLineEstimate:
        points = cluster.points
        count = len(points)
        mean_x = sum(point.x_right_mm for point in points) / count
        mean_y = sum(point.y_back_mm for point in points) / count
        variance_x = sum(
            (point.x_right_mm - mean_x) ** 2 for point in points
        ) / count
        variance_y = sum(
            (point.y_back_mm - mean_y) ** 2 for point in points
        ) / count
        covariance = sum(
            (point.x_right_mm - mean_x)
            * (point.y_back_mm - mean_y)
            for point in points
        ) / count
        trace = variance_x + variance_y
        discriminant = sqrt(
            max(
                0.0,
                (variance_x - variance_y) ** 2
                + 4.0 * covariance * covariance,
            )
        )
        major = (trace + discriminant) / 2.0
        minor = (trace - discriminant) / 2.0
        linearity = major / max(1.0, minor)
        # Heading is measured from the ego rear axis.  A line parallel to
        # the vehicle is 0 degrees; positive values lean toward the right.
        heading = 0.5 * degrees(
            atan2(
                2.0 * covariance,
                variance_y - variance_x,
            )
        )
        score = (
            cluster.extent_mm
            * sqrt(min(100.0, max(1.0, linearity)))
        )
        return SideLineEstimate(
            valid=True,
            heading_deg=heading,
            point_count=count,
            extent_mm=cluster.extent_mm,
            linearity=linearity,
            score=score,
        )

    def _combine_side_headings(
        self,
        left: SideLineEstimate,
        right: SideLineEstimate,
    ) -> Optional[float]:
        if left.valid and right.valid:
            if (
                abs(left.heading_deg - right.heading_deg)
                > self.config.side_line_max_disagreement_deg
            ):
                return (
                    left.heading_deg
                    if left.score >= right.score
                    else right.heading_deg
                )
            total = left.score + right.score
            return (
                left.heading_deg * left.score
                + right.heading_deg * right.score
            ) / max(1.0, total)
        if left.valid:
            return left.heading_deg
        if right.valid:
            return right.heading_deg
        return None


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
