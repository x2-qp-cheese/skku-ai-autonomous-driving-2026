from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Optional, Sequence, Tuple

import numpy as np

from .lidar_slot_geometry import LidarSlotGeometryProjector
from .parking_geometry import ParkingGeometry
from .parking_lidar import LidarParkingObservation, infer_dynamic_slot_polygon


Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class LockedSlotTrackerConfig:
    """Short-range LiDAR odometry used after the parking bay is locked.

    The locked rectangle is expressed in the current LiDAR/vehicle frame. Fresh
    observations of both bordering cars re-anchor its pose, while ICP estimates
    the rigid motion between consecutive scans during temporary occlusion.
    """

    min_points: int = 12
    max_points: int = 180
    min_range_mm: float = 200.0
    max_range_mm: float = 3500.0
    max_correspondence_mm: float = 320.0
    trim_ratio: float = 0.65
    iterations: int = 6
    max_translation_per_scan_mm: float = 300.0
    max_rotation_per_scan_deg: float = 15.0
    max_hold_scans: int = 3


@dataclass(frozen=True)
class LockedSlotPose:
    polygon: Optional[Polygon] = None
    locked: bool = False
    tracked: bool = False
    held: bool = False
    lost: bool = False
    translation_mm: float = 0.0
    rotation_deg: float = 0.0
    reason: str = "slot_not_locked"


class LockedSlotTracker:
    """Keep one physical parking rectangle stable while the vehicle reverses."""

    def __init__(self, config: LockedSlotTrackerConfig = LockedSlotTrackerConfig()):
        self.config = config
        self._polygon: Optional[Polygon] = None
        self._previous_points: Optional[np.ndarray] = None
        self._hold_scans = 0

    @property
    def locked(self) -> bool:
        return self._polygon is not None

    def reset(self) -> None:
        self._polygon = None
        self._previous_points = None
        self._hold_scans = 0

    def lock(self, polygon: Sequence[Point], points: Sequence[Point]) -> LockedSlotPose:
        return self._set_polygon(polygon, points, reason="slot_locked")

    def reanchor(
        self,
        polygon: Sequence[Point],
        points: Sequence[Point],
    ) -> LockedSlotPose:
        """Replace accumulated ICP pose drift with a fresh two-car gap pose."""

        return self._set_polygon(
            polygon,
            points,
            reason="locked_slot_reanchored",
        )

    def _set_polygon(
        self,
        polygon: Sequence[Point],
        points: Sequence[Point],
        *,
        reason: str,
    ) -> LockedSlotPose:
        if len(polygon) != 4:
            return LockedSlotPose(reason="slot_lock_requires_four_corners")
        sampled = self._prepare_points(points)
        if sampled is None:
            return LockedSlotPose(reason="slot_lock_requires_lidar_points")
        self._polygon = tuple(
            (float(point[0]), float(point[1])) for point in polygon
        )  # type: ignore[assignment]
        self._previous_points = sampled
        self._hold_scans = 0
        return LockedSlotPose(
            polygon=self._polygon,
            locked=True,
            tracked=True,
            reason=reason,
        )

    def update(self, points: Sequence[Point]) -> LockedSlotPose:
        if self._polygon is None or self._previous_points is None:
            return LockedSlotPose(reason="slot_not_locked")
        current = self._prepare_points(points)
        if current is None:
            return self._hold("locked_slot_insufficient_points")

        transform = estimate_current_to_previous_transform(
            current,
            self._previous_points,
            max_correspondence_mm=max(1.0, self.config.max_correspondence_mm),
            trim_ratio=self.config.trim_ratio,
            iterations=max(1, self.config.iterations),
            min_correspondences=max(3, self.config.min_points // 2),
        )
        if transform is None:
            return self._hold("locked_slot_scan_match_failed")
        rotation, translation = transform
        rotation_deg = degrees(atan2(rotation[1, 0], rotation[0, 0]))
        translation_mm = hypot(float(translation[0]), float(translation[1]))
        if (
            translation_mm > max(0.0, self.config.max_translation_per_scan_mm)
            or abs(rotation_deg) > max(0.0, self.config.max_rotation_per_scan_deg)
        ):
            return self._hold("locked_slot_motion_gate_rejected")

        # ICP returns previous ~= R * current + t.  The slot is stored in the
        # previous frame, so apply the inverse to express it in the current frame.
        inverse_rotation = rotation.T
        transformed = []
        for point in self._polygon:
            previous = np.asarray(point, dtype=np.float64)
            current_point = inverse_rotation @ (previous - translation)
            transformed.append((float(current_point[0]), float(current_point[1])))
        self._polygon = tuple(transformed)  # type: ignore[assignment]
        self._previous_points = current
        self._hold_scans = 0
        return LockedSlotPose(
            polygon=self._polygon,
            locked=True,
            tracked=True,
            translation_mm=translation_mm,
            rotation_deg=rotation_deg,
            reason="locked_slot_tracked",
        )

    def _prepare_points(self, points: Sequence[Point]) -> Optional[np.ndarray]:
        filtered = [
            (float(x), float(y))
            for x, y in points
            if self.config.min_range_mm
            <= hypot(float(x), float(y))
            <= self.config.max_range_mm
        ]
        if len(filtered) < max(3, self.config.min_points):
            return None
        maximum = max(3, int(self.config.max_points))
        if len(filtered) > maximum:
            indexes = np.linspace(0, len(filtered) - 1, maximum, dtype=int)
            filtered = [filtered[index] for index in indexes]
        return np.asarray(filtered, dtype=np.float64)

    def _hold(self, reason: str) -> LockedSlotPose:
        self._hold_scans += 1
        lost = self._hold_scans > max(0, self.config.max_hold_scans)
        return LockedSlotPose(
            polygon=self._polygon,
            locked=True,
            held=not lost,
            lost=lost,
            reason="locked_slot_lost" if lost else reason,
        )


class LockedSlotGeometryEstimator:
    """One shared locked-bay pipeline for live control and recorded replay."""

    def __init__(
        self,
        projector: LidarSlotGeometryProjector,
        tracker: LockedSlotTracker,
        *,
        width_mm: float,
        depth_mm: float,
        enabled: bool = True,
    ) -> None:
        self.projector = projector
        self.tracker = tracker
        self.width_mm = max(1.0, float(width_mm))
        self.depth_mm = max(1.0, float(depth_mm))
        self.enabled = bool(enabled)
        self.pose = LockedSlotPose()

    def reset(self) -> None:
        self.tracker.reset()
        self.pose = LockedSlotPose()

    def update(
        self,
        observation: LidarParkingObservation,
        points: Sequence[Point],
        *,
        lock_requested: bool,
    ) -> ParkingGeometry:
        if not self.enabled:
            return self.projector.project(observation)

        candidate = (
            infer_dynamic_slot_polygon(
                observation,
                self.depth_mm,
                self.width_mm,
            )
            if lock_requested
            else None
        )
        if lock_requested and not self.tracker.locked:
            if candidate is not None:
                self.pose = self.tracker.lock(candidate, points)
        elif self.tracker.locked:
            # A current, consistent pair of bordering cars is the strongest
            # source of the physical bay pose. Re-anchor the official-size box
            # on every such scan so small ICP errors cannot accumulate until
            # the target overlaps a parked car. ICP remains the fallback while
            # one or both bordering cars are occluded.
            strong_pair_pose = (
                candidate is not None
                and observation.gap_confirmed
                and observation.gap_pair_observed
                and not observation.coasted
            )
            if strong_pair_pose:
                reanchored = self.tracker.reanchor(candidate, points)
                self.pose = (
                    reanchored
                    if reanchored.tracked
                    else self.tracker.update(points)
                )
            else:
                self.pose = self.tracker.update(points)

        if self.tracker.locked:
            if self.pose.lost or self.pose.polygon is None:
                return ParkingGeometry(reason=self.pose.reason)
            return self.projector.project_polygon(
                self.pose.polygon,
                confirmed=True,
                coasted=self.pose.held,
                reason=self.pose.reason,
            )
        if lock_requested:
            return ParkingGeometry(reason=self.pose.reason)
        return self.projector.project(observation)


def estimate_current_to_previous_transform(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    max_correspondence_mm: float,
    trim_ratio: float,
    iterations: int,
    min_correspondences: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Trimmed point-to-point ICP returning ``previous ~= R*current + t``."""

    if current.ndim != 2 or previous.ndim != 2 or current.shape[1:] != (2,) or previous.shape[1:] != (2,):
        return None
    if len(current) < min_correspondences or len(previous) < min_correspondences:
        return None

    rotation = np.eye(2, dtype=np.float64)
    translation = np.zeros(2, dtype=np.float64)
    keep_ratio = min(1.0, max(0.25, float(trim_ratio)))
    max_distance_squared = float(max_correspondence_mm) ** 2

    for _ in range(max(1, int(iterations))):
        transformed = current @ rotation.T + translation
        differences = transformed[:, None, :] - previous[None, :, :]
        distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
        nearest_indexes = np.argmin(distances_squared, axis=1)
        nearest_distances = distances_squared[np.arange(len(transformed)), nearest_indexes]
        valid_indexes = np.flatnonzero(nearest_distances <= max_distance_squared)
        if len(valid_indexes) < min_correspondences:
            return None
        keep_count = max(min_correspondences, int(round(len(valid_indexes) * keep_ratio)))
        ordered = valid_indexes[np.argsort(nearest_distances[valid_indexes])[:keep_count]]
        source = transformed[ordered]
        target = previous[nearest_indexes[ordered]]
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        u, _, vt = np.linalg.svd(covariance)
        delta_rotation = vt.T @ u.T
        if np.linalg.det(delta_rotation) < 0.0:
            vt[-1, :] *= -1.0
            delta_rotation = vt.T @ u.T
        delta_translation = target_center - delta_rotation @ source_center
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation

    return rotation, translation
