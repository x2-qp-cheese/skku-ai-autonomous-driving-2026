from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin
from typing import List, Optional, Sequence, Tuple

from ..sensors.lidar import LidarPoint, LidarScan


@dataclass(frozen=True)
class RectangleRoi:
    """Axis-aligned ROI in vehicle coordinates.

    x is positive to vehicle-right. ``y_back`` is positive behind the vehicle.
    """

    x_min_mm: float
    x_max_mm: float
    y_back_min_mm: float
    y_back_max_mm: float

    def contains(self, x_right_mm: float, y_back_mm: float) -> bool:
        return (
            self.x_min_mm <= x_right_mm <= self.x_max_mm
            and self.y_back_min_mm <= y_back_mm <= self.y_back_max_mm
        )


@dataclass(frozen=True)
class CarCluster:
    """One parked-car surface observed in the side-facing LiDAR ROI."""

    point_count: int
    x_min_mm: float
    x_max_mm: float
    y_back_min_mm: float
    y_back_max_mm: float

    @property
    def center_y_back_mm(self) -> float:
        return (self.y_back_min_mm + self.y_back_max_mm) / 2.0

    @property
    def center_x_right_mm(self) -> float:
        return (self.x_min_mm + self.x_max_mm) / 2.0

    @property
    def span_x_mm(self) -> float:
        return self.x_max_mm - self.x_min_mm

    @property
    def span_y_back_mm(self) -> float:
        return self.y_back_max_mm - self.y_back_min_mm

    @property
    def max_span_mm(self) -> float:
        return max(self.span_x_mm, self.span_y_back_mm)


@dataclass(frozen=True)
class LidarParkingConfig:
    quality_min: int = 1
    min_distance_mm: float = 120.0
    max_distance_mm: float = 12000.0
    # On the supplied installation, raw 90 deg was displayed at screen-right
    # but that direction is the physical vehicle front. Subtract 90 deg so raw
    # 90 becomes vehicle bearing 0 (front/up in the debug vehicle view).
    angle_offset_deg: float = -90.0
    clockwise_angles: bool = False
    sensor_x_right_mm: float = 0.0
    sensor_y_forward_mm: float = 0.0

    # This ROI looks sideways at the row of parked cars. It is not a parking
    # line or a claim that the bay interior is empty.
    car_detection_roi: RectangleRoi = RectangleRoi(250.0, 2600.0, -2500.0, 2500.0)
    # Once the slot is confirmed, the ego vehicle may rotate enough that one
    # bordering car moves out of the initial right-side ROI. Use this wider ROI
    # only for tracking an already confirmed slot.
    slot_tracking_roi: Optional[RectangleRoi] = RectangleRoi(-1800.0, 2600.0, -2500.0, 2500.0)
    # Separate collision envelope used while reversing.
    safety_roi: RectangleRoi = RectangleRoi(-300.0, 300.0, 100.0, 450.0)
    min_observed_points: int = 5
    car_cluster_radius_mm: float = 350.0
    car_cluster_min_points: int = 3
    # First-car detection stays intentionally sensitive, but a parking bay
    # must be bordered by two stronger parked-car clusters. Otherwise a partial
    # return from the first obstacle plus a stray reflection can look like a
    # false slot.
    gap_cluster_min_points: int = 5
    gap_pair_min_points: int = 10

    # Official painted bay dimensions. The observed surface gap is larger
    # because LiDAR sees the neighboring vehicles, not the painted boundaries.
    parking_space_width_mm: float = 950.0
    parking_space_depth_mm: float = 1500.0
    expected_observed_gap_mm: float = 1375.0
    observed_gap_min_mm: float = 1100.0
    observed_gap_max_mm: float = 1650.0
    # The mission slot starts on the vehicle-right, although either bordering
    # car may move into the left half of the vehicle frame while turning.
    gap_center_x_min_mm: float = 0.0
    gap_center_y_back_min_mm: float = 200.0
    gap_confirm_scans: int = 3
    gap_candidate_hold_s: float = 1.2
    gap_single_cluster_track_enabled: bool = True
    gap_single_cluster_max_edge_jump_mm: float = 700.0
    gap_coast_scans: int = 15
    # Keep displaying the last confirmed slot for the current mission when
    # both bordering cars are temporarily unavailable. Coasted observations
    # remain marked HOLD and are never used to advance POSITIONING.
    gap_hold_confirmed_until_reset: bool = True
    gap_center_smooth_alpha: float = 0.35
    gap_max_center_jump_mm: float = 400.0
    gap_orientation_smooth_alpha: float = 0.25
    gap_max_orientation_jump_deg: float = 35.0

    # The competition guarantees that the first parked car is immediately
    # followed by the mission bay.  Its front-most longitudinal surface is a
    # useful early trigger, before the second bordering car is visible.  With
    # the rear-mounted LiDAR, -650 mm places that surface roughly 350 mm ahead
    # of the provisional rear axle (-300 mm).
    first_car_confirm_scans: int = 6
    first_car_min_x_right_mm: float = 450.0
    # A lone weak cluster must not be enough to start the parking maneuver.
    # People and cable reflections often produce one or two LiDAR points on the
    # right; require a denser, physically wider cluster before it is treated as
    # the first parked car.
    first_car_cluster_min_points: int = 5
    first_car_min_span_mm: float = 80.0
    first_car_max_center_jump_mm: float = 450.0
    first_car_turn_target_y_back_mm: float = -650.0

    # Positive y_back is behind the sensor. The LiDAR is provisionally 30 cm
    # behind the rear axle, so the rear axle is at negative y_back. Replace this
    # signed offset with the measured vehicle value before motor testing.
    sensor_to_rear_axle_y_back_mm: float = -300.0
    entry_rear_axle_offset_from_gap_center_mm: float = 0.0
    entry_tolerance_mm: float = 100.0
    stale_after_s: float = 0.35


@dataclass(frozen=True)
class LidarParkingObservation:
    timestamp: float = 0.0
    valid: bool = False
    unsafe: bool = False
    observed_points: int = 0
    car_roi_points: int = 0
    car_count: int = 0
    first_car_seen: bool = False
    first_car_confirmed: bool = False
    first_car_slot_edge_x_right_mm: Optional[float] = None
    first_car_slot_edge_y_back_mm: Optional[float] = None
    first_car_turn_error_mm: Optional[float] = None
    first_car_turn_reached: bool = False
    second_car_seen: bool = False
    gap_found: bool = False
    gap_confirmed: bool = False
    gap_pair_observed: bool = False
    gap_width_mm: Optional[float] = None
    gap_near_edge_x_right_mm: Optional[float] = None
    gap_near_edge_y_back_mm: Optional[float] = None
    gap_far_edge_x_right_mm: Optional[float] = None
    gap_far_edge_y_back_mm: Optional[float] = None
    gap_center_x_right_mm: Optional[float] = None
    gap_center_y_back_mm: Optional[float] = None
    slot_depth_x_right: Optional[float] = None
    slot_depth_y_back: Optional[float] = None
    entry_target_y_back_mm: Optional[float] = None
    entry_error_mm: Optional[float] = None
    entry_reached: bool = False
    nearest_safety_mm: Optional[float] = None
    car_clusters: Tuple[CarCluster, ...] = ()
    coasted: bool = False
    reason: str = "no_scan"


class LidarParkingSpaceEstimator:
    """Find a bay from the two parked cars that border it.

    The estimator deliberately does not treat LiDAR returns as painted parking
    lines. A candidate is the longitudinal gap between two distinct parked-car
    surface clusters. The bay interior is guaranteed empty by the competition,
    while ``safety_roi`` remains an independent emergency envelope.
    """

    def __init__(self, config: LidarParkingConfig = LidarParkingConfig()):
        self.config = config
        self._last_timestamp: Optional[float] = None
        self._first_car_confirm_scans = 0
        self._first_car_center_x_mm: Optional[float] = None
        self._first_car_center_y_mm: Optional[float] = None
        self._candidate_center_x_mm: Optional[float] = None
        self._candidate_center_mm: Optional[float] = None
        self._last_candidate_timestamp: Optional[float] = None
        self._smoothed_center_x_mm: Optional[float] = None
        self._smoothed_center_mm: Optional[float] = None
        self._confirm_scans = 0
        self._coast_scans = 0
        self._confirmed = False
        self._tracked_axis_x: Optional[float] = None
        self._tracked_axis_y: Optional[float] = None
        self._tracked_depth_x: Optional[float] = None
        self._tracked_depth_y: Optional[float] = None
        self._smoothed_edge_span_mm: Optional[float] = None
        self._smoothed_gap_width_mm: Optional[float] = None
        self._last_gap: Optional[
            Tuple[float, float, float, float, float, float, float]
        ] = None

    def reset(self) -> None:
        self._last_timestamp = None
        self._first_car_confirm_scans = 0
        self._first_car_center_x_mm = None
        self._first_car_center_y_mm = None
        self._candidate_center_x_mm = None
        self._candidate_center_mm = None
        self._last_candidate_timestamp = None
        self._smoothed_center_x_mm = None
        self._smoothed_center_mm = None
        self._confirm_scans = 0
        self._coast_scans = 0
        self._confirmed = False
        self._tracked_axis_x = None
        self._tracked_axis_y = None
        self._tracked_depth_x = None
        self._tracked_depth_y = None
        self._smoothed_edge_span_mm = None
        self._smoothed_gap_width_mm = None
        self._last_gap = None

    def vehicle_points(self, scan: Optional[LidarScan]) -> List[Tuple[float, float]]:
        if scan is None:
            return []
        result = []
        for point in scan.points:
            transformed = self._transform(point)
            if transformed is not None:
                result.append(transformed)
        return result

    def estimate(self, scan: Optional[LidarScan], now: Optional[float] = None) -> LidarParkingObservation:
        if scan is None:
            self.reset()
            return LidarParkingObservation(reason="no_scan")
        if now is not None and now - scan.timestamp > self.config.stale_after_s:
            self.reset()
            return LidarParkingObservation(timestamp=scan.timestamp, reason="stale_scan")

        transformed = [value for value in (self._transform(point) for point in scan.points) if value]
        cluster_roi = (
            self.config.slot_tracking_roi
            if self._confirmed and self.config.slot_tracking_roi is not None
            else self.config.car_detection_roi
        )
        car_points = [
            point for point in transformed
            if cluster_roi.contains(point[0], point[1])
        ]
        cluster_points_list = cluster_points(
            car_points,
            self.config.car_cluster_radius_mm,
            self.config.car_cluster_min_points,
        )
        clusters = tuple(sorted(
            (summarize_cluster(points) for points in cluster_points_list),
            key=lambda cluster: cluster.y_back_min_mm,
        ))
        safety_distances = [
            hypot(x_right, y_back)
            for x_right, y_back in transformed
            if self.config.safety_roi.contains(x_right, y_back)
        ]
        nearest_safety = min(safety_distances) if safety_distances else None
        unsafe = nearest_safety is not None
        valid = (
            len(transformed) >= max(1, self.config.min_observed_points)
            or len(car_points) >= max(1, self.config.car_cluster_min_points)
        )
        reference_depth = (
            (self._tracked_depth_x, self._tracked_depth_y)
            if self._tracked_depth_x is not None and self._tracked_depth_y is not None
            else None
        )
        reference_center = (
            (self._candidate_center_x_mm, self._candidate_center_mm)
            if self._candidate_center_x_mm is not None and self._candidate_center_mm is not None
            else None
        )
        candidate = (
            choose_gap(
                clusters,
                self.config,
                reference_depth=reference_depth,
                reference_center=reference_center,
            )
            if valid
            else None
        )
        if candidate is not None and self._confirmed and not self._track_consistent(candidate):
            # A confirmed physical slot must not be replaced by a different
            # cluster pair or a 90/180-degree pose jump. Retain the previous
            # pose briefly and wait for a consistent observation.
            candidate = None
        gap_pair_observed = candidate is not None
        if candidate is None and valid and self._confirmed:
            candidate = self._track_gap_from_single_cluster(clusters)
        new_scan = self._last_timestamp != scan.timestamp

        first_car = select_first_approach_car(clusters, self.config) if valid else None
        if new_scan:
            if first_car is None:
                self._clear_first_car()
            else:
                if not self._first_car_track_consistent(first_car):
                    self._first_car_confirm_scans = 1
                else:
                    self._first_car_confirm_scans += 1
                self._first_car_center_x_mm = first_car.center_x_right_mm
                self._first_car_center_y_mm = first_car.center_y_back_mm
        first_car_confirmed = (
            first_car is not None
            and self._first_car_confirm_scans
            >= max(1, self.config.first_car_confirm_scans)
        )
        first_car_edge_x = (
            (first_car.x_min_mm + first_car.x_max_mm) / 2.0
            if first_car is not None
            else None
        )
        # During the straight approach, decreasing y_back points toward the
        # upcoming bay.  The minimum-y surface is therefore the edge adjacent
        # to the guaranteed empty slot.
        first_car_edge_y = (
            first_car.y_back_min_mm if first_car is not None else None
        )
        first_car_turn_error = (
            self.config.first_car_turn_target_y_back_mm - first_car_edge_y
            if first_car_edge_y is not None
            else None
        )
        first_car_turn_reached = (
            first_car_confirmed
            and first_car_turn_error is not None
            and first_car_turn_error <= 0.0
        )

        if new_scan:
            self._update_tracking(candidate, scan.timestamp)
            self._last_timestamp = scan.timestamp

        gap = self._last_gap if candidate is not None and self._last_gap is not None else candidate
        coasted = False
        if gap is None and self._confirmed and self._last_gap is not None:
            if (
                self.config.gap_hold_confirmed_until_reset
                or self._coast_scans <= max(0, self.config.gap_coast_scans)
            ):
                gap = self._last_gap
                coasted = True

        gap_found = gap is not None
        gap_width = gap[0] if gap else None
        near_edge = gap[1] if gap else None
        far_edge = gap[2] if gap else None
        near_edge_x = gap[3] if gap else None
        far_edge_x = gap[4] if gap else None
        slot_depth_x = gap[5] if gap else None
        slot_depth_y = gap[6] if gap else None
        center_x = self._smoothed_center_x_mm if gap_found else None
        center = self._smoothed_center_mm if gap_found else None
        if center_x is None and gap_found:
            center_x = (near_edge_x + far_edge_x) / 2.0
        if center is None and gap_found:
            center = (near_edge + far_edge) / 2.0
        entry_target = (
            self.config.sensor_to_rear_axle_y_back_mm
            + self.config.entry_rear_axle_offset_from_gap_center_mm
        )
        entry_error = center - entry_target if center is not None else None
        entry_reached = (
            self._confirmed
            and entry_error is not None
            and abs(entry_error) <= self.config.entry_tolerance_mm
        )

        if unsafe:
            reason = "safety_obstacle"
        elif not valid:
            reason = "insufficient_lidar_points"
        elif self._confirmed and entry_reached:
            reason = "rear_axle_aligned"
        elif self._confirmed and gap_found:
            reason = "gap_confirmed"
        elif candidate is not None:
            reason = "gap_confirming:%d/%d" % (
                self._confirm_scans,
                max(1, self.config.gap_confirm_scans),
            )
        elif len(clusters) == 1 and first_car is None:
            reason = "one_cluster_filtered"
        elif len(clusters) == 1:
            reason = "one_parked_car"
        elif len(clusters) >= 2:
            reason = "two_cars_no_valid_gap"
        else:
            reason = "searching_for_parked_cars"

        return LidarParkingObservation(
            timestamp=scan.timestamp,
            valid=valid,
            unsafe=unsafe,
            observed_points=len(transformed),
            car_roi_points=len(car_points),
            car_count=len(clusters),
            first_car_seen=first_car is not None,
            first_car_confirmed=first_car_confirmed,
            first_car_slot_edge_x_right_mm=first_car_edge_x,
            first_car_slot_edge_y_back_mm=first_car_edge_y,
            first_car_turn_error_mm=first_car_turn_error,
            first_car_turn_reached=first_car_turn_reached,
            second_car_seen=len(clusters) >= 2,
            gap_found=gap_found,
            gap_confirmed=self._confirmed and gap_found,
            gap_pair_observed=gap_pair_observed,
            gap_width_mm=gap_width,
            gap_near_edge_x_right_mm=near_edge_x,
            gap_near_edge_y_back_mm=near_edge,
            gap_far_edge_x_right_mm=far_edge_x,
            gap_far_edge_y_back_mm=far_edge,
            gap_center_x_right_mm=center_x,
            gap_center_y_back_mm=center,
            slot_depth_x_right=slot_depth_x,
            slot_depth_y_back=slot_depth_y,
            entry_target_y_back_mm=entry_target,
            entry_error_mm=entry_error,
            entry_reached=entry_reached,
            nearest_safety_mm=nearest_safety,
            car_clusters=clusters,
            coasted=coasted,
            reason=reason,
        )

    def _update_tracking(
        self,
        gap: Optional[Tuple[float, float, float, float, float, float, float]],
        timestamp: float,
    ) -> None:
        if gap is None:
            if self._confirmed and self._last_gap is not None:
                self._coast_scans += 1
                if (
                    not self.config.gap_hold_confirmed_until_reset
                    and self._coast_scans > max(0, self.config.gap_coast_scans)
                ):
                    self._clear_candidate()
            elif (
                self._last_candidate_timestamp is not None
                and timestamp - self._last_candidate_timestamp
                <= max(0.0, self.config.gap_candidate_hold_s)
            ):
                self._coast_scans += 1
            else:
                self._clear_candidate()
            return

        self._last_candidate_timestamp = timestamp
        first_x, first_y = gap[3], gap[1]
        second_x, second_y = gap[4], gap[2]
        axis_x = second_x - first_x
        axis_y = second_y - first_y
        edge_span = hypot(axis_x, axis_y)
        if edge_span <= 1e-6:
            return
        axis_x /= edge_span
        axis_y /= edge_span
        if (
            self._tracked_axis_x is not None
            and self._tracked_axis_y is not None
            and axis_x * self._tracked_axis_x + axis_y * self._tracked_axis_y < 0.0
        ):
            first_x, second_x = second_x, first_x
            first_y, second_y = second_y, first_y
            axis_x = -axis_x
            axis_y = -axis_y
        center_x = (first_x + second_x) / 2.0
        center = (first_y + second_y) / 2.0
        jump = (
            self._candidate_center_x_mm is not None
            and
            self._candidate_center_mm is not None
            and hypot(
                center_x - self._candidate_center_x_mm,
                center - self._candidate_center_mm,
            ) > self.config.gap_max_center_jump_mm
        )
        if jump:
            self._confirm_scans = 1
            self._smoothed_center_x_mm = center_x
            self._smoothed_center_mm = center
            self._tracked_axis_x = axis_x
            self._tracked_axis_y = axis_y
            self._tracked_depth_x = gap[5]
            self._tracked_depth_y = gap[6]
            self._smoothed_edge_span_mm = edge_span
            self._smoothed_gap_width_mm = gap[0]
            self._confirmed = False
        else:
            self._confirm_scans += 1
            alpha = clip(self.config.gap_center_smooth_alpha, 0.0, 1.0)
            self._smoothed_center_x_mm = (
                center_x if self._smoothed_center_x_mm is None
                else alpha * center_x + (1.0 - alpha) * self._smoothed_center_x_mm
            )
            self._smoothed_center_mm = (
                center if self._smoothed_center_mm is None
                else alpha * center + (1.0 - alpha) * self._smoothed_center_mm
            )
            orientation_alpha = clip(
                self.config.gap_orientation_smooth_alpha,
                0.0,
                1.0,
            )
            self._tracked_axis_x, self._tracked_axis_y = smooth_unit_vector(
                self._tracked_axis_x,
                self._tracked_axis_y,
                axis_x,
                axis_y,
                orientation_alpha,
            )
            self._tracked_depth_x, self._tracked_depth_y = smooth_unit_vector(
                self._tracked_depth_x,
                self._tracked_depth_y,
                gap[5],
                gap[6],
                orientation_alpha,
            )
            self._smoothed_edge_span_mm = smooth_value(
                self._smoothed_edge_span_mm,
                edge_span,
                alpha,
            )
            self._smoothed_gap_width_mm = smooth_value(
                self._smoothed_gap_width_mm,
                gap[0],
                alpha,
            )
        self._candidate_center_x_mm = center_x
        self._candidate_center_mm = center
        if (
            self._smoothed_center_x_mm is not None
            and self._smoothed_center_mm is not None
            and self._tracked_axis_x is not None
            and self._tracked_axis_y is not None
            and self._tracked_depth_x is not None
            and self._tracked_depth_y is not None
            and self._smoothed_edge_span_mm is not None
            and self._smoothed_gap_width_mm is not None
        ):
            half_span = self._smoothed_edge_span_mm / 2.0
            tracked_first_x = self._smoothed_center_x_mm - self._tracked_axis_x * half_span
            tracked_first_y = self._smoothed_center_mm - self._tracked_axis_y * half_span
            tracked_second_x = self._smoothed_center_x_mm + self._tracked_axis_x * half_span
            tracked_second_y = self._smoothed_center_mm + self._tracked_axis_y * half_span
            self._last_gap = (
                self._smoothed_gap_width_mm,
                tracked_first_y,
                tracked_second_y,
                tracked_first_x,
                tracked_second_x,
                self._tracked_depth_x,
                self._tracked_depth_y,
            )
        self._coast_scans = 0
        if self._confirm_scans >= max(1, self.config.gap_confirm_scans):
            self._confirmed = True

    def _track_gap_from_single_cluster(
        self,
        clusters: Sequence[CarCluster],
    ) -> Optional[Tuple[float, float, float, float, float, float, float]]:
        if (
            not self.config.gap_single_cluster_track_enabled
            or self._last_gap is None
            or self._tracked_axis_x is None
            or self._tracked_axis_y is None
            or self._tracked_depth_x is None
            or self._tracked_depth_y is None
        ):
            return None
        eligible = [
            cluster for cluster in clusters
            if is_gap_cluster_eligible(cluster, self.config)
        ]
        if not eligible:
            return None

        last_width, first_y, second_y, first_x, second_x, depth_x, depth_y = self._last_gap
        first_edge = (first_x, first_y)
        second_edge = (second_x, second_y)
        best: Optional[Tuple[float, float, float, float]] = None
        for cluster in eligible:
            first_candidate = cluster_slot_edge(
                cluster,
                self._tracked_axis_x,
                self._tracked_axis_y,
                self._tracked_depth_x,
                self._tracked_depth_y,
                side=1,
            )
            first_distance = hypot(
                first_candidate[0] - first_edge[0],
                first_candidate[1] - first_edge[1],
            )
            second_candidate = cluster_slot_edge(
                cluster,
                self._tracked_axis_x,
                self._tracked_axis_y,
                self._tracked_depth_x,
                self._tracked_depth_y,
                side=-1,
            )
            second_distance = hypot(
                second_candidate[0] - second_edge[0],
                second_candidate[1] - second_edge[1],
            )
            for distance, candidate_edge, reference_edge in (
                (first_distance, first_candidate, first_edge),
                (second_distance, second_candidate, second_edge),
            ):
                if best is None or distance < best[0]:
                    best = (
                        distance,
                        candidate_edge[0] - reference_edge[0],
                        candidate_edge[1] - reference_edge[1],
                        last_width,
                    )
        if (
            best is None
            or best[0] > max(0.0, self.config.gap_single_cluster_max_edge_jump_mm)
        ):
            return None
        _, delta_x, delta_y, width = best
        return (
            width,
            first_y + delta_y,
            second_y + delta_y,
            first_x + delta_x,
            second_x + delta_x,
            depth_x,
            depth_y,
        )

    def _track_consistent(
        self,
        gap: Tuple[float, float, float, float, float, float, float],
    ) -> bool:
        center_x = (gap[3] + gap[4]) / 2.0
        center_y = (gap[1] + gap[2]) / 2.0
        if (
            self._candidate_center_x_mm is not None
            and self._candidate_center_mm is not None
            and hypot(
                center_x - self._candidate_center_x_mm,
                center_y - self._candidate_center_mm,
            ) > self.config.gap_max_center_jump_mm
        ):
            return False
        if self._tracked_axis_x is None or self._tracked_axis_y is None:
            return True
        axis_x = gap[4] - gap[3]
        axis_y = gap[2] - gap[1]
        length = hypot(axis_x, axis_y)
        if length <= 1e-6:
            return False
        axis_x /= length
        axis_y /= length
        axis_alignment = abs(
            axis_x * self._tracked_axis_x + axis_y * self._tracked_axis_y
        )
        minimum_alignment = cos(radians(self.config.gap_max_orientation_jump_deg))
        return axis_alignment >= minimum_alignment

    def _first_car_track_consistent(self, cluster: CarCluster) -> bool:
        if self._first_car_center_x_mm is None or self._first_car_center_y_mm is None:
            return True
        jump = hypot(
            cluster.center_x_right_mm - self._first_car_center_x_mm,
            cluster.center_y_back_mm - self._first_car_center_y_mm,
        )
        return jump <= max(0.0, self.config.first_car_max_center_jump_mm)

    def _clear_first_car(self) -> None:
        self._first_car_confirm_scans = 0
        self._first_car_center_x_mm = None
        self._first_car_center_y_mm = None

    def _clear_candidate(self) -> None:
        self._candidate_center_x_mm = None
        self._candidate_center_mm = None
        self._last_candidate_timestamp = None
        self._smoothed_center_x_mm = None
        self._smoothed_center_mm = None
        self._confirm_scans = 0
        self._coast_scans = 0
        self._confirmed = False
        self._tracked_axis_x = None
        self._tracked_axis_y = None
        self._tracked_depth_x = None
        self._tracked_depth_y = None
        self._smoothed_edge_span_mm = None
        self._smoothed_gap_width_mm = None
        self._last_gap = None

    def _transform(self, point: LidarPoint) -> Optional[Tuple[float, float]]:
        if point.quality < self.config.quality_min:
            return None
        if not (self.config.min_distance_mm <= point.distance_mm <= self.config.max_distance_mm):
            return None
        angle = point.angle_deg + self.config.angle_offset_deg
        if self.config.clockwise_angles:
            angle = -angle
        bearing = radians(angle)
        # Bearing zero is vehicle-forward; x right and y_back positive rearward.
        beam_x = sin(bearing)
        beam_y_back = -cos(bearing)
        return (
            self.config.sensor_x_right_mm + point.distance_mm * beam_x,
            -self.config.sensor_y_forward_mm + point.distance_mm * beam_y_back,
        )


def select_first_approach_car(
    clusters: Sequence[CarCluster],
    config: LidarParkingConfig,
) -> Optional[CarCluster]:
    """Select the first bordering car encountered along the approach.

    Stationary objects move from negative to positive ``y_back`` while the ego
    vehicle drives forward.  The first car is consequently the right-side
    cluster with the largest longitudinal center once multiple clusters enter
    the ROI.
    """

    candidates = [
        cluster
        for cluster in clusters
        if first_car_cluster_eligible(cluster, config)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda cluster: cluster.center_y_back_mm)


def first_car_cluster_eligible(
    cluster: CarCluster,
    config: LidarParkingConfig,
) -> bool:
    if cluster.center_x_right_mm < config.first_car_min_x_right_mm:
        return False
    if cluster.point_count < max(1, config.first_car_cluster_min_points):
        return False
    if cluster.max_span_mm < max(0.0, config.first_car_min_span_mm):
        return False
    return True


def is_gap_cluster_eligible(
    cluster: CarCluster,
    config: LidarParkingConfig,
) -> bool:
    min_points = max(1, config.car_cluster_min_points, config.gap_cluster_min_points)
    return cluster.point_count >= min_points


def cluster_slot_edge(
    cluster: CarCluster,
    axis_x: float,
    axis_y: float,
    depth_x: float,
    depth_y: float,
    side: int,
) -> Tuple[float, float]:
    center_x = cluster.center_x_right_mm
    center_y = cluster.center_y_back_mm
    axis_radius = (
        abs(axis_x) * (cluster.x_max_mm - cluster.x_min_mm) / 2.0
        + abs(axis_y) * (cluster.y_back_max_mm - cluster.y_back_min_mm) / 2.0
    )
    depth_radius = (
        abs(depth_x) * (cluster.x_max_mm - cluster.x_min_mm) / 2.0
        + abs(depth_y) * (cluster.y_back_max_mm - cluster.y_back_min_mm) / 2.0
    )
    direction = 1.0 if side >= 0 else -1.0
    return (
        center_x + direction * axis_x * axis_radius - depth_x * depth_radius,
        center_y + direction * axis_y * axis_radius - depth_y * depth_radius,
    )


def choose_gap(
    clusters: Sequence[CarCluster],
    config: LidarParkingConfig,
    reference_depth: Optional[Tuple[float, float]] = None,
    reference_center: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """Return the best oriented gap between any two parked-car clusters.

    The tuple is ``(width, first_y, second_y, first_x, second_x, depth_x,
    depth_y)``. The returned
    edge points lie on the sensor-facing surfaces of the two parked cars. This
    lets the rendered bay begin at the obstacle boundary instead of centering
    its depth on the obstacle clusters. Measuring the gap along the line between
    cluster centers keeps the same two cars trackable after the ego vehicle
    rotates and one car enters the left half-plane.
    """

    gap_clusters = tuple(
        cluster for cluster in clusters
        if is_gap_cluster_eligible(cluster, config)
    )
    min_pair_points = max(
        2,
        config.gap_pair_min_points,
        2 * max(1, config.gap_cluster_min_points),
    )
    candidates = []
    for first_index, first in enumerate(gap_clusters):
        first_center_x = first.center_x_right_mm
        first_center_y = first.center_y_back_mm
        for second in gap_clusters[first_index + 1:]:
            if first.point_count + second.point_count < min_pair_points:
                continue
            second_center_x = second.center_x_right_mm
            second_center_y = second.center_y_back_mm
            axis_x = second_center_x - first_center_x
            axis_y = second_center_y - first_center_y
            center_distance = hypot(axis_x, axis_y)
            if center_distance <= 1e-6:
                continue
            unit_x = axis_x / center_distance
            unit_y = axis_y / center_distance
            first_radius = (
                abs(unit_x) * (first.x_max_mm - first.x_min_mm) / 2.0
                + abs(unit_y) * (first.y_back_max_mm - first.y_back_min_mm) / 2.0
            )
            second_radius = (
                abs(unit_x) * (second.x_max_mm - second.x_min_mm) / 2.0
                + abs(unit_y) * (second.y_back_max_mm - second.y_back_min_mm) / 2.0
            )
            width = center_distance - first_radius - second_radius
            if not config.observed_gap_min_mm <= width <= config.observed_gap_max_mm:
                continue
            first_edge_x = first_center_x + unit_x * first_radius
            first_edge_y = first_center_y + unit_y * first_radius
            second_edge_x = second_center_x - unit_x * second_radius
            second_edge_y = second_center_y - unit_y * second_radius
            center_x = (first_edge_x + second_edge_x) / 2.0
            center_y = (first_edge_y + second_edge_y) / 2.0
            if center_x < config.gap_center_x_min_mm:
                continue
            if (
                reference_center is None
                and center_y < config.gap_center_y_back_min_mm
            ):
                continue

            # On the first observation, choose the normal that points from the
            # LiDAR toward the slot. Once a slot is tracked, keep the normal
            # continuous with that initial depth direction even after the
            # LiDAR crosses the entrance line.
            depth_x = -unit_y
            depth_y = unit_x
            if reference_depth is not None:
                flip_depth = (
                    depth_x * reference_depth[0] + depth_y * reference_depth[1]
                    < 0.0
                )
            else:
                flip_depth = center_x * depth_x + center_y * depth_y < 0.0
            if flip_depth:
                depth_x = -depth_x
                depth_y = -depth_y
            first_depth_radius = (
                abs(depth_x) * (first.x_max_mm - first.x_min_mm) / 2.0
                + abs(depth_y) * (first.y_back_max_mm - first.y_back_min_mm) / 2.0
            )
            second_depth_radius = (
                abs(depth_x) * (second.x_max_mm - second.x_min_mm) / 2.0
                + abs(depth_y) * (second.y_back_max_mm - second.y_back_min_mm) / 2.0
            )
            first_edge_x -= depth_x * first_depth_radius
            first_edge_y -= depth_y * first_depth_radius
            second_edge_x -= depth_x * second_depth_radius
            second_edge_y -= depth_y * second_depth_radius
            entrance_center_x = (first_edge_x + second_edge_x) / 2.0
            entrance_center_y = (first_edge_y + second_edge_y) / 2.0
            continuity_distance = (
                hypot(
                    entrance_center_x - reference_center[0],
                    entrance_center_y - reference_center[1],
                )
                if reference_center is not None
                else 0.0
            )
            candidates.append((
                continuity_distance,
                abs(width - config.expected_observed_gap_mm),
                width,
                first_edge_y,
                second_edge_y,
                first_edge_x,
                second_edge_x,
                depth_x,
                depth_y,
            ))
    if not candidates:
        return None
    _, _, width, first_y, second_y, first_x, second_x, depth_x, depth_y = min(
        candidates,
        key=(
            (lambda item: (item[0], item[1]))
            if reference_center is not None
            else (lambda item: (item[1], item[0]))
        ),
    )
    return width, first_y, second_y, first_x, second_x, depth_x, depth_y


def infer_dynamic_slot_polygon(
    observation: LidarParkingObservation,
    depth_mm: float,
    width_mm: Optional[float] = None,
) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Infer a moving parking-space rectangle from the two bordering cars.

    The entrance side passes through the sensor-facing obstacle edges that
    delimit the empty gap. The rectangle rotates with the vector between those
    edges and extends the official parking-space depth away from the LiDAR. It
    is no longer centered on the parked cars. A short coast keeps the last valid
    pose visible through momentary LiDAR occlusion.
    """

    if (
        not observation.gap_found
        or observation.gap_near_edge_x_right_mm is None
        or observation.gap_near_edge_y_back_mm is None
        or observation.gap_far_edge_x_right_mm is None
        or observation.gap_far_edge_y_back_mm is None
    ):
        return None

    detected_first_edge = (
        observation.gap_near_edge_x_right_mm,
        observation.gap_near_edge_y_back_mm,
    )
    detected_second_edge = (
        observation.gap_far_edge_x_right_mm,
        observation.gap_far_edge_y_back_mm,
    )
    axis_x = detected_second_edge[0] - detected_first_edge[0]
    axis_y = detected_second_edge[1] - detected_first_edge[1]
    axis_length = hypot(axis_x, axis_y)
    if axis_length <= 1e-6:
        return None

    axis_x /= axis_length
    axis_y /= axis_length
    center_x = (detected_first_edge[0] + detected_second_edge[0]) / 2.0
    center_y = (detected_first_edge[1] + detected_second_edge[1]) / 2.0
    locked_width = axis_length if width_mm is None else max(0.0, width_mm)
    half_width = locked_width / 2.0
    first_edge = (
        center_x - axis_x * half_width,
        center_y - axis_y * half_width,
    )
    second_edge = (
        center_x + axis_x * half_width,
        center_y + axis_y * half_width,
    )

    # Use the depth direction locked by the tracker. Only the sign of the
    # perpendicular is selected here, so the rectangle remains orthogonal while
    # never flipping 180 degrees as the LiDAR crosses the entrance line.
    normal_x = -axis_y
    normal_y = axis_x
    if (
        observation.slot_depth_x_right is not None
        and observation.slot_depth_y_back is not None
    ):
        flip_normal = (
            normal_x * observation.slot_depth_x_right
            + normal_y * observation.slot_depth_y_back
            < 0.0
        )
    else:
        flip_normal = center_x * normal_x + center_y * normal_y < 0.0
    if flip_normal:
        normal_x = -normal_x
        normal_y = -normal_y
    offset_x = normal_x * max(0.0, depth_mm)
    offset_y = normal_y * max(0.0, depth_mm)
    return (
        first_edge,
        second_edge,
        (second_edge[0] + offset_x, second_edge[1] + offset_y),
        (first_edge[0] + offset_x, first_edge[1] + offset_y),
    )


def summarize_cluster(points: Sequence[Tuple[float, float]]) -> CarCluster:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return CarCluster(
        point_count=len(points),
        x_min_mm=min(xs),
        x_max_mm=max(xs),
        y_back_min_mm=min(ys),
        y_back_max_mm=max(ys),
    )


def cluster_points(
    points: Sequence[Tuple[float, float]],
    radius_mm: float,
    minimum_points: int,
) -> List[List[Tuple[float, float]]]:
    if not points:
        return []
    radius_squared = max(0.0, radius_mm) ** 2
    unvisited = set(range(len(points)))
    clusters = []
    while unvisited:
        seed = unvisited.pop()
        component = [seed]
        pending = [seed]
        while pending:
            current = pending.pop()
            cx, cy = points[current]
            neighbors = []
            for index in unvisited:
                px, py = points[index]
                if (px - cx) ** 2 + (py - cy) ** 2 <= radius_squared:
                    neighbors.append(index)
            for index in neighbors:
                unvisited.remove(index)
                pending.append(index)
                component.append(index)
        if len(component) >= max(1, minimum_points):
            clusters.append([points[index] for index in component])
    return clusters


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smooth_value(previous: Optional[float], current: float, alpha: float) -> float:
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


def smooth_unit_vector(
    previous_x: Optional[float],
    previous_y: Optional[float],
    current_x: float,
    current_y: float,
    alpha: float,
) -> Tuple[float, float]:
    if previous_x is None or previous_y is None:
        length = hypot(current_x, current_y)
        return (
            (current_x / length, current_y / length)
            if length > 1e-6
            else (1.0, 0.0)
        )
    if current_x * previous_x + current_y * previous_y < 0.0:
        current_x = -current_x
        current_y = -current_y
    mixed_x = alpha * current_x + (1.0 - alpha) * previous_x
    mixed_y = alpha * current_y + (1.0 - alpha) * previous_y
    length = hypot(mixed_x, mixed_y)
    if length <= 1e-6:
        return previous_x, previous_y
    return mixed_x / length, mixed_y / length
