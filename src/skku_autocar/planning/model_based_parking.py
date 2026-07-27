from __future__ import annotations

from dataclasses import dataclass
from math import atan, atan2, degrees, hypot
from typing import Optional, Sequence, Tuple

from ..estimation.lidar_triangulation import (
    LidarDecisionTriangle,
    decision_triangle_from_observation,
)
from ..estimation.parking_geometry import ParkingGeometry
from ..estimation.parking_lidar import LidarParkingObservation
from ..types import ControlCommand
from .hybrid_parking_path import HybridPathConfig, VehicleModel
from .t_parking_planner import ParkingPlan, ParkingState


Point = Tuple[float, float]


@dataclass(frozen=True)
class ModelBasedParkingConfig:
    """LiDAR-only reactive triangulation controller parameters."""

    search_speed: int = 100
    gap_tracking_speed: int = 50
    maneuver_forward_speed: int = 90
    maneuver_reverse_speed: int = -90
    final_reverse_speed: int = -60
    exit_speed: int = 50
    straight_steering_trim: int = 0
    max_steering_command: int = 150

    lidar_first_car_gate_min_x_mm: float = 350.0
    lidar_first_car_gate_max_x_mm: float = 1900.0
    lidar_first_car_gate_max_range_mm: float = 3000.0
    lidar_first_car_confirm_scans: int = 3
    lidar_first_car_lost_scans: int = 6
    lidar_first_car_speed: int = 100
    right_ultrasonic_first_car_max_mm: float = 2500.0
    right_ultrasonic_first_car_confirm_scans: int = 1
    right_ultrasonic_open_confirm_scans: int = 1

    entry_setup_speed: int = 50
    entry_setup_steering: int = -150
    entry_setup_duration_s: float = 7.0
    pair_confirm_scans: int = 3
    reverse_pair_coast_scans: int = 2
    gear_change_stop_frames: int = 2

    reverse_lookahead_mm: float = 650.0
    # Signed perpendicular distance of the rear LiDAR from the line joining
    # the two inner car edges. In 205519 a correctly inserted vehicle put this
    # line almost exactly through the LiDAR origin.
    target_sensor_depth_mm: float = 0.0
    final_slow_distance_mm: float = 550.0
    final_slow_heading_deg: float = 15.0
    final_slow_lateral_mm: float = 180.0
    steering_filter_alpha: float = 0.40
    steering_max_delta_per_scan: int = 35

    parking_complete_lateral_mm: float = 120.0
    parking_complete_heading_deg: float = 8.0
    parking_complete_depth_tolerance_mm: float = 90.0
    parking_complete_confirm_scans: int = 3
    park_hold_s: float = 3.4
    auto_exit_enabled: bool = False

    search_timeout_s: float = 60.0
    maneuver_watchdog_s: float = 75.0
    ultrasonic_stale_after_s: float = 0.8


@dataclass(frozen=True)
class TriangulationControlDebug:
    lidar_timestamp: float = 0.0
    pair_valid: bool = False
    car1_x_mm: Optional[float] = None
    car1_y_back_mm: Optional[float] = None
    car2_x_mm: Optional[float] = None
    car2_y_back_mm: Optional[float] = None
    entrance_x_mm: Optional[float] = None
    entrance_y_back_mm: Optional[float] = None
    depth_x: Optional[float] = None
    depth_y_back: Optional[float] = None
    gap_width_mm: Optional[float] = None
    decision_angle_deg: Optional[float] = None
    heading_error_deg: Optional[float] = None
    lateral_error_mm: Optional[float] = None
    depth_progress_mm: Optional[float] = None
    depth_remaining_mm: Optional[float] = None
    target_x_mm: Optional[float] = None
    target_y_back_mm: Optional[float] = None
    curvature_per_mm: Optional[float] = None
    geometric_steering: Optional[int] = None
    physical_steering: Optional[int] = None


@dataclass(frozen=True)
class _LiveTriangleGeometry:
    entrance: Point
    axis: Point
    depth: Point
    lateral_error_mm: float
    depth_progress_mm: float
    depth_remaining_mm: float
    heading_error_deg: float


class ModelBasedTParkingPlanner:
    """Park between two cars using a fresh LiDAR triangle on every scan.

    No camera, ultrasonic value, locked rectangle, ICP pose, frozen path, or
    elapsed-time driving segment participates in control.  The only moving
    commands are:

    * straight search until the first car disappears,
    * forward-left until three fresh two-car triangles are observed,
    * reverse pure pursuit of the live triangle centreline.
    """

    ENTRY_ACQUIRE_PAIR = "acquire_live_pair"
    ENTRY_TIMED_LEFT = "timed_left_setup"
    ENTRY_REVERSE_LIVE = "reverse_live_triangle"

    def __init__(
        self,
        config: ModelBasedParkingConfig = ModelBasedParkingConfig(),
        vehicle: VehicleModel = VehicleModel(),
        path_config: HybridPathConfig = HybridPathConfig(),
        *,
        sensor_to_rear_axle_y_back_mm: float = -300.0,
    ) -> None:
        del path_config
        self.config = config
        self.vehicle = vehicle
        self.sensor_to_rear_axle_y_back_mm = float(
            sensor_to_rear_axle_y_back_mm
        )
        self.state = ParkingState.IDLE
        self._state_started_at = 0.0
        self._entry_phase = self.ENTRY_ACQUIRE_PAIR
        self._lidar_reset_requested = False
        self._last_lidar_timestamp: Optional[float] = None
        self._first_car_scans = 0
        self._first_car_seen = False
        self._first_car_lost_scans = 0
        self._ultrasonic_close_scans = 0
        self._ultrasonic_none_scans = 0
        self._last_ultrasonic_timestamp: Optional[float] = None
        self._pair_scans = 0
        self._pair_candidate_seen = False
        self._reverse_pair_coast_scans = 0
        self._gear_stop_frames = 0
        self._complete_scans = 0
        self._decision_triangle = LidarDecisionTriangle()
        self._filtered_geometric_steering: Optional[float] = None
        self._debug = TriangulationControlDebug()

    @property
    def entry_phase(self) -> str:
        return self._entry_phase

    @property
    def decision_triangle(self) -> LidarDecisionTriangle:
        return self._decision_triangle

    @property
    def debug_snapshot(self) -> TriangulationControlDebug:
        return self._debug

    @property
    def lidar_slot_lock_allowed(self) -> bool:
        return False

    def accepts_lidar_slot_lock(
        self,
        observation: LidarParkingObservation,
    ) -> bool:
        del observation
        return False

    def consume_lidar_reset_request(self) -> bool:
        requested = self._lidar_reset_requested
        self._lidar_reset_requested = False
        return requested

    def start(self, now: float) -> bool:
        if self.state not in (
            ParkingState.IDLE,
            ParkingState.ABORTED,
            ParkingState.EMERGENCY_STOP,
            ParkingState.PARKED,
            ParkingState.EXIT_DONE,
        ):
            return False
        self._reset_counters()
        self._enter(ParkingState.SEARCH_CARS, now)
        return True

    def reset(self, now: float = 0.0) -> None:
        self.state = ParkingState.IDLE
        self._state_started_at = now
        self._reset_counters()

    def update(
        self,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        slot_polygon: Optional[Sequence[Point]],
        now: float,
        *,
        enabled: bool = True,
        left_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_mm: Optional[float] = None,
        front_left_ultrasonic_mm: Optional[float] = None,
        front_center_ultrasonic_mm: Optional[float] = None,
        front_right_ultrasonic_mm: Optional[float] = None,
        right_ultrasonic_reported: bool = False,
        right_ultrasonic_timestamp: Optional[float] = None,
    ) -> ParkingPlan:
        del (
            geometry,
            slot_polygon,
            left_ultrasonic_mm,
            front_left_ultrasonic_mm,
            front_center_ultrasonic_mm,
            front_right_ultrasonic_mm,
        )
        if not enabled:
            self.reset(now)
            return self._stop("parking_disabled")
        if self.state == ParkingState.IDLE:
            return self._stop("waiting_for_start")
        if self.state == ParkingState.ABORTED:
            return self._stop("parking_aborted")
        if self.state == ParkingState.PARKED:
            return self._stop("parked_lidar_triangle_complete")

        if self.state in (ParkingState.SEARCH_CARS, ParkingState.TRACK_GAP):
            return self._search(
                lidar,
                now,
                right_ultrasonic_mm,
                right_ultrasonic_reported,
                right_ultrasonic_timestamp,
            )
        if self.state == ParkingState.ENTRY_SETUP:
            if self._entry_phase == self.ENTRY_TIMED_LEFT:
                return self._timed_left_setup(now)
            return self._acquire_pair(lidar, now)
        if self.state in (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
        ):
            return self._reverse_live(lidar, now)
        return self._abort(now, "unsupported_lidar_triangle_state")

    def _search(
        self,
        lidar: LidarParkingObservation,
        now: float,
        right_ultrasonic_mm: Optional[float],
        right_ultrasonic_reported: bool,
        right_ultrasonic_timestamp: Optional[float],
    ) -> ParkingPlan:
        if self._expired(now, self.config.search_timeout_s):
            return self._abort(now, "lidar_search_watchdog")
        del lidar
        new_ultrasonic = (
            right_ultrasonic_reported
            and right_ultrasonic_timestamp is not None
            and right_ultrasonic_timestamp
            != self._last_ultrasonic_timestamp
        )
        if new_ultrasonic:
            self._last_ultrasonic_timestamp = right_ultrasonic_timestamp

        if not self._first_car_seen:
            if new_ultrasonic:
                # Any valid SR echo means the right-side sensor is alongside
                # the first car.  215034 measured that car at 1186-1485 mm;
                # a fixed 1100 mm ceiling incorrectly rejected every sample.
                close = right_ultrasonic_mm is not None
                self._ultrasonic_close_scans = (
                    self._ultrasonic_close_scans + 1 if close else 0
                )
            if self._ultrasonic_close_scans >= max(
                1,
                self.config.right_ultrasonic_first_car_confirm_scans,
            ):
                self._first_car_seen = True
                self._ultrasonic_none_scans = 0
            if not self._first_car_seen:
                self.state = ParkingState.SEARCH_CARS
                return self._drive(
                    self.config.search_speed,
                    self._straight_steering(),
                    "ultrasonic_search_first_car SR=%s"
                    % (
                        "None"
                        if right_ultrasonic_mm is None
                        else "%.0fmm" % right_ultrasonic_mm,
                    ),
                )

        if new_ultrasonic:
            self._ultrasonic_none_scans = (
                self._ultrasonic_none_scans + 1
                if right_ultrasonic_mm is None
                else 0
            )
        if self._ultrasonic_none_scans < max(
            1,
            self.config.right_ultrasonic_open_confirm_scans,
        ):
            self.state = ParkingState.TRACK_GAP
            return self._drive(
                abs(self.config.lidar_first_car_speed),
                self._straight_steering(),
                "ultrasonic_first_car_seen_wait_none SR=%s"
                % (
                    "None"
                    if right_ultrasonic_mm is None
                    else "%.0fmm" % right_ultrasonic_mm,
                ),
            )

        self._enter(ParkingState.ENTRY_SETUP, now)
        self._entry_phase = self.ENTRY_TIMED_LEFT
        return self._drive(
            abs(self.config.entry_setup_speed),
            self._clamp(self.config.entry_setup_steering),
            "ultrasonic_none_start_timed_left",
        )

    def _timed_left_setup(self, now: float) -> ParkingPlan:
        elapsed = max(0.0, now - self._state_started_at)
        if elapsed < max(0.0, self.config.entry_setup_duration_s):
            return self._drive(
                abs(self.config.entry_setup_speed),
                self._clamp(self.config.entry_setup_steering),
                "timed_left_setup:%.2f/%.2fs"
                % (elapsed, self.config.entry_setup_duration_s),
            )
        self._entry_phase = self.ENTRY_ACQUIRE_PAIR
        self._lidar_reset_requested = True
        self._pair_scans = 0
        self._pair_candidate_seen = False
        self._decision_triangle = LidarDecisionTriangle(
            reason="waiting_for_post_timed_turn_live_pair"
        )
        return self._stop("timed_left_complete_reset_lidar")

    def _acquire_pair(
        self,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.maneuver_watchdog_s):
            return self._abort(now, "triangulation_maneuver_watchdog")
        if not lidar.valid:
            self._pair_scans = 0
            return self._stop(
                "lidar_not_fresh_hold_pair_acquire:%s"
                % (lidar.reason or "invalid_scan")
            )

        triangle = decision_triangle_from_observation(lidar)
        fresh_pair = self._fresh_pair(lidar, triangle)
        new_scan = self._new_scan(lidar)
        if new_scan:
            self._pair_scans = self._pair_scans + 1 if fresh_pair else 0
        if triangle.valid:
            self._pair_candidate_seen = True
            self._decision_triangle = triangle
            self._update_debug(lidar, triangle, self._live_geometry(lidar, triangle))

        if self._pair_scans < max(1, self.config.pair_confirm_scans):
            return self._stop(
                "stationary_post_turn_triangle:%d/%d %s"
                % (
                    self._pair_scans,
                    max(1, self.config.pair_confirm_scans),
                    triangle.reason,
                ),
            )

        self._entry_phase = self.ENTRY_REVERSE_LIVE
        self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
        self._gear_stop_frames = max(1, self.config.gear_change_stop_frames)
        return self._reverse_live(lidar, now)

    def _reverse_live(
        self,
        lidar: LidarParkingObservation,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.maneuver_watchdog_s):
            return self._abort(now, "triangulation_maneuver_watchdog")
        if not lidar.valid:
            return self._stop(
                "lidar_not_fresh_hold_reverse:%s"
                % (lidar.reason or "invalid_scan")
            )

        triangle = decision_triangle_from_observation(lidar)
        fresh_pair = self._fresh_pair(lidar, triangle)
        new_scan = self._new_scan(lidar)
        coasting_pair = False
        if fresh_pair:
            self._reverse_pair_coast_scans = 0
            self._decision_triangle = triangle
        elif (
            lidar.valid
            and lidar.coasted
            and self._decision_triangle.valid
            and self._reverse_pair_coast_scans
            < max(0, self.config.reverse_pair_coast_scans)
        ):
            if new_scan:
                self._reverse_pair_coast_scans += 1
            triangle = self._decision_triangle
            coasting_pair = True
        else:
            self._pair_scans = 0
            self._filtered_geometric_steering = None
            self._update_debug(lidar, triangle, None)
            return self._stop(
                "live_two_car_pair_lost_hold:%s"
                % (lidar.reason or triangle.reason)
            )

        live = self._live_geometry(lidar, triangle)
        if live is None:
            return self._stop("live_triangle_geometry_invalid")

        depth_ready = (
            live.depth_remaining_mm
            <= max(0.0, self.config.parking_complete_depth_tolerance_mm)
        )
        aligned = (
            abs(live.lateral_error_mm)
            <= max(0.0, self.config.parking_complete_lateral_mm)
            and abs(live.heading_error_deg)
            <= max(0.0, self.config.parking_complete_heading_deg)
        )
        if new_scan:
            self._complete_scans = (
                self._complete_scans + 1
                if depth_ready and aligned
                else 0
            )
        if self._complete_scans >= max(
            1,
            self.config.parking_complete_confirm_scans,
        ):
            self._enter(ParkingState.PARKED, now)
            self._update_debug(lidar, triangle, live)
            return self._stop("live_triangle_goal_reached")
        target, curvature = self._live_target(live)
        geometric = self._steering_for_curvature(curvature, new_scan)
        physical = self._physical_steering_command(geometric)
        self._update_debug(
            lidar,
            triangle,
            live,
            target=target,
            curvature=curvature,
            geometric_steering=geometric,
            physical_steering=physical,
        )

        if self._gear_stop_frames > 0:
            self._gear_stop_frames -= 1
            return self._stop(
                "gear_change_presteer_live_triangle:%d"
                % self._gear_stop_frames,
                steering=physical,
            )

        slow = (
            live.depth_remaining_mm
            <= max(0.0, self.config.final_slow_distance_mm)
            or abs(live.heading_error_deg)
            <= max(0.0, self.config.final_slow_heading_deg)
            and abs(live.lateral_error_mm)
            <= max(0.0, self.config.final_slow_lateral_mm)
        )
        speed = (
            -abs(self.config.final_reverse_speed)
            if slow
            else -abs(self.config.maneuver_reverse_speed)
        )
        self.state = (
            ParkingState.FOLLOW_SLOT_CENTER
            if slow
            else ParkingState.FOLLOW_ENTRY_CURVE
        )
        reason_prefix = (
            "live_triangle_reverse_coast=%d "
            % self._reverse_pair_coast_scans
            if coasting_pair
            else "live_triangle_reverse "
        )
        return self._drive(
            speed,
            physical,
            (
                reason_prefix
                + "head=%+.1f lat=%+.0f depth=%.0f remain=%.0f"
            )
            % (
                live.heading_error_deg,
                live.lateral_error_mm,
                live.depth_progress_mm,
                live.depth_remaining_mm,
            ),
        )

    def _fresh_pair(
        self,
        lidar: LidarParkingObservation,
        triangle: LidarDecisionTriangle,
    ) -> bool:
        return (
            lidar.valid
            and lidar.gap_pair_observed
            and lidar.second_car_seen
            and not lidar.coasted
            and triangle.valid
        )

    def _live_geometry(
        self,
        lidar: LidarParkingObservation,
        triangle: LidarDecisionTriangle,
    ) -> Optional[_LiveTriangleGeometry]:
        if (
            not triangle.valid
            or lidar.slot_depth_x_right is None
            or lidar.slot_depth_y_back is None
        ):
            return None
        axis_x = triangle.car2_edge[0] - triangle.car1_edge[0]
        axis_y = triangle.car2_edge[1] - triangle.car1_edge[1]
        axis_length = hypot(axis_x, axis_y)
        depth_x = float(lidar.slot_depth_x_right)
        depth_y = float(lidar.slot_depth_y_back)
        depth_length = hypot(depth_x, depth_y)
        if axis_length <= 1.0 or depth_length <= 1e-6:
            return None
        axis = (axis_x / axis_length, axis_y / axis_length)
        depth = (depth_x / depth_length, depth_y / depth_length)
        entrance = (
            (triangle.car1_edge[0] + triangle.car2_edge[0]) / 2.0,
            (triangle.car1_edge[1] + triangle.car2_edge[1]) / 2.0,
        )
        from_entrance = (-entrance[0], -entrance[1])
        lateral = (
            from_entrance[0] * axis[0]
            + from_entrance[1] * axis[1]
        )
        progress = (
            from_entrance[0] * depth[0]
            + from_entrance[1] * depth[1]
        )
        return _LiveTriangleGeometry(
            entrance=entrance,
            axis=axis,
            depth=depth,
            lateral_error_mm=lateral,
            depth_progress_mm=progress,
            depth_remaining_mm=(
                self.config.target_sensor_depth_mm - progress
            ),
            heading_error_deg=degrees(atan2(depth[0], depth[1])),
        )

    def _live_target(
        self,
        live: _LiveTriangleGeometry,
    ) -> Tuple[Point, float]:
        rear_axle = (0.0, self.sensor_to_rear_axle_y_back_mm)
        rear_progress = (
            live.depth_progress_mm
            + rear_axle[0] * live.depth[0]
            + rear_axle[1] * live.depth[1]
        )
        target_rear_depth = (
            self.config.target_sensor_depth_mm
            + self.sensor_to_rear_axle_y_back_mm
        )
        lookahead_depth = min(
            target_rear_depth,
            rear_progress + max(100.0, self.config.reverse_lookahead_mm),
        )
        target = (
            live.entrance[0] + live.depth[0] * lookahead_depth,
            live.entrance[1] + live.depth[1] * lookahead_depth,
        )
        relative_x = target[0] - rear_axle[0]
        relative_y_back = target[1] - rear_axle[1]
        distance_squared = (
            relative_x * relative_x
            + relative_y_back * relative_y_back
        )
        curvature = (
            0.0
            if distance_squared <= 1.0
            else 2.0 * relative_x / distance_squared
        )
        return target, curvature

    def _steering_for_curvature(
        self,
        curvature_per_mm: float,
        new_scan: bool,
    ) -> int:
        maximum = max(1, abs(self.config.max_steering_command))
        maximum_angle = max(0.1, abs(self.vehicle.max_steering_angle_deg))
        angle_deg = degrees(
            atan(self.vehicle.wheelbase_mm * curvature_per_mm)
        )
        raw = max(-maximum, min(maximum, maximum * angle_deg / maximum_angle))
        previous = self._filtered_geometric_steering
        if previous is None:
            filtered = raw
        elif new_scan:
            alpha = max(0.0, min(1.0, self.config.steering_filter_alpha))
            filtered = previous + alpha * (raw - previous)
            delta = max(1, abs(self.config.steering_max_delta_per_scan))
            filtered = max(previous - delta, min(previous + delta, filtered))
        else:
            filtered = previous
        self._filtered_geometric_steering = filtered
        return int(round(filtered))

    def _new_scan(self, lidar: LidarParkingObservation) -> bool:
        if (
            self._last_lidar_timestamp is None
            or lidar.timestamp > self._last_lidar_timestamp + 1e-6
        ):
            self._last_lidar_timestamp = lidar.timestamp
            return True
        return False

    def _update_debug(
        self,
        lidar: LidarParkingObservation,
        triangle: LidarDecisionTriangle,
        live: Optional[_LiveTriangleGeometry],
        *,
        target: Optional[Point] = None,
        curvature: Optional[float] = None,
        geometric_steering: Optional[int] = None,
        physical_steering: Optional[int] = None,
    ) -> None:
        self._debug = TriangulationControlDebug(
            lidar_timestamp=lidar.timestamp,
            pair_valid=triangle.valid and live is not None,
            car1_x_mm=triangle.car1_edge[0] if triangle.valid else None,
            car1_y_back_mm=triangle.car1_edge[1] if triangle.valid else None,
            car2_x_mm=triangle.car2_edge[0] if triangle.valid else None,
            car2_y_back_mm=triangle.car2_edge[1] if triangle.valid else None,
            entrance_x_mm=live.entrance[0] if live is not None else None,
            entrance_y_back_mm=live.entrance[1] if live is not None else None,
            depth_x=live.depth[0] if live is not None else None,
            depth_y_back=live.depth[1] if live is not None else None,
            gap_width_mm=triangle.car_gap_mm if triangle.valid else None,
            decision_angle_deg=(
                triangle.decision_angle_deg if triangle.valid else None
            ),
            heading_error_deg=(
                live.heading_error_deg if live is not None else None
            ),
            lateral_error_mm=(
                live.lateral_error_mm if live is not None else None
            ),
            depth_progress_mm=(
                live.depth_progress_mm if live is not None else None
            ),
            depth_remaining_mm=(
                live.depth_remaining_mm if live is not None else None
            ),
            target_x_mm=target[0] if target is not None else None,
            target_y_back_mm=target[1] if target is not None else None,
            curvature_per_mm=curvature,
            geometric_steering=geometric_steering,
            physical_steering=physical_steering,
        )

    def _straight_steering(self) -> int:
        return self._clamp(self.config.straight_steering_trim)

    def _physical_steering_command(self, geometric_command: int) -> int:
        maximum = max(1, abs(self.config.max_steering_command))
        geometric = max(-maximum, min(maximum, int(geometric_command)))
        neutral = max(
            -maximum,
            min(maximum, int(self.config.straight_steering_trim)),
        )
        if geometric >= 0:
            command = neutral + (maximum - neutral) * geometric / maximum
        else:
            command = neutral + (maximum + neutral) * geometric / maximum
        return self._clamp(int(round(command)))

    def _drive(self, speed: int, steering: int, reason: str) -> ParkingPlan:
        command = ControlCommand(
            speed=int(speed),
            steering=self._clamp(steering),
            brake=False,
            reason=reason,
        )
        return ParkingPlan(self.state, command, reason)

    def _stop(
        self,
        reason: str,
        *,
        steering: Optional[int] = None,
    ) -> ParkingPlan:
        command = ControlCommand(
            speed=0,
            steering=(
                self._straight_steering()
                if steering is None
                else self._clamp(steering)
            ),
            brake=True,
            reason=reason,
        )
        return ParkingPlan(self.state, command, reason)

    def _abort(self, now: float, reason: str) -> ParkingPlan:
        self._enter(ParkingState.ABORTED, now)
        return self._stop(reason)

    def _clamp(self, value: int) -> int:
        maximum = abs(self.config.max_steering_command)
        return int(max(-maximum, min(maximum, int(value))))

    def _enter(self, state: ParkingState, now: float) -> None:
        self.state = state
        self._state_started_at = now

    def _expired(self, now: float, timeout_s: float) -> bool:
        return timeout_s > 0.0 and now - self._state_started_at >= timeout_s

    def _reset_counters(self) -> None:
        self._entry_phase = self.ENTRY_ACQUIRE_PAIR
        self._lidar_reset_requested = False
        self._last_lidar_timestamp = None
        self._first_car_scans = 0
        self._first_car_seen = False
        self._first_car_lost_scans = 0
        self._ultrasonic_close_scans = 0
        self._ultrasonic_none_scans = 0
        self._last_ultrasonic_timestamp = None
        self._pair_scans = 0
        self._pair_candidate_seen = False
        self._reverse_pair_coast_scans = 0
        self._gear_stop_frames = 0
        self._complete_scans = 0
        self._decision_triangle = LidarDecisionTriangle()
        self._filtered_geometric_steering = None
        self._debug = TriangulationControlDebug()
