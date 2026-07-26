from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, sin
from typing import Optional, Sequence, Tuple

from ..estimation.parking_geometry import ParkingGeometry
from ..estimation.parking_lidar import LidarParkingObservation
from ..types import ControlCommand
from .hybrid_parking_path import (
    HybridAStarParkingPathPlanner,
    HybridParkingPath,
    HybridPathConfig,
    SlotManeuverModel,
    PathPose,
    VehicleModel,
    build_slot_maneuver_model,
    pure_pursuit_curvature,
    steering_command_for_curvature,
    wrap_angle,
)
from .t_parking_planner import ParkingPlan, ParkingState


Point = Tuple[float, float]


@dataclass(frozen=True)
class ModelBasedParkingConfig:
    """Closed-loop mission policy.

    Durations are used only for a mandated stationary hold and fault
    watchdogs. Every moving transition is based on the live slot pose, planned
    path, vehicle footprint, or a distance sensor.
    """

    search_speed: int = 55
    gap_tracking_speed: int = 42
    maneuver_forward_speed: int = 58
    maneuver_reverse_speed: int = -55
    final_reverse_speed: int = -38
    exit_speed: int = 50
    straight_steering_trim: int = -30
    max_steering_command: int = 150
    forward_lookahead_mm: float = 430.0
    reverse_lookahead_mm: float = 340.0
    final_slow_distance_mm: float = 550.0
    slot_lock_confirm_scans: int = 3
    path_failure_limit: int = 5
    gear_change_stop_frames: int = 2
    parking_complete_confirm_scans: int = 3
    goal_position_tolerance_mm: float = 115.0
    goal_heading_tolerance_deg: float = 7.0
    minimum_inside_ratio: float = 0.96
    ultrasonic_kp_steering_per_mm: float = 0.18
    ultrasonic_max_correction: int = 28
    ultrasonic_balance_enable_inside_ratio: float = 0.45
    ultrasonic_max_valid_mm: float = 2500.0
    ultrasonic_stale_after_s: float = 0.8
    ultrasonic_emergency_mm: float = 100.0
    emergency_stop_enabled: bool = True
    park_hold_s: float = 3.4
    auto_exit_enabled: bool = True
    back_clearance_mm: float = 120.0
    inferred_neighbor_width_mm: float = 600.0
    back_wall_depth_mm: float = 500.0
    exit_lane_offset_mm: float = 700.0
    exit_lane_advance_mm: float = 1200.0
    search_timeout_s: float = 60.0
    maneuver_watchdog_s: float = 75.0
    exit_watchdog_s: float = 30.0
    path_replan_deviation_mm: float = 260.0


@dataclass(frozen=True)
class _SlotRelativePathPose:
    axis_mm: float
    depth_mm: float
    forward_axis: float
    forward_depth: float
    gear: int
    steering_rad: float


class ModelBasedTParkingPlanner:
    """LiDAR-relative Hybrid A* plus pure-pursuit T-parking controller."""

    def __init__(
        self,
        config: ModelBasedParkingConfig = ModelBasedParkingConfig(),
        vehicle: VehicleModel = VehicleModel(),
        path_config: HybridPathConfig = HybridPathConfig(),
        *,
        sensor_to_rear_axle_y_back_mm: float = -300.0,
    ) -> None:
        self.config = config
        self.vehicle = vehicle
        self.path_planner = HybridAStarParkingPathPlanner(vehicle, path_config)
        self.sensor_to_rear_axle_y_back_mm = float(
            sensor_to_rear_axle_y_back_mm
        )
        self.state = ParkingState.IDLE
        self._state_started_at = 0.0
        self._slot_confirm_scans = 0
        self._path_failures = 0
        self._complete_scans = 0
        self._active_gear = 0
        self._pending_gear = 0
        self._gear_stop_frames = 0
        self._last_path = HybridParkingPath()
        self._slot_path: Tuple[_SlotRelativePathPose, ...] = ()
        self._slot_path_progress = 0
        self._slot_path_kind = ""
        self._slot_path_cost = 0.0
        self._slot_path_expansions = 0

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
    ) -> ParkingPlan:
        if not enabled:
            self.reset(now)
            return self._stop("parking_disabled")
        if self.state == ParkingState.IDLE:
            return self._stop("waiting_for_start")
        if self.state == ParkingState.ABORTED:
            return self._stop("parking_aborted")
        if self.state == ParkingState.EMERGENCY_STOP:
            return self._stop("emergency_stop_latched")
        if self.state == ParkingState.EXIT_DONE:
            return self._stop("exit_pose_reached")

        emergency = self._emergency_reason(
            lidar,
            left_ultrasonic_mm,
            right_ultrasonic_mm,
            front_left_ultrasonic_mm,
            front_center_ultrasonic_mm,
            front_right_ultrasonic_mm,
        )
        if emergency is not None:
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop(emergency)

        if self.state == ParkingState.PARKED:
            if self._state_elapsed(now) < self.config.park_hold_s:
                return self._stop("parked_hold_3_to_5_seconds")
            if not self.config.auto_exit_enabled:
                return self._stop("parked_hold_complete_auto_exit_disabled")
            self._enter(ParkingState.EXIT_RIGHT, now)
            self._active_gear = 0
            self._clear_slot_path()
            return self._stop("parked_hold_complete_plan_exit")

        if self.state in (ParkingState.SEARCH_CARS, ParkingState.TRACK_GAP):
            return self._search_plan(lidar, slot_polygon, now)

        if not lidar.valid:
            return self._stop("waiting_for_fresh_lidar")
        model = self._slot_model(slot_polygon)
        if model is None:
            self._path_failures += 1
            if self._path_failures >= max(1, self.config.path_failure_limit):
                return self._abort(now, "locked_slot_pose_lost")
            return self._stop("waiting_for_locked_slot_pose")

        if self.state == ParkingState.VERIFY_SLOT_BOX:
            return self._begin_parking_path(model, now)
        if self.state in (
            ParkingState.PLAN_REVERSE_PATH,
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
        ):
            return self._follow_parking_path(
                model,
                geometry,
                lidar,
                now,
                left_ultrasonic_mm,
                right_ultrasonic_mm,
            )
        if self.state in (ParkingState.EXIT_RIGHT, ParkingState.EXIT_STRAIGHT):
            return self._follow_exit_path(model, now)
        return self._abort(now, "unsupported_state:%s" % self.state.value)

    def _search_plan(
        self,
        lidar: LidarParkingObservation,
        slot_polygon: Optional[Sequence[Point]],
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.search_timeout_s):
            return self._abort(now, "slot_search_watchdog")
        ready = (
            lidar.valid
            and lidar.gap_confirmed
            and slot_polygon is not None
            and len(slot_polygon) == 4
        )
        self._slot_confirm_scans = self._slot_confirm_scans + 1 if ready else 0
        if self._slot_confirm_scans >= max(
            1,
            self.config.slot_lock_confirm_scans,
        ):
            self._enter(ParkingState.VERIFY_SLOT_BOX, now)
            return self._stop("slot_pose_locked")
        if not lidar.valid:
            return self._stop("waiting_for_lidar_scan")
        if lidar.gap_found:
            self.state = ParkingState.TRACK_GAP
            return self._drive(
                self.config.gap_tracking_speed,
                self._straight_steering(),
                "tracking_slot_pose:%d/%d"
                % (
                    self._slot_confirm_scans,
                    max(1, self.config.slot_lock_confirm_scans),
                ),
            )
        self.state = ParkingState.SEARCH_CARS
        return self._drive(
            self.config.search_speed,
            self._straight_steering(),
            "straight_search_for_two_bordering_cars",
        )

    def _begin_parking_path(
        self,
        model: SlotManeuverModel,
        now: float,
    ) -> ParkingPlan:
        path = self.path_planner.plan(
            model.parking_goal,
            model.obstacles,
            initial_gear=self._active_gear,
            allowed_gears=(-1, 1),
        )
        self._last_path = path
        if not path.found:
            self._path_failures += 1
            if self._path_failures >= max(1, self.config.path_failure_limit):
                return self._abort(now, path.reason)
            return self._stop(
                "parking_path_retry:%s" % path.reason,
                world_path=path,
            )
        self._path_failures = 0
        self._freeze_path_in_slot(path, model, "parking")
        self._enter(ParkingState.FOLLOW_ENTRY_CURVE, now)
        return self._stop("model_path_armed", world_path=path)

    def _follow_parking_path(
        self,
        model: SlotManeuverModel,
        geometry: ParkingGeometry,
        lidar: LidarParkingObservation,
        now: float,
        left_ultrasonic_mm: Optional[float],
        right_ultrasonic_mm: Optional[float],
    ) -> ParkingPlan:
        if self._expired(now, self.config.maneuver_watchdog_s):
            return self._abort(now, "parking_motion_watchdog")
        if self._parking_complete(model, geometry):
            self._complete_scans += 1
            if self._complete_scans >= max(
                1,
                self.config.parking_complete_confirm_scans,
            ):
                self._enter(ParkingState.PARKED, now)
                self._active_gear = 0
                return self._stop("vehicle_inside_slot_goal_reached")
            return self._stop(
                "parking_complete_confirming:%d/%d"
                % (
                    self._complete_scans,
                    max(1, self.config.parking_complete_confirm_scans),
                )
            )
        self._complete_scans = 0

        if not geometry.found:
            return self._stop("locked_slot_geometry_not_fresh")
        if self._active_gear < 0 and lidar.unsafe:
            self._enter(ParkingState.EMERGENCY_STOP, now)
            return self._stop("rear_lidar_safety_roi_blocked")

        path, deviation = self._path_from_live_slot(model, "parking")
        if (
            not path.found
            or deviation > max(1.0, self.config.path_replan_deviation_mm)
        ):
            path = self.path_planner.plan(
                model.parking_goal,
                model.obstacles,
                initial_gear=self._active_gear,
                allowed_gears=(-1, 1),
            )
            if path.found:
                self._freeze_path_in_slot(path, model, "parking")
                path, deviation = self._path_from_live_slot(model, "parking")
        self._last_path = path
        if not path.found:
            self._path_failures += 1
            if self._path_failures >= max(1, self.config.path_failure_limit):
                return self._abort(now, path.reason)
            return self._stop(
                "parking_replan_retry:%s" % path.reason,
                world_path=path,
            )
        self._path_failures = 0

        if (
            geometry.vehicle_inside_ratio
            >= self.config.ultrasonic_balance_enable_inside_ratio
        ):
            self.state = ParkingState.FOLLOW_SLOT_CENTER
        else:
            self.state = ParkingState.FOLLOW_ENTRY_CURVE
        return self._path_command(
            path,
            model,
            geometry,
            left_ultrasonic_mm,
            right_ultrasonic_mm,
            exiting=False,
        )

    def _follow_exit_path(
        self,
        model: SlotManeuverModel,
        now: float,
    ) -> ParkingPlan:
        if self._expired(now, self.config.exit_watchdog_s):
            return self._abort(now, "exit_motion_watchdog")
        if self._goal_reached(model.exit_goal):
            self._enter(ParkingState.EXIT_DONE, now)
            self._active_gear = 0
            return self._stop("slot_exit_pose_reached")

        path, deviation = self._path_from_live_slot(model, "exit")
        if (
            not path.found
            or deviation > max(1.0, self.config.path_replan_deviation_mm)
        ):
            path = self.path_planner.plan(
                model.exit_goal,
                model.obstacles,
                initial_gear=1,
                allowed_gears=(1,),
            )
            if path.found:
                self._freeze_path_in_slot(path, model, "exit")
                path, deviation = self._path_from_live_slot(model, "exit")
        self._last_path = path
        if not path.found:
            self._path_failures += 1
            if self._path_failures >= max(1, self.config.path_failure_limit):
                return self._abort(now, "exit_%s" % path.reason)
            return self._stop(
                "exit_path_retry:%s" % path.reason,
                world_path=path,
            )
        self._path_failures = 0
        self.state = ParkingState.EXIT_STRAIGHT
        return self._path_command(
            path,
            model,
            ParkingGeometry(),
            None,
            None,
            exiting=True,
        )

    def _path_command(
        self,
        path: HybridParkingPath,
        model: SlotManeuverModel,
        geometry: ParkingGeometry,
        left_ultrasonic_mm: Optional[float],
        right_ultrasonic_mm: Optional[float],
        *,
        exiting: bool,
    ) -> ParkingPlan:
        requested_gear = path.first_gear
        if not requested_gear:
            return self._stop("path_has_no_motion", world_path=path)
        gear_change = self._prepare_gear(requested_gear)
        lookahead_distance = (
            self.config.forward_lookahead_mm
            if requested_gear > 0
            else self.config.reverse_lookahead_mm
        )
        target = path.lookahead(lookahead_distance, requested_gear)
        if target is None:
            return self._stop("path_lookahead_missing", world_path=path)
        curvature = pure_pursuit_curvature(target)
        steering = steering_command_for_curvature(
            curvature,
            self.vehicle,
            self.config.max_steering_command,
        )
        if (
            not exiting
            and requested_gear < 0
            and geometry.vehicle_inside_ratio
            >= self.config.ultrasonic_balance_enable_inside_ratio
        ):
            steering += self._ultrasonic_correction(
                left_ultrasonic_mm,
                right_ultrasonic_mm,
            )
        steering = int(
            max(
                -abs(self.config.max_steering_command),
                min(abs(self.config.max_steering_command), steering),
            )
        )
        if gear_change:
            return self._drive(
                0,
                steering,
                "gear_change_stationary_settle:%+d" % requested_gear,
                world_path=path,
            )

        distance = hypot(
            model.parking_goal.x_right_mm,
            model.parking_goal.y_forward_mm,
        )
        if exiting:
            speed = abs(self.config.exit_speed)
            reason = "follow_exit_hybrid_path"
        elif requested_gear > 0:
            speed = abs(self.config.maneuver_forward_speed)
            reason = "follow_parking_setup_path_forward"
        elif distance <= max(1.0, self.config.final_slow_distance_mm):
            speed = -abs(self.config.final_reverse_speed)
            reason = "follow_parking_path_final_reverse"
        else:
            speed = -abs(self.config.maneuver_reverse_speed)
            reason = "follow_parking_path_reverse"
        return self._drive(
            speed,
            steering,
            "%s goal=%.0fmm head=%+.1fdeg exp=%d"
            % (
                reason,
                distance,
                degrees(model.parking_goal.heading_rad),
                path.expansions,
            ),
            world_path=path,
        )

    def _prepare_gear(self, requested_gear: int) -> bool:
        requested = 1 if requested_gear > 0 else -1
        if self._active_gear == requested:
            self._pending_gear = 0
            self._gear_stop_frames = 0
            return False
        if self._pending_gear != requested:
            self._pending_gear = requested
            self._gear_stop_frames = 1
            return True
        self._gear_stop_frames += 1
        if self._gear_stop_frames <= max(1, self.config.gear_change_stop_frames):
            return True
        self._active_gear = requested
        self._pending_gear = 0
        self._gear_stop_frames = 0
        return False

    def _slot_model(
        self,
        slot_polygon: Optional[Sequence[Point]],
    ) -> Optional[SlotManeuverModel]:
        if slot_polygon is None:
            return None
        return build_slot_maneuver_model(
            slot_polygon,
            sensor_to_rear_axle_y_back_mm=(
                self.sensor_to_rear_axle_y_back_mm
            ),
            vehicle=self.vehicle,
            back_clearance_mm=self.config.back_clearance_mm,
            inferred_neighbor_width_mm=self.config.inferred_neighbor_width_mm,
            back_wall_depth_mm=self.config.back_wall_depth_mm,
            exit_lane_offset_mm=self.config.exit_lane_offset_mm,
            exit_lane_advance_mm=self.config.exit_lane_advance_mm,
        )

    def _freeze_path_in_slot(
        self,
        path: HybridParkingPath,
        model: SlotManeuverModel,
        kind: str,
    ) -> None:
        relative = []
        axis = model.slot_axis
        depth = model.depth_direction
        entrance = model.entrance_center
        for pose in path.poses:
            delta_x = pose.x_right_mm - entrance[0]
            delta_y = pose.y_forward_mm - entrance[1]
            forward = (sin(pose.heading_rad), cos(pose.heading_rad))
            relative.append(
                _SlotRelativePathPose(
                    axis_mm=delta_x * axis[0] + delta_y * axis[1],
                    depth_mm=delta_x * depth[0] + delta_y * depth[1],
                    forward_axis=forward[0] * axis[0] + forward[1] * axis[1],
                    forward_depth=(
                        forward[0] * depth[0] + forward[1] * depth[1]
                    ),
                    gear=pose.gear,
                    steering_rad=pose.steering_rad,
                )
            )
        self._slot_path = tuple(relative)
        self._slot_path_progress = 0
        self._slot_path_kind = kind
        self._slot_path_cost = path.cost
        self._slot_path_expansions = path.expansions

    def _path_from_live_slot(
        self,
        model: SlotManeuverModel,
        kind: str,
    ) -> Tuple[HybridParkingPath, float]:
        if self._slot_path_kind != kind or not self._slot_path:
            return HybridParkingPath(reason="no_frozen_slot_path"), float("inf")
        axis = model.slot_axis
        depth = model.depth_direction
        entrance = model.entrance_center
        live = []
        for pose in self._slot_path:
            x = (
                entrance[0]
                + axis[0] * pose.axis_mm
                + depth[0] * pose.depth_mm
            )
            y = (
                entrance[1]
                + axis[1] * pose.axis_mm
                + depth[1] * pose.depth_mm
            )
            forward_x = (
                axis[0] * pose.forward_axis
                + depth[0] * pose.forward_depth
            )
            forward_y = (
                axis[1] * pose.forward_axis
                + depth[1] * pose.forward_depth
            )
            live.append(
                PathPose(
                    x_right_mm=x,
                    y_forward_mm=y,
                    heading_rad=atan2(forward_x, forward_y),
                    gear=pose.gear,
                    steering_rad=pose.steering_rad,
                )
            )

        start = max(0, self._slot_path_progress - 2)
        stop = min(len(live), self._slot_path_progress + 14)
        nearest = min(
            range(start, stop),
            key=lambda index: hypot(
                live[index].x_right_mm,
                live[index].y_forward_mm,
            ),
        )
        self._slot_path_progress = max(self._slot_path_progress, nearest)
        deviation = hypot(
            live[self._slot_path_progress].x_right_mm,
            live[self._slot_path_progress].y_forward_mm,
        )
        remaining = tuple(live[self._slot_path_progress + 1 :])
        goal = (
            model.parking_goal if kind == "parking" else model.exit_goal
        )
        if not remaining:
            return (
                HybridParkingPath(
                    goal=goal,
                    reason="frozen_slot_path_consumed",
                ),
                deviation,
            )
        current = PathPose(0.0, 0.0, 0.0)
        return (
            HybridParkingPath(
                found=True,
                poses=(current,) + remaining,
                goal=goal,
                cost=self._slot_path_cost,
                expansions=self._slot_path_expansions,
                reason="frozen_slot_path_tracked",
            ),
            deviation,
        )

    def _parking_complete(
        self,
        model: SlotManeuverModel,
        geometry: ParkingGeometry,
    ) -> bool:
        return (
            self._goal_reached(model.parking_goal)
            and geometry.vehicle_inside_ratio >= self.config.minimum_inside_ratio
            and geometry.vehicle_fully_inside
        )

    def _goal_reached(self, goal: object) -> bool:
        position = hypot(
            float(getattr(goal, "x_right_mm")),
            float(getattr(goal, "y_forward_mm")),
        )
        heading = abs(
            degrees(
                wrap_angle(float(getattr(goal, "heading_rad")))
            )
        )
        return (
            position <= self.config.goal_position_tolerance_mm
            and heading <= self.config.goal_heading_tolerance_deg
        )

    def _emergency_reason(
        self,
        lidar: LidarParkingObservation,
        left_mm: Optional[float],
        right_mm: Optional[float],
        front_left_mm: Optional[float],
        front_center_mm: Optional[float],
        front_right_mm: Optional[float],
    ) -> Optional[str]:
        if not self.config.emergency_stop_enabled:
            return None
        if self._active_gear >= 0 or self.state in (
            ParkingState.SEARCH_CARS,
            ParkingState.TRACK_GAP,
            ParkingState.EXIT_RIGHT,
            ParkingState.EXIT_STRAIGHT,
        ):
            if any(
                self._ultrasonic_emergency(value)
                for value in (
                    front_left_mm,
                    front_center_mm,
                    front_right_mm,
                )
            ):
                return "front_ultrasonic_emergency"
        if any(
            self._ultrasonic_emergency(value)
            for value in (left_mm, right_mm)
        ):
            return "side_ultrasonic_emergency"
        if self._active_gear < 0 and lidar.valid and lidar.unsafe:
            return "rear_lidar_emergency"
        return None

    def _ultrasonic_correction(
        self,
        left_mm: Optional[float],
        right_mm: Optional[float],
    ) -> int:
        if not self._usable_ultrasonic(left_mm) or not self._usable_ultrasonic(
            right_mm
        ):
            return 0
        correction = self.config.ultrasonic_kp_steering_per_mm * (
            float(right_mm) - float(left_mm)
        )
        limit = abs(self.config.ultrasonic_max_correction)
        return int(round(max(-limit, min(limit, correction))))

    def _ultrasonic_emergency(self, value_mm: Optional[float]) -> bool:
        return (
            self._usable_ultrasonic(value_mm)
            and float(value_mm) <= self.config.ultrasonic_emergency_mm
        )

    def _usable_ultrasonic(self, value_mm: Optional[float]) -> bool:
        return (
            value_mm is not None
            and 0.0 < float(value_mm) <= self.config.ultrasonic_max_valid_mm
        )

    def _straight_steering(self) -> int:
        limit = abs(self.config.max_steering_command)
        return int(max(-limit, min(limit, self.config.straight_steering_trim)))

    def _enter(self, state: ParkingState, now: float) -> None:
        self.state = state
        self._state_started_at = now
        if state not in (
            ParkingState.FOLLOW_ENTRY_CURVE,
            ParkingState.FOLLOW_SLOT_CENTER,
        ):
            self._complete_scans = 0

    def _state_elapsed(self, now: float) -> float:
        return max(0.0, now - self._state_started_at)

    def _expired(self, now: float, watchdog_s: float) -> bool:
        return watchdog_s > 0.0 and self._state_elapsed(now) >= watchdog_s

    def _reset_counters(self) -> None:
        self._slot_confirm_scans = 0
        self._path_failures = 0
        self._complete_scans = 0
        self._active_gear = 0
        self._pending_gear = 0
        self._gear_stop_frames = 0
        self._last_path = HybridParkingPath()
        self._clear_slot_path()

    def _clear_slot_path(self) -> None:
        self._slot_path = ()
        self._slot_path_progress = 0
        self._slot_path_kind = ""
        self._slot_path_cost = 0.0
        self._slot_path_expansions = 0

    def _abort(self, now: float, reason: str) -> ParkingPlan:
        self._enter(ParkingState.ABORTED, now)
        self._active_gear = 0
        return self._stop(reason)

    def _stop(
        self,
        reason: str,
        *,
        world_path: Optional[HybridParkingPath] = None,
    ) -> ParkingPlan:
        return ParkingPlan(
            state=self.state,
            command=ControlCommand.stop(reason),
            reason=reason,
            world_path=world_path,
        )

    def _drive(
        self,
        speed: int,
        steering: int,
        reason: str,
        *,
        world_path: Optional[HybridParkingPath] = None,
    ) -> ParkingPlan:
        return ParkingPlan(
            state=self.state,
            command=ControlCommand(
                speed=int(speed),
                steering=int(steering),
                brake=False,
                reason=reason,
            ),
            reason=reason,
            world_path=world_path,
        )
