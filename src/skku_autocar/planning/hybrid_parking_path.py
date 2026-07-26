from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin, tan
from typing import Dict, Iterable, Optional, Sequence, Tuple


Point = Tuple[float, float]
Polygon = Tuple[Point, ...]


@dataclass(frozen=True)
class VehicleModel:
    """Kinematic bicycle and body dimensions in millimetres."""

    wheelbase_mm: float = 620.0
    width_mm: float = 600.0
    length_mm: float = 1000.0
    rear_axle_to_rear_bumper_mm: float = 200.0
    max_steering_angle_deg: float = 30.0
    collision_clearance_mm: float = 25.0

    @property
    def front_overhang_from_rear_axle_mm(self) -> float:
        return max(
            1.0,
            self.length_mm - self.rear_axle_to_rear_bumper_mm,
        )

    @property
    def maximum_curvature_per_mm(self) -> float:
        return tan(radians(abs(self.max_steering_angle_deg))) / max(
            1.0,
            self.wheelbase_mm,
        )


@dataclass(frozen=True)
class HybridPathConfig:
    """Search resolution and cost policy for a short parking manoeuvre."""

    primitive_length_mm: float = 160.0
    integration_step_mm: float = 40.0
    xy_resolution_mm: float = 90.0
    heading_resolution_deg: float = 10.0
    steering_samples: int = 3
    goal_position_tolerance_mm: float = 90.0
    goal_heading_tolerance_deg: float = 6.0
    maximum_expansions: int = 22000
    search_margin_mm: float = 1000.0
    gear_change_cost_mm: float = 650.0
    steering_change_cost_mm: float = 35.0
    steering_magnitude_cost: float = 0.05
    forward_cost_multiplier: float = 1.02
    reverse_cost_multiplier: float = 1.0
    heading_heuristic_weight: float = 0.35


@dataclass(frozen=True)
class PathPose:
    x_right_mm: float
    y_forward_mm: float
    heading_rad: float
    gear: int = 0
    steering_rad: float = 0.0


@dataclass(frozen=True)
class HybridParkingPath:
    found: bool = False
    poses: Tuple[PathPose, ...] = ()
    goal: Optional[PathPose] = None
    cost: float = 0.0
    expansions: int = 0
    reason: str = "not_planned"

    @property
    def first_gear(self) -> int:
        for pose in self.poses:
            if pose.gear:
                return 1 if pose.gear > 0 else -1
        return 0

    def lookahead(self, distance_mm: float, gear: Optional[int] = None) -> Optional[PathPose]:
        if len(self.poses) < 2:
            return self.poses[-1] if self.poses else None
        requested_gear = self.first_gear if gear is None else (1 if gear > 0 else -1)
        accumulated = 0.0
        previous = self.poses[0]
        last_matching: Optional[PathPose] = None
        for pose in self.poses[1:]:
            if pose.gear != requested_gear:
                if last_matching is not None:
                    break
                previous = pose
                continue
            accumulated += hypot(
                pose.x_right_mm - previous.x_right_mm,
                pose.y_forward_mm - previous.y_forward_mm,
            )
            last_matching = pose
            if accumulated >= max(1.0, distance_mm):
                return pose
            previous = pose
        return last_matching


@dataclass(frozen=True)
class SlotManeuverModel:
    """Official-size bay, neighbouring vehicles, and relative target poses."""

    slot_polygon: Polygon
    obstacles: Tuple[Polygon, ...]
    parking_goal: PathPose
    exit_goal: PathPose
    entrance_center: Point
    back_center: Point
    depth_direction: Point
    slot_axis: Point
    lane_direction: Point


@dataclass
class _SearchNode:
    pose: PathPose
    cost: float
    parent: Optional[int]
    segment: Tuple[PathPose, ...]


class HybridAStarParkingPathPlanner:
    """Collision-aware Hybrid A* in the live rear-axle frame.

    A new search can start at ``(0, 0, 0)`` after every LiDAR update because
    the locked slot is also expressed in that same current vehicle frame. This
    makes the controller independent of elapsed motor time and wheel odometry.
    """

    def __init__(
        self,
        vehicle: VehicleModel = VehicleModel(),
        config: HybridPathConfig = HybridPathConfig(),
    ) -> None:
        self.vehicle = vehicle
        self.config = config

    def plan(
        self,
        goal: PathPose,
        obstacles: Sequence[Polygon],
        *,
        initial_gear: int = 0,
        allowed_gears: Sequence[int] = (-1, 1),
    ) -> HybridParkingPath:
        start = PathPose(0.0, 0.0, 0.0, gear=0, steering_rad=0.0)
        if self._goal_reached(start, goal):
            return HybridParkingPath(
                found=True,
                poses=(start,),
                goal=goal,
                reason="already_at_goal",
            )
        if self._collides(start, obstacles):
            return HybridParkingPath(
                goal=goal,
                reason="vehicle_pose_collides_with_inferred_obstacle",
            )

        gears = tuple(
            dict.fromkeys(1 if gear > 0 else -1 for gear in allowed_gears if gear)
        )
        if not gears:
            return HybridParkingPath(goal=goal, reason="no_allowed_gear")
        preferred_gear = 1 if initial_gear > 0 else -1 if initial_gear < 0 else 0
        search_start = PathPose(
            0.0,
            0.0,
            0.0,
            gear=preferred_gear,
            steering_rad=0.0,
        )
        ordered_gears = tuple(
            sorted(gears, key=lambda gear: 0 if gear == preferred_gear else 1)
        )
        steering_values = self._steering_values()
        bounds = self._search_bounds(goal, obstacles)

        nodes = [_SearchNode(search_start, 0.0, None, ())]
        queue: list[Tuple[float, int, int]] = []
        counter = 0
        heapq.heappush(queue, (self._heuristic(search_start, goal), counter, 0))
        best_cost: Dict[Tuple[int, int, int, int], float] = {
            self._state_key(search_start): 0.0
        }
        expansions = 0

        while queue and expansions < max(1, self.config.maximum_expansions):
            _, _, node_index = heapq.heappop(queue)
            node = nodes[node_index]
            node_key = self._state_key(node.pose)
            if node.cost > best_cost.get(node_key, float("inf")) + 1e-6:
                continue
            if self._goal_reached(node.pose, goal):
                poses = self._reconstruct(nodes, node_index)
                return HybridParkingPath(
                    found=True,
                    poses=poses,
                    goal=goal,
                    cost=node.cost,
                    expansions=expansions,
                    reason="hybrid_astar_ready",
                )

            expansions += 1
            controls = self._ordered_controls(
                ordered_gears,
                steering_values,
                node.pose,
                goal,
            )
            for gear, steering in controls:
                segment = self._propagate(node.pose, gear, steering)
                if not segment:
                    continue
                successor = segment[-1]
                if not self._inside_bounds(successor, bounds):
                    continue
                if any(self._collides(pose, obstacles) for pose in segment):
                    continue

                transition_cost = self._transition_cost(
                    node.pose,
                    gear,
                    steering,
                )
                successor_cost = node.cost + transition_cost
                successor_key = self._state_key(successor)
                if successor_cost >= best_cost.get(successor_key, float("inf")):
                    continue

                best_cost[successor_key] = successor_cost
                nodes.append(
                    _SearchNode(
                        pose=successor,
                        cost=successor_cost,
                        parent=node_index,
                        segment=segment,
                    )
                )
                successor_index = len(nodes) - 1
                counter += 1
                priority = successor_cost + self._heuristic(successor, goal)
                heapq.heappush(queue, (priority, counter, successor_index))

        reason = (
            "hybrid_astar_expansion_limit"
            if expansions >= max(1, self.config.maximum_expansions)
            else "hybrid_astar_no_collision_free_path"
        )
        return HybridParkingPath(
            goal=goal,
            expansions=expansions,
            reason=reason,
        )

    def _steering_values(self) -> Tuple[float, ...]:
        maximum = radians(abs(self.vehicle.max_steering_angle_deg))
        samples = max(3, int(self.config.steering_samples))
        if samples % 2 == 0:
            samples += 1
        return tuple(
            -maximum + 2.0 * maximum * index / float(samples - 1)
            for index in range(samples)
        )

    def _ordered_controls(
        self,
        gears: Sequence[int],
        steering_values: Sequence[float],
        pose: PathPose,
        goal: PathPose,
    ) -> Tuple[Tuple[int, float], ...]:
        bearing = atan2(
            goal.x_right_mm - pose.x_right_mm,
            goal.y_forward_mm - pose.y_forward_mm,
        )
        heading_error = wrap_angle(bearing - pose.heading_rad)
        desired_sign = 1 if heading_error >= 0.0 else -1
        return tuple(
            sorted(
                (
                    (gear, steering)
                    for gear in gears
                    for steering in steering_values
                ),
                key=lambda control: (
                    0 if control[0] == pose.gear and pose.gear else 1,
                    0
                    if control[1] == 0.0
                    else 1
                    if (control[1] > 0.0) == (desired_sign > 0)
                    else 2,
                    abs(control[1] - pose.steering_rad),
                ),
            )
        )

    def _propagate(
        self,
        start: PathPose,
        gear: int,
        steering: float,
    ) -> Tuple[PathPose, ...]:
        primitive = max(1.0, abs(self.config.primitive_length_mm))
        integration = max(1.0, abs(self.config.integration_step_mm))
        steps = max(1, int(round(primitive / integration)))
        distance = gear * primitive / float(steps)
        curvature = tan(steering) / max(1.0, self.vehicle.wheelbase_mm)
        x = start.x_right_mm
        y = start.y_forward_mm
        heading = start.heading_rad
        poses = []
        for _ in range(steps):
            heading_mid = heading + 0.5 * curvature * distance
            x += sin(heading_mid) * distance
            y += cos(heading_mid) * distance
            heading = wrap_angle(heading + curvature * distance)
            poses.append(
                PathPose(
                    x_right_mm=x,
                    y_forward_mm=y,
                    heading_rad=heading,
                    gear=gear,
                    steering_rad=steering,
                )
            )
        return tuple(poses)

    def _transition_cost(
        self,
        previous: PathPose,
        gear: int,
        steering: float,
    ) -> float:
        primitive = max(1.0, abs(self.config.primitive_length_mm))
        multiplier = (
            self.config.forward_cost_multiplier
            if gear > 0
            else self.config.reverse_cost_multiplier
        )
        cost = primitive * max(0.01, multiplier)
        if previous.gear and previous.gear != gear:
            cost += max(0.0, self.config.gear_change_cost_mm)
        maximum = max(1e-6, radians(abs(self.vehicle.max_steering_angle_deg)))
        cost += (
            max(0.0, self.config.steering_change_cost_mm)
            * abs(steering - previous.steering_rad)
            / maximum
        )
        cost += primitive * max(0.0, self.config.steering_magnitude_cost) * (
            abs(steering) / maximum
        )
        return cost

    def _heuristic(self, pose: PathPose, goal: PathPose) -> float:
        distance = hypot(
            goal.x_right_mm - pose.x_right_mm,
            goal.y_forward_mm - pose.y_forward_mm,
        )
        heading_error = abs(wrap_angle(goal.heading_rad - pose.heading_rad))
        minimum_radius = 1.0 / max(
            1e-9,
            self.vehicle.maximum_curvature_per_mm,
        )
        return distance + (
            max(0.0, self.config.heading_heuristic_weight)
            * minimum_radius
            * heading_error
        )

    def _goal_reached(self, pose: PathPose, goal: PathPose) -> bool:
        return (
            hypot(
                goal.x_right_mm - pose.x_right_mm,
                goal.y_forward_mm - pose.y_forward_mm,
            )
            <= max(1.0, self.config.goal_position_tolerance_mm)
            and abs(wrap_angle(goal.heading_rad - pose.heading_rad))
            <= radians(max(0.1, self.config.goal_heading_tolerance_deg))
        )

    def _state_key(self, pose: PathPose) -> Tuple[int, int, int, int]:
        heading_bins = max(
            4,
            int(round(360.0 / max(1.0, self.config.heading_resolution_deg))),
        )
        heading_index = int(
            round(
                (wrap_angle(pose.heading_rad) + pi)
                / (2.0 * pi)
                * heading_bins
            )
        ) % heading_bins
        return (
            int(round(pose.x_right_mm / max(1.0, self.config.xy_resolution_mm))),
            int(round(pose.y_forward_mm / max(1.0, self.config.xy_resolution_mm))),
            heading_index,
            1 if pose.gear > 0 else -1 if pose.gear < 0 else 0,
        )

    def _search_bounds(
        self,
        goal: PathPose,
        obstacles: Sequence[Polygon],
    ) -> Tuple[float, float, float, float]:
        points = [(0.0, 0.0), (goal.x_right_mm, goal.y_forward_mm)]
        points.extend(point for polygon in obstacles for point in polygon)
        margin = max(100.0, self.config.search_margin_mm)
        return (
            min(point[0] for point in points) - margin,
            max(point[0] for point in points) + margin,
            min(point[1] for point in points) - margin,
            max(point[1] for point in points) + margin,
        )

    @staticmethod
    def _inside_bounds(
        pose: PathPose,
        bounds: Tuple[float, float, float, float],
    ) -> bool:
        return (
            bounds[0] <= pose.x_right_mm <= bounds[1]
            and bounds[2] <= pose.y_forward_mm <= bounds[3]
        )

    def _collides(
        self,
        pose: PathPose,
        obstacles: Sequence[Polygon],
    ) -> bool:
        footprint = vehicle_footprint(pose, self.vehicle)
        return any(polygons_intersect(footprint, obstacle) for obstacle in obstacles)

    @staticmethod
    def _reconstruct(
        nodes: Sequence[_SearchNode],
        node_index: int,
    ) -> Tuple[PathPose, ...]:
        segments = []
        current: Optional[int] = node_index
        while current is not None:
            node = nodes[current]
            segments.append(node.segment)
            current = node.parent
        poses = [PathPose(0.0, 0.0, 0.0)]
        for segment in reversed(segments):
            poses.extend(segment)
        return tuple(poses)


def build_slot_maneuver_model(
    lidar_slot_polygon: Sequence[Point],
    *,
    sensor_to_rear_axle_y_back_mm: float,
    vehicle: VehicleModel,
    back_clearance_mm: float = 120.0,
    inferred_neighbor_width_mm: float = 600.0,
    back_wall_depth_mm: float = 500.0,
    exit_lane_offset_mm: float = 700.0,
    exit_lane_advance_mm: float = 1200.0,
) -> Optional[SlotManeuverModel]:
    """Convert the locked LiDAR bay rectangle into the rear-axle planning frame.

    LiDAR uses ``x=right, y=back``. Planning uses ``x=right, y=forward``.
    The polygon ordering is entrance-first, entrance-second, far-second,
    far-first, matching ``infer_dynamic_slot_polygon``.
    """

    if len(lidar_slot_polygon) != 4:
        return None
    converted = tuple(
        (
            float(point[0]),
            float(sensor_to_rear_axle_y_back_mm) - float(point[1]),
        )
        for point in lidar_slot_polygon
    )
    entrance_first, entrance_second, far_second, far_first = converted
    entrance_center = midpoint(entrance_first, entrance_second)
    back_center = midpoint(far_first, far_second)
    axis = normalize(subtract(entrance_second, entrance_first))
    depth = normalize(subtract(back_center, entrance_center))
    if axis is None or depth is None:
        return None
    # The detected bordering-car order can swap after a re-anchor. Use a
    # canonical side axis derived from the sign-locked bay depth for frozen
    # path coordinates, while keeping the detected axis for obstacle roles.
    canonical_slot_axis = (depth[1], -depth[0])

    rear_target_offset = (
        max(0.0, vehicle.rear_axle_to_rear_bumper_mm)
        + max(0.0, back_clearance_mm)
    )
    parking_goal_xy = (
        back_center[0] - depth[0] * rear_target_offset,
        back_center[1] - depth[1] * rear_target_offset,
    )
    parked_forward = (-depth[0], -depth[1])
    parking_heading = atan2(parked_forward[0], parked_forward[1])

    first_neighbor_width = max(1.0, inferred_neighbor_width_mm)
    first_outward = (-axis[0], -axis[1])
    second_outward = axis
    first_neighbor = (
        add_scaled(entrance_first, first_outward, first_neighbor_width),
        entrance_first,
        far_first,
        add_scaled(far_first, first_outward, first_neighbor_width),
    )
    second_neighbor = (
        entrance_second,
        add_scaled(entrance_second, second_outward, first_neighbor_width),
        add_scaled(far_second, second_outward, first_neighbor_width),
        far_second,
    )
    wall_depth = max(1.0, back_wall_depth_mm)
    back_wall = (
        far_first,
        far_second,
        add_scaled(far_second, depth, wall_depth),
        add_scaled(far_first, depth, wall_depth),
    )

    positive_axis_heading = atan2(axis[0], axis[1])
    negative_axis_heading = wrap_angle(positive_axis_heading + pi)
    right_turn_positive = wrap_angle(positive_axis_heading - parking_heading)
    right_turn_negative = wrap_angle(negative_axis_heading - parking_heading)
    candidates = (
        (positive_axis_heading, axis, right_turn_positive),
        (negative_axis_heading, (-axis[0], -axis[1]), right_turn_negative),
    )
    positive_turns = [candidate for candidate in candidates if candidate[2] >= 0.0]
    exit_heading, lane_direction, _ = min(
        positive_turns or list(candidates),
        key=lambda candidate: abs(candidate[2]),
    )
    exit_base = add_scaled(
        entrance_center,
        (-depth[0], -depth[1]),
        max(100.0, exit_lane_offset_mm),
    )
    exit_xy = add_scaled(
        exit_base,
        lane_direction,
        max(0.0, exit_lane_advance_mm),
    )

    return SlotManeuverModel(
        slot_polygon=converted,
        obstacles=(first_neighbor, second_neighbor, back_wall),
        parking_goal=PathPose(
            parking_goal_xy[0],
            parking_goal_xy[1],
            parking_heading,
        ),
        exit_goal=PathPose(exit_xy[0], exit_xy[1], exit_heading),
        entrance_center=entrance_center,
        back_center=back_center,
        depth_direction=depth,
        slot_axis=canonical_slot_axis,
        lane_direction=lane_direction,
    )


def vehicle_footprint(pose: PathPose, vehicle: VehicleModel) -> Polygon:
    forward = (sin(pose.heading_rad), cos(pose.heading_rad))
    right = (cos(pose.heading_rad), -sin(pose.heading_rad))
    clearance = max(0.0, vehicle.collision_clearance_mm)
    half_width = vehicle.width_mm / 2.0 + clearance
    rear_distance = vehicle.rear_axle_to_rear_bumper_mm + clearance
    front_distance = vehicle.front_overhang_from_rear_axle_mm + clearance
    rear_center = (
        pose.x_right_mm - forward[0] * rear_distance,
        pose.y_forward_mm - forward[1] * rear_distance,
    )
    front_center = (
        pose.x_right_mm + forward[0] * front_distance,
        pose.y_forward_mm + forward[1] * front_distance,
    )
    return (
        add_scaled(rear_center, right, -half_width),
        add_scaled(rear_center, right, half_width),
        add_scaled(front_center, right, half_width),
        add_scaled(front_center, right, -half_width),
    )


def polygons_intersect(first: Sequence[Point], second: Sequence[Point]) -> bool:
    if len(first) < 3 or len(second) < 3:
        return False
    for polygon in (first, second):
        for start, end in zip(polygon, tuple(polygon[1:]) + (polygon[0],)):
            edge = subtract(end, start)
            axis = normalize((-edge[1], edge[0]))
            if axis is None:
                continue
            first_projection = tuple(dot(point, axis) for point in first)
            second_projection = tuple(dot(point, axis) for point in second)
            if (
                max(first_projection) < min(second_projection) - 1e-6
                or max(second_projection) < min(first_projection) - 1e-6
            ):
                return False
    return True


def pure_pursuit_curvature(target: PathPose) -> float:
    distance_squared = (
        target.x_right_mm * target.x_right_mm
        + target.y_forward_mm * target.y_forward_mm
    )
    if distance_squared <= 1.0:
        return 0.0
    return 2.0 * target.x_right_mm / distance_squared


def steering_command_for_curvature(
    curvature_per_mm: float,
    vehicle: VehicleModel,
    maximum_command: int,
) -> int:
    from math import atan

    maximum_angle = max(0.1, abs(vehicle.max_steering_angle_deg))
    steering_angle = degrees(
        atan(vehicle.wheelbase_mm * float(curvature_per_mm))
    )
    normalized = max(-1.0, min(1.0, steering_angle / maximum_angle))
    return int(round(abs(int(maximum_command)) * normalized))


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def normalize(vector: Point) -> Optional[Point]:
    length = hypot(vector[0], vector[1])
    if length <= 1e-9:
        return None
    return vector[0] / length, vector[1] / length


def midpoint(first: Point, second: Point) -> Point:
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def subtract(first: Point, second: Point) -> Point:
    return first[0] - second[0], first[1] - second[1]


def add_scaled(point: Point, direction: Point, scale: float) -> Point:
    return point[0] + direction[0] * scale, point[1] + direction[1] * scale


def dot(first: Point, second: Point) -> float:
    return first[0] * second[0] + first[1] * second[1]
