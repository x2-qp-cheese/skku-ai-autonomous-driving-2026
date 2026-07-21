from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, hypot
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParkingLine:
    """A straight line fitted to one YOLO instance mask in rear-camera BEV."""

    center_x: float
    center_y: float
    direction_x: float
    direction_y: float
    length_px: float
    residual_px: float
    quality: float
    point_count: int
    mask_index: int = -1

    @property
    def angle_deg(self) -> float:
        angle = degrees(atan2(self.direction_y, self.direction_x)) % 180.0
        return angle

    def point_at_y(self, y: float) -> Optional[Tuple[float, float]]:
        if abs(self.direction_y) < 1e-6:
            return None
        scale = (y - self.center_y) / self.direction_y
        return (
            self.center_x + scale * self.direction_x,
            y,
        )

    def supports(self, point: Tuple[float, float], margin_px: float) -> bool:
        dx = point[0] - self.center_x
        dy = point[1] - self.center_y
        along = abs(dx * self.direction_x + dy * self.direction_y)
        return along <= self.length_px / 2.0 + margin_px


@dataclass(frozen=True)
class ParkingGeometry:
    found: bool = False
    has_side_pair: bool = False
    has_back_line: bool = False
    left: Optional[ParkingLine] = None
    right: Optional[ParkingLine] = None
    back: Optional[ParkingLine] = None
    lateral_error_px: float = 0.0
    lateral_error_norm: float = 0.0
    heading_error_deg: float = 0.0
    depth_to_back_px: Optional[float] = None
    depth_remaining_px: Optional[float] = None
    slot_width_px: Optional[float] = None
    vehicle_x_px: float = 0.0
    vehicle_y_px: float = 0.0
    slot_center_x_px: Optional[float] = None
    slot_center_y_px: Optional[float] = None
    slot_direction_x: float = 0.0
    slot_direction_y: float = -1.0
    back_center_x_px: Optional[float] = None
    back_center_y_px: Optional[float] = None
    stop_target_x_px: Optional[float] = None
    stop_target_y_px: Optional[float] = None
    confidence: float = 0.0
    observed_line_count: int = 0
    coasted: bool = False
    reason: str = "not_found"


@dataclass(frozen=True)
class ParkingGeometryConfig:
    min_line_pixels: int = 30
    max_fit_points: int = 5000
    max_line_residual_px: float = 10.0
    parallel_tolerance_deg: float = 15.0
    perpendicular_tolerance_deg: float = 18.0
    slot_width_min_px: float = 60.0
    slot_width_max_px: float = 360.0
    expected_slot_width_px: float = 180.0
    intersection_margin_px: float = 45.0
    vehicle_center_x_ratio: float = 0.50
    vehicle_reference_y_ratio: float = 0.95
    control_target_y_ratio: float = 0.70
    desired_back_clearance_px: float = 50.0
    min_geometry_confidence: float = 0.20
    min_confirm_frames: int = 3
    max_coast_frames: int = 3
    smooth_alpha: float = 0.35
    max_lateral_jump_px: float = 90.0
    max_heading_jump_deg: float = 35.0
    max_depth_jump_px: float = 140.0
    synthesize_back_from_side_pair: bool = False
    synthetic_back_depth_to_width_ratio: float = 1500.0 / 950.0
    synthetic_back_quality_scale: float = 0.70


class ParkingGeometryEstimator:
    """Resolve separate ``line`` masks into the topology of a parking bay.

    Coordinates are rear-camera BEV pixels: x grows to vehicle-right, y grows
    toward the vehicle.  The desired reverse direction is therefore negative y.
    Angles are compared only after the ground-plane BEV transform.
    """

    def __init__(self, config: ParkingGeometryConfig = ParkingGeometryConfig()):
        self.config = config
        self._candidate: Optional[ParkingGeometry] = None
        self._confirm_frames = 0
        self._last: Optional[ParkingGeometry] = None
        self._coast_frames = 0

    def reset(self) -> None:
        self._candidate = None
        self._confirm_frames = 0
        self._last = None
        self._coast_frames = 0

    def estimate(
        self,
        masks: Sequence[Any],
        confidence: float = 1.0,
    ) -> ParkingGeometry:
        raw = self._estimate_raw(masks, confidence)
        if not raw.found:
            self._candidate = None
            self._confirm_frames = 0
            if self._last is not None and self._coast_frames < self.config.max_coast_frames:
                self._coast_frames += 1
                decay = max(0.0, 1.0 - self._coast_frames / float(self.config.max_coast_frames + 1))
                return replace(
                    self._last,
                    confidence=self._last.confidence * decay,
                    coasted=True,
                    reason="coast:%s" % raw.reason,
                )
            self._last = None
            self._coast_frames = 0
            return raw

        if self._candidate is None or self._is_jump(self._candidate, raw):
            self._candidate = raw
            self._confirm_frames = 1
        else:
            self._candidate = raw
            self._confirm_frames += 1

        if self._confirm_frames < max(1, self.config.min_confirm_frames):
            return replace(
                raw,
                found=False,
                confidence=raw.confidence * self._confirm_frames / max(1, self.config.min_confirm_frames),
                reason="confirming:%d/%d" % (
                    self._confirm_frames,
                    max(1, self.config.min_confirm_frames),
                ),
            )

        result = self._smooth(self._last, raw)
        self._last = result
        self._coast_frames = 0
        return result

    def _estimate_raw(self, masks: Sequence[Any], detection_confidence: float) -> ParkingGeometry:
        lines = [
            line
            for line in (
                self._fit_line(mask, mask_index)
                for mask_index, mask in enumerate(masks)
            )
            if line is not None
        ]
        line_count = len(lines)
        if line_count < 2:
            return ParkingGeometry(observed_line_count=line_count, reason="need_two_lines")

        pair_candidates = []
        for first_index in range(line_count - 1):
            for second_index in range(first_index + 1, line_count):
                first = lines[first_index]
                second = lines[second_index]
                angle_error = axial_angle_difference(first.angle_deg, second.angle_deg)
                if angle_error > self.config.parallel_tolerance_deg:
                    continue
                direction = average_direction(first, second)
                separation = parallel_line_distance(first, second, direction)
                if not (
                    self.config.slot_width_min_px
                    <= separation
                    <= self.config.slot_width_max_px
                ):
                    continue
                parallel_score = 1.0 - angle_error / max(1e-6, self.config.parallel_tolerance_deg)
                width_span = max(
                    1.0,
                    self.config.slot_width_max_px - self.config.slot_width_min_px,
                )
                width_score = max(
                    0.0,
                    1.0 - abs(separation - self.config.expected_slot_width_px) / width_span,
                )
                score = 0.55 * parallel_score + 0.45 * width_score
                pair_candidates.append(
                    (score, first_index, second_index, direction, separation)
                )

        if not pair_candidates:
            return ParkingGeometry(observed_line_count=line_count, reason="no_parallel_side_pair")

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        best_topology = None
        for pair_score, first_index, second_index, direction, separation in pair_candidates:
            first = lines[first_index]
            second = lines[second_index]
            for back_index, back in enumerate(lines):
                if back_index in (first_index, second_index):
                    continue
                perpendicular_error = abs(
                    90.0 - axial_angle_difference(direction_angle(direction), back.angle_deg)
                )
                if perpendicular_error > self.config.perpendicular_tolerance_deg:
                    continue
                first_corner = line_intersection(first, back)
                second_corner = line_intersection(second, back)
                if first_corner is None or second_corner is None:
                    continue
                support_count = sum(
                    (
                        first.supports(first_corner, self.config.intersection_margin_px),
                        second.supports(second_corner, self.config.intersection_margin_px),
                        back.supports(first_corner, self.config.intersection_margin_px),
                        back.supports(second_corner, self.config.intersection_margin_px),
                    )
                )
                if support_count < 3:
                    continue
                support_score = support_count / 4.0
                perpendicular_score = 1.0 - perpendicular_error / max(
                    1e-6, self.config.perpendicular_tolerance_deg
                )
                topology_score = (
                    0.45 * pair_score
                    + 0.35 * perpendicular_score
                    + 0.20 * support_score
                )
                candidate = (
                    topology_score,
                    first,
                    second,
                    back,
                    direction,
                    separation,
                )
                if best_topology is None or candidate[0] > best_topology[0]:
                    best_topology = candidate

        if best_topology is not None:
            topology_score, first, second, back, direction, separation = best_topology
            return self._build_geometry(
                first,
                second,
                direction,
                separation,
                detection_confidence,
                topology_score,
                line_count,
                back=back,
            )

        pair_score, first_index, second_index, direction, separation = pair_candidates[0]
        return self._build_geometry(
            lines[first_index],
            lines[second_index],
            direction,
            separation,
            detection_confidence,
            pair_score * 0.65,
            line_count,
            back=None,
        )

    def _build_geometry(
        self,
        first: ParkingLine,
        second: ParkingLine,
        direction: Tuple[float, float],
        separation: float,
        detection_confidence: float,
        topology_score: float,
        line_count: int,
        back: Optional[ParkingLine],
    ) -> ParkingGeometry:
        import numpy as np

        if direction[1] > 0.0:
            direction = (-direction[0], -direction[1])
        height, width = self._shape
        target_y = height * self.config.control_target_y_ratio
        first_point = point_on_average_direction(first, direction, target_y)
        second_point = point_on_average_direction(second, direction, target_y)
        if first_point is None or second_point is None:
            return ParkingGeometry(observed_line_count=line_count, reason="side_pair_horizontal")

        if first_point[0] <= second_point[0]:
            left, right = first, second
            left_point, right_point = first_point, second_point
        else:
            left, right = second, first
            left_point, right_point = second_point, first_point

        center_x = (left_point[0] + right_point[0]) / 2.0
        vehicle_center_x = width * self.config.vehicle_center_x_ratio
        lateral_error = center_x - vehicle_center_x
        lateral_norm = lateral_error / max(1.0, separation / 2.0)
        heading_error = degrees(atan2(direction[0], -direction[1]))

        center_line = ParkingLine(
            center_x=(left.center_x + right.center_x) / 2.0,
            center_y=(left.center_y + right.center_y) / 2.0,
            direction_x=direction[0],
            direction_y=direction[1],
            length_px=(left.length_px + right.length_px) / 2.0,
            residual_px=(left.residual_px + right.residual_px) / 2.0,
            quality=(left.quality + right.quality) / 2.0,
            point_count=left.point_count + right.point_count,
            mask_index=-1,
        )

        synthetic_back = False
        if back is None and self.config.synthesize_back_from_side_pair:
            depth_px = separation * self.config.synthetic_back_depth_to_width_ratio
            near_x = center_line.center_x - direction[0] * center_line.length_px / 2.0
            near_y = center_line.center_y - direction[1] * center_line.length_px / 2.0
            back_x = near_x + direction[0] * depth_px
            back_y = near_y + direction[1] * depth_px
            back_quality = (
                (left.quality + right.quality)
                / 2.0
                * self.config.synthetic_back_quality_scale
            )
            back = ParkingLine(
                center_x=back_x,
                center_y=back_y,
                direction_x=-direction[1],
                direction_y=direction[0],
                length_px=separation,
                residual_px=(left.residual_px + right.residual_px) / 2.0,
                quality=clip(back_quality, 0.0, 1.0),
                point_count=left.point_count + right.point_count,
                mask_index=-1,
            )
            synthetic_back = True

        depth = None
        remaining = None
        back_center_x = None
        back_center_y = None
        stop_target_x = None
        stop_target_y = None
        if back is not None:
            back_center = line_intersection(center_line, back)
            if back_center is not None:
                vehicle_reference_y = height * self.config.vehicle_reference_y_ratio
                depth = vehicle_reference_y - back_center[1]
                remaining = depth - self.config.desired_back_clearance_px
                back_center_x = back_center[0]
                back_center_y = back_center[1]
                # direction points from the bay mouth toward the back line.
                # Move opposite that vector to keep the rear reference point a
                # configured clearance in front of the painted back line.
                stop_target_x = back_center[0] - direction[0] * self.config.desired_back_clearance_px
                stop_target_y = back_center[1] - direction[1] * self.config.desired_back_clearance_px

        line_quality = float(np.mean([left.quality, right.quality] + ([back.quality] if back else [])))
        confidence = clip(
            float(detection_confidence) * line_quality * topology_score,
            0.0,
            1.0,
        )
        found = confidence >= self.config.min_geometry_confidence
        return ParkingGeometry(
            found=found,
            has_side_pair=True,
            has_back_line=back is not None and depth is not None,
            left=left,
            right=right,
            back=back,
            lateral_error_px=lateral_error,
            lateral_error_norm=clip(lateral_norm, -2.0, 2.0),
            heading_error_deg=heading_error,
            depth_to_back_px=depth,
            depth_remaining_px=remaining,
            slot_width_px=separation,
            vehicle_x_px=vehicle_center_x,
            vehicle_y_px=height * self.config.vehicle_reference_y_ratio,
            slot_center_x_px=center_line.center_x,
            slot_center_y_px=center_line.center_y,
            slot_direction_x=direction[0],
            slot_direction_y=direction[1],
            back_center_x_px=back_center_x,
            back_center_y_px=back_center_y,
            stop_target_x_px=stop_target_x,
            stop_target_y_px=stop_target_y,
            confidence=confidence,
            observed_line_count=line_count,
            reason=(
                (
                    "parking_bay_synthetic_back"
                    if synthetic_back
                    else ("parking_bay" if back is not None else "side_pair")
                )
                if found
                else "low_confidence"
            ),
        )

    def _fit_line(self, mask: Any, mask_index: int = -1) -> Optional[ParkingLine]:
        import numpy as np

        array = np.asarray(mask)
        self._shape = array.shape[:2]
        ys, xs = np.nonzero(array > 0)
        count = len(xs)
        if count < self.config.min_line_pixels:
            return None
        if count > self.config.max_fit_points:
            step = max(1, count // self.config.max_fit_points)
            xs = xs[::step]
            ys = ys[::step]

        points = np.column_stack((xs.astype(float), ys.astype(float)))
        center = np.mean(points, axis=0)
        centered = points - center
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        if direction[1] > 0.0 or (abs(direction[1]) < 1e-6 and direction[0] < 0.0):
            direction = -direction
        projections = centered @ direction
        perpendicular = centered - np.outer(projections, direction)
        distances = np.linalg.norm(perpendicular, axis=1)
        residual = float(np.sqrt(np.mean(distances ** 2)))
        if residual > self.config.max_line_residual_px:
            return None
        length = float(np.max(projections) - np.min(projections))
        if length < 5.0:
            return None
        quality = clip(1.0 - residual / max(1e-6, self.config.max_line_residual_px), 0.0, 1.0)
        return ParkingLine(
            center_x=float(center[0]),
            center_y=float(center[1]),
            direction_x=float(direction[0]),
            direction_y=float(direction[1]),
            length_px=length,
            residual_px=residual,
            quality=quality,
            point_count=count,
            mask_index=mask_index,
        )

    def _smooth(
        self,
        previous: Optional[ParkingGeometry],
        current: ParkingGeometry,
    ) -> ParkingGeometry:
        if previous is None:
            return current
        alpha = clip(self.config.smooth_alpha, 0.0, 1.0)

        def blend(old: float, new: float) -> float:
            return alpha * new + (1.0 - alpha) * old

        def blend_optional(old: Optional[float], new: Optional[float]) -> Optional[float]:
            if old is None or new is None:
                return new
            return blend(old, new)

        depth = current.depth_to_back_px
        remaining = current.depth_remaining_px
        if previous.depth_to_back_px is not None and depth is not None:
            depth = blend(previous.depth_to_back_px, depth)
        if previous.depth_remaining_px is not None and remaining is not None:
            remaining = blend(previous.depth_remaining_px, remaining)
        direction_x = blend(previous.slot_direction_x, current.slot_direction_x)
        direction_y = blend(previous.slot_direction_y, current.slot_direction_y)
        direction_length = hypot(direction_x, direction_y)
        if direction_length > 1e-9:
            direction_x /= direction_length
            direction_y /= direction_length
        return replace(
            current,
            lateral_error_px=blend(previous.lateral_error_px, current.lateral_error_px),
            lateral_error_norm=blend(previous.lateral_error_norm, current.lateral_error_norm),
            heading_error_deg=blend(previous.heading_error_deg, current.heading_error_deg),
            depth_to_back_px=depth,
            depth_remaining_px=remaining,
            slot_center_x_px=blend_optional(previous.slot_center_x_px, current.slot_center_x_px),
            slot_center_y_px=blend_optional(previous.slot_center_y_px, current.slot_center_y_px),
            slot_direction_x=direction_x,
            slot_direction_y=direction_y,
            back_center_x_px=blend_optional(previous.back_center_x_px, current.back_center_x_px),
            back_center_y_px=blend_optional(previous.back_center_y_px, current.back_center_y_px),
            stop_target_x_px=blend_optional(previous.stop_target_x_px, current.stop_target_x_px),
            stop_target_y_px=blend_optional(previous.stop_target_y_px, current.stop_target_y_px),
            confidence=blend(previous.confidence, current.confidence),
            coasted=False,
        )

    def _is_jump(self, previous: ParkingGeometry, current: ParkingGeometry) -> bool:
        if abs(current.lateral_error_px - previous.lateral_error_px) > self.config.max_lateral_jump_px:
            return True
        if (
            axial_signed_difference(current.heading_error_deg, previous.heading_error_deg)
            > self.config.max_heading_jump_deg
        ):
            return True
        if previous.depth_to_back_px is not None and current.depth_to_back_px is not None:
            if abs(current.depth_to_back_px - previous.depth_to_back_px) > self.config.max_depth_jump_px:
                return True
        return False


def axial_angle_difference(first_deg: float, second_deg: float) -> float:
    difference = abs((first_deg - second_deg) % 180.0)
    return min(difference, 180.0 - difference)


def axial_signed_difference(first_deg: float, second_deg: float) -> float:
    return abs(((first_deg - second_deg + 90.0) % 180.0) - 90.0)


def direction_angle(direction: Tuple[float, float]) -> float:
    return degrees(atan2(direction[1], direction[0])) % 180.0


def average_direction(first: ParkingLine, second: ParkingLine) -> Tuple[float, float]:
    first_vector = (first.direction_x, first.direction_y)
    second_vector = (second.direction_x, second.direction_y)
    if first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1] < 0.0:
        second_vector = (-second_vector[0], -second_vector[1])
    dx = first_vector[0] + second_vector[0]
    dy = first_vector[1] + second_vector[1]
    length = hypot(dx, dy)
    if length < 1e-9:
        return first_vector
    return (dx / length, dy / length)


def parallel_line_distance(
    first: ParkingLine,
    second: ParkingLine,
    direction: Tuple[float, float],
) -> float:
    normal = (-direction[1], direction[0])
    delta = (second.center_x - first.center_x, second.center_y - first.center_y)
    return abs(delta[0] * normal[0] + delta[1] * normal[1])


def point_on_average_direction(
    line: ParkingLine,
    direction: Tuple[float, float],
    target_y: float,
) -> Optional[Tuple[float, float]]:
    if abs(direction[1]) < 1e-6:
        return None
    scale = (target_y - line.center_y) / direction[1]
    return (
        line.center_x + scale * direction[0],
        target_y,
    )


def line_intersection(
    first: ParkingLine,
    second: ParkingLine,
) -> Optional[Tuple[float, float]]:
    p = (first.center_x, first.center_y)
    r = (first.direction_x, first.direction_y)
    q = (second.center_x, second.center_y)
    s = (second.direction_x, second.direction_y)
    denominator = cross(r, s)
    if abs(denominator) < 1e-8:
        return None
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    scale = cross(q_minus_p, s) / denominator
    return (p[0] + scale * r[0], p[1] + scale * r[1])


def cross(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
