from typing import Iterable, Optional, Sequence, Tuple


ScanPoint = Tuple[float, float]


def angle_in_window(angle_deg: float, start_deg: float, end_deg: float) -> bool:
    angle = angle_deg % 360.0
    start = start_deg % 360.0
    end = end_deg % 360.0
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def nearest_distance_mm(
    points: Iterable[ScanPoint],
    angle_min: float,
    angle_max: float,
) -> Optional[float]:
    distances = [
        distance
        for angle, distance in points
        if distance > 0 and angle_in_window(angle, angle_min, angle_max)
    ]
    if not distances:
        return None
    return min(distances)


def normalize_rplidar_scan(raw_scan: Sequence[Sequence[float]]) -> Iterable[ScanPoint]:
    for point in raw_scan:
        if len(point) < 3:
            continue
        angle = float(point[1])
        distance = float(point[2])
        yield angle, distance
