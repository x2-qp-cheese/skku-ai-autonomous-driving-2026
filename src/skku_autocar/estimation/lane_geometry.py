from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LaneGeometry:
    """Lane target geometry produced by the BEV corridor estimator."""

    found: bool
    center_x: float
    vehicle_center_x: float
    target_y: float
    lateral_error_px: float
    lateral_error_norm: float
    heading_error: float
    confidence: float
    reason: str = ""
    # Height of the BEV space. Pure pursuit uses height-target_y as its forward
    # distance. Zero keeps compatibility with synthetic planner tests.
    height: float = 0.0
    # A second lateral measurement close to the vehicle. The normal target is a
    # forward lookahead point and can cross the next-lane center while the rear of
    # a diagonally moving car is still beside the obstacle. Obstacle lane changes
    # use this near-field value before releasing maximum steering.
    near_center_x: Optional[float] = None
    near_target_y: Optional[float] = None
    near_lateral_error_px: Optional[float] = None
    near_lateral_error_norm: Optional[float] = None
    # Full, temporally filtered BEV driving path. Controllers should prefer this
    # over steering at one lookahead dot; the dot fields remain for diagnostics,
    # compatibility and lane-change completion checks.
    path_points: Tuple[Tuple[float, float], ...] = ()
