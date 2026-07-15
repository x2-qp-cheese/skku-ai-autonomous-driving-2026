from dataclasses import dataclass


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
