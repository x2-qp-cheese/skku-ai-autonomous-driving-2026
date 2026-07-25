from __future__ import annotations

"""Competition-time behavior fixes for the full-speed autonomous-driving runtime.

This module intentionally patches four competition-critical behaviors:

1. Prevent one obstacle mask from falsely blocking both the current lane and the
   destination lane.
2. Use the measured BEV lane width as the adjacent-lane center offset.
3. Use the existing smoothstep trajectory for obstacle-triggered lane changes
   instead of jumping the target by one full lane in a single frame.
4. Keep crosswalk pixels out of lane geometry without activating the stale
   crosswalk-cache trajectory. Traffic-light stopping remains independent.

Call ``install_competition_patch()`` before importing ``skku_autocar.runtime.yolo_drive_app``.
"""

from typing import Sequence

_INSTALLED = False


def _assess_measurements_exclusive(
    self,
    measurements: Sequence[object],
    current_y_threshold: float,
    target_y_threshold: float,
):
    """Classify obstacle instances by the lane they are closest to.

    The original implementation treated any target-lane lookahead candidate as
    ``target_blocked``. A wide mask belonging to the current lane could therefore
    touch both projected paths and veto an otherwise safe lane change.

    This replacement keeps the current-lane test tolerant, but requires clear
    target-lane dominance before declaring the destination blocked.
    """
    from .planning.obstacle_fusion import PathAssessment

    overlap_min = max(
        0.0,
        min(1.0, float(self.config.min_path_overlap_ratio)),
    )
    max_current_distance = max(
        0.0,
        float(self.config.max_current_path_distance_ratio),
    )

    # Pixel-space hysteresis prevents near-equal path distances from flipping
    # assignment every frame. Ambiguous masks remain attributable to the current
    # lane when current overlap is at least as large, but do not block the target.
    assignment_margin_px = 8.0

    def inside_current_path(item: object) -> bool:
        return float(item.current_distance_ratio) <= max_current_distance

    def current_preferred(item: object) -> bool:
        # Ties and small projection errors are assigned to the current path.
        # This preserves obstacle detection without falsely vetoing the escape lane.
        return (
            float(item.current_overlap) >= overlap_min
            and inside_current_path(item)
            and float(item.current_distance_px)
            <= float(item.target_distance_px) + assignment_margin_px
        )

    def target_preferred(item: object) -> bool:
        # The destination is blocked only when the obstacle is unambiguously
        # closer to the destination trajectory by more than the hysteresis margin.
        return (
            float(item.target_overlap) >= overlap_min
            and float(item.target_distance_px) + assignment_margin_px
            < float(item.current_distance_px)
        )

    current = [
        item
        for item in measurements
        if float(item.bottom_y_ratio) >= float(current_y_threshold)
        and float(item.current_overlap) >= overlap_min
        and inside_current_path(item)
        and current_preferred(item)
    ]

    # Important: unlike the original code, a far target lookahead does not
    # automatically block the destination. The obstacle must be sufficiently
    # close and be unambiguously assigned to that lane.
    target = [
        item
        for item in measurements
        if float(item.bottom_y_ratio) >= float(target_y_threshold)
        and float(item.target_overlap) >= overlap_min
        and target_preferred(item)
    ]

    range_fallback = any(
        float(item.bottom_y_ratio) >= float(current_y_threshold)
        and inside_current_path(item)
        and current_preferred(item)
        and not target_preferred(item)
        for item in measurements
    )

    return PathAssessment(
        current_detected=bool(current),
        target_blocked=bool(target),
        range_fallback_candidate=range_fallback,
        closest_y_ratio=max(
            (float(item.bottom_y_ratio) for item in current),
            default=0.0,
        ),
        obstacle_count=len(measurements),
    )


def _use_smooth_lane_change(self, state: str) -> bool:
    """Disable the one-frame full-lane target jump for avoidance maneuvers.

    Returning False makes the existing controller use its smoothstep transition
    for both ordinary and obstacle-triggered lane changes.
    """
    return False


def _measured_lane_change_width(self, lane_width_px: float) -> float:
    """Use configured width when explicit, otherwise use measured BEV width."""
    configured = max(
        0.0,
        float(self._lane_change.config.target_lane_width_px),
    )
    if configured > 0.0:
        return configured
    return max(0.0, float(lane_width_px))


def _resolve_lane_change_target_width_px(args) -> float:
    """Keep obstacle path projection consistent with the measured corridor width."""
    configured = max(
        0.0,
        float(args.lane_change_target_width_px),
    )
    if configured > 0.0:
        return configured
    return max(0.0, float(args.corridor_lane_width_px))



def _ignore_crosswalk_for_lane_geometry(self, bev) -> bool:
    """Do not switch the lane estimator into its crosswalk-cache mode.

    The segmentation model already exposes crosswalk pixels as a separate class,
    so they are not part of the center/side lane fits. The stock crosswalk mode
    can replace a recent curved trajectory with an older cached trajectory after
    a heading/center guard fires. At full speed that creates a large target jump.

    Returning False keeps the ordinary BEV corridor, coast, and virtual-lane
    fallbacks active. The traffic-light controller still receives the original
    crosswalk masks directly from the runtime and can stop the vehicle normally.
    """
    return False

def install_competition_patch() -> None:
    """Install patches once for the current Python process."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .estimation.bev_corridor import BevCorridorLaneEstimator
    from .planning.lane_change import LaneChangeController
    from .planning.obstacle_fusion import ObstacleFusionPlanner
    from .runtime import obstacle_mode as obstacle_mode_module

    BevCorridorLaneEstimator._crosswalk_visible = (
        _ignore_crosswalk_for_lane_geometry
    )
    ObstacleFusionPlanner._assess_measurements = _assess_measurements_exclusive
    LaneChangeController._uses_target_arrival = _use_smooth_lane_change
    obstacle_mode_module.ObstacleDriveMode._lane_change_width_px = (
        _measured_lane_change_width
    )
    obstacle_mode_module.resolve_lane_change_target_width_px = (
        _resolve_lane_change_target_width_px
    )

    _INSTALLED = True
