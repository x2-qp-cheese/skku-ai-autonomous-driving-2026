"""Plan A: build the driving corridor in bird's-eye view instead of frame space.

Old pipeline (yolo_lane._select_group):
    YOLO mask -> fit + constant-pixel offset IN FRAME SPACE -> corridor raster
    -> warp -> re-fit in BEV
The constant pixel offset is only correct on straight road; in a curve the true
lane width in the frame shrinks with perspective, so the virtual boundary and the
corridor centerline drift outward -> the car rides the outer (side) line (P2).

New pipeline (this module):
    YOLO per-class masks -> warp EACH to BEV first -> fit center / side lines in BEV
    -> offset by ONE physical constant (lane_width_px) -> single centerline fit
In BEV the two lane lines are (nearly) parallel, so the gap between them is
constant in y. A virtual boundary is therefore always a parallel curve of the
real one, and a single lane_width_px replaces the four frame-space corridor knobs
(fallback / min / max / smooth) it used to take to approximate that gap.

The 3-tier fallback is preserved (center+right / center+virtual / side+virtual),
plus a lane-class fallback for models that only emit a drivable-area class.

Convention (shared with bev_lane.py):
  - x grows to the right, forward is UP (bottom row y=H-1 is closest to the car).
  - The car drives on the RIGHT of the center line, so the driven lane is bounded
    on the left by the center line and on the right by the outer side line, and
    the target centerline is the midpoint of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .lane_geometry import LaneGeometry


@dataclass(frozen=True)
class BevCorridorConfig:
    # Physical lane width in BEV pixels: the gap between the center line and the
    # outer side line. Because BEV rectifies perspective this is a constant in y,
    # so it is measured ONCE on a clip (bev_replay --corridor prints the live
    # measured value on the overlay whenever both lines are seen) and pasted here.
    lane_width_px: float = 150.0
    lane_width_smooth_alpha: float = 0.15  # EMA on the measured width (tier 1)
    min_lane_width_px: float = 60.0
    max_lane_width_px: float = 320.0

    # Row (ratio of BEV height) where lateral error is measured / width sampled.
    lookahead_y_ratio: float = 0.45
    # Near-field row used only to verify that the vehicle body has completed an
    # obstacle lane change. This must stay below/closer than the lookahead row.
    lane_change_near_y_ratio: float = 0.88
    sample_top_y_ratio: float = 0.02
    sample_bottom_y_ratio: float = 0.98
    num_samples: int = 24
    band_height_ratio: float = 0.02
    # Quadratic. An S-curve (inflection) never appears whole in this low camera's
    # BEV window -- only single left/right curves do -- so degree 2 is enough and
    # is more rigid/stable on sparse dashed lines than a cubic. (Raise with
    # --poly-degree only if a real S ever fits in one BEV frame.)
    poly_degree: int = 2
    min_line_area_ratio: float = 0.0003

    # The camera is mounted slightly left of the car's true centerline, so the
    # true vehicle center sits a bit RIGHT of the frame center. Measured off the
    # last good run (blue vehicle-axis marker at frame x-ratio ~0.585), so the
    # lateral-error reference is width*(0.5 + 0.085). Positive = shift the vehicle
    # center right (a centered road then reads as slightly left -> steers left).
    # Tune with --vehicle-center-offset.
    vehicle_center_x_offset_ratio: float = 0.085
    # Where the driving centerline sits between the two lane boundaries:
    # 0.0 = on the center line (innermost), 0.5 = midpoint (geometric center),
    # 1.0 = on the outer side line. Raise above 0.5 if the car rides too far inside
    # (toward the infield); lower it to hug the center line.
    centerline_bias: float = 0.5
    heading_gain: float = 1.6
    center_smooth_alpha: float = 0.4
    heading_smooth_alpha: float = 0.4

    # The outer side line must sit at least this far right of the center line at
    # the lookahead row to be accepted as the right boundary.
    side_min_gap_px: float = 20.0

    # Anchor the driving centerline on the (usually most reliable) center line,
    # offset by half the SMOOTHED lane width, instead of the raw midpoint of the
    # center and side lines. This keeps the centerline from inheriting the side
    # line's frame-to-frame jitter and makes the tier-1 -> tier-2 transition
    # seamless (both become center + W/2), removing a common source of the target
    # point "jumping". False = raw midpoint (tracks the true width better but
    # jitters more).
    center_anchor: bool = True

    # Crosswalk handling. YOLO's crosswalk pixels are their own class, so they are
    # already kept out of the center/side line fits; a lane line partly covered by
    # a crosswalk is still followed, and if the center line is fully covered the
    # visible side line drives the corridor via tier 3. We therefore do NOT stop
    # for a crosswalk by default -- dropping good, clearly-visible lanes just
    # because a crosswalk is in view is wrong. crosswalk_halt is opt-in for the
    # traffic-light mission that must hold before the line; crosswalk_min_area_ratio
    # is the BEV-area fraction at which the crosswalk counts as "present" (the
    # last_crosswalk_visible flag, and the optional halt).
    crosswalk_halt: bool = False
    crosswalk_min_area_ratio: float = 0.02
    # While a crosswalk is in view, the zebra stripes make the measured lane width
    # unreliable, so build the corridor's virtual centerline from a FIXED lane
    # width (BEV px between the center line and the outer side line) instead of the
    # live-measured one.
    crosswalk_lane_width_px: float = 155.0
    # Keep following the real lanes through the crosswalk but
    # damp the zebra/pedestrian jitter that swings the target -- smooth the center
    # harder (lower alpha = laggier/steadier) and reject smaller center-x jumps
    # (coast on the last good geometry) ONLY while a crosswalk is in view. Normal
    # driving keeps its responsive values.
    crosswalk_center_smooth_alpha: float = 0.15
    crosswalk_max_center_jump_px: float = 30.0
    # Crosswalk option A follows the center line with a virtual right boundary.
    # Option B follows a detected right boundary at a fixed inward BEV offset.
    crosswalk_option: str = "a"
    crosswalk_right_offset_px: float = 90.0
    # A real lane line is thin at every BEV row; rows spanning wider than this
    # (ratio of BEV width) are dropped from a single line's fit, so a stray wide
    # blob (e.g. a mislabeled crosswalk stripe leaking into center/side) can't
    # skew it even when it is not on the crosswalk class.
    max_line_span_ratio: float = 0.18
    # Reject a measured lane width that jumps more than this many px from the
    # current smoothed value (keeps lane_width_px stable through noise/crosswalks).
    max_width_jump_px: float = 40.0

    # Temporal gating (Plan C). EMA follows an outlier alpha-much every frame; a
    # low-confidence YOLO fit shows up as a big lookahead center_x or heading
    # jump. Reject those frames and coast on the last good geometry for up to
    # max_coast_frames before declaring the lane lost. Coast confidence decays
    # each held frame.
    max_center_jump_px: float = 80.0
    max_heading_jump: float = 0.22
    max_coast_frames: int = 10
    coast_confidence_decay: float = 0.8

    # Vehicle-width virtual-lane hold (last resort). When NO lane evidence is left
    # (all tiers fail) and coasting on the last good geometry is exhausted, instead
    # of immediately declaring the lane lost (which brakes the car), synthesize a
    # straight virtual lane one vehicle-width wide and keep the car centered in it.
    # The centerline is anchored on the last known lane center and eased back toward
    # the vehicle axis so the car straightens out rather than freezing a mid-curve
    # bias. It is held for at most virtual_hold_max_frames, after which the lane is
    # truly lost (and the follower/safety layer stops the car). The reason/class name
    # contains "virtual" so the drive-time safety guard caps its speed/steering.
    virtual_hold: bool = True
    # BEV pixel width of the car; the virtual lane boundaries are drawn at
    # +/- vehicle_width_px/2 around the held center. Center-holding does not depend
    # on this value (it only shapes the drawn corridor), so a rough estimate is fine.
    vehicle_width_px: float = 120.0
    virtual_hold_max_frames: int = 45
    virtual_hold_confidence: float = 0.3
    # Per-frame easing of the held center toward the vehicle axis (0 = freeze at the
    # last lane center, 1 = snap straight to center). A small value straightens the
    # car out gradually while blind.
    virtual_hold_recenter_alpha: float = 0.05


@dataclass
class BevClassMasks:
    """Per-class YOLO masks already warped into BEV (lists keep side instances
    separate so left/right side lines are not merged before fitting)."""

    center: List[Any] = field(default_factory=list)
    side: List[Any] = field(default_factory=list)
    lane: List[Any] = field(default_factory=list)
    crosswalk: List[Any] = field(default_factory=list)
    obstacle: List[Any] = field(default_factory=list)
    center_conf: float = 1.0
    side_conf: float = 1.0
    lane_conf: float = 1.0
    crosswalk_conf: float = 1.0
    obstacle_conf: float = 0.0
    shape: Tuple[int, int] = (0, 0)  # (height, width) of the BEV canvas


def warp_class_masks(
    transformer: Any,
    class_masks: Any,
    include_obstacle: bool = True,
) -> BevClassMasks:
    """Warp a frame-space YoloClassMasks bundle into BEV, preserving instances."""
    out_w, out_h = transformer.out_size
    return BevClassMasks(
        center=[transformer.warp_mask(m) for m in class_masks.center],
        side=[transformer.warp_mask(m) for m in class_masks.side],
        lane=[transformer.warp_mask(m) for m in class_masks.lane],
        crosswalk=[transformer.warp_mask(m) for m in getattr(class_masks, "crosswalk", ())],
        obstacle=(
            [transformer.warp_mask(m) for m in getattr(class_masks, "obstacle", ())]
            if include_obstacle
            else []
        ),
        center_conf=class_masks.center_conf,
        side_conf=class_masks.side_conf,
        lane_conf=class_masks.lane_conf,
        crosswalk_conf=getattr(class_masks, "crosswalk_conf", 0.0),
        obstacle_conf=(
            getattr(class_masks, "obstacle_conf", 0.0)
            if include_obstacle
            else 0.0
        ),
        shape=(out_h, out_w),
    )


class BevCorridorLaneEstimator:
    def __init__(self, config: BevCorridorConfig = BevCorridorConfig()):
        self.config = config
        self._smoothed_center_x: Optional[float] = None
        self._smoothed_near_center_x: Optional[float] = None
        self._smoothed_heading: Optional[float] = None
        self._lane_width_px: Optional[float] = None

        # Debug overlays (BEV pixel coords), consumed by bev_replay / drive_app.
        self.last_centerline_bev: List[Tuple[float, float]] = []
        self.last_center_line_bev: List[Tuple[float, float]] = []
        self.last_right_line_bev: List[Tuple[float, float]] = []
        self.last_class_name: str = "none"
        self.last_tier: int = 0
        self.last_lane_width_px: float = config.lane_width_px
        self.last_crosswalk_visible: bool = False

        # Temporal gating state (Plan C).
        self._coast_frames: int = 0
        self._last_lane: Optional[LaneGeometry] = None
        self._last_raw_center_x: Optional[float] = None
        self._last_raw_heading: Optional[float] = None
        self._last_overlays: Tuple[list, list, list] = ([], [], [])

        # Vehicle-width virtual-lane hold state.
        self._virtual_hold_frames: int = 0
        self._virtual_center_x: Optional[float] = None

        # True while a crosswalk is in view: build the corridor from the fixed
        # crosswalk_lane_width_px instead of the (zebra-contaminated) measured width.
        self._crosswalk_active: bool = False

    # ------------------------------------------------------------------
    def estimate(self, bev: BevClassMasks) -> LaneGeometry:
        import numpy as np  # noqa: F401  (kept for import-cost parity / clarity)

        self.last_centerline_bev = []
        self.last_center_line_bev = []
        self.last_right_line_bev = []

        height, width = bev.shape
        if height <= 0 or width <= 0:
            return self._lost((0, 0), "no_bev_shape")

        vehicle_center_x = self._vehicle_center_x(width)
        target_y = height * self.config.lookahead_y_ratio
        near_target_y = height * self.config.lane_change_near_y_ratio

        # A crosswalk in view does NOT halt driving: its pixels are a separate
        # class (kept out of the lane fits), so we keep following whatever lane
        # lines are still visible -- e.g. the right side line, which stays clean
        # through a crosswalk and drives the corridor via tier 3. Only the opt-in
        # crosswalk_halt (traffic-light mission) stops here.
        self.last_crosswalk_visible = self._crosswalk_visible(bev)
        if self.config.crosswalk_halt and self.last_crosswalk_visible:
            self.last_class_name = "crosswalk"
            self.last_tier = 0
            self._reset_temporal()
            return self._lost(bev.shape, "crosswalk")

        # Through a crosswalk (not halting), build the virtual centerline from the
        # fixed crosswalk_lane_width_px rather than the zebra-contaminated measured
        # width, so the car stays centered on a stable guessed lane.
        self._crosswalk_active = self.last_crosswalk_visible
        if self._crosswalk_active:
            self.last_lane_width_px = self.config.crosswalk_lane_width_px

        center_fit = self._fit_line(bev.center, bev.shape)
        side_fits = [f for f in (self._fit_line([m], bev.shape) for m in bev.side) if f]

        resolved = self._resolve(center_fit, side_fits, bev, vehicle_center_x, target_y)
        if resolved is None:
            # A momentary YOLO miss: coast rather than dropping the lane outright.
            return self._coast_or_lost(bev.shape, "no_corridor")

        centerline_fit, left_fit, right_fit, tier, class_name, det_conf = resolved
        raw_center_x = float(np.polyval(centerline_fit["fit"], target_y))
        raw_near_center_x = float(np.polyval(centerline_fit["fit"], near_target_y))
        raw_heading = self._heading_error(centerline_fit["fit"], height)

        # Temporal gate: outlier fits show up as a big lookahead center_x or
        # heading jump. Reject the frame and coast on the last good geometry
        # instead of letting the EMA follow it.
        if self._is_center_jump(raw_center_x):
            return self._coast_or_lost(bev.shape, "center_jump")
        if self._is_heading_jump(raw_heading):
            return self._coast_or_lost(bev.shape, "heading_jump")

        self._coast_frames = 0
        self._virtual_hold_frames = 0
        self._virtual_center_x = None
        self._last_raw_center_x = raw_center_x
        self._last_raw_heading = raw_heading
        self.last_class_name = class_name
        self.last_tier = tier
        self.last_centerline_bev = centerline_fit["points"]
        self.last_center_line_bev = self._line_points(left_fit)
        self.last_right_line_bev = self._line_points(right_fit)
        self._last_overlays = (self.last_centerline_bev, self.last_center_line_bev, self.last_right_line_bev)

        center_x = self._clip(self._smooth_center(raw_center_x), 0.0, float(width - 1))
        near_center_x = self._clip(
            self._smooth_near_center(raw_near_center_x),
            0.0,
            float(width - 1),
        )
        heading_error = self._smooth_heading(raw_heading)
        lateral_error_px = center_x - vehicle_center_x
        lateral_error_norm = self._clip(lateral_error_px / (width / 2.0), -1.0, 1.0)
        near_lateral_error_px = near_center_x - vehicle_center_x
        near_lateral_error_norm = self._clip(
            near_lateral_error_px / (width / 2.0),
            -1.0,
            1.0,
        )

        row_coverage = min(1.0, centerline_fit["n"] / float(self.config.num_samples))
        tier_base = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.5}.get(tier, 0.5)
        # Confidence is driven by the YOLO detection confidence of the class the
        # corridor was built from, scaled by tier reliability and row coverage --
        # not by corridor width.
        confidence = self._clip(det_conf * tier_base * (0.5 + 0.5 * row_coverage), 0.0, 1.0)

        lane = LaneGeometry(
            found=True,
            center_x=center_x,
            vehicle_center_x=vehicle_center_x,
            target_y=target_y,
            lateral_error_px=lateral_error_px,
            lateral_error_norm=lateral_error_norm,
            heading_error=heading_error,
            confidence=confidence,
            reason="corridor_tier%d" % tier,
            height=float(height),
            near_center_x=near_center_x,
            near_target_y=near_target_y,
            near_lateral_error_px=near_lateral_error_px,
            near_lateral_error_norm=near_lateral_error_norm,
        )
        self._last_lane = lane
        return lane

    # ------------------------------------------------------------------
    # 3-tier corridor resolution (+ lane-class fallback), all in BEV pixels.
    # returns (centerline_fit, left_boundary_fit, right_boundary_fit, tier,
    #          class_name, detection_confidence) or None.
    # ------------------------------------------------------------------
    def _resolve(
        self,
        center_fit: Optional[dict],
        side_fits: List[dict],
        bev: BevClassMasks,
        vehicle_center_x: float,
        target_y: float,
    ):
        # Crosswalk option B: the right boundary normally remains visible while
        # zebra markings obscure the center line. Drive a fixed distance inward
        # from that real boundary instead of caching the pre-crosswalk curve.
        if self._crosswalk_active and self.config.crosswalk_option.lower() == "b":
            right = self._select_crosswalk_right_side(side_fits, vehicle_center_x, target_y)
            if right is not None:
                offset_px = max(0.0, self.config.crosswalk_right_offset_px)
                centerline = self._offset(right, -offset_px)
                centerline["points"] = self._line_points(centerline)
                left = self._offset(right, -2.0 * offset_px)
                return centerline, left, right, 3, "crosswalk-right-side-b", bev.side_conf

        # Tier 1: center line + a real right-side line.
        if center_fit is not None and side_fits:
            center_x = self._x_at(center_fit, target_y)
            right = self._select_right_side(side_fits, center_x, target_y)
            if right is not None:
                if self._crosswalk_active:
                    # Through a crosswalk: trust the center line but build the
                    # centerline from the fixed crosswalk width, not the measured
                    # (zebra-contaminated) gap. Show the virtual boundary too.
                    width_px = self.config.crosswalk_lane_width_px
                    right_boundary = self._offset(center_fit, width_px)
                    name = "crosswalk-virtual-center"
                else:
                    width_px = self._update_width(self._x_at(right, target_y) - center_x)
                    # centerline = center line offset by half the smoothed width;
                    # the real side line only refines the width (via _update_width)
                    # so its jitter does not reach the centerline. Overlay still
                    # shows the real detected side.
                    right_boundary = self._offset(center_fit, width_px) if self.config.center_anchor else right
                    name = "center+right-side"
                midline = self._midline(center_fit, right_boundary)
                if midline is not None:
                    return midline, center_fit, right_boundary, 1, name, bev.center_conf

        # Tier 2: center line only -> virtual parallel right boundary.
        if center_fit is not None:
            width_px = self._corridor_lane_width()
            right = self._offset(center_fit, width_px)
            midline = self._midline(center_fit, right)
            if midline is not None:
                name = "crosswalk-virtual-center" if self._crosswalk_active else "center+virtual-right-side"
                return midline, center_fit, right, 2, name, bev.center_conf

        # Tier 3: side line only -> place the center line one lane width away.
        if side_fits:
            nearest = min(side_fits, key=lambda f: abs(self._x_at(f, target_y) - vehicle_center_x))
            width_px = self._corridor_lane_width()
            side_x = self._x_at(nearest, target_y)
            if side_x >= vehicle_center_x:
                # treat as the right boundary: center line is to its left.
                left = self._offset(nearest, -width_px)
                right = nearest
                name = "virtual-lane-center+right-side"
            else:
                # treat as the left/center boundary: outer line is to its right.
                left = nearest
                right = self._offset(nearest, width_px)
                name = "left-side+virtual-right-side"
            midline = self._midline(left, right)
            if midline is not None:
                return midline, left, right, 3, name, bev.side_conf

        # Tier 4: only a drivable/lane-area class -> follow its own centerline.
        # A filled region is legitimately wide, so span filtering is disabled.
        lane_fit = self._fit_line(bev.lane, bev.shape, apply_span_filter=False)
        if lane_fit is not None:
            midline = {
                "fit": lane_fit["fit"],
                "min_y": lane_fit["min_y"],
                "max_y": lane_fit["max_y"],
                "n": lane_fit["n"],
                "points": self._line_points(lane_fit),
            }
            return midline, lane_fit, lane_fit, 4, "lane-area", bev.lane_conf

        return None

    def _select_right_side(self, side_fits: List[dict], center_x: float, target_y: float) -> Optional[dict]:
        candidates = []
        for fit in side_fits:
            side_x = self._x_at(fit, target_y)
            gap = side_x - center_x
            if gap < self.config.side_min_gap_px:
                continue
            candidates.append((gap, fit))
        if not candidates:
            return None
        # Nearest line to the right of the center line = the boundary of our lane.
        return min(candidates, key=lambda item: item[0])[1]

    def _select_crosswalk_right_side(
        self,
        side_fits: List[dict],
        vehicle_center_x: float,
        target_y: float,
    ) -> Optional[dict]:
        candidates = [
            fit for fit in side_fits
            if self._x_at(fit, target_y) >= vehicle_center_x
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda fit: self._x_at(fit, target_y))

    # ------------------------------------------------------------------
    # crosswalk detection
    # ------------------------------------------------------------------
    def _crosswalk_visible(self, bev: BevClassMasks) -> bool:
        """True when YOLO's dedicated crosswalk class covers at least
        crosswalk_min_area_ratio of the BEV canvas (proximity-based: a far/small
        crosswalk stays below it). Used only for the last_crosswalk_visible flag
        and the opt-in crosswalk_halt -- it does not, by itself, stop driving."""
        import numpy as np

        if not bev.crosswalk:
            return False
        height, width = bev.shape
        if height <= 0 or width <= 0:
            return False
        crosswalk_area = 0
        for mask in bev.crosswalk:
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            crosswalk_area += int((arr > 0).sum())
        return crosswalk_area >= int(width * height * self.config.crosswalk_min_area_ratio)

    # ------------------------------------------------------------------
    # line fitting / geometry
    # ------------------------------------------------------------------
    def _fit_line(self, masks: List[Any], bev_shape: Tuple[int, int], apply_span_filter: bool = True) -> Optional[dict]:
        import numpy as np

        if not masks:
            return None
        binary = None
        for mask in masks:
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            layer = arr > 0
            binary = layer if binary is None else (binary | layer)
        if binary is None:
            return None

        height, width = bev_shape
        area = int(binary.sum())
        if area < int(width * height * self.config.min_line_area_ratio):
            return None

        top = int(height * self.config.sample_top_y_ratio)
        bottom = int(height * self.config.sample_bottom_y_ratio)
        band_half = max(1, int(height * self.config.band_height_ratio / 2.0))
        sample_ys = np.linspace(top, bottom, self.config.num_samples).astype(int)
        max_span = width * self.config.max_line_span_ratio if apply_span_filter else None

        xs: List[float] = []
        ys: List[float] = []
        for y in sample_ys:
            y0 = max(0, y - band_half)
            y1 = min(height, y + band_half + 1)
            columns = np.where(binary[y0:y1, :].any(axis=0))[0]
            if len(columns) == 0:
                continue
            # A crosswalk stripe / white blob spans much wider than a lane line;
            # drop those rows so they can't skew the line fit.
            if max_span is not None and (columns[-1] - columns[0]) > max_span:
                continue
            # Midpoint of the (thin) line's span at this row.
            xs.append(float((columns[0] + columns[-1]) / 2.0))
            ys.append(float(y))
        if len(xs) < 2:
            return None

        degree = min(self.config.poly_degree, len(xs) - 1)
        fit = np.polyfit(np.array(ys), np.array(xs), degree)
        return {"fit": fit, "min_y": min(ys), "max_y": max(ys), "n": len(xs)}

    def _midline(self, a: dict, b: dict) -> Optional[dict]:
        import numpy as np

        y0 = max(a["min_y"], b["min_y"])
        y1 = min(a["max_y"], b["max_y"])
        if y1 - y0 < 1.0:
            return None
        ys = np.linspace(y0, y1, self.config.num_samples)
        # a = center/left boundary, b = outer/right boundary. bias 0.5 = midpoint;
        # >0.5 pulls the driving line toward the outer line (less inside).
        bias = self._clip(self.config.centerline_bias, 0.0, 1.0)
        xc = (1.0 - bias) * np.polyval(a["fit"], ys) + bias * np.polyval(b["fit"], ys)
        degree = min(self.config.poly_degree, len(ys) - 1)
        fit = np.polyfit(ys, xc, degree)
        points = [(float(x), float(y)) for x, y in zip(xc, ys)]
        return {"fit": fit, "min_y": float(y0), "max_y": float(y1), "n": len(ys), "points": points}

    def _line_points(self, fit_info: Optional[dict], num: int = 20) -> List[Tuple[float, float]]:
        import numpy as np

        if fit_info is None:
            return []
        ys = np.linspace(fit_info["min_y"], fit_info["max_y"], num)
        xs = np.polyval(fit_info["fit"], ys)
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def _offset(self, fit_info: dict, dx: float) -> dict:
        import numpy as np

        fit = np.array(fit_info["fit"], dtype=float).copy()
        fit[-1] += dx  # horizontal shift of x = f(y)
        return {"fit": fit, "min_y": fit_info["min_y"], "max_y": fit_info["max_y"], "n": fit_info.get("n", 0)}

    def _heading_error(self, fit: Any, height: int) -> float:
        import numpy as np

        slope = float(np.polyval(np.polyder(fit), height - 1))
        # y grows downward, so forward is -y; a right-bending path has slope < 0.
        return self._clip(-slope * self.config.heading_gain, -1.0, 1.0)

    def _x_at(self, fit_info: dict, y: float) -> float:
        import numpy as np

        return float(np.polyval(fit_info["fit"], y))

    # ------------------------------------------------------------------
    # lane-width memory
    # ------------------------------------------------------------------
    def _lane_width(self) -> float:
        if self._lane_width_px is not None:
            return self._lane_width_px
        return self.config.lane_width_px

    def _corridor_lane_width(self) -> float:
        """Lane width to build virtual boundaries from: a fixed value through a
        crosswalk (zebra makes the measured width unreliable), else the remembered
        one."""
        if self._crosswalk_active:
            return self.config.crosswalk_lane_width_px
        return self._lane_width()

    def _update_width(self, measured: float) -> float:
        if not (self.config.min_lane_width_px <= measured <= self.config.max_lane_width_px):
            return self._lane_width()
        if self._lane_width_px is None:
            self._lane_width_px = measured
        else:
            # Reject sudden jumps (crosswalk contamination, a mis-picked side
            # line) so one bad frame can't yank the remembered width.
            if abs(measured - self._lane_width_px) > self.config.max_width_jump_px:
                return self._lane_width_px
            alpha = self.config.lane_width_smooth_alpha
            self._lane_width_px = alpha * measured + (1.0 - alpha) * self._lane_width_px
        self.last_lane_width_px = self._lane_width_px
        return self._lane_width_px

    # ------------------------------------------------------------------
    def _vehicle_center_x(self, width: int) -> float:
        center = width * (0.5 + self.config.vehicle_center_x_offset_ratio)
        return self._clip(center, 0.0, float(width - 1))

    # ------------------------------------------------------------------
    # temporal gating (Plan C)
    # ------------------------------------------------------------------
    def _is_center_jump(self, raw_center_x: float) -> bool:
        if self._last_raw_center_x is None:
            return False
        max_jump = (
            self.config.crosswalk_max_center_jump_px
            if self._crosswalk_active
            else self.config.max_center_jump_px
        )
        return abs(raw_center_x - self._last_raw_center_x) > max_jump

    def _is_heading_jump(self, raw_heading: float) -> bool:
        if self._last_raw_heading is None:
            return False
        return (
            abs(raw_heading - self._last_raw_heading)
            > max(0.0, float(self.config.max_heading_jump))
        )

    def _coast_or_lost(self, bev_shape: Tuple[int, int], reason: str) -> LaneGeometry:
        had_last = self._last_lane is not None
        # 1) Coast on the last good geometry for a few frames (momentary YOLO miss).
        if had_last and self._coast_frames < self.config.max_coast_frames:
            self._coast_frames += 1
            prev = self._last_lane
            self.last_class_name = "coast"
            # keep the last good line visible while coasting.
            self.last_centerline_bev, self.last_center_line_bev, self.last_right_line_bev = self._last_overlays
            confidence = prev.confidence * (self.config.coast_confidence_decay ** self._coast_frames)
            return LaneGeometry(
                found=True,
                center_x=prev.center_x,
                vehicle_center_x=prev.vehicle_center_x,
                target_y=prev.target_y,
                lateral_error_px=prev.lateral_error_px,
                lateral_error_norm=prev.lateral_error_norm,
                heading_error=prev.heading_error,
                confidence=self._clip(confidence, 0.0, 1.0),
                reason="coast:%s(%d)" % (reason, self._coast_frames),
                height=prev.height,
                near_center_x=prev.near_center_x,
                near_target_y=prev.near_target_y,
                near_lateral_error_px=prev.near_lateral_error_px,
                near_lateral_error_norm=prev.near_lateral_error_norm,
            )

        # 2) Coasting exhausted (or nothing was ever detected): hold a vehicle-width
        # virtual lane and keep the car centered before giving up entirely.
        if self.config.virtual_hold and self._virtual_hold_frames < self.config.virtual_hold_max_frames:
            return self._virtual_hold_lane(bev_shape, reason)

        # 3) Give up: the lane is truly lost.
        self._reset_temporal()
        self.last_class_name = "none"
        self.last_tier = 0
        return self._lost(bev_shape, ("lost:%s" % reason) if had_last else reason)

    def _virtual_hold_lane(self, bev_shape: Tuple[int, int], reason: str) -> LaneGeometry:
        """Last-resort fallback: no lane evidence remains, so build a straight
        virtual lane one vehicle-width wide and keep the car centered in it.

        The centerline is anchored on the last known lane center (falling back to
        the vehicle axis when nothing was ever seen) and eased toward the vehicle
        axis each frame, so the car straightens out gradually instead of freezing a
        mid-curve steering bias. The lane is reported as found (low confidence) so
        the follower keeps driving and re-centers; the "virtual" class name lets the
        drive-time safety guard cap its speed/steering."""
        import numpy as np

        height, width = bev_shape
        vehicle_center_x = self._vehicle_center_x(width)
        target_y = height * self.config.lookahead_y_ratio
        near_target_y = height * self.config.lane_change_near_y_ratio

        if self._virtual_center_x is None:
            self._virtual_center_x = (
                self._smoothed_center_x if self._smoothed_center_x is not None else vehicle_center_x
            )
        # Ease the held center back toward the vehicle axis so we straighten out.
        alpha = self._clip(self.config.virtual_hold_recenter_alpha, 0.0, 1.0)
        self._virtual_center_x = (1.0 - alpha) * self._virtual_center_x + alpha * vehicle_center_x
        center_x = self._clip(self._virtual_center_x, 0.0, float(width - 1))
        self._virtual_hold_frames += 1

        # Straight vertical virtual lane centered on the hold: one vehicle-width
        # wide normally, or one crosswalk-lane-width wide through a crosswalk.
        lane_w = self.config.crosswalk_lane_width_px if self._crosswalk_active else self.config.vehicle_width_px
        half = 0.5 * lane_w
        top = height * self.config.sample_top_y_ratio
        bottom = height * self.config.sample_bottom_y_ratio
        ys = np.linspace(top, bottom, self.config.num_samples)
        self.last_centerline_bev = [(center_x, float(y)) for y in ys]
        self.last_center_line_bev = [(center_x - half, float(y)) for y in ys]
        self.last_right_line_bev = [(center_x + half, float(y)) for y in ys]
        self.last_class_name = "virtual-hold"
        self.last_tier = 5
        self.last_lane_width_px = self.config.vehicle_width_px

        # Keep temporal state consistent so a recovered real frame transitions
        # smoothly (gated/smoothed relative to the held center, not a stale one).
        self._smoothed_center_x = center_x
        self._smoothed_near_center_x = center_x
        self._last_raw_center_x = center_x
        self._last_raw_heading = 0.0
        self._smoothed_heading = 0.0

        lateral_error_px = center_x - vehicle_center_x
        lateral_error_norm = self._clip(lateral_error_px / (width / 2.0), -1.0, 1.0)
        return LaneGeometry(
            found=True,
            center_x=center_x,
            vehicle_center_x=vehicle_center_x,
            target_y=target_y,
            lateral_error_px=lateral_error_px,
            lateral_error_norm=lateral_error_norm,
            heading_error=0.0,
            confidence=self._clip(self.config.virtual_hold_confidence, 0.0, 1.0),
            reason="virtual_hold:%s(%d)" % (reason, self._virtual_hold_frames),
            height=float(height),
            near_center_x=center_x,
            near_target_y=near_target_y,
            near_lateral_error_px=lateral_error_px,
            near_lateral_error_norm=lateral_error_norm,
        )

    def _reset_temporal(self) -> None:
        self._coast_frames = 0
        self._last_lane = None
        self._last_raw_center_x = None
        self._last_raw_heading = None
        self._smoothed_center_x = None
        self._smoothed_near_center_x = None
        self._smoothed_heading = None
        self._last_overlays = ([], [], [])
        self._virtual_hold_frames = 0
        self._virtual_center_x = None

    def _smooth_center(self, value: float) -> float:
        alpha = (
            self.config.crosswalk_center_smooth_alpha
            if self._crosswalk_active
            else self.config.center_smooth_alpha
        )
        if self._smoothed_center_x is None:
            self._smoothed_center_x = value
        else:
            self._smoothed_center_x = alpha * value + (1.0 - alpha) * self._smoothed_center_x
        return self._smoothed_center_x

    def _smooth_heading(self, value: float) -> float:
        alpha = self.config.heading_smooth_alpha
        if self._smoothed_heading is None:
            self._smoothed_heading = value
        else:
            self._smoothed_heading = alpha * value + (1.0 - alpha) * self._smoothed_heading
        return self._smoothed_heading

    def _smooth_near_center(self, value: float) -> float:
        alpha = (
            self.config.crosswalk_center_smooth_alpha
            if self._crosswalk_active
            else self.config.center_smooth_alpha
        )
        if self._smoothed_near_center_x is None:
            self._smoothed_near_center_x = value
        else:
            self._smoothed_near_center_x = (
                alpha * value + (1.0 - alpha) * self._smoothed_near_center_x
            )
        return self._smoothed_near_center_x

    def reset(self) -> None:
        self._lane_width_px = None
        self.last_centerline_bev = []
        self.last_center_line_bev = []
        self.last_right_line_bev = []
        self.last_class_name = "none"
        self.last_tier = 0
        self._reset_temporal()

    def _lost(self, bev_shape: Tuple[int, int], reason: str) -> LaneGeometry:
        height, width = bev_shape if bev_shape[1] > 0 else (1, 2)
        vehicle_center_x = self._vehicle_center_x(width)
        target_y = height * self.config.lookahead_y_ratio
        near_target_y = height * self.config.lane_change_near_y_ratio
        return LaneGeometry(
            found=False,
            center_x=vehicle_center_x,
            vehicle_center_x=vehicle_center_x,
            target_y=target_y,
            lateral_error_px=0.0,
            lateral_error_norm=0.0,
            heading_error=0.0,
            confidence=0.0,
            reason=reason,
            height=float(height),
            near_center_x=vehicle_center_x,
            near_target_y=near_target_y,
            near_lateral_error_px=0.0,
            near_lateral_error_norm=0.0,
        )

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
