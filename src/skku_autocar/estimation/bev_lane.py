from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .lane_geometry import LaneGeometry


@dataclass(frozen=True)
class BevLaneConfig:
    """Lane-center estimation on a bird's-eye-view mask.

    BEV coordinate conventions:
      - x = out_width / 2 is the vehicle centerline (camera mounted on center).
      - Forward is UP: the bottom row (y = H-1) is closest to the car.
      - After warping, lanes are (nearly) straight vertical bands, so a low-degree
        polyfit x = f(y) is stable and heading maps directly to the fit slope.
    """

    # Row (ratio of BEV height) where lateral error is measured. Lower = farther
    # ahead of the car. 0.45 sits roughly mid-lookahead.
    lookahead_y_ratio: float = 0.45
    # Rows sampled for the centerline fit, top and bottom of the scan band.
    sample_top_y_ratio: float = 0.05
    sample_bottom_y_ratio: float = 0.98
    num_samples: int = 20
    band_height_ratio: float = 0.02
    poly_degree: int = 2

    # Camera mounted slightly left of the car centerline -> the true vehicle
    # center is a bit right of frame center (measured ~0.585 on a good run).
    # Positive shifts the lateral-error reference right. Tune with
    # --vehicle-center-offset. See BevCorridorConfig for the rationale.
    vehicle_center_x_offset_ratio: float = 0.085
    min_mask_area_ratio: float = 0.001
    # Slope (dx/dy in BEV px) is dimensionless; this scales it into [-1, 1].
    heading_gain: float = 1.6

    center_smooth_alpha: float = 0.35
    heading_smooth_alpha: float = 0.40


class BevLaneGeometryEstimator:
    def __init__(self, config: BevLaneConfig = BevLaneConfig()):
        self.config = config
        self._smoothed_center_x: Optional[float] = None
        self._smoothed_heading: Optional[float] = None
        # Centerline sample points in BEV pixel coords, exposed for debug overlay.
        self.last_centerline_bev: List[Tuple[float, float]] = []

    def estimate(self, bev_mask: Optional[Any]) -> LaneGeometry:
        import numpy as np

        self.last_centerline_bev = []
        if bev_mask is None:
            return self._lost(0.0, 0.0, "no_mask")

        binary = np.asarray(bev_mask)
        if binary.ndim == 3:
            binary = binary[:, :, 0]
        binary = binary > 0

        height, width = binary.shape[:2]
        vehicle_center_x = self._vehicle_center_x(width)
        target_y = height * self.config.lookahead_y_ratio

        area = int(binary.sum())
        min_area = int(width * height * self.config.min_mask_area_ratio)
        if area < min_area:
            return self._lost(vehicle_center_x, target_y, "mask_too_small")

        centers = self._sample_centers(binary)
        if len(centers) < 2:
            return self._lost(vehicle_center_x, target_y, "no_sampled_rows")

        ys = np.array([c[0] for c in centers], dtype=float)
        xs = np.array([c[1] for c in centers], dtype=float)
        degree = min(self.config.poly_degree, len(centers) - 1)
        fit = np.polyfit(ys, xs, degree)

        center_x = float(np.polyval(fit, target_y))
        heading_error = self._heading_error(fit, height)

        self.last_centerline_bev = self._centerline_points(fit, ys)

        center_x = self._smooth_center(center_x)
        heading_error = self._smooth_heading(heading_error)
        lateral_error_px = center_x - vehicle_center_x
        lateral_error_norm = self._clip(lateral_error_px / (width / 2.0), -1.0, 1.0)

        row_coverage = min(1.0, len(centers) / float(self.config.num_samples))
        area_ratio = min(1.0, area / float(width * height) / 0.35)
        confidence = 0.2 + 0.5 * row_coverage + 0.3 * area_ratio
        return LaneGeometry(
            found=True,
            center_x=center_x,
            vehicle_center_x=vehicle_center_x,
            target_y=target_y,
            lateral_error_px=lateral_error_px,
            lateral_error_norm=lateral_error_norm,
            heading_error=heading_error,
            confidence=self._clip(confidence, 0.0, 1.0),
            reason="ok",
            height=float(height),
        )

    def _sample_centers(self, binary: Any) -> List[Tuple[float, float]]:
        import numpy as np

        height, width = binary.shape[:2]
        top = int(height * self.config.sample_top_y_ratio)
        bottom = int(height * self.config.sample_bottom_y_ratio)
        band_half = max(1, int(height * self.config.band_height_ratio / 2.0))
        sample_ys = np.linspace(top, bottom, self.config.num_samples).astype(int)

        centers: List[Tuple[float, float]] = []
        for y in sample_ys:
            y0 = max(0, y - band_half)
            y1 = min(height, y + band_half + 1)
            columns = np.where(binary[y0:y1, :].any(axis=0))[0]
            if len(columns) == 0:
                continue
            # Midpoint of the mask span. TODO: for a filled corridor mask this is
            # the corridor center; for thin lane lines consider mean instead.
            center_x = float((columns[0] + columns[-1]) / 2.0)
            centers.append((float(y), self._clip(center_x, 0.0, float(width - 1))))
        return centers

    def _heading_error(self, fit: Any, height: int) -> float:
        import numpy as np

        # Tangent of the centerline near the car (bottom row). y grows downward,
        # so forward is -y; a right-bending path has negative dx/dy.
        derivative = np.polyder(fit)
        slope = float(np.polyval(derivative, height - 1))
        return self._clip(-slope * self.config.heading_gain, -1.0, 1.0)

    def _centerline_points(self, fit: Any, ys: Any) -> List[Tuple[float, float]]:
        import numpy as np

        y_line = np.linspace(float(ys.min()), float(ys.max()), 20)
        x_line = np.polyval(fit, y_line)
        return [(float(x), float(y)) for x, y in zip(x_line, y_line)]

    def _vehicle_center_x(self, width: int) -> float:
        center = width * (0.5 + self.config.vehicle_center_x_offset_ratio)
        return self._clip(center, 0.0, float(width - 1))

    def _smooth_center(self, value: float) -> float:
        alpha = self.config.center_smooth_alpha
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

    def _lost(self, vehicle_center_x: float, target_y: float, reason: str) -> LaneGeometry:
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
        )

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
