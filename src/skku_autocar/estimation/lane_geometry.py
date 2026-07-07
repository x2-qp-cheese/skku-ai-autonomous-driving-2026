from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class LaneGeometryConfig:
    lookahead_y_ratio: float = 0.72
    sample_top_y_ratio: float = 0.45
    sample_bottom_y_ratio: float = 0.92
    vehicle_center_x_offset_ratio: float = 0.0
    band_height_ratio: float = 0.035
    min_mask_area_ratio: float = 0.001
    center_smooth_alpha: float = 0.35
    heading_smooth_alpha: float = 0.40


@dataclass(frozen=True)
class LaneGeometry:
    found: bool
    center_x: float
    vehicle_center_x: float
    target_y: float
    lateral_error_px: float
    lateral_error_norm: float
    heading_error: float
    confidence: float
    reason: str = ""


class MaskLaneGeometryEstimator:
    def __init__(self, config: LaneGeometryConfig = LaneGeometryConfig()):
        self.config = config
        self._smoothed_center_x: Optional[float] = None
        self._smoothed_heading: Optional[float] = None

    def estimate(self, mask: Optional[Any], frame_shape: Tuple[int, int, int]) -> LaneGeometry:
        import numpy as np

        height, width = frame_shape[:2]
        vehicle_center_x = self._vehicle_center_x(width)
        target_y = height * self.config.lookahead_y_ratio

        if mask is None:
            return self._lost(width, vehicle_center_x, target_y, "no_mask")

        binary = np.asarray(mask)
        if binary.ndim == 3:
            binary = binary[:, :, 0]
        binary = binary > 0

        area = int(binary.sum())
        min_area = int(width * height * self.config.min_mask_area_ratio)
        if area < min_area:
            return self._lost(width, vehicle_center_x, target_y, "mask_too_small")

        centers = self._sample_centers(binary)
        if len(centers) == 0:
            return self._lost(width, vehicle_center_x, target_y, "no_sampled_rows")

        ys = np.array([item[0] for item in centers], dtype=float)
        xs = np.array([item[1] for item in centers], dtype=float)
        center_x = float(np.interp(target_y, ys, xs))
        heading_error = self._heading_error(xs, ys, width, height)

        center_x = self._smooth_center(center_x)
        heading_error = self._smooth_heading(heading_error)
        lateral_error_px = center_x - vehicle_center_x
        lateral_error_norm = self._clip(lateral_error_px / (width / 2.0), -1.0, 1.0)

        row_coverage = min(1.0, len(centers) / 8.0)
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
        )

    def _sample_centers(self, binary: Any) -> list:
        import numpy as np

        height, width = binary.shape[:2]
        top = int(height * self.config.sample_top_y_ratio)
        bottom = int(height * self.config.sample_bottom_y_ratio)
        band_half = max(1, int(height * self.config.band_height_ratio / 2.0))
        sample_ys = np.linspace(top, bottom, 10).astype(int)

        centers = []
        for y in sample_ys:
            y0 = max(0, y - band_half)
            y1 = min(height, y + band_half + 1)
            columns = np.where(binary[y0:y1, :].any(axis=0))[0]
            if len(columns) == 0:
                continue
            center_x = float((columns[0] + columns[-1]) / 2.0)
            centers.append((float(y), self._clip(center_x, 0.0, float(width - 1))))
        return centers

    def _heading_error(self, xs: Any, ys: Any, width: int, height: int) -> float:
        import numpy as np

        if len(xs) < 2:
            return 0.0
        slope_x_per_y = float(np.polyfit(ys, xs, 1)[0])
        # y grows downward in images, so a right-bending path has a negative slope.
        normalized = -slope_x_per_y * (height / max(1.0, width / 2.0))
        return self._clip(normalized, -1.0, 1.0)

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

    def _lost(
        self,
        width: int,
        vehicle_center_x: float,
        target_y: float,
        reason: str,
    ) -> LaneGeometry:
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
