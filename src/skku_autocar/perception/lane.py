from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..config import LaneConfig
from ..types import LaneEstimate


@dataclass(frozen=True)
class LaneDetectionResult:
    estimate: LaneEstimate
    binary: Any           # combined color+edge binary (pre-BEV)
    warped_binary: Any    # Bird's Eye View binary
    lane_viz: Any         # BEV with sliding windows + fitted curves
    left_fit: Any         # polynomial coefficients for left lane (None if undetected)
    right_fit: Any        # polynomial coefficients for right lane (None if undetected)


class LaneDetector:
    def __init__(self, config: LaneConfig):
        self.config = config
        self._M: Any = None
        self._Minv: Any = None
        self._warped_size: Optional[Tuple[int, int]] = None
        self._frame_shape: Optional[Tuple[int, int]] = None

    def detect(self, frame: Any) -> LaneDetectionResult:
        cv2, np = _load_cv()
        h, w = frame.shape[:2]

        binary = self._make_binary(frame, cv2, np)

        if self._frame_shape != (h, w):
            self._init_transform(h, w, np, cv2)
            self._frame_shape = (h, w)

        warped = cv2.warpPerspective(binary, self._M, self._warped_size)
        left_fit, right_fit, lane_viz = self._fit_lanes(warped, cv2, np)
        estimate = self._estimate(warped.shape, left_fit, right_fit, np)

        return LaneDetectionResult(
            estimate=estimate,
            binary=binary,
            warped_binary=warped,
            lane_viz=lane_viz,
            left_fit=left_fit,
            right_fit=right_fit,
        )

    # ------------------------------------------------------------------
    # Step 1+2: Binary image (color filter + Canny edge)
    # ------------------------------------------------------------------
    def _make_binary(self, frame: Any, cv2: Any, np: Any) -> Any:
        # White lane in HLS: high L (lightness), low S (saturation)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower = np.array([0, self.config.white_l_min, 0], dtype=np.uint8)
        upper = np.array([180, 255, self.config.white_s_max], dtype=np.uint8)
        color_mask = cv2.inRange(hls, lower, upper)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.config.canny_low, self.config.canny_high)

        binary = np.zeros(gray.shape, dtype=np.uint8)
        binary[(color_mask == 255) | (edges == 255)] = 255
        return binary

    # ------------------------------------------------------------------
    # Step 3: Perspective transform (Bird's Eye View)
    # ------------------------------------------------------------------
    def _init_transform(self, h: int, w: int, np: Any, cv2: Any) -> None:
        cfg = self.config
        src = np.float32([
            [w * cfg.bev_src_bottom_left_x,  h * cfg.bev_src_bottom_y],
            [w * cfg.bev_src_bottom_right_x, h * cfg.bev_src_bottom_y],
            [w * cfg.bev_src_top_right_x,    h * cfg.bev_src_top_y],
            [w * cfg.bev_src_top_left_x,     h * cfg.bev_src_top_y],
        ])
        dst = np.float32([
            [w * cfg.bev_dst_left_x,  h],
            [w * cfg.bev_dst_right_x, h],
            [w * cfg.bev_dst_right_x, 0],
            [w * cfg.bev_dst_left_x,  0],
        ])
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)
        self._warped_size = (w, h)

    # ------------------------------------------------------------------
    # Step 4+5: Sliding window + Polynomial fitting
    # ------------------------------------------------------------------
    def _fit_lanes(
        self, warped: Any, cv2: Any, np: Any
    ) -> Tuple[Any, Any, Any]:
        wh, ww = warped.shape[:2]

        # Histogram of bottom half to find initial x positions
        histogram = np.sum(warped[wh // 2:, :], axis=0)
        mid = ww // 2
        left_x_cur = int(np.argmax(histogram[:mid]))
        right_x_cur = int(np.argmax(histogram[mid:])) + mid

        n_win = self.config.n_windows
        margin = self.config.window_margin
        min_pix = self.config.min_pix
        win_h = wh // n_win

        nonzero = warped.nonzero()
        nzy = np.array(nonzero[0])
        nzx = np.array(nonzero[1])

        left_idx_parts, right_idx_parts = [], []
        viz = np.dstack([warped, warped, warped])

        for i in range(n_win):
            y_lo = wh - (i + 1) * win_h
            y_hi = wh - i * win_h
            xl_lo, xl_hi = left_x_cur - margin,  left_x_cur + margin
            xr_lo, xr_hi = right_x_cur - margin, right_x_cur + margin

            cv2.rectangle(viz, (xl_lo, y_lo), (xl_hi, y_hi), (0, 255, 0), 2)
            cv2.rectangle(viz, (xr_lo, y_lo), (xr_hi, y_hi), (0, 255, 0), 2)

            good_l = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xl_lo) & (nzx < xl_hi)).nonzero()[0]
            good_r = ((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= xr_lo) & (nzx < xr_hi)).nonzero()[0]
            left_idx_parts.append(good_l)
            right_idx_parts.append(good_r)

            if len(good_l) >= min_pix:
                left_x_cur = int(np.mean(nzx[good_l]))
            if len(good_r) >= min_pix:
                right_x_cur = int(np.mean(nzx[good_r]))

        left_idx = np.concatenate(left_idx_parts)
        right_idx = np.concatenate(right_idx_parts)
        deg = self.config.poly_deg

        left_fit = right_fit = None
        if len(left_idx) > deg + 1:
            left_fit = np.polyfit(nzy[left_idx], nzx[left_idx], deg)
            viz[nzy[left_idx], nzx[left_idx]] = [255, 80, 80]

        if len(right_idx) > deg + 1:
            right_fit = np.polyfit(nzy[right_idx], nzx[right_idx], deg)
            viz[nzy[right_idx], nzx[right_idx]] = [80, 80, 255]

        # Draw fitted curves
        y_vals = np.arange(wh)
        for fit, color in ((left_fit, (0, 220, 255)), (right_fit, (0, 220, 255))):
            if fit is None:
                continue
            x_vals = np.polyval(fit, y_vals).astype(np.int32)
            for y, x in zip(y_vals, x_vals):
                if 0 <= x < ww:
                    cv2.circle(viz, (x, int(y)), 2, color, -1)

        return left_fit, right_fit, viz

    # ------------------------------------------------------------------
    # Step 6: Lane offset and heading estimate
    # ------------------------------------------------------------------
    def _estimate(
        self,
        warped_shape: Tuple[int, int],
        left_fit: Any,
        right_fit: Any,
        np: Any,
    ) -> LaneEstimate:
        if left_fit is None and right_fit is None:
            return LaneEstimate(confidence=0.0)

        wh, ww = warped_shape[:2]
        y_eval = float(wh - 1)
        assumed_half_lane = ww * (self.config.bev_dst_right_x - self.config.bev_dst_left_x) * 0.5

        if left_fit is not None and right_fit is not None:
            lx = float(np.polyval(left_fit, y_eval))
            rx = float(np.polyval(right_fit, y_eval))
            center_x = (lx + rx) / 2.0
            l_slope = float(np.polyval(np.polyder(left_fit), y_eval))
            r_slope = float(np.polyval(np.polyder(right_fit), y_eval))
            heading_err = (l_slope + r_slope) / 2.0
            confidence = 1.0
        elif left_fit is not None:
            center_x = float(np.polyval(left_fit, y_eval)) + assumed_half_lane
            heading_err = float(np.polyval(np.polyder(left_fit), y_eval))
            confidence = 0.6
        else:
            center_x = float(np.polyval(right_fit, y_eval)) - assumed_half_lane
            heading_err = float(np.polyval(np.polyder(right_fit), y_eval))
            confidence = 0.6

        offset_norm = (center_x - ww / 2.0) / (ww / 2.0)
        return LaneEstimate(
            center_offset_norm=float(offset_norm),
            heading_error=float(heading_err),
            confidence=confidence,
        )


def _load_cv() -> Tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for lane detection") from exc
    return cv2, np
