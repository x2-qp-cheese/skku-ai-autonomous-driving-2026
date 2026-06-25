from dataclasses import dataclass
from typing import Any, List, Tuple

from ..config import LaneConfig
from ..types import LaneEstimate


@dataclass(frozen=True)
class LaneDetectionResult:
    estimate: LaneEstimate
    gray: Any
    binary_mask: Any
    lane_mask: Any
    roi_points: Any
    contours: List[Any]


class LaneDetector:
    def __init__(self, config: LaneConfig):
        self.config = config

    def detect(self, frame: Any) -> LaneDetectionResult:
        cv2, np = _load_cv()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_mask = cv2.threshold(
            blurred,
            self.config.gray_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        roi_mask, roi_points = self._make_roi_mask(frame.shape, np, cv2)
        lane_mask = cv2.bitwise_and(binary_mask, roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours = self._select_contours(lane_mask, cv2)
        estimate = self._estimate_lane(frame.shape, contours, cv2)
        return LaneDetectionResult(
            estimate=estimate,
            gray=gray,
            binary_mask=binary_mask,
            lane_mask=lane_mask,
            roi_points=roi_points,
            contours=contours,
        )

    def _make_roi_mask(self, frame_shape: Tuple[int, int, int], np: Any, cv2: Any):
        height, width = frame_shape[:2]
        points = np.array(
            [
                [
                    (int(width * self.config.roi_bottom_left_x), height),
                    (int(width * self.config.roi_bottom_right_x), height),
                    (int(width * self.config.roi_top_right_x), int(height * self.config.roi_top_y)),
                    (int(width * self.config.roi_top_left_x), int(height * self.config.roi_top_y)),
                ]
            ],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, points, 255)
        return mask, points

    def _select_contours(self, lane_mask: Any, cv2: Any) -> List[Any]:
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 15 or height < 10:
                continue
            valid.append(contour)
        valid.sort(key=cv2.contourArea, reverse=True)
        return valid[:4]

    def _estimate_lane(self, frame_shape: Tuple[int, int, int], contours: List[Any], cv2: Any) -> LaneEstimate:
        if not contours:
            return LaneEstimate(confidence=0.0)

        total_area = 0.0
        weighted_x = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            moments = cv2.moments(contour)
            if area <= 0.0 or moments["m00"] == 0:
                continue
            center_x = moments["m10"] / moments["m00"]
            weighted_x += center_x * area
            total_area += area

        if total_area <= 0.0:
            return LaneEstimate(confidence=0.0)

        width = frame_shape[1]
        lane_x = weighted_x / total_area
        offset_norm = (lane_x - (width / 2.0)) / (width / 2.0)
        confidence = min(1.0, total_area / (width * frame_shape[0] * 0.08))
        return LaneEstimate(center_offset_norm=offset_norm, confidence=confidence)


def _load_cv():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for lane detection") from exc
    return cv2, np
