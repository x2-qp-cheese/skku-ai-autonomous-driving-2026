from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import sys
import time

import cv2
import numpy as np
import serial


# =========================================================
# Settings
# =========================================================


@dataclass(frozen=True)
class SerialSettings:
    port: Optional[str] = "COM4" if sys.platform.startswith("win") else None
    baud: int = 115200
    ready_timeout_s: float = 3.0


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 360
    fourcc: str = "MJPG"


@dataclass(frozen=True)
class RoiSettings:
    bottom_left_x: float = 0.08
    bottom_right_x: float = 0.92
    bottom_y: float = 0.96
    top_left_x: float = 0.28
    top_right_x: float = 0.72
    top_y: float = 0.55


@dataclass(frozen=True)
class VisionSettings:
    # Color filtering: white lane extraction in HSV/HLS.
    white_s_max: int = 90
    white_v_min: int = 165
    white_l_min: int = 170
    dynamic_white_percentile: int = 88
    dynamic_white_margin: int = 8

    # Edge detection: PDF pipeline uses Canny edge after color filtering.
    blur_kernel: int = 5
    canny_low: int = 45
    canny_high: int = 135
    edge_dilate: int = 5

    # Small bright line recovery.
    tophat_kernel_size: int = 21
    tophat_threshold: int = 18
    max_roi_white_ratio: float = 0.22

    # BEV cleanup: remove stop lines/crosswalk bars before lane fitting.
    horizontal_kernel_width: int = 65
    horizontal_min_width_ratio: float = 0.38
    horizontal_max_height_ratio: float = 0.12
    dense_row_ratio: float = 0.38

    # Sliding window tracking.
    n_windows: int = 12
    window_margin: int = 58
    min_pixels_recenter: int = 8
    min_fit_pixels: int = 70
    min_y_span_ratio: float = 0.22
    peak_min_count: int = 14
    peak_min_ratio: float = 0.18
    peak_min_separation: int = 42
    max_lane_candidates: int = 4
    fit_degree: int = 2

    # Lane model selection.
    lookahead_y: float = 0.76
    lane_pair_min_width: float = 0.18
    lane_pair_max_width: float = 0.74
    assumed_half_lane_width: float = 0.25
    target_min_x: float = 0.10
    target_max_x: float = 0.90

    # Temporal validation.
    pair_confidence: float = 0.95
    single_confidence: float = 0.66
    max_low_confidence_jump_px: float = 150.0
    max_target_step_px: float = 135.0
    pair_target_alpha: float = 0.78
    single_target_alpha: float = 0.58

    # Debug windows.
    show_raw_mask: bool = False


@dataclass(frozen=True)
class ControlSettings:
    base_speed: int = 70
    turn_speed: int = 88
    hold_speed: int = 70
    max_steer: int = 130
    hold_max_steer: int = 105
    steer_sign: float = 1.0
    lateral_gain: float = 2.15
    heading_gain: float = 78.0
    steer_smooth: float = 0.10
    turn_deadband_px: float = 5.0
    min_turn_steer: int = 30
    hard_turn_error_px: float = 45.0
    hard_turn_steer: int = 105
    lost_frames_before_stop: int = 12
    center_hold_frames: int = 8


@dataclass(frozen=True)
class WindowSettings:
    drive: str = "Drive"
    lane_mask: str = "ROI Lane Mask"
    raw_mask: str = "Raw ROI Mask"


SERIAL = SerialSettings()
CAMERA = CameraSettings()
ROI = RoiSettings()
VISION = VisionSettings()
CONTROL = ControlSettings()
WINDOWS = WindowSettings()


# =========================================================
# Data objects
# =========================================================


@dataclass
class LaneFit:
    fit: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    score: float
    source_x: int

    def x_at(self, y: float) -> float:
        return float(np.polyval(self.fit, y))

    def slope_at(self, y: float) -> float:
        if len(self.fit) == 3:
            return float(2.0 * self.fit[0] * y + self.fit[1])
        if len(self.fit) == 2:
            return float(self.fit[0])
        return 0.0

    @property
    def y_span(self) -> float:
        if len(self.point_y) == 0:
            return 0.0
        return float(np.max(self.point_y) - np.min(self.point_y))


@dataclass
class LaneResult:
    target_x: Optional[float]
    target_point: Optional[Tuple[int, int]]
    heading_error: float
    confidence: float
    reason: str
    raw_mask: np.ndarray
    selected_mask: np.ndarray
    roi_points: np.ndarray
    lane_polylines: List[np.ndarray]
    center_polyline: Optional[np.ndarray]


@dataclass
class TrackResult:
    target_x: Optional[float]
    target_point: Optional[Tuple[int, int]]
    heading_error: float
    confidence: float
    reason: str
    lost_frames: int


@dataclass
class DriveCommand:
    speed: int
    steer: int


# =========================================================
# Hardware
# =========================================================


def find_serial_port() -> Optional[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return None

    preferred_ports = []
    fallback_ports = []
    for port in list_ports.comports():
        device = port.device
        if sys.platform == "darwin" and not device.startswith("/dev/cu."):
            continue

        fallback_ports.append(device)
        details = "%s %s %s" % (device, port.description, port.hwid)
        details = details.lower()
        if any(token in details for token in ("arduino", "usbmodem", "usbserial", "ch340")):
            preferred_ports.append(device)

    if preferred_ports:
        return preferred_ports[0]
    if fallback_ports:
        return fallback_ports[0]
    return None


class ArduinoLink:
    def __init__(self, settings: SerialSettings, control: ControlSettings):
        self.settings = settings
        self.control = control
        self.serial: Optional[serial.Serial] = None

    @property
    def port(self) -> str:
        return self.serial.port if self.serial is not None else ""

    def open(self) -> None:
        port = self.settings.port or find_serial_port()
        if not port:
            raise RuntimeError("아두이노 시리얼 포트를 찾지 못했습니다. scripts/list_serial_ports.py로 포트를 확인하세요.")

        self.serial = serial.Serial(port, self.settings.baud, timeout=1)
        time.sleep(2.0)

        start = time.time()
        while time.time() - start < self.settings.ready_timeout_s:
            line = self.serial.readline().decode(errors="ignore").strip()
            if line:
                print("[ARDUINO]", line)
            if "READY" in line:
                break

    def _write(self, data: bytes, label: str) -> bool:
        if self.serial is None:
            return False
        port = self.port
        try:
            self.serial.write(data)
            self._drain_responses()
            return True
        except (OSError, serial.SerialException) as exc:
            print(f"시리얼 쓰기 실패({label}, {port}): {exc}")
            self.close()
            return False

    def _drain_responses(self) -> None:
        if self.serial is None:
            return
        while self.serial.in_waiting > 0:
            line = self.serial.readline().decode(errors="ignore").strip()
            if line and not line.startswith("OK "):
                print("[ARDUINO]", line)

    def drive(self, command: DriveCommand) -> bool:
        speed = int(max(0, min(255, command.speed)))
        steer = int(max(-self.control.max_steer, min(self.control.max_steer, command.steer)))
        return self._write(f"DRIVE {speed} {steer}\n".encode(), "DRIVE")

    def stop(self) -> bool:
        return self._write(b"STOP\n", "STOP")

    def close(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except (OSError, serial.SerialException) as exc:
                print(f"시리얼 닫기 실패({self.port}): {exc}")
            self.serial = None


class CameraSource:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.settings.index, cv2.CAP_DSHOW)
        elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(self.settings.index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(self.settings.index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.settings.fourcc))

        if not cap.isOpened():
            cap.release()
            raise RuntimeError("카메라를 열 수 없습니다. CAMERA_INDEX를 0, 1, 2로 바꿔보세요.")

        self.capture = cap

    def read(self) -> Optional[np.ndarray]:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        return cv2.resize(frame, (self.settings.width, self.settings.height))

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


# =========================================================
# Perception: color + edge + BEV + sliding window + polynomial
# =========================================================


class RoiProjector:
    def __init__(self, settings: RoiSettings):
        self.settings = settings
        self._shape: Optional[Tuple[int, int]] = None
        self._roi_points: Optional[np.ndarray] = None
        self._roi_mask: Optional[np.ndarray] = None
        self._M: Optional[np.ndarray] = None
        self._Minv: Optional[np.ndarray] = None

    def ensure(self, frame_shape: Tuple[int, int, int]) -> None:
        h, w = frame_shape[:2]
        if self._shape == (h, w):
            return

        cfg = self.settings
        points = np.array([
            [
                (int(w * cfg.bottom_left_x), int(h * cfg.bottom_y)),
                (int(w * cfg.bottom_right_x), int(h * cfg.bottom_y)),
                (int(w * cfg.top_right_x), int(h * cfg.top_y)),
                (int(w * cfg.top_left_x), int(h * cfg.top_y)),
            ]
        ], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, points, 255)

        src = points[0].astype(np.float32)
        dst = np.array([
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0],
            [0, 0],
        ], dtype=np.float32)

        self._shape = (h, w)
        self._roi_points = points
        self._roi_mask = mask
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)

    def points(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        self.ensure(frame_shape)
        return self._roi_points.copy()

    def mask(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        self.ensure(frame_shape)
        return self._roi_mask.copy()

    def warp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self._M, (w, h))

    def unwarp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self._Minv, (w, h))

    def unwarp_point(self, x: float, y: float) -> Tuple[int, int]:
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self._Minv)[0][0]
        return int(round(mapped[0])), int(round(mapped[1]))

    def unwarp_polyline(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points.reshape(-1, 2)
        mapped = cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), self._Minv)
        return np.round(mapped.reshape(-1, 2)).astype(np.int32)


class LaneBinaryExtractor:
    def __init__(self, settings: VisionSettings):
        self.settings = settings

    def make_binary(self, frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        cfg = self.settings
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_size = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        roi_gray = blurred[roi_mask > 0]
        dynamic_v = cfg.white_v_min
        dynamic_l = cfg.white_l_min
        if len(roi_gray) > 0:
            dynamic = int(np.percentile(roi_gray, cfg.dynamic_white_percentile)) + cfg.dynamic_white_margin
            dynamic_v = min(245, max(cfg.white_v_min, dynamic))
            dynamic_l = min(245, max(cfg.white_l_min, dynamic))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_white = cv2.inRange(
            hsv,
            np.array([0, 0, dynamic_v], dtype=np.uint8),
            np.array([180, cfg.white_s_max, 255], dtype=np.uint8),
        )

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hls_white = cv2.inRange(
            hls,
            np.array([0, dynamic_l, 0], dtype=np.uint8),
            np.array([180, 255, cfg.white_s_max], dtype=np.uint8),
        )

        tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.tophat_kernel_size, cfg.tophat_kernel_size),
        )
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, tophat_kernel)
        _, tophat_mask = cv2.threshold(tophat, cfg.tophat_threshold, 255, cv2.THRESH_BINARY)

        white_mask = cv2.bitwise_or(hsv_white, hls_white)
        white_mask = cv2.bitwise_or(white_mask, tophat_mask)
        white_mask = cv2.bitwise_and(white_mask, roi_mask)

        roi_area = max(1, cv2.countNonZero(roi_mask))
        if cv2.countNonZero(white_mask) / float(roi_area) > cfg.max_roi_white_ratio:
            strict_threshold = min(250, max(dynamic_v, int(np.percentile(roi_gray, 96)) if len(roi_gray) else 230))
            _, strict = cv2.threshold(blurred, strict_threshold, 255, cv2.THRESH_BINARY)
            white_mask = cv2.bitwise_and(white_mask, strict)

        edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
        near_white = cv2.dilate(
            white_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.edge_dilate, cfg.edge_dilate)),
            iterations=1,
        )
        edge_mask = cv2.bitwise_and(edges, near_white)

        binary = cv2.bitwise_or(white_mask, edge_mask)
        binary = cv2.bitwise_and(binary, roi_mask)

        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return binary


class SlidingWindowLaneDetector:
    def __init__(self, roi: RoiProjector, extractor: LaneBinaryExtractor, settings: VisionSettings):
        self.roi = roi
        self.extractor = extractor
        self.settings = settings

    def detect(self, frame: np.ndarray, previous_target_x: Optional[float]) -> LaneResult:
        roi_points = self.roi.points(frame.shape)
        roi_mask = self.roi.mask(frame.shape)
        raw_mask = self.extractor.make_binary(frame, roi_mask)

        bev_raw = self.roi.warp(raw_mask)
        bev_clean = self._remove_transverse_markings(bev_raw)
        candidates = self._fit_lane_candidates(bev_clean)
        selected, center_fit, confidence, reason = self._select_lane_model(
            candidates,
            previous_target_x,
            bev_clean.shape,
        )

        selected_mask = np.zeros_like(raw_mask)
        lane_polylines: List[np.ndarray] = []
        center_polyline: Optional[np.ndarray] = None
        target_x = None
        target_point = None
        heading_error = 0.0

        if center_fit is not None:
            h, w = bev_clean.shape[:2]
            lookahead_y = h * self.settings.lookahead_y
            target_bev_x = self._clip_target_x(float(np.polyval(center_fit, lookahead_y)), w)
            target_point = self.roi.unwarp_point(target_bev_x, lookahead_y)
            target_x = float(target_point[0])
            heading_error = float(np.arctan(np.polyval(np.polyder(center_fit), lookahead_y)))

            selected_bev = np.zeros_like(bev_clean)
            for lane in selected:
                polyline_bev = self._polyline_from_fit(lane.fit, bev_clean.shape)
                if len(polyline_bev) > 0:
                    cv2.polylines(selected_bev, [polyline_bev], False, 255, thickness=8)
                    lane_polylines.append(self.roi.unwarp_polyline(polyline_bev))

            center_bev = self._polyline_from_fit(center_fit, bev_clean.shape)
            if len(center_bev) > 0:
                cv2.polylines(selected_bev, [center_bev], False, 255, thickness=3)
                center_polyline = self.roi.unwarp_polyline(center_bev)

            selected_mask = self.roi.unwarp(selected_bev)
            _, selected_mask = cv2.threshold(selected_mask, 127, 255, cv2.THRESH_BINARY)

        return LaneResult(
            target_x=target_x,
            target_point=target_point,
            heading_error=heading_error,
            confidence=confidence,
            reason=reason,
            raw_mask=raw_mask,
            selected_mask=selected_mask,
            roi_points=roi_points,
            lane_polylines=lane_polylines,
            center_polyline=center_polyline,
        )

    def _remove_transverse_markings(self, mask: np.ndarray) -> np.ndarray:
        cfg = self.settings
        h, w = mask.shape[:2]
        clean = mask.copy()

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.horizontal_kernel_width, 3))
        horizontal = cv2.morphologyEx(clean, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        transverse = np.zeros_like(clean)
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            _, _, bw, bh = cv2.boundingRect(contour)
            if bw >= w * cfg.horizontal_min_width_ratio and bh <= h * cfg.horizontal_max_height_ratio:
                cv2.drawContours(transverse, [contour], -1, 255, thickness=cv2.FILLED)
        clean = cv2.subtract(clean, transverse)

        row_counts = np.count_nonzero(clean > 0, axis=1) / float(w)
        dense_rows = np.where(row_counts > cfg.dense_row_ratio)[0]
        for y in dense_rows:
            clean[max(0, y - 2):min(h, y + 3), :] = 0

        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        return clean

    def _fit_lane_candidates(self, binary: np.ndarray) -> List[LaneFit]:
        cfg = self.settings
        h, w = binary.shape[:2]
        histogram = np.sum(binary[h // 2:, :] > 0, axis=0).astype(np.float32)
        if np.max(histogram) < cfg.peak_min_count:
            return []

        histogram = np.convolve(histogram, np.ones(11, dtype=np.float32) / 11.0, mode="same")
        peak_threshold = max(cfg.peak_min_count, float(np.max(histogram)) * cfg.peak_min_ratio)
        peaks = self._find_histogram_peaks(histogram, peak_threshold, cfg.peak_min_separation)
        peaks = peaks[:cfg.max_lane_candidates]

        nonzero_y, nonzero_x = binary.nonzero()
        candidates = []
        for peak in peaks:
            fit = self._sliding_window_fit(binary, nonzero_x, nonzero_y, int(peak))
            if fit is not None:
                candidates.append(fit)

        return self._merge_duplicate_fits(candidates, binary.shape)

    def _find_histogram_peaks(self, histogram: np.ndarray, threshold: float, min_separation: int) -> List[int]:
        peaks = []
        for x in range(1, len(histogram) - 1):
            if histogram[x] < threshold:
                continue
            if histogram[x] >= histogram[x - 1] and histogram[x] >= histogram[x + 1]:
                peaks.append((histogram[x], x))

        peaks.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for _, x in peaks:
            if all(abs(x - existing) >= min_separation for existing in selected):
                selected.append(int(x))
        selected.sort()
        return selected

    def _sliding_window_fit(
        self,
        binary: np.ndarray,
        nonzero_x: np.ndarray,
        nonzero_y: np.ndarray,
        start_x: int,
    ) -> Optional[LaneFit]:
        cfg = self.settings
        h, w = binary.shape[:2]
        window_height = max(1, h // cfg.n_windows)
        current_x = start_x
        lane_indices = []

        for window in range(cfg.n_windows):
            y_low = h - (window + 1) * window_height
            y_high = h - window * window_height
            x_low = max(0, current_x - cfg.window_margin)
            x_high = min(w, current_x + cfg.window_margin)

            good = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            ).nonzero()[0]
            if len(good) > 0:
                lane_indices.append(good)
            if len(good) >= cfg.min_pixels_recenter:
                current_x = int(np.mean(nonzero_x[good]))

        if not lane_indices:
            return None

        lane_indices = np.concatenate(lane_indices)
        if len(lane_indices) < cfg.min_fit_pixels:
            return None

        point_x = nonzero_x[lane_indices].astype(np.float32)
        point_y = nonzero_y[lane_indices].astype(np.float32)
        y_span = float(np.max(point_y) - np.min(point_y))
        if y_span < h * cfg.min_y_span_ratio:
            return None

        degree = min(cfg.fit_degree, len(point_y) - 1)
        fit = np.polyfit(point_y, point_x, degree)
        if len(fit) < 3:
            fit = np.pad(fit, (3 - len(fit), 0), mode="constant")

        score = float(len(point_x) + y_span * 2.0)
        return LaneFit(fit=fit, point_x=point_x, point_y=point_y, score=score, source_x=start_x)

    def _merge_duplicate_fits(self, candidates: Sequence[LaneFit], shape: Tuple[int, int]) -> List[LaneFit]:
        if not candidates:
            return []

        h, _ = shape[:2]
        y_eval = h * self.settings.lookahead_y
        sorted_candidates = sorted(candidates, key=lambda lane: lane.x_at(y_eval))
        merged: List[LaneFit] = []
        for candidate in sorted_candidates:
            if not merged:
                merged.append(candidate)
                continue
            if abs(candidate.x_at(y_eval) - merged[-1].x_at(y_eval)) < self.settings.peak_min_separation:
                if candidate.score > merged[-1].score:
                    merged[-1] = candidate
            else:
                merged.append(candidate)
        return merged

    def _select_lane_model(
        self,
        candidates: Sequence[LaneFit],
        previous_target_x: Optional[float],
        shape: Tuple[int, int],
    ) -> Tuple[List[LaneFit], Optional[np.ndarray], float, str]:
        if not candidates:
            return [], None, 0.0, "lost"

        cfg = self.settings
        h, w = shape[:2]
        y_eval = h * cfg.lookahead_y
        frame_center = w * 0.5
        reference = previous_target_x if previous_target_x is not None else frame_center

        samples = sorted((lane.x_at(y_eval), lane) for lane in candidates)
        pair_candidates = []
        for i in range(len(samples) - 1):
            left_x, left_lane = samples[i]
            right_x, right_lane = samples[i + 1]
            lane_width = right_x - left_x
            if not (w * cfg.lane_pair_min_width <= lane_width <= w * cfg.lane_pair_max_width):
                continue

            center_fit = (left_lane.fit + right_lane.fit) / 2.0
            center_x = float(np.polyval(center_fit, y_eval))
            left_bottom = left_lane.x_at(h - 1)
            right_bottom = right_lane.x_at(h - 1)
            contains_vehicle = left_bottom <= frame_center <= right_bottom
            score = abs(center_x - reference)
            if contains_vehicle:
                score -= w * 0.22
            score -= (left_lane.score + right_lane.score) / 1000.0
            pair_candidates.append((score, [left_lane, right_lane], center_fit))

        if pair_candidates:
            pair_candidates.sort(key=lambda item: item[0])
            _, lanes, center_fit = pair_candidates[0]
            return lanes, center_fit, cfg.pair_confidence, "pair"

        best = max(samples, key=lambda item: item[1].score - abs(item[0] - frame_center) * 1.5)[1]
        offset = w * cfg.assumed_half_lane_width
        bottom_x = best.x_at(h - 1)
        if bottom_x < frame_center:
            offset_sign = 1.0
        else:
            offset_sign = -1.0

        center_fit = best.fit.copy()
        center_fit[-1] += offset_sign * offset
        return [best], center_fit, cfg.single_confidence, "single"

    def _polyline_from_fit(self, fit: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        ys = np.linspace(0, h - 1, 80)
        xs = np.polyval(fit, ys)
        valid = (xs >= 0) & (xs < w)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32)
        points = np.column_stack([xs[valid], ys[valid]])
        return np.round(points).astype(np.int32)

    def _clip_target_x(self, x: float, image_width: int) -> float:
        cfg = self.settings
        return float(max(image_width * cfg.target_min_x, min(image_width * cfg.target_max_x, x)))


# =========================================================
# Tracking and control
# =========================================================


class LaneTracker:
    def __init__(self, vision: VisionSettings, control: ControlSettings):
        self.vision = vision
        self.control = control
        self.last_target_x: Optional[float] = None
        self.last_target_point: Optional[Tuple[int, int]] = None
        self.last_heading_error: float = 0.0
        self.hold_frames = 0
        self.lost_frames = 0

    def update(self, lane: LaneResult) -> TrackResult:
        if lane.target_x is not None and lane.confidence > 0.0:
            target_x = self._stabilize_target(lane)
            if target_x is None:
                return self._hold_or_lost()

            heading = 0.7 * lane.heading_error + 0.3 * self.last_heading_error
            self.last_target_x = target_x
            self.last_target_point = lane.target_point
            self.last_heading_error = heading
            self.hold_frames = 0
            self.lost_frames = 0
            return TrackResult(
                target_x=target_x,
                target_point=lane.target_point,
                heading_error=heading,
                confidence=lane.confidence,
                reason=lane.reason,
                lost_frames=self.lost_frames,
            )

        return self._hold_or_lost()

    def _stabilize_target(self, lane: LaneResult) -> Optional[float]:
        measured_x = lane.target_x
        if measured_x is None:
            return None
        if self.last_target_x is None:
            return measured_x

        delta = measured_x - self.last_target_x
        jump = abs(delta)
        if lane.confidence < 0.75 and jump > self.vision.max_low_confidence_jump_px:
            measured_x = self.last_target_x + self.vision.max_low_confidence_jump_px * (1.0 if delta > 0 else -1.0)
            delta = measured_x - self.last_target_x
            jump = abs(delta)

        alpha = self.vision.pair_target_alpha if lane.reason == "pair" else self.vision.single_target_alpha
        if jump > self.vision.max_target_step_px:
            measured_x = self.last_target_x + self.vision.max_target_step_px * (1.0 if delta > 0 else -1.0)
        return self.last_target_x + alpha * (measured_x - self.last_target_x)

    def _hold_or_lost(self) -> TrackResult:
        self.lost_frames += 1
        if self.last_target_x is not None and self.hold_frames < self.control.center_hold_frames:
            self.hold_frames += 1
            return TrackResult(
                target_x=self.last_target_x,
                target_point=self.last_target_point,
                heading_error=self.last_heading_error,
                confidence=0.25,
                reason="hold",
                lost_frames=self.lost_frames,
            )
        return TrackResult(
            target_x=None,
            target_point=None,
            heading_error=0.0,
            confidence=0.0,
            reason="lost",
            lost_frames=self.lost_frames,
        )

    def reset(self) -> None:
        self.hold_frames = 0
        self.lost_frames = 0


class DriveController:
    def __init__(self, control: ControlSettings, camera: CameraSettings):
        self.control = control
        self.camera = camera
        self.smoothed_steer = 0.0

    def make_command(self, target_x: float, heading_error: float) -> DriveCommand:
        frame_center_x = self.camera.width * 0.5
        lateral_error = target_x - frame_center_x
        abs_error = abs(lateral_error)

        if abs_error <= self.control.turn_deadband_px:
            lateral_term = 0.0
        else:
            lateral_term = np.sign(lateral_error) * (abs_error - self.control.turn_deadband_px)

        raw_steer = self.control.steer_sign * (
            self.control.lateral_gain * lateral_term
            + self.control.heading_gain * heading_error
        )

        if abs(raw_steer) > 0:
            steer_sign = 1.0 if raw_steer > 0 else -1.0
            if abs(raw_steer) < self.control.min_turn_steer and abs_error > self.control.turn_deadband_px:
                raw_steer = steer_sign * self.control.min_turn_steer
            if abs_error >= self.control.hard_turn_error_px:
                raw_steer = steer_sign * max(abs(raw_steer), self.control.hard_turn_steer)

        self.smoothed_steer = (
            self.control.steer_smooth * self.smoothed_steer
            + (1.0 - self.control.steer_smooth) * raw_steer
        )
        steer = int(max(-self.control.max_steer, min(self.control.max_steer, self.smoothed_steer)))
        turn_ratio = abs(steer) / float(self.control.max_steer)
        speed = int(self.control.base_speed + (self.control.turn_speed - self.control.base_speed) * turn_ratio)
        return DriveCommand(speed=speed, steer=steer)

    def reset(self) -> None:
        self.smoothed_steer = 0.0


# =========================================================
# Display
# =========================================================


def setup_display_windows() -> None:
    cv2.namedWindow(WINDOWS.drive, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS.drive, CAMERA.width, CAMERA.height)
    cv2.moveWindow(WINDOWS.drive, 40, 80)

    cv2.namedWindow(WINDOWS.lane_mask, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS.lane_mask, CAMERA.width, CAMERA.height)
    cv2.moveWindow(WINDOWS.lane_mask, CAMERA.width + 80, 80)

    if VISION.show_raw_mask:
        cv2.namedWindow(WINDOWS.raw_mask, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOWS.raw_mask, CAMERA.width, CAMERA.height)
        cv2.moveWindow(WINDOWS.raw_mask, CAMERA.width + 80, CAMERA.height + 140)


def draw_overlay(
    frame: np.ndarray,
    lane: LaneResult,
    track: TrackResult,
    command: Optional[DriveCommand],
    driving: bool,
    fps: float,
) -> np.ndarray:
    result = frame.copy()
    cv2.polylines(result, lane.roi_points, True, (255, 0, 0), 2)

    for polyline in lane.lane_polylines:
        if len(polyline) >= 2:
            cv2.polylines(result, [polyline], False, (0, 255, 0), 3)

    if lane.center_polyline is not None and len(lane.center_polyline) >= 2:
        cv2.polylines(result, [lane.center_polyline], False, (255, 0, 255), 2)

    frame_center_x = CAMERA.width * 0.5
    y0 = int(CAMERA.height * ROI.top_y)
    y1 = CAMERA.height
    cv2.line(result, (int(frame_center_x), y1), (int(frame_center_x), y0), (255, 255, 0), 1)

    if track.target_point is not None:
        cv2.circle(result, track.target_point, 7, (0, 0, 255), -1)
        cv2.line(result, (track.target_point[0], y1), (track.target_point[0], y0), (0, 0, 255), 2)

    mode = "RUN" if driving else "PAUSE"
    mode_color = (0, 255, 0) if driving else (0, 0, 255)
    cv2.putText(result, mode, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)

    if command is not None:
        status = f"{track.reason.upper()} s={command.speed} st={command.steer}"
    elif track.target_x is not None:
        status = f"{track.reason.upper()} target"
    elif driving and track.lost_frames >= CONTROL.lost_frames_before_stop:
        status = "LANE LOST -> STOP"
    else:
        status = f"LANE LOST ({track.lost_frames})"

    cv2.putText(result, status, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(result, f"FPS: {fps:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return result


# =========================================================
# Main loop
# =========================================================


def main() -> None:
    arduino = ArduinoLink(SERIAL, CONTROL)
    camera = CameraSource(CAMERA)

    try:
        arduino.open()
        print("시리얼 연결 완료:", arduino.port)
    except Exception as exc:
        print("시리얼 연결 실패:", exc)
        print("포트 설정과 시리얼 모니터/다른 프로그램 점유 여부를 확인하세요.")
        return

    try:
        camera.open()
    except Exception as exc:
        print(exc)
        arduino.stop()
        arduino.close()
        return

    roi = RoiProjector(ROI)
    extractor = LaneBinaryExtractor(VISION)
    perception = SlidingWindowLaneDetector(roi, extractor, VISION)
    tracker = LaneTracker(VISION, CONTROL)
    controller = DriveController(CONTROL, CAMERA)

    driving = False
    prev_time = time.time()
    setup_display_windows()
    print("스페이스바: 주행 시작/정지 토글  |  q: 종료")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("프레임을 읽지 못했습니다.")
                break

            lane = perception.detect(frame, tracker.last_target_x)
            track = tracker.update(lane)

            command: Optional[DriveCommand] = None
            if track.target_x is not None:
                command = controller.make_command(track.target_x, track.heading_error)
                if track.reason == "hold":
                    command = DriveCommand(
                        speed=CONTROL.hold_speed,
                        steer=int(max(-CONTROL.hold_max_steer, min(CONTROL.hold_max_steer, command.steer))),
                    )
                if driving and not arduino.drive(command):
                    print("시리얼 연결이 끊겨 주행을 종료합니다. USB/전원 연결을 확인하세요.")
                    driving = False
                    break
            elif driving and track.lost_frames >= CONTROL.lost_frames_before_stop:
                if not arduino.stop():
                    print("시리얼 연결이 끊겨 주행을 종료합니다. USB/전원 연결을 확인하세요.")
                    driving = False
                    break

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            cv2.imshow(WINDOWS.drive, draw_overlay(frame, lane, track, command, driving, fps))
            cv2.imshow(WINDOWS.lane_mask, lane.selected_mask)
            if VISION.show_raw_mask:
                cv2.imshow(WINDOWS.raw_mask, lane.raw_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                driving = not driving
                if not driving:
                    if not arduino.stop():
                        print("시리얼 연결 없이 주행 상태를 해제합니다.")
                    controller.reset()
                    tracker.reset()
                print("주행:", "ON" if driving else "OFF")

    finally:
        stopped = arduino.stop()
        time.sleep(0.2)
        arduino.close()
        camera.release()
        cv2.destroyAllWindows()
        if stopped:
            print("정지 명령 전송 후 종료.")
        else:
            print("시리얼 연결 없이 종료.")


if __name__ == "__main__":
    main()
