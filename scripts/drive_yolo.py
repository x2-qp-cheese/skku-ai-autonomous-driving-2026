from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import sys
import time

import cv2
import numpy as np
import serial
from ultralytics import YOLO


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
    index: int = 1
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
    # YOLO detection.
    yolo_conf: float = 0.4
    conf_select_bonus_px: float = 60.0  # 검출 conf 1.0당 후보 선정에서 60px 이득

    # Polynomial fitting on YOLO masks.
    min_fit_pixels: int = 40
    min_y_span_ratio: float = 0.15
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

MODEL_PATH: str = "scripts/best.pt"
CLASS_LANE_CENTER: int = 3  # ['arrow', 'car-front', 'crosswalk', 'lane-center', 'lane-side', 'light']
CLASS_LANE_SIDE: int = 4

# =========================================================
# Data objects
# =========================================================


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

        # 카메라 워밍업: 초기 프레임이 비어있는 경우가 있어 몇 프레임 읽어 버림
        for _ in range(5):
            cap.read()

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
# ROI (표시용 사다리꼴)
# =========================================================


class RoiProjector:
    def __init__(self, settings: RoiSettings):
        self.settings = settings
        self._shape: Optional[Tuple[int, int]] = None
        self._roi_points: Optional[np.ndarray] = None

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
        self._shape = (h, w)
        self._roi_points = points

    def points(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        self.ensure(frame_shape)
        return self._roi_points.copy()


# =========================================================
# Perception: YOLO segmentation → image-space polynomial fit
# =========================================================


class YoloLaneDetector:
    def __init__(self, model_path: str, roi: RoiProjector, settings: VisionSettings):
        self.model = YOLO(model_path)
        self.roi = roi
        self.settings = settings

    @staticmethod
    def _conf_scale(conf: float) -> float:
        # 검출 신뢰도를 0.6~1.0 배율로 변환해 최종 confidence에 반영.
        return 0.6 + 0.4 * float(np.clip(conf, 0.0, 1.0))

    def detect(self, frame: np.ndarray, previous_target_x: Optional[float]) -> LaneResult:
        cfg = self.settings
        h, w = frame.shape[:2]
        roi_points = self.roi.points(frame.shape)
        lookahead_y = h * cfg.lookahead_y
        frame_cx = w * 0.5
        reference = previous_target_x if previous_target_x is not None else frame_cx

        results = self.model(frame, conf=cfg.yolo_conf, verbose=False)

        raw_mask = np.zeros((h, w), dtype=np.uint8)
        center_fits: List[Tuple[np.ndarray, float]] = []
        side_fits: List[Tuple[np.ndarray, float]] = []

        for r in results:
            if r.masks is None:
                continue
            for mask_xy, cls, conf in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
                cls_id = int(cls)
                if cls_id not in (CLASS_LANE_CENTER, CLASS_LANE_SIDE):
                    continue
                det_conf = float(conf)
                pts = np.array(mask_xy, dtype=np.int32)
                if len(pts) < 3:
                    continue

                mask_img = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_img, [pts], 255)
                raw_mask = cv2.bitwise_or(raw_mask, mask_img)

                # YOLO 마스크를 이미지 좌표에서 그대로 피팅 (ROI 클리핑 없음)
                ys, xs = np.where(mask_img > 0)
                fit = self._fit_poly(xs.astype(np.float32), ys.astype(np.float32), h)
                if fit is None:
                    continue

                if cls_id == CLASS_LANE_CENTER:
                    center_fits.append((fit, det_conf))
                else:
                    side_fits.append((fit, det_conf))

        final_center_fit: Optional[np.ndarray] = None
        selected_fits: List[np.ndarray] = []
        confidence = 0.0
        reason = "lost"

        if center_fits:
            fit, det_conf = self._pick_closest(center_fits, lookahead_y, reference, cfg)
            final_center_fit = fit
            selected_fits = [fit]
            confidence = cfg.pair_confidence * self._conf_scale(det_conf)
            reason = "center"
        elif len(side_fits) >= 2:
            pair = self._pick_side_pair(side_fits, lookahead_y, w, cfg, reference)
            if pair is not None:
                left_fit, right_fit, mean_conf = pair
                final_center_fit = (left_fit + right_fit) / 2.0
                selected_fits = [left_fit, right_fit]
                confidence = cfg.pair_confidence * self._conf_scale(mean_conf)
                reason = "pair"

        if final_center_fit is None and side_fits:
            best_fit, det_conf = self._pick_closest(side_fits, lookahead_y, reference, cfg)
            bottom_x = float(np.polyval(best_fit, h - 1))
            offset = w * cfg.assumed_half_lane_width
            offset_sign = 1.0 if bottom_x < frame_cx else -1.0
            final_center_fit = best_fit.copy()
            final_center_fit[-1] += offset_sign * offset
            selected_fits = [best_fit]
            confidence = cfg.single_confidence * self._conf_scale(det_conf)
            reason = "single"

        target_x = None
        target_point = None
        heading_error = 0.0
        selected_mask = np.zeros((h, w), dtype=np.uint8)
        lane_polylines: List[np.ndarray] = []
        center_polyline: Optional[np.ndarray] = None

        if final_center_fit is not None:
            cx = float(np.polyval(final_center_fit, lookahead_y))
            cx = max(w * cfg.target_min_x, min(w * cfg.target_max_x, cx))
            target_x = cx
            target_point = (int(round(cx)), int(round(lookahead_y)))
            heading_error = float(np.arctan(np.polyval(np.polyder(final_center_fit), lookahead_y)))

            for fit in selected_fits:
                poly = self._make_polyline(fit, h, w)
                if len(poly) >= 2:
                    cv2.polylines(selected_mask, [poly], False, 255, thickness=8)
                    lane_polylines.append(poly)

            cpoly = self._make_polyline(final_center_fit, h, w)
            if len(cpoly) >= 2:
                center_polyline = cpoly

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

    def _fit_poly(self, xs: np.ndarray, ys: np.ndarray, h: int) -> Optional[np.ndarray]:
        cfg = self.settings
        if len(xs) < cfg.min_fit_pixels:
            return None
        y_span = float(np.max(ys) - np.min(ys))
        if y_span < h * cfg.min_y_span_ratio:
            return None
        degree = min(cfg.fit_degree, len(xs) - 1)
        fit = np.polyfit(ys, xs, degree)
        if len(fit) < 3:
            fit = np.pad(fit, (3 - len(fit), 0), mode="constant")
        return fit

    def _pick_closest(
        self,
        fits: List[Tuple[np.ndarray, float]],
        lookahead_y: float,
        reference: float,
        cfg: VisionSettings,
    ) -> Tuple[np.ndarray, float]:
        # 이전 목표와의 거리에서 검출 conf만큼 이득을 줘서 오검출에 덜 끌리게.
        def score(item: Tuple[np.ndarray, float]) -> float:
            fit, conf = item
            pos = float(np.polyval(fit, lookahead_y))
            return abs(pos - reference) - cfg.conf_select_bonus_px * conf
        return min(fits, key=score)

    def _pick_side_pair(
        self,
        side_fits: List[Tuple[np.ndarray, float]],
        lookahead_y: float,
        w: int,
        cfg: VisionSettings,
        reference: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        evaluated = sorted(
            ((float(np.polyval(fit, lookahead_y)), fit, conf) for fit, conf in side_fits),
            key=lambda item: item[0],
        )
        best_score = None
        best_pair = None
        for i in range(len(evaluated) - 1):
            lx, lf, lc = evaluated[i]
            rx, rf, rc = evaluated[i + 1]
            lane_width = rx - lx
            if not (w * cfg.lane_pair_min_width <= lane_width <= w * cfg.lane_pair_max_width):
                continue
            center_x = (lx + rx) / 2.0
            mean_conf = 0.5 * (lc + rc)
            score = abs(center_x - reference) - cfg.conf_select_bonus_px * mean_conf
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (lf, rf, mean_conf)
        return best_pair

    def _make_polyline(self, fit: np.ndarray, h: int, w: int) -> np.ndarray:
        ys = np.linspace(0, h - 1, 80)
        xs = np.polyval(fit, ys)
        valid = (xs >= 0) & (xs < w)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int32)
        points = np.column_stack([xs[valid], ys[valid]])
        return np.round(points).astype(np.int32)


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
    perception = YoloLaneDetector(MODEL_PATH, roi, VISION)
    tracker = LaneTracker(VISION, CONTROL)
    controller = DriveController(CONTROL, CAMERA)

    driving = False
    prev_time = time.time()
    setup_display_windows()
    print("스페이스바: 주행 시작/정지 토글  |  q: 종료")

    consecutive_failures = 0
    max_consecutive_failures = 10

    try:
        while True:
            frame = camera.read()
            if frame is None:
                consecutive_failures += 1
                print(f"프레임을 읽지 못했습니다. ({consecutive_failures}/{max_consecutive_failures})")
                if consecutive_failures >= max_consecutive_failures:
                    print("카메라 연결을 확인하세요. CAMERA_INDEX를 0, 1, 2로 바꿔보세요.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

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