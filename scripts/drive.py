import cv2
import numpy as np
import sys
import time
import serial

# =========================================================
# 시리얼 (아두이노 메가) 설정
# =========================================================

WINDOWS_SERIAL_PORT = "COM4"
SERIAL_PORT = WINDOWS_SERIAL_PORT if sys.platform.startswith("win") else None
SERIAL_BAUD = 115200

# =========================================================
# 주행 제어 파라미터
# =========================================================

# 기본 전진 속도. 90이 바닥에서 차를 못 미는 것 같으면 천천히 올려보기.
# (210~220에서 드라이버가 탔으니 절대 거기 근처로는 가지 말 것)
BASE_SPEED = 70

# 곡선/조향 클 때 감속 하한. 너무 낮으면 바닥에서 안 움직임.
MIN_SPEED = 90

MAX_STEER = 130
STEER_GAIN = 2
STEER_SIGN = 1.0
STEER_SMOOTH = 0.5
LOST_FRAMES_BEFORE_STOP = 8

# =========================================================
# 카메라 설정
# =========================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
SHOW_RAW_MASK = False
DRIVE_WINDOW = "Drive"
LANE_MASK_WINDOW = "ROI Lane Mask"
RAW_MASK_WINDOW = "Raw ROI Mask"

# =========================================================
# 차선 검출 설정
# =========================================================

# --- top-hat (빛 반사 제거 핵심) ---
# 이 커널보다 '두꺼운' 밝은 영역(바닥 반사 덩어리)은 제거되고,
# 이보다 얇은 밝은 선(차선)만 남는다.
# 차선이 통째로 사라지면 값을 키우고(예: 35), 반사가 덜 지워지면 줄인다(예: 15).
TOPHAT_KERNEL_SIZE = 25

# top-hat 결과를 이진화하는 임계값. 차선이 잘 안 잡히면 낮추고, 잡티 많으면 높임.
TOPHAT_THRESHOLD = 20
WHITE_L_MIN = 185
WHITE_S_MAX = 75
BRIGHT_THRESHOLD = 180
BRIGHT_PERCENTILE = 92
BRIGHT_MARGIN = 12
STRICT_BRIGHT_PERCENTILE = 97
MAX_ROI_WHITE_RATIO = 0.18

# --- 컨투어 크기 필터 ---
MIN_CONTOUR_AREA = 120
MAX_LANE_PARTS = 2

# --- 모양 필터 (반사 덩어리 제거 핵심) ---
# 차선은 '가늘고 길다'. 길이/폭 비율이 이 값 이상인 것만 차선으로 인정.
# 반사 덩어리는 뭉툭해서 비율이 낮아 걸러진다.
# 너무 빡세서 차선까지 지워지면 1.8 정도로 낮추기.
MIN_ELONGATION = 1.6
MIN_CONTOUR_HEIGHT = 25
MIN_CONTOUR_BOTTOM_Y = 0.55
MAX_FILL_RATIO = 0.96
MAX_ANGLE_FROM_VERTICAL = 78.0
LANE_LOOKAHEAD_Y = 0.82
ASSUMED_HALF_LANE_WIDTH = 0.25

# =========================================================
# ROI 설정
# =========================================================

ROI_BOTTOM_LEFT_X = 0.08
ROI_BOTTOM_RIGHT_X = 0.92
ROI_BOTTOM_Y = 0.96
ROI_TOP_LEFT_X = 0.28
ROI_TOP_RIGHT_X = 0.72
ROI_TOP_Y = 0.55


def make_roi_mask(frame_shape):
    h, w = frame_shape[:2]
    roi_points = np.array([
        [
            (int(w * ROI_BOTTOM_LEFT_X), int(h * ROI_BOTTOM_Y)),
            (int(w * ROI_BOTTOM_RIGHT_X), int(h * ROI_BOTTOM_Y)),
            (int(w * ROI_TOP_RIGHT_X), int(h * ROI_TOP_Y)),
            (int(w * ROI_TOP_LEFT_X), int(h * ROI_TOP_Y))
        ]
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, roi_points, 255)
    return mask, roi_points


def detect_lane(frame):
    """
    빛 반사에 강한 차선 검출.
    1. grayscale + blur
    2. white top-hat + HLS 흰색 필터로 흰 차선 후보 생성
    3. 밝기 조건을 같이 걸어 어두운 잡음 제거
    4. ROI 적용
    5. 작은 노이즈 정리
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_mask, roi_points = make_roi_mask(frame.shape)

    # --- 핵심: white top-hat ---
    th_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE)
    )
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, th_kernel)

    _, tophat_mask = cv2.threshold(
        tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(
        hls,
        np.array([0, WHITE_L_MIN, 0], dtype=np.uint8),
        np.array([180, 255, WHITE_S_MAX], dtype=np.uint8),
    )

    roi_values = blurred[roi_mask > 0]
    if len(roi_values) > 0:
        dynamic_bright = int(np.percentile(roi_values, BRIGHT_PERCENTILE)) + BRIGHT_MARGIN
        bright_threshold = min(245, max(BRIGHT_THRESHOLD, dynamic_bright))
    else:
        bright_threshold = BRIGHT_THRESHOLD

    _, bright_mask = cv2.threshold(
        blurred, bright_threshold, 255, cv2.THRESH_BINARY
    )
    color_mask = cv2.bitwise_and(white_mask, bright_mask)
    binary_mask = cv2.bitwise_or(tophat_mask, color_mask)

    lane_mask = cv2.bitwise_and(binary_mask, roi_mask)

    roi_area = max(1, cv2.countNonZero(roi_mask))
    if cv2.countNonZero(lane_mask) / float(roi_area) > MAX_ROI_WHITE_RATIO:
        strict_bright = int(np.percentile(roi_values, STRICT_BRIGHT_PERCENTILE)) if len(roi_values) > 0 else 230
        strict_bright = min(250, max(bright_threshold, strict_bright))
        _, strict_bright_mask = cv2.threshold(
            blurred, strict_bright, 255, cv2.THRESH_BINARY
        )
        strict_color_mask = cv2.bitwise_and(white_mask, strict_bright_mask)
        lane_mask = cv2.bitwise_and(
            cv2.bitwise_or(tophat_mask, strict_color_mask),
            roi_mask,
        )

    # 작은 노이즈 정리 + 끊긴 선 연결
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return lane_mask, roi_points


def select_lane_contours(lane_mask):
    """
    크기 + 모양 + 방향으로 컨투어를 거른다.
    흰색 잡음 중 차선처럼 길고, 아래쪽 ROI에 걸치고, 세로 방향 성분이 큰 것만 남긴다.
    """
    contours, _ = cv2.findContours(
        lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    frame_h, frame_w = lane_mask.shape[:2]
    frame_cx = frame_w / 2.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h < MIN_CONTOUR_HEIGHT:
            continue
        if y + h < frame_h * MIN_CONTOUR_BOTTOM_Y:
            continue

        fill_ratio = area / float(max(1, w * h))
        if fill_ratio > MAX_FILL_RATIO:
            continue

        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        if short_side < 1:
            continue

        elongation = long_side / short_side
        if elongation < MIN_ELONGATION:
            continue

        line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(line[0][0])
        vy = float(line[1][0])
        angle_from_vertical = np.degrees(np.arctan2(abs(vx), abs(vy)))
        if angle_from_vertical > MAX_ANGLE_FROM_VERTICAL:
            continue

        bottom_weight = (y + h) / float(frame_h)
        side_weight = abs(cx - frame_cx) / frame_cx
        score = area * elongation * bottom_weight * (1.0 + side_weight)
        candidates.append((score, cx, contour))

    if not candidates:
        return []

    left = [item for item in candidates if item[1] < frame_cx]
    right = [item for item in candidates if item[1] >= frame_cx]
    selected = []
    for group in (left, right):
        if group:
            selected.append(max(group, key=lambda item: item[0])[2])

    if not selected:
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [item[2] for item in candidates[:MAX_LANE_PARTS]]

    return selected[:MAX_LANE_PARTS]


def make_selected_lane_mask(mask_shape, lane_contours):
    filtered = np.zeros(mask_shape, dtype=np.uint8)
    if lane_contours:
        cv2.drawContours(filtered, lane_contours, -1, 255, thickness=cv2.FILLED)
    return filtered


def contour_x_at_y(contour, target_y):
    points = contour.reshape(-1, 2)
    if len(points) == 0:
        return None

    distances = np.abs(points[:, 1] - target_y)
    near_band = max(8, int(FRAME_HEIGHT * 0.04))
    near_points = points[distances <= near_band]
    if len(near_points) == 0:
        nearest_count = min(8, len(points))
        near_points = points[np.argsort(distances)[:nearest_count]]

    return float(np.mean(near_points[:, 0]))


def compute_lane_center_x(lane_contours):
    if not lane_contours:
        return None

    centers = []
    target_y = FRAME_HEIGHT * LANE_LOOKAHEAD_Y
    for contour in lane_contours:
        cx = contour_x_at_y(contour, target_y)
        if cx is None:
            continue
        area = cv2.contourArea(contour)
        centers.append((cx, area))

    if not centers:
        return None

    frame_cx = FRAME_WIDTH / 2.0
    left = [(cx, a) for cx, a in centers if cx < frame_cx]
    right = [(cx, a) for cx, a in centers if cx >= frame_cx]

    def group_center(group):
        total = sum(a for _, a in group)
        if total == 0:
            return None
        return sum(cx * a for cx, a in group) / total

    left_x = group_center(left)
    right_x = group_center(right)

    if left_x is not None and right_x is not None:
        return (left_x + right_x) / 2.0
    elif left_x is not None:
        return left_x + FRAME_WIDTH * ASSUMED_HALF_LANE_WIDTH
    else:
        return right_x - FRAME_WIDTH * ASSUMED_HALF_LANE_WIDTH


# =========================================================
# 시리얼 헬퍼
# =========================================================

def find_serial_port():
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


def open_serial():
    port = SERIAL_PORT or find_serial_port()
    if not port:
        raise RuntimeError("아두이노 시리얼 포트를 찾지 못했습니다. scripts/list_serial_ports.py로 포트를 확인하세요.")

    ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
    time.sleep(2.0)
    start = time.time()
    while time.time() - start < 3.0:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("[ARDUINO]", line)
        if "READY" in line:
            break
    return ser


def send_drive(ser, speed, steer):
    speed = int(max(0, min(255, speed)))
    steer = int(max(-MAX_STEER, min(MAX_STEER, steer)))
    ser.write(f"DRIVE {speed} {steer}\n".encode())


def send_stop(ser):
    ser.write(b"STOP\n")


def setup_display_windows():
    cv2.namedWindow(DRIVE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DRIVE_WINDOW, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.moveWindow(DRIVE_WINDOW, 40, 80)

    cv2.namedWindow(LANE_MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LANE_MASK_WINDOW, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.moveWindow(LANE_MASK_WINDOW, FRAME_WIDTH + 80, 80)

    if SHOW_RAW_MASK:
        cv2.namedWindow(RAW_MASK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RAW_MASK_WINDOW, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.moveWindow(RAW_MASK_WINDOW, FRAME_WIDTH + 80, FRAME_HEIGHT + 140)


# =========================================================
# 메인 주행 루프
# =========================================================

def main():
    try:
        ser = open_serial()
        print("시리얼 연결 완료:", ser.port)
    except Exception as e:
        print("시리얼 연결 실패:", e)
        print("포트 설정과 시리얼 모니터/다른 프로그램 점유 여부를 확인하세요.")
        return

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("카메라를 열 수 없습니다. CAMERA_INDEX를 0, 1, 2로 바꿔보세요.")
        send_stop(ser)
        ser.close()
        return

    prev_time = time.time()
    smoothed_steer = 0.0
    lost_count = 0
    driving = False
    setup_display_windows()
    print("스페이스바: 주행 시작/정지 토글  |  q: 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            lane_mask, roi_points = detect_lane(frame)
            lane_contours = select_lane_contours(lane_mask)
            selected_lane_mask = make_selected_lane_mask(lane_mask.shape, lane_contours)
            lane_center_x = compute_lane_center_x(lane_contours)

            result = frame.copy()
            cv2.polylines(result, roi_points, True, (255, 0, 0), 2)
            for c in lane_contours:
                cv2.drawContours(result, [c], -1, (0, 255, 0), 3)

            frame_center_x = FRAME_WIDTH / 2.0

            if lane_center_x is not None:
                lost_count = 0
                error = lane_center_x - frame_center_x
                raw_steer = STEER_SIGN * STEER_GAIN * error
                smoothed_steer = (
                    STEER_SMOOTH * smoothed_steer
                    + (1 - STEER_SMOOTH) * raw_steer
                )
                steer_cmd = int(max(-MAX_STEER, min(MAX_STEER, smoothed_steer)))
                speed_cmd = int(
                    BASE_SPEED
                    - (BASE_SPEED - MIN_SPEED) * (abs(steer_cmd) / MAX_STEER)
                )

                cv2.line(result,
                         (int(lane_center_x), FRAME_HEIGHT),
                         (int(lane_center_x), int(FRAME_HEIGHT * ROI_TOP_Y)),
                         (0, 0, 255), 2)
                cv2.line(result,
                         (int(frame_center_x), FRAME_HEIGHT),
                         (int(frame_center_x), int(FRAME_HEIGHT * ROI_TOP_Y)),
                         (255, 255, 0), 1)

                if driving:
                    send_drive(ser, speed_cmd, steer_cmd)
                status = f"DRIVE s={speed_cmd} st={steer_cmd}"
            else:
                lost_count += 1
                if driving and lost_count >= LOST_FRAMES_BEFORE_STOP:
                    send_stop(ser)
                    status = "LANE LOST -> STOP"
                else:
                    status = f"LANE LOST ({lost_count})"

            mode = "RUN" if driving else "PAUSE"
            color = (0, 255, 0) if driving else (0, 0, 255)
            cv2.putText(result, mode, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(result, status, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            cv2.putText(result, f"FPS: {fps:.1f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(DRIVE_WINDOW, result)
            cv2.imshow(LANE_MASK_WINDOW, selected_lane_mask)
            if SHOW_RAW_MASK:
                cv2.imshow(RAW_MASK_WINDOW, lane_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                driving = not driving
                if not driving:
                    send_stop(ser)
                    smoothed_steer = 0.0
                print("주행:", "ON" if driving else "OFF")

    finally:
        send_stop(ser)
        time.sleep(0.2)
        ser.close()
        cap.release()
        cv2.destroyAllWindows()
        print("정지 명령 전송 후 종료.")


if __name__ == "__main__":
    main()
