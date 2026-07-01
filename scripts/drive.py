import cv2
import numpy as np
import time
import serial

# =========================================================
# 시리얼 (아두이노 메가) 설정
# =========================================================

SERIAL_PORT = "COM6"
SERIAL_BAUD = 115200

# =========================================================
# 주행 제어 파라미터
# =========================================================

# 기본 전진 속도. 90이 바닥에서 차를 못 미는 것 같으면 천천히 올려보기.
# (210~220에서 드라이버가 탔으니 절대 거기 근처로는 가지 말 것)
BASE_SPEED = 50

# 곡선/조향 클 때 감속 하한. 너무 낮으면 바닥에서 안 움직임.
MIN_SPEED = 90

MAX_STEER = 150
STEER_GAIN = 0.85
STEER_SIGN = 1.0
STEER_SMOOTH = 0.5
LOST_FRAMES_BEFORE_STOP = 8

# =========================================================
# 카메라 설정
# =========================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# =========================================================
# 차선 검출 설정
# =========================================================

# --- top-hat (빛 반사 제거 핵심) ---
# 이 커널보다 '두꺼운' 밝은 영역(바닥 반사 덩어리)은 제거되고,
# 이보다 얇은 밝은 선(차선)만 남는다.
# 차선이 통째로 사라지면 값을 키우고(예: 35), 반사가 덜 지워지면 줄인다(예: 15).
TOPHAT_KERNEL_SIZE = 25

# top-hat 결과를 이진화하는 임계값. 차선이 잘 안 잡히면 낮추고, 잡티 많으면 높임.
TOPHAT_THRESHOLD = 30

# --- 컨투어 크기 필터 ---
MIN_CONTOUR_AREA = 250
MAX_LANE_PARTS = 4

# --- 모양 필터 (반사 덩어리 제거 핵심) ---
# 차선은 '가늘고 길다'. 길이/폭 비율이 이 값 이상인 것만 차선으로 인정.
# 반사 덩어리는 뭉툭해서 비율이 낮아 걸러진다.
# 너무 빡세서 차선까지 지워지면 1.8 정도로 낮추기.
MIN_ELONGATION = 2.2

# =========================================================
# ROI 설정
# =========================================================

ROI_BOTTOM_LEFT_X = 0
ROI_BOTTOM_RIGHT_X = 1
ROI_TOP_LEFT_X = 0.2
ROI_TOP_RIGHT_X = 0.8
ROI_TOP_Y = 0.5


def make_roi_mask(frame_shape):
    h, w = frame_shape[:2]
    roi_points = np.array([
        [
            (int(w * ROI_BOTTOM_LEFT_X), h),
            (int(w * ROI_BOTTOM_RIGHT_X), h),
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
    2. white top-hat 으로 넓은 반사 영역 제거, 얇은 밝은 선만 강조
    3. 이진화
    4. ROI 적용
    5. 작은 노이즈 정리
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 핵심: white top-hat ---
    th_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE)
    )
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, th_kernel)

    # top-hat 결과 이진화 (넓은 반사는 이미 거의 0으로 죽어 있음)
    _, binary_mask = cv2.threshold(
        tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    # ROI
    roi_mask, roi_points = make_roi_mask(frame.shape)
    lane_mask = cv2.bitwise_and(binary_mask, roi_mask)

    # 작은 노이즈 정리 + 끊긴 선 연결
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return lane_mask, roi_points


def select_lane_contours(lane_mask):
    """
    크기 + 모양(가늘고 긴 것만)으로 컨투어를 거른다.
    반사 덩어리는 뭉툭해서 elongation 필터에서 떨어진다.
    """
    contours, _ = cv2.findContours(
        lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        # 최소 회전 사각형으로 진짜 길쭉함 측정 (기울어진 차선도 OK)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        if short_side < 1:
            continue

        elongation = long_side / short_side
        if elongation < MIN_ELONGATION:
            # 뭉툭한 반사 덩어리 -> 버림
            continue

        valid_contours.append(contour)

    valid_contours.sort(key=cv2.contourArea, reverse=True)
    return valid_contours[:MAX_LANE_PARTS]


def compute_lane_center_x(lane_contours):
    if not lane_contours:
        return None

    total_area = 0.0
    weighted_x = 0.0
    for contour in lane_contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        area = cv2.contourArea(contour)
        weighted_x += cx * area
        total_area += area

    if total_area == 0:
        return None
    return weighted_x / total_area


# =========================================================
# 시리얼 헬퍼
# =========================================================

def open_serial():
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
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


# =========================================================
# 메인 주행 루프
# =========================================================

def main():
    try:
        ser = open_serial()
        print("시리얼 연결 완료:", SERIAL_PORT)
    except Exception as e:
        print("시리얼 연결 실패:", e)
        print("COM6가 맞는지, 시리얼 모니터/다른 프로그램이 점유 중인지 확인하세요.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
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

            cv2.imshow("Drive", result)
            cv2.imshow("ROI Lane Mask", lane_mask)

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