import cv2
import numpy as np
import time

# Logitech 카메라가 기본 카메라면 0, 노트북 내장 카메라가 있으면 보통 1일 수 있음
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 화면에서 바닥 부분만 볼 비율
# 0.45면 위쪽 45%는 버리고 아래쪽 55%만 차선 탐지
ROI_TOP_RATIO = 0.45

# 흰색 차선 검출 HSV 범위
# 사진처럼 흰 선이 밝으면 V_MIN을 160~200 사이에서 조절
WHITE_S_MAX = 90
WHITE_V_MIN = 160

MIN_CONTOUR_AREA = 300


def make_roi_mask(frame_shape):
    h, w = frame_shape[:2]

    roi_top = int(h * ROI_TOP_RATIO)

    # 바닥 영역만 남기는 사각형 ROI
    polygon = np.array([
        [(0, h), (w, h), (w, roi_top), (0, roi_top)]
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)

    return mask


def detect_white_lane_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, WHITE_V_MIN], dtype=np.uint8)
    upper_white = np.array([180, WHITE_S_MAX, 255], dtype=np.uint8)

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 바닥 영역 ROI 적용
    roi_mask = make_roi_mask(frame.shape)
    white_mask = cv2.bitwise_and(white_mask, roi_mask)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)

    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return white_mask


def draw_lane_contours(frame, lane_mask):
    output = frame.copy()

    contours, _ = cv2.findContours(
        lane_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # 너무 작은 점, 반사광 같은 것 제거
        if w < 10 or h < 5:
            continue

        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return output


def draw_hough_lines(frame, lane_mask):
    output = frame.copy()

    edges = cv2.Canny(lane_mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=25
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            length = np.hypot(x2 - x1, y2 - y1)
            if length < 40:
                continue

            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return output


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Logitech 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 일부 Logitech 카메라에서 FPS 안정화에 도움
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("카메라를 열 수 없습니다. CAMERA_INDEX를 0, 1, 2로 바꿔보세요.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽지 못했습니다.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        lane_mask = detect_white_lane_mask(frame)

        contour_view = draw_lane_contours(frame, lane_mask)
        hough_view = draw_hough_lines(contour_view, lane_mask)

        # FPS 표시
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(
            hough_view,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("Original", frame)
        cv2.imshow("Lane Mask", lane_mask)
        cv2.imshow("Lane Detection", hough_view)

        key = cv2.waitKey(1) & 0xFF

        # q 누르면 종료
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()