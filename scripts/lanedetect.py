import cv2
import numpy as np
import time

# =========================
# 카메라 설정
# =========================

CAMERA_INDEX = 1

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# =========================
# 흑백 threshold 설정
# =========================

# 흑백 이미지에서 이 값보다 밝으면 흰색 차선으로 판단
# 차선이 잘 안 잡히면 낮추고, 너무 많이 잡히면 높이면 됨
GRAY_THRESHOLD = 175

# 너무 작은 흰색 조각 제거
MIN_CONTOUR_AREA = 700

# 가장 큰 차선 조각 몇 개만 표시할지
MAX_LANE_PARTS = 4

# =========================
# ROI 설정
# =========================
# 화면 전체가 아니라 중앙 하단 도로 부분만 검사
# 필요 없는 벽, 의자, 커튼 쪽을 제거하기 위한 영역

ROI_BOTTOM_LEFT_X = 0
ROI_BOTTOM_RIGHT_X = 1

ROI_TOP_LEFT_X = 0.1
ROI_TOP_RIGHT_X = 0.9

ROI_TOP_Y = 0.6


def make_roi_mask(frame_shape):
    """
    차선을 찾을 영역만 남기는 ROI mask 생성 함수.
    중앙 하단 사다리꼴 영역만 흰색으로 만들고,
    나머지 영역은 검은색으로 만든다.
    """
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


def detect_lane_by_grayscale(frame):
    """
    흑백 처리 기반 차선 검출 함수.

    처리 순서:
    1. 원본 BGR 이미지를 grayscale로 변환
    2. blur로 노이즈 완화
    3. threshold로 밝은 흰색 차선만 남김
    4. ROI를 적용해서 필요한 바닥 영역만 남김
    5. morphology 연산으로 작은 노이즈 제거
    """

    # 1. 컬러 이미지를 흑백 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 작은 노이즈 제거를 위해 blur 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 밝은 부분만 흰색으로 남기는 이진화 처리
    # GRAY_THRESHOLD보다 밝은 픽셀은 255, 나머지는 0
    _, binary_mask = cv2.threshold(
        blurred,
        GRAY_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # 4. ROI 적용
    roi_mask, roi_points = make_roi_mask(frame.shape)
    lane_mask = cv2.bitwise_and(binary_mask, roi_mask)

    # 5. 노이즈 제거 및 끊긴 차선 연결
    kernel = np.ones((5, 5), np.uint8)

    # 작은 점 제거
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 끊어진 흰색 차선 연결
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return gray, binary_mask, lane_mask, roi_points


def select_lane_contours(lane_mask):
    """
    흑백 mask에서 contour를 찾고,
    너무 작은 조각은 제거한 뒤,
    큰 차선 조각만 선택한다.
    """
    contours, _ = cv2.findContours(
        lane_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # 너무 작은 영역은 노이즈로 판단
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # 폭이나 높이가 너무 작으면 제거
        if w < 15 or h < 10:
            continue

        valid_contours.append(contour)

    # 큰 contour부터 정렬
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    # 가장 큰 몇 개만 사용
    return valid_contours[:MAX_LANE_PARTS]


def draw_result(frame, lane_contours, roi_points):
    """
    최종 차선 인식 결과를 그리는 함수.
    """
    result = frame.copy()

    # ROI 영역 표시
    # 필요 없으면 이 줄을 주석 처리해도 됨
    cv2.polylines(result, roi_points, True, (255, 0, 0), 2)

    # 검출된 차선 contour 표시
    for contour in lane_contours:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)

    return result


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
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

        # 흑백 기반 차선 검출
        gray, binary_mask, lane_mask, roi_points = detect_lane_by_grayscale(frame)

        # 필요한 차선 contour만 선택
        lane_contours = select_lane_contours(lane_mask)

        # 결과 화면 생성
        result = draw_result(frame, lane_contours, roi_points)

        # FPS 계산
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(
            result,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 화면 출력
        cv2.imshow("Original with Lane Detection", result)
        cv2.imshow("Gray Image", gray)
        cv2.imshow("Binary Mask", binary_mask)
        cv2.imshow("ROI Lane Mask", lane_mask)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()