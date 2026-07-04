from ultralytics import YOLO
import numpy as np
import cv2
import sys
import time

CLASS_CENTER = 0
CLASS_LEFT = 1
CLASS_RIGHT = 2

CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def get_lane_x_position(mask_xy, frame_height, y_target_ratio=0.8):
    y_target = frame_height * y_target_ratio
    points = np.array(mask_xy)
    closest_idx = np.argsort(np.abs(points[:, 1] - y_target))[:5]
    x_values = points[closest_idx, 0]
    return np.mean(x_values)

def compute_lane_center(left_x, center_x, right_x, frame_width, lane_width_px=None):
    car_center = frame_width / 2
    if left_x and center_x and right_x:
        if car_center < center_x:
            return {'current_lane': 'lane1', 'target_center': (left_x + center_x) / 2}
        else:
            return {'current_lane': 'lane2', 'target_center': (center_x + right_x) / 2}
    if left_x and center_x and not right_x:
        return {'current_lane': 'lane1', 'target_center': (left_x + center_x) / 2}
    if center_x and right_x and not left_x:
        return {'current_lane': 'lane2', 'target_center': (center_x + right_x) / 2}
    if left_x and not center_x and not right_x:
        estimated_lane_width = lane_width_px or frame_width * 0.3
        return {'current_lane': 'lane1', 'target_center': left_x + estimated_lane_width / 2}
    if right_x and not center_x and not left_x:
        estimated_lane_width = lane_width_px or frame_width * 0.3
        return {'current_lane': 'lane2', 'target_center': right_x - estimated_lane_width / 2}
    return {'current_lane': None, 'target_center': None}


def main():
    model = YOLO('best_v2.pt')

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
        return

    print("q: 종료")
    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            h, w = frame.shape[:2]
            results = model(frame, conf=0.5, verbose=False)

            left_x, center_x, right_x = None, None, None

            for r in results:
                if r.masks is not None:
                    for mask, cls in zip(r.masks.xy, r.boxes.cls):
                        cls_id = int(cls)
                        x_pos = get_lane_x_position(mask, h)
                        if cls_id == CLASS_LEFT:
                            left_x = x_pos
                        elif cls_id == CLASS_CENTER:
                            center_x = x_pos
                        elif cls_id == CLASS_RIGHT:
                            right_x = x_pos

            lane_info = compute_lane_center(left_x, center_x, right_x, w)

            annotated_frame = results[0].plot()

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            center_str = f"{lane_info['target_center']:.1f}" if lane_info['target_center'] is not None else "N/A"
            lane_text = f"Lane: {lane_info['current_lane']}  Center: {center_str}"
            cv2.putText(annotated_frame, lane_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 차량 중심선 (파란색)
            cv2.line(annotated_frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)

            # 목표 중심점 (빨간 원)
            if lane_info['target_center'] is not None:
                tx = int(lane_info['target_center'])
                cv2.circle(annotated_frame, (tx, int(h * 0.8)), 8, (0, 0, 255), -1)

            cv2.imshow('Lane Detection (YOLO)', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("종료.")


if __name__ == "__main__":
    main()
