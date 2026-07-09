"""
YOLO + 카메라 점검 스크립트 (아두이노/시리얼 불필요)

목적: 모델이 lane-center / lane-side 마스크를 제대로 내보내는지 확인.
- 시작 시 모델 task와 클래스 이름을 출력
- 매 프레임 검출된 boxes/masks/클래스 id를 터미널에 출력
- Camera 창: 원본 + 검출된 lane 마스크 오버레이(초록)
- Mask 창: lane 마스크만 흰색으로

조작:  q 종료  |  s 현재 프레임 저장(debug_frame.png)
"""

import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 설정 — 필요에 맞게 여기만 바꾸세요
# =========================================================

MODEL_PATH = "scripts/best_v2.pt"
CAMERA_INDEX = 0           # 프레임이 안 나오면 0, 1, 2 로 바꿔보세요
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
YOLO_CONF = 0.25           # 검출이 안 되면 더 낮춰서(예: 0.15) 테스트

CLASS_LANE_CENTER = 3      # names 확인 후 다르면 수정
CLASS_LANE_SIDE = 4

# 클래스별 오버레이 색(BGR)
CLASS_COLORS = {
    CLASS_LANE_CENTER: (255, 0, 255),   # 자홍
    CLASS_LANE_SIDE: (0, 255, 0),       # 초록
}


# =========================================================
# 카메라
# =========================================================

def open_camera(index: int):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        cap.release()
        return None

    # 워밍업: 초기 빈 프레임 버리기
    for _ in range(5):
        cap.read()
    return cap


def probe_camera_indices(max_index: int = 4) -> None:
    """어느 인덱스가 프레임을 주는지 스캔."""
    print("=== 카메라 인덱스 스캔 ===")
    for i in range(max_index):
        cap = open_camera(i)
        if cap is None:
            print(f"index {i}: 열기 실패")
            continue
        ok, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"index {i}: opened=True read_ok={ok} shape={shape}")
        cap.release()
    print("========================")


# =========================================================
# 메인
# =========================================================

def main() -> None:
    # 인자로 'probe'를 주면 인덱스만 스캔하고 종료
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe_camera_indices()
        return

    print("모델 로딩:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("[YOLO] task:", model.task, "| names:", model.names)
    if model.task != "segment":
        print("!! 경고: 이 모델의 task가 'segment'가 아닙니다. "
              "세그멘테이션 마스크가 나오지 않으므로 차선 마스크를 얻을 수 없습니다.")

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        print(f"카메라를 열 수 없습니다 (index={CAMERA_INDEX}).")
        print("인덱스 스캔을 하려면:  python check_yolo.py probe")
        return

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", FRAME_WIDTH, FRAME_HEIGHT)
    cv2.namedWindow("Lane Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Mask", FRAME_WIDTH, FRAME_HEIGHT)

    print("실행 중 —  q: 종료  |  s: 프레임 저장")

    prev_time = time.time()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("프레임을 읽지 못했습니다.")
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            h, w = frame.shape[:2]
            frame_idx += 1

            results = model(frame, conf=YOLO_CONF, verbose=False)

            overlay = frame.copy()
            lane_mask = np.zeros((h, w), dtype=np.uint8)

            n_boxes_total = 0
            any_masks = False
            all_cls = []
            lane_hits = 0

            for r in results:
                any_masks = any_masks or (r.masks is not None)
                if r.boxes is not None:
                    n_boxes_total += len(r.boxes)
                    all_cls.extend(int(c) for c in r.boxes.cls)

                if r.masks is None or r.boxes is None:
                    continue

                for mask_xy, cls, conf in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
                    cls_id = int(cls)
                    if cls_id not in (CLASS_LANE_CENTER, CLASS_LANE_SIDE):
                        continue
                    pts = np.array(mask_xy, dtype=np.int32)
                    if len(pts) < 3:
                        continue

                    lane_hits += 1
                    color = CLASS_COLORS.get(cls_id, (0, 255, 255))

                    # 흰색 마스크 창용
                    cv2.fillPoly(lane_mask, [pts], 255)

                    # 오버레이: 반투명 채움 + 외곽선 + 라벨
                    filled = overlay.copy()
                    cv2.fillPoly(filled, [pts], color)
                    overlay = cv2.addWeighted(filled, 0.4, overlay, 0.6, 0)
                    cv2.polylines(overlay, [pts], True, color, 2)

                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    name = model.names.get(cls_id, str(cls_id))
                    cv2.putText(overlay, f"{name} {float(conf):.2f}", (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 터미널 로그 (매 프레임)
            # print(f"[{frame_idx:04d}] boxes={n_boxes_total} masks={any_masks} "
            #       f"cls={all_cls} lane_hits={lane_hits}")

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            status = f"boxes={n_boxes_total} lane={lane_hits} FPS={fps:.1f}"
            color = (0, 255, 0) if lane_hits > 0 else (0, 0, 255)
            cv2.putText(overlay, status, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if not any_masks:
                cv2.putText(overlay, "NO MASKS (segment model?)", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Camera", overlay)
            cv2.imshow("Lane Mask", lane_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite("debug_frame.png", frame)
                cv2.imwrite("debug_overlay.png", overlay)
                print("저장: debug_frame.png, debug_overlay.png")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("종료.")


if __name__ == "__main__":
    main()