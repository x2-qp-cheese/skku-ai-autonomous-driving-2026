#!/usr/bin/env python3
# scripts/drive.py
# 차선 인식 기반 주행 (자체 완결형)
#   카메라 -> 차선 검출 -> 조향/속도 계산 -> 시리얼로 아두이노에 전송
#   펌웨어(vehicle_controller.ino)와 1:1 호환: "DRIVE <speed> <steer>\n" / "STOP\n"
#
# 실행:  (가상환경 활성화 후, 레포 루트에서)
#   python scripts/drive.py
# 종료:  영상 창에서 q
#
# 필요 패키지: opencv-python, numpy, pyserial  (requirements.txt에 포함)

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import serial

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.json"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_roi_mask(h, w, lane):
    """사다리꼴 관심영역(ROI) 마스크 생성"""
    pts = np.array([[
        (int(w * lane["roi_bottom_left_x"]),  h),
        (int(w * lane["roi_top_left_x"]),     int(h * lane["roi_top_y"])),
        (int(w * lane["roi_top_right_x"]),    int(h * lane["roi_top_y"])),
        (int(w * lane["roi_bottom_right_x"]), h),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def detect_lane_offset(frame, lane, roi_mask):
    """차선 중심의 좌우 치우침을 반환.
    반환: (offset, binary)
      offset: -1.0 ~ +1.0  (양수 = 차선 중심이 화면 오른쪽 -> 오른쪽으로 조향)
              차선을 못 찾으면 None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, lane["gray_threshold"], 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, roi_mask)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= lane["min_contour_area"]]

    h, w = gray.shape
    cx_img = w / 2.0
    if not big:
        return None, binary

    xs = []
    for c in big:
        M = cv2.moments(c)
        if M["m00"] > 0:
            xs.append(M["m10"] / M["m00"])
    if not xs:
        return None, binary

    lane_center = sum(xs) / len(xs)
    offset = (lane_center - cx_img) / cx_img   # -1 ~ +1
    return offset, binary


def main():
    cfg = load_config(CONFIG_PATH)
    cam, lane = cfg["camera"], cfg["lane"]
    ser_cfg, ctl = cfg["serial"], cfg["control"]

    port = ser_cfg["arduino_port"]
    if not port:
        print('ERROR: configs/default.json 의 serial.arduino_port 를 "COM6" 으로 설정하세요.')
        sys.exit(1)

    # 1) 시리얼 연결 (아두이노는 연결 시 리셋되므로 잠깐 대기)
    ser = serial.Serial(port, ser_cfg["baudrate"], timeout=ser_cfg["timeout_s"])
    time.sleep(2.0)
    ser.reset_input_buffer()
    print(f"[serial] {port} @ {ser_cfg['baudrate']} 연결됨")

    # 2) 카메라 (Windows는 CAP_DSHOW가 빠르게 열림)
    cap = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    if cam.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
    if not cap.isOpened():
        print("ERROR: 카메라를 열 수 없습니다. camera.index 를 확인하세요.")
        ser.close()
        sys.exit(1)

    base_speed   = int(ctl["base_speed"])
    max_steering = int(ctl["max_steering"])
    kp           = float(ctl["lane_kp"])
    lost_brake   = bool(ctl["lane_lost_brake"])

    def send(line):
        ser.write(line.encode("ascii"))

    roi_mask = None
    print("[run] 주행 시작 — 영상 창에서 q 누르면 정지/종료")
    try:
        send("PING\n")
        while True:
            ok, frame = cap.read()
            if not ok:
                send("STOP\n")
                continue

            h, w = frame.shape[:2]
            if roi_mask is None or roi_mask.shape != (h, w):
                roi_mask = make_roi_mask(h, w, lane)

            offset, dbg = detect_lane_offset(frame, lane, roi_mask)

            if offset is None:
                # 차선 분실
                if lost_brake:
                    send("STOP\n")
                    speed, steer = 0, 0
                    status = "LANE LOST -> STOP"
                else:
                    speed, steer = base_speed, 0
                    send(f"DRIVE {speed} {steer}\n")
                    status = "LANE LOST -> keep straight"
            else:
                # P 제어: 치우친 만큼 비례해서 조향
                steer = int(kp * offset)
                steer = max(-max_steering, min(max_steering, steer))
                speed = base_speed
                send(f"DRIVE {speed} {steer}\n")
                status = f"offset={offset:+.2f}  steer={steer:+d}  speed={speed}"

            # ---- 디버그 화면 ----
            vis = frame.copy()
            cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            cv2.putText(vis, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("drive", vis)
            cv2.imshow("lane_mask", dbg)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.02)  # 약 50Hz (펌웨어 안전 타임아웃 500ms보다 충분히 빠름)
    finally:
        send("STOP\n")
        time.sleep(0.1)
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        print("[exit] STOP 전송 후 종료")


if __name__ == "__main__":
    main()
