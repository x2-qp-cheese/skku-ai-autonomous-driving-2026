#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.control.serial_vehicle import SerialVehicleClient, SerialVehicleConfig
from skku_autocar.types import ControlCommand


WINDOW_NAME = "Keyboard Drive and Label Capture"
DEFAULT_OUTPUT_DIR = Path("data/raw/labeling")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the car with the keyboard and save raw camera frames for labeling"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--backend",
        choices=("auto", "default", "avfoundation", "dshow"),
        default="auto",
    )
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--auto-capture-interval", type=float, default=0.5)

    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--ready-timeout", type=float, default=3.0)
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--command-rate", type=float, default=20.0)

    parser.add_argument("--forward-speed", type=int, default=80)
    parser.add_argument("--reverse-speed", type=int, default=65)
    parser.add_argument("--speed-step", type=int, default=15)
    parser.add_argument("--max-speed", type=int, default=120)
    parser.add_argument("--max-reverse-speed", type=int, default=90)
    parser.add_argument("--steering-step", type=int, default=15)
    parser.add_argument("--max-steering", type=int, default=150)
    return parser.parse_args(argv)


def open_camera(cv2: Any, args: argparse.Namespace) -> Any:
    index = args.camera_index
    if args.backend == "auto":
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
    elif args.backend == "avfoundation":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif args.backend == "dshow":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if len(args.fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    return cap


def create_session(output_dir: Path) -> tuple[Path, Path]:
    session_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)
    return session_dir, images_dir


def save_frame(
    cv2: Any,
    frame: Any,
    images_dir: Path,
    metadata_writer: Any,
    metadata_file: Any,
    sequence: int,
    speed: int,
    steering: int,
    source: str,
    jpeg_quality: int,
) -> bool:
    captured_at = datetime.now()
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = "image_%06d_%s.jpg" % (sequence, timestamp)
    image_path = images_dir / filename
    quality = max(1, min(100, int(jpeg_quality)))
    saved = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not saved:
        return False
    metadata_writer.writerow(
        [filename, captured_at.isoformat(timespec="milliseconds"), speed, steering, source]
    )
    metadata_file.flush()
    print("saved:", image_path)
    return True


def draw_status(
    cv2: Any,
    frame: Any,
    speed: int,
    steering: int,
    braking: bool,
    auto_capture: bool,
    saved_images: int,
) -> Any:
    display = frame.copy()
    mode = "STOP" if braking else "DRIVE"
    color = (0, 0, 255) if braking else (0, 255, 0)
    lines = [
        "%s  speed=%d  steering=%d" % (mode, speed, steering),
        "saved=%d  auto=%s" % (saved_images, "ON" if auto_capture else "OFF"),
        "W/S speed  A/D steer  X coast  Z center  SPACE stop  C capture  T auto  Q quit",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (20, 42 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color if index == 0 else (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def current_command(speed: int, steering: int, braking: bool) -> ControlCommand:
    if braking:
        return ControlCommand.stop("keyboard_emergency_stop")
    return ControlCommand(speed=speed, steering=steering, reason="keyboard")


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required") from exc

    session_dir, images_dir = create_session(args.output_dir)
    metadata_path = session_dir / "metadata.csv"
    cap = open_camera(cv2, args)
    if not cap.isOpened():
        raise RuntimeError("camera index %d could not be opened" % args.camera_index)

    vehicle = None
    metadata_file = None
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("failed to read the first camera frame")

        if not args.no_serial:
            vehicle = SerialVehicleClient(
                SerialVehicleConfig(
                    port=args.serial_port,
                    baudrate=args.baudrate,
                    ready_timeout_s=args.ready_timeout,
                ),
                max_speed=args.max_speed,
                max_steering=args.max_steering,
            )
            vehicle.connect()
            print("serial connected:", vehicle.port)
        else:
            print("serial disabled")

        metadata_file = metadata_path.open("w", newline="", encoding="utf-8")
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(["filename", "captured_at", "speed", "steering", "source"])
        metadata_file.flush()

        frame_height, frame_width = frame.shape[:2]
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, frame_width, frame_height)
        print("session:", session_dir)
        print("focus the camera window before using W/A/S/D")

        speed = 0
        steering = 0
        braking = True
        auto_capture = False
        saved_images = 0
        last_command_at = 0.0
        next_auto_capture_at = time.monotonic()
        pending_frame = frame

        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("camera frame read failed")
            else:
                frame = pending_frame
                pending_frame = None

            raw_frame = frame.copy()
            now = time.monotonic()
            if auto_capture and now >= next_auto_capture_at:
                saved_images += 1
                if not save_frame(
                    cv2,
                    raw_frame,
                    images_dir,
                    metadata_writer,
                    metadata_file,
                    saved_images,
                    speed,
                    steering,
                    "auto",
                    args.jpeg_quality,
                ):
                    saved_images -= 1
                    raise RuntimeError("failed to save an auto-captured image")
                next_auto_capture_at = now + max(0.05, args.auto_capture_interval)

            if vehicle is not None and now - last_command_at >= 1.0 / max(1.0, args.command_rate):
                vehicle.send(current_command(speed, steering, braking))
                last_command_at = now

            display = draw_status(
                cv2, raw_frame, speed, steering, braking, auto_capture, saved_images
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("w"):
                if speed < 0:
                    speed = 0
                else:
                    speed = min(
                        args.max_speed,
                        args.forward_speed if speed == 0 else speed + args.speed_step,
                    )
                braking = False
            elif key == ord("s"):
                if speed > 0:
                    speed = 0
                else:
                    speed = max(
                        -args.max_reverse_speed,
                        -args.reverse_speed if speed == 0 else speed - args.speed_step,
                    )
                braking = False
            elif key == ord("a"):
                steering = max(-args.max_steering, steering - args.steering_step)
                braking = False
            elif key == ord("d"):
                steering = min(args.max_steering, steering + args.steering_step)
                braking = False
            elif key == ord("x"):
                speed = 0
                braking = False
            elif key == ord("z"):
                steering = 0
                braking = False
            elif key == ord(" "):
                speed = 0
                steering = 0
                braking = True
                if vehicle is not None:
                    vehicle.stop("keyboard_emergency_stop")
                    last_command_at = now
            elif key == ord("c"):
                saved_images += 1
                if not save_frame(
                    cv2,
                    raw_frame,
                    images_dir,
                    metadata_writer,
                    metadata_file,
                    saved_images,
                    speed,
                    steering,
                    "manual",
                    args.jpeg_quality,
                ):
                    saved_images -= 1
                    raise RuntimeError("failed to save a captured image")
            elif key == ord("t"):
                auto_capture = not auto_capture
                next_auto_capture_at = now
                print("auto capture:", "ON" if auto_capture else "OFF")
    finally:
        if vehicle is not None:
            try:
                vehicle.stop("keyboard_capture_shutdown")
            except Exception as exc:
                print("warning: final serial stop failed:", exc, file=sys.stderr)
            vehicle.close()
        if metadata_file is not None:
            metadata_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print("images:", images_dir)

    return 0


def main(argv: Optional[list] = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print("error:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
