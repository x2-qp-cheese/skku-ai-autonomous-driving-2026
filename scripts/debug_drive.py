#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.control.serial_vehicle import SerialVehicleClient, SerialVehicleConfig
from skku_autocar.types import ControlCommand


WINDOW_NAME = "Debug Keyboard Drive"
DEFAULT_CAPTURE_DIR = "data/raw/debug_drive"


def main(argv: Optional[list] = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print("error:", exc, file=sys.stderr)
        return 1


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyboard-drive the Arduino car while saving camera images every second for labeling"
    )
    parser.add_argument("--camera", default="0", help="front camera index or video path; default=0")
    parser.add_argument(
        "--rear-camera",
        default=None,
        help="rear camera index or video path; when set, saves front and rear images together",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--backend",
        choices=("auto", "default", "avfoundation", "dshow"),
        default="auto",
        help="camera backend for numeric camera indexes",
    )

    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--ready-timeout", type=float, default=3.0)
    parser.add_argument("--no-serial", action="store_true", help="preview/capture without sending Arduino commands")
    parser.add_argument("--command-rate", type=float, default=50.0)

    parser.add_argument(
        "--control-style",
        choices=("direct", "step"),
        default="direct",
        help="direct: W/S/A/D immediately set target speed/steering; step: each key press increments",
    )
    parser.add_argument("--start-speed", type=int, default=100, help="W target speed in direct mode; first W jump in step mode")
    parser.add_argument("--reverse-speed", type=int, default=60, help="first S press from zero jumps to this reverse speed")
    parser.add_argument("--speed-step", type=int, default=15)
    parser.add_argument("--steering-step", type=int, default=15)
    parser.add_argument("--steering-command", type=int, default=100, help="A/D target steering in direct mode")
    parser.add_argument("--max-speed", type=int, default=100)
    parser.add_argument("--max-reverse-speed", type=int, default=120)
    parser.add_argument("--max-steering", type=int, default=150)
    parser.add_argument(
        "--hold-to-run",
        choices=("on", "off"),
        default="on",
        help="when on, stop automatically if W/A/S/D input is not repeated before --key-timeout",
    )
    parser.add_argument(
        "--key-timeout",
        type=float,
        default=0.55,
        help="seconds without a movement key before automatic STOP in --hold-to-run mode",
    )

    parser.add_argument("--capture", choices=("on", "off"), default="on")
    parser.add_argument("--capture-interval", type=float, default=1.0, help="seconds between auto-saved label images")
    parser.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    cv2 = load_cv2()
    front_cap = open_camera(cv2, args, args.camera)
    if not front_cap.isOpened():
        raise RuntimeError("front camera could not be opened: %s" % args.camera)

    ok, first_front_frame = front_cap.read()
    if not ok:
        front_cap.release()
        raise RuntimeError("failed to read first front camera frame")

    rear_cap = None
    first_rear_frame = None
    if args.rear_camera is not None:
        rear_cap = open_camera(cv2, args, args.rear_camera)
        if not rear_cap.isOpened():
            front_cap.release()
            raise RuntimeError("rear camera could not be opened: %s" % args.rear_camera)
        ok, first_rear_frame = rear_cap.read()
        if not ok:
            front_cap.release()
            rear_cap.release()
            raise RuntimeError("failed to read first rear camera frame")

    session_dir, image_dirs = create_session(resolve_path(args.capture_dir), rear_cap is not None)
    metadata_file = None
    vehicle = None

    try:
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
            print("serial disabled: preview/capture only")

        metadata_path = session_dir / "metadata.csv"
        metadata_file = metadata_path.open("w", newline="", encoding="utf-8")
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(["camera", "filename", "captured_at", "speed", "steering", "brake", "source"])
        metadata_file.flush()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        preview = compose_preview(cv2, first_front_frame, first_rear_frame)
        frame_h, frame_w = preview.shape[:2]
        cv2.resizeWindow(WINDOW_NAME, frame_w, frame_h)

        print("session:", session_dir)
        for name, directory in image_dirs.items():
            print("%s images:" % name, directory)
        print("keys: W/S speed | A/D steer | Z center | X speed0 | SPACE stop | C snapshot | T auto capture | Q quit")

        speed = 0
        steering = 0
        braking = True
        auto_capture = args.capture == "on"
        saved_images = 0
        last_command_at = 0.0
        last_motion_key_at = None
        next_capture_at = time.monotonic()
        pending_front_frame = first_front_frame
        pending_rear_frame = first_rear_frame

        while True:
            if pending_front_frame is None:
                ok, front_frame = front_cap.read()
                if not ok:
                    raise RuntimeError("front camera frame read failed")
            else:
                front_frame = pending_front_frame
                pending_front_frame = None

            rear_frame = None
            if rear_cap is not None:
                if pending_rear_frame is None:
                    ok, rear_frame = rear_cap.read()
                    if not ok:
                        raise RuntimeError("rear camera frame read failed")
                else:
                    rear_frame = pending_rear_frame
                    pending_rear_frame = None

            raw_frames = {"front": front_frame.copy()}
            if rear_frame is not None:
                raw_frames["rear"] = rear_frame.copy()
            now = time.monotonic()
            timed_out = hold_timeout_expired(args, last_motion_key_at, now)
            if timed_out:
                speed = 0
                steering = 0
                braking = True
                last_motion_key_at = None

            display = draw_overlay(
                cv2,
                compose_preview(cv2, raw_frames["front"], raw_frames.get("rear")),
                speed,
                steering,
                braking,
                auto_capture,
                saved_images,
                next_capture_at - now,
                args.hold_to_run == "on",
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            command_changed = timed_out
            now = time.monotonic()
            if key in (ord("q"), 27):
                break
            if key == ord("w"):
                speed = increase_speed(speed, args)
                braking = False
                last_motion_key_at = now
                command_changed = True
            elif key == ord("s"):
                speed = decrease_speed(speed, args)
                braking = False
                last_motion_key_at = now
                command_changed = True
            elif key == ord("a"):
                steering = steer_left(steering, args)
                braking = False
                last_motion_key_at = now
                command_changed = True
            elif key == ord("d"):
                steering = steer_right(steering, args)
                braking = False
                last_motion_key_at = now
                command_changed = True
            elif key == ord("z"):
                steering = 0
                braking = False
                last_motion_key_at = now
                command_changed = True
            elif key == ord("x"):
                speed = 0
                braking = False
                last_motion_key_at = now
                command_changed = True
            elif key == ord(" "):
                speed = 0
                steering = 0
                braking = True
                last_motion_key_at = None
                command_changed = True
            elif key == ord("c"):
                saved_images = save_label_images(
                    cv2,
                    raw_frames,
                    image_dirs,
                    metadata_writer,
                    metadata_file,
                    saved_images,
                    speed,
                    steering,
                    braking,
                    "manual",
                    args.jpeg_quality,
                )
            elif key == ord("t"):
                auto_capture = not auto_capture
                next_capture_at = now
                print("auto capture:", "ON" if auto_capture else "OFF")

            if command_changed and vehicle is not None:
                vehicle.send(current_command(speed, steering, braking))
                last_command_at = now

            if vehicle is not None and now - last_command_at >= 1.0 / max(1.0, args.command_rate):
                vehicle.send(current_command(speed, steering, braking))
                last_command_at = now

            if auto_capture and now >= next_capture_at:
                saved_images = save_label_images(
                    cv2,
                    raw_frames,
                    image_dirs,
                    metadata_writer,
                    metadata_file,
                    saved_images,
                    speed,
                    steering,
                    braking,
                    "auto",
                    args.jpeg_quality,
                )
                next_capture_at = now + max(0.05, args.capture_interval)
    finally:
        if vehicle is not None:
            try:
                vehicle.stop("debug_drive_shutdown")
            except Exception as exc:
                print("warning: final serial stop failed:", exc, file=sys.stderr)
            vehicle.close()
        if metadata_file is not None:
            metadata_file.close()
        front_cap.release()
        if rear_cap is not None:
            rear_cap.release()
        cv2.destroyAllWindows()
        for name, directory in image_dirs.items():
            print("saved %s images:" % name, directory)

    return 0


def open_camera(cv2: Any, args: argparse.Namespace, camera: str) -> Any:
    source: Any = int(camera) if str(camera).isdigit() else camera
    if not isinstance(source, int):
        return cv2.VideoCapture(source)
    if args.backend == "auto":
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
    elif args.backend == "avfoundation":
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    elif args.backend == "dshow":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if len(args.fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    return cap


def create_session(capture_dir: Path, dual_camera: bool) -> tuple[Path, Dict[str, Path]]:
    session_dir = capture_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    images_root = session_dir / "images"
    if dual_camera:
        image_dirs = {
            "front": images_root / "front",
            "rear": images_root / "rear",
        }
    else:
        image_dirs = {"front": images_root}
    for directory in image_dirs.values():
        directory.mkdir(parents=True, exist_ok=False)
    return session_dir, image_dirs


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def save_label_images(
    cv2: Any,
    frames: Dict[str, Any],
    image_dirs: Dict[str, Path],
    metadata_writer: Any,
    metadata_file: Any,
    saved_images: int,
    speed: int,
    steering: int,
    braking: bool,
    source: str,
    jpeg_quality: int,
) -> int:
    sequence = saved_images + 1
    captured_at = datetime.now()
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    quality = max(1, min(100, int(jpeg_quality)))
    for camera_name, frame in frames.items():
        filename = "%s_%06d_%s.jpg" % (camera_name, sequence, timestamp)
        image_path = image_dirs[camera_name] / filename
        ok = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("failed to save image: %s" % image_path)
        metadata_writer.writerow(
            [
                camera_name,
                filename,
                captured_at.isoformat(timespec="milliseconds"),
                speed,
                steering,
                int(braking),
                source,
            ]
        )
    metadata_file.flush()
    print("saved frame set %06d: %s" % (sequence, ", ".join(sorted(frames.keys()))))
    return sequence


def compose_preview(cv2: Any, front_frame: Any, rear_frame: Optional[Any]) -> Any:
    front = front_frame.copy()
    draw_camera_name(cv2, front, "FRONT")
    if rear_frame is None:
        return front

    rear = rear_frame.copy()
    draw_camera_name(cv2, rear, "REAR")
    front_h, front_w = front.shape[:2]
    rear_h, rear_w = rear.shape[:2]
    if rear_h != front_h:
        width = max(1, int(round(rear_w * (front_h / max(1, rear_h)))))
        rear = cv2.resize(rear, (width, front_h))
    if rear.shape[1] != front_w:
        rear = cv2.resize(rear, (front_w, front_h))
    return cv2.hconcat([front, rear])


def draw_camera_name(cv2: Any, frame: Any, name: str) -> None:
    cv2.putText(
        frame,
        name,
        (22, frame.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def draw_overlay(
    cv2: Any,
    frame: Any,
    speed: int,
    steering: int,
    braking: bool,
    auto_capture: bool,
    saved_images: int,
    next_capture_seconds: float,
    hold_to_run: bool,
) -> Any:
    display = frame.copy()
    status = "STOP" if braking else "DRIVE"
    color = (0, 0, 255) if braking else (0, 255, 0)
    lines = [
        "%s speed=%d steering=%d" % (status, speed, steering),
        "auto_capture=%s saved=%d next=%.1fs" % (
            "ON" if auto_capture else "OFF",
            saved_images,
            max(0.0, next_capture_seconds),
        ),
        "hold-to-run=%s: keep W/S/A/D pressed; release = STOP" % ("ON" if hold_to_run else "OFF"),
        "W/S speed  A/D steer  Z center  X speed0  SPACE stop",
        "C snapshot  T auto capture  Q/ESC quit",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (22, 42 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color if index == 0 else (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def current_command(speed: int, steering: int, braking: bool) -> ControlCommand:
    if braking:
        return ControlCommand.stop("debug_keyboard_stop")
    return ControlCommand(speed=speed, steering=steering, brake=False, reason="debug_keyboard")


def hold_timeout_expired(args: argparse.Namespace, last_motion_key_at: Optional[float], now: float) -> bool:
    if args.hold_to_run != "on" or last_motion_key_at is None:
        return False
    return now - last_motion_key_at > max(0.05, args.key_timeout)


def increase_speed(speed: int, args: argparse.Namespace) -> int:
    if args.control_style == "direct":
        return min(args.max_speed, args.start_speed)
    if speed < 0:
        return 0
    if speed == 0:
        return min(args.max_speed, args.start_speed)
    return min(args.max_speed, speed + args.speed_step)


def decrease_speed(speed: int, args: argparse.Namespace) -> int:
    if args.control_style == "direct":
        return -min(args.max_reverse_speed, args.reverse_speed)
    if speed > 0:
        return max(0, speed - args.speed_step)
    if speed == 0:
        return -min(args.max_reverse_speed, args.reverse_speed)
    return max(-args.max_reverse_speed, speed - args.speed_step)


def steer_left(steering: int, args: argparse.Namespace) -> int:
    if args.control_style == "direct":
        return -min(args.max_steering, args.steering_command)
    return max(-args.max_steering, steering - args.steering_step)


def steer_right(steering: int, args: argparse.Namespace) -> int:
    if args.control_style == "direct":
        return min(args.max_steering, args.steering_command)
    return min(args.max_steering, steering + args.steering_step)


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for camera preview/capture") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
