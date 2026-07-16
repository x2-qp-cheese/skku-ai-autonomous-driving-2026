#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.control.serial_vehicle import find_arduino_port


def main(argv: Optional[list] = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print("error:", exc, file=sys.stderr)
        return 1


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read four ultrasonic sensors from the Arduino vehicle controller")
    parser.add_argument("--serial-port", default=None, help="Arduino port; auto-detected when omitted")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--interval", type=float, default=0.2, help="seconds between US requests")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means forever")
    parser.add_argument("--ready-timeout", type=float, default=3.0)
    parser.add_argument(
        "--sensor",
        choices=("all", "front_right", "front_left", "side_right", "side_left"),
        default="all",
        help="read all sensors or only one sensor for wiring diagnosis",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required. Run: pip install pyserial") from exc

    port = find_arduino_port(args.serial_port)
    if not port:
        raise RuntimeError("Arduino serial port was not found. Pass --serial-port explicitly.")

    print("opening:", port)
    with serial.Serial(port, args.baudrate, timeout=0.05) as conn:
        wait_ready(conn, args.ready_timeout)
        deadline = None if args.duration <= 0 else time.monotonic() + args.duration
        next_request_at = 0.0
        command = ultrasonic_command(args.sensor)
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_request_at:
                conn.write(command)
                next_request_at = now + max(0.05, args.interval)

            line = conn.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line.startswith("US "):
                print(format_ultrasonic_line(line))
            elif line.startswith("ERR"):
                print(line)
    return 0


def ultrasonic_command(sensor: str) -> bytes:
    commands = {
        "all": b"US\n",
        "front_right": b"USFR\n",
        "front_left": b"USFL\n",
        "side_right": b"USSR\n",
        "side_left": b"USSL\n",
    }
    return commands[sensor]


def wait_ready(conn: object, timeout_s: float) -> None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    seen = []
    while time.monotonic() < deadline:
        line = conn.readline().decode("ascii", errors="replace").strip()
        if not line:
            continue
        seen.append(line)
        print(line)
        if "READY" in line or "PONG" in line:
            return
    detail = "; ".join(seen[-3:]) if seen else "no serial output"
    raise RuntimeError("Arduino READY response was not received: %s" % detail)


def format_ultrasonic_line(line: str) -> str:
    values = parse_key_values(line)
    return (
        "front_right={FR:>4}mm  front_left={FL:>4}mm  "
        "side_right={SR:>4}mm  side_left={SL:>4}mm"
    ).format(
        FR=values.get("FR", "0"),
        FL=values.get("FL", "0"),
        SR=values.get("SR", "0"),
        SL=values.get("SL", "0"),
    )


def parse_key_values(line: str) -> dict:
    result = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
