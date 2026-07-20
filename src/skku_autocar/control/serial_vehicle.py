from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from ..types import ControlCommand
from .protocol import encode_command


@dataclass(frozen=True)
class SerialVehicleConfig:
    port: Optional[str] = None
    baudrate: int = 115200
    timeout_s: float = 0.1
    ready_timeout_s: float = 3.0


@dataclass(frozen=True)
class UltrasonicReadings:
    """One Arduino ultrasonic report in millimeters.

    The vehicle firmware returns zero when an echo was not measured. Such
    values are represented as ``None`` so they cannot trigger a false 10 cm
    emergency stop.
    """

    front_right_mm: Optional[float] = None
    front_left_mm: Optional[float] = None
    side_right_mm: Optional[float] = None
    side_left_mm: Optional[float] = None


def parse_ultrasonic_line(line: str) -> Optional[UltrasonicReadings]:
    if not line.startswith("US "):
        return None
    values = {
        key: _valid_ultrasonic_mm(float(value))
        for key, value in re.findall(r"\b(FR|FL|SR|SL)=(-?\d+(?:\.\d+)?)", line)
    }
    if not values:
        return None
    return UltrasonicReadings(
        front_right_mm=values.get("FR"),
        front_left_mm=values.get("FL"),
        side_right_mm=values.get("SR"),
        side_left_mm=values.get("SL"),
    )


def _valid_ultrasonic_mm(value: float) -> Optional[float]:
    return value if value > 0.0 else None


def find_arduino_port(explicit_port: Optional[str] = None) -> Optional[str]:
    if explicit_port:
        return explicit_port

    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required for Arduino serial control") from exc

    ports = list(list_ports.comports())
    # macOS exposes the same USB serial device as both /dev/tty.* and /dev/cu.*.
    # The cu device is the correct endpoint for an app initiating the connection;
    # preferring it also avoids a lexicographic tie accidentally selecting tty.
    if sys.platform == "darwin":
        callout_ports = [port for port in ports if str(port.device).startswith("/dev/cu.")]
        if callout_ports:
            ports = callout_ports
    scored = []
    for port in ports:
        score = _score_port(port)
        if score > 0:
            scored.append((score, port.device))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _score_port(port: Any) -> int:
    text = " ".join(
        str(value).lower()
        for value in (
            getattr(port, "device", ""),
            getattr(port, "description", ""),
            getattr(port, "hwid", ""),
            getattr(port, "manufacturer", ""),
        )
    )

    if "debug-console" in text or "bluetooth" in text:
        return 0

    score = 0
    for token in ("arduino", "usbmodem", "usbserial", "wchusbserial", "ch340", "cp210", "ttyacm"):
        if token in text:
            score += 10
    if "vid:pid" in text:
        score += 2
    if "com" in str(getattr(port, "device", "")).lower():
        score += 1
    return score


class SerialVehicleClient:
    def __init__(self, config: SerialVehicleConfig, max_speed: int = 255, max_steering: int = 120):
        self.config = config
        self.max_speed = max_speed
        self.max_steering = max_steering
        self.port: Optional[str] = None
        self._serial = None
        self._rx_buffer = ""

    def connect(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for Arduino serial control") from exc

        port = find_arduino_port(self.config.port)
        if not port:
            raise RuntimeError("Arduino serial port was not found. Pass --serial-port COM3 or /dev/cu.usbmodemXXXX.")

        self._serial = serial.Serial(port, self.config.baudrate, timeout=self.config.timeout_s)
        self.port = port
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def send(self, command: ControlCommand) -> List[str]:
        serial_conn = self._require_open()
        # The firmware acknowledges every DRIVE/STOP command. Drain replies from
        # the previous command so a long run cannot fill the host receive buffer.
        lines = self.read_lines()
        line = encode_command(command, self.max_speed, self.max_steering)
        serial_conn.write(line.encode("ascii"))
        return lines

    def write_line(self, line: str) -> List[str]:
        serial_conn = self._require_open()
        lines = self.read_lines()
        text = line if line.endswith("\n") else line + "\n"
        serial_conn.write(text.encode("ascii"))
        return lines

    def read_lines(self) -> List[str]:
        serial_conn = self._require_open()
        waiting = int(getattr(serial_conn, "in_waiting", 0))
        if waiting <= 0:
            return []
        raw = serial_conn.read(waiting)
        if not raw:
            return []
        self._rx_buffer += raw.decode("ascii", errors="replace")
        parts = self._rx_buffer.split("\n")
        self._rx_buffer = parts[-1]
        return [part.strip("\r") for part in parts[:-1] if part.strip("\r")]

    def stop(self, reason: str = "stop") -> None:
        if self._serial is not None:
            self.send(ControlCommand.stop(reason))

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._rx_buffer = ""

    def _wait_ready(self) -> None:
        serial_conn = self._require_open()
        deadline = time.monotonic() + self.config.ready_timeout_s
        ready_lines = []
        while time.monotonic() < deadline:
            raw = serial_conn.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            ready_lines.append(line)
            if "READY" in line or "PONG" in line:
                return
        detail = "; ".join(ready_lines[-3:]) if ready_lines else "no serial output"
        raise RuntimeError("Arduino READY response was not received from %s: %s" % (self.port, detail))

    def _require_open(self) -> Any:
        if self._serial is None:
            raise RuntimeError("serial connection is not open")
        return self._serial

    def __enter__(self) -> "SerialVehicleClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
