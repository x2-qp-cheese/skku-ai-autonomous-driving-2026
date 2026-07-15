from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from typing import Any, Optional

from ..types import ControlCommand
from .protocol import encode_command


@dataclass(frozen=True)
class SerialVehicleConfig:
    port: Optional[str] = None
    baudrate: int = 115200
    timeout_s: float = 0.1
    ready_timeout_s: float = 3.0


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

    def send(self, command: ControlCommand) -> None:
        serial_conn = self._require_open()
        line = encode_command(command, self.max_speed, self.max_steering)
        serial_conn.write(line.encode("ascii"))

    def stop(self, reason: str = "stop") -> None:
        if self._serial is not None:
            self.send(ControlCommand.stop(reason))

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

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
