from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from ..config import SerialConfig
from ..types import ControlCommand


def find_arduino_port(
    explicit_port: Optional[str] = None,
    ports: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    candidates = _arduino_port_candidates(explicit_port, ports)
    return candidates[0] if candidates else None


def _arduino_port_candidates(
    explicit_port: Optional[str] = None,
    ports: Optional[Sequence[Any]] = None,
) -> list[str]:
    requested = (explicit_port or "").strip()
    if requested and requested.lower() != "auto":
        if requested.upper().startswith("COM") or Path(requested).exists():
            return [requested]
        return []
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required") from exc
    candidates = list(ports) if ports is not None else list(list_ports.comports())
    if sys.platform == "darwin":
        callout = [
            port for port in candidates
            if str(getattr(port, "device", "")).startswith("/dev/cu.")
        ]
        if callout:
            candidates = callout
    scored = []
    for port in candidates:
        text = " ".join(
            str(value).lower()
            for value in (
                getattr(port, "device", ""),
                getattr(port, "description", ""),
                getattr(port, "hwid", ""),
                getattr(port, "manufacturer", ""),
            )
        )
        if "bluetooth" in text or "debug-console" in text:
            continue
        score = sum(
            weight for token, weight in (
                ("arduino", 50),
                ("usbmodem", 45),
                ("ttyacm", 40),
                ("wchusbserial", 20),
                ("ch340", 20),
                ("cp210", 15),
                ("usbserial", 5),
            )
            if token in text
        )
        if score:
            scored.append((score, str(getattr(port, "device", ""))))
    scored.sort(reverse=True)
    return [device for _, device in scored]


class SerialVehicle:
    """Host-side DRIVE/STOP client for the existing Arduino firmware."""

    def __init__(
        self,
        config: SerialConfig,
        *,
        max_speed: int = 255,
        max_steering: int = 150,
    ):
        self.config = config
        self.max_speed = max_speed
        self.max_steering = max_steering
        self.port: Optional[str] = None
        self._serial = None

    def connect(
        self,
        *,
        excluded_ports: Sequence[str] = (),
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required") from exc
        excluded = set(excluded_ports)
        candidates = [
            port
            for port in _arduino_port_candidates(self.config.port)
            if port not in excluded
        ]
        if not candidates:
            raise RuntimeError(
                "Arduino port not found; pass --serial-port explicitly"
            )
        failures = []
        for port in candidates:
            try:
                self._serial = serial.Serial(
                    port,
                    self.config.baudrate,
                    timeout=self.config.timeout_s,
                )
                self.port = port
                if self.config.startup_delay_s > 0.0:
                    time.sleep(self.config.startup_delay_s)
                self._wait_ready()
                return
            except Exception as exc:
                failures.append("%s: %s" % (port, exc))
                if self._serial is not None:
                    self._serial.close()
                self._serial = None
                self.port = None
        raise RuntimeError(
            "No Arduino protocol response from automatic candidates: %s"
            % " | ".join(failures)
        )

    def send(self, command: ControlCommand) -> None:
        serial_conn = self._require_open()
        command = command.clipped(self.max_speed, self.max_steering)
        line = (
            "STOP\n"
            if command.brake
            else "DRIVE %d %d\n" % (command.speed, command.steering)
        )
        serial_conn.write(line.encode("ascii"))
        self._drain()

    def stop(self) -> None:
        if self._serial is not None:
            self.send(ControlCommand.stop("runtime_stop"))

    def close(self) -> None:
        if self._serial is not None:
            try:
                self.stop()
            finally:
                self._serial.close()
                self._serial = None

    def _drain(self) -> None:
        serial_conn = self._require_open()
        waiting = int(getattr(serial_conn, "in_waiting", 0))
        if waiting > 0:
            serial_conn.read(waiting)

    def _wait_ready(self) -> None:
        serial_conn = self._require_open()
        deadline = time.monotonic() + max(
            0.1,
            self.config.ready_timeout_s,
        )
        next_ping_at = time.monotonic() + 0.25
        received = []
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_ping_at:
                serial_conn.write(b"PING\n")
                next_ping_at = now + 0.5
            raw = serial_conn.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line:
                received.append(line)
            if "READY" in line or "PONG" in line:
                return
        detail = "; ".join(received[-3:]) if received else "no serial output"
        raise RuntimeError(
            "Arduino READY response not received from %s: %s"
            % (self.port, detail)
        )

    def _require_open(self):
        if self._serial is None:
            raise RuntimeError("Arduino serial connection is not open")
        return self._serial
