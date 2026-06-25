from typing import Optional

from ..config import SerialConfig
from ..types import ControlCommand
from .protocol import encode_command


class ArduinoController:
    def __init__(self, config: SerialConfig, max_speed: int = 255, max_steering: int = 255):
        self.config = config
        self.max_speed = max_speed
        self.max_steering = max_steering
        self._serial = None

    def connect(self) -> None:
        if not self.config.arduino_port:
            raise ValueError("serial.arduino_port is not configured")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for Arduino control") from exc
        self._serial = serial.Serial(
            self.config.arduino_port,
            self.config.baudrate,
            timeout=self.config.timeout_s,
        )

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send(self, command: ControlCommand) -> None:
        serial_conn = self._require_connection()
        line = encode_command(command, self.max_speed, self.max_steering)
        serial_conn.write(line.encode("ascii"))

    def read_line(self) -> Optional[str]:
        serial_conn = self._require_connection()
        raw = serial_conn.readline()
        if not raw:
            return None
        return raw.decode("ascii", errors="replace").strip()

    def _require_connection(self):
        if self._serial is None:
            raise RuntimeError("Arduino serial connection is not open")
        return self._serial

    def __enter__(self) -> "ArduinoController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
