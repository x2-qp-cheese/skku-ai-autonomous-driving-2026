from __future__ import annotations

import bisect
import csv
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union


ScanPoint = Tuple[float, float]


@dataclass(frozen=True)
class LidarPoint:
    quality: int
    angle_deg: float
    distance_mm: float


@dataclass(frozen=True)
class LidarScan:
    timestamp: float
    points: Tuple[LidarPoint, ...]


class LidarCsvRecorder:
    """Persist each distinct live LiDAR scan in the replay CSV format."""

    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(("timestamp", "quality", "angle_deg", "distance_mm"))
        self._last_timestamp: Optional[float] = None
        self.scans_written = 0
        self.points_written = 0

    def write(self, scan: Optional[LidarScan]) -> bool:
        if scan is None or scan.timestamp == self._last_timestamp:
            return False
        for point in scan.points:
            self._writer.writerow(
                (
                    "%.9f" % scan.timestamp,
                    point.quality,
                    "%.6f" % point.angle_deg,
                    "%.3f" % point.distance_mm,
                )
            )
        self._handle.flush()
        self._last_timestamp = scan.timestamp
        self.scans_written += 1
        self.points_written += len(scan.points)
        return True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def angle_in_window(angle_deg: float, start_deg: float, end_deg: float) -> bool:
    angle = angle_deg % 360.0
    start = start_deg % 360.0
    end = end_deg % 360.0
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def nearest_distance_mm(
    points: Iterable[ScanPoint],
    angle_min: float,
    angle_max: float,
) -> Optional[float]:
    distances = [
        distance
        for angle, distance in points
        if distance > 0 and angle_in_window(angle, angle_min, angle_max)
    ]
    if not distances:
        return None
    return min(distances)


def normalize_rplidar_scan(raw_scan: Sequence[Sequence[float]]) -> Iterable[ScanPoint]:
    for point in raw_scan:
        if len(point) < 3:
            continue
        angle = float(point[1])
        distance = float(point[2])
        yield angle, distance


def load_lidar_csv(path: str) -> List[LidarScan]:
    """Load the timestamp/quality/angle/distance format used by recordings."""

    groups = []
    current_timestamp = None
    current_points = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "quality", "angle_deg", "distance_mm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "LiDAR CSV must contain: timestamp,quality,angle_deg,distance_mm"
            )
        for row in reader:
            timestamp = float(row["timestamp"])
            if current_timestamp is not None and timestamp != current_timestamp:
                groups.append(LidarScan(current_timestamp, tuple(current_points)))
                current_points = []
            current_timestamp = timestamp
            current_points.append(
                LidarPoint(
                    quality=int(float(row["quality"])),
                    angle_deg=float(row["angle_deg"]),
                    distance_mm=float(row["distance_mm"]),
                )
            )
    if current_timestamp is not None:
        groups.append(LidarScan(current_timestamp, tuple(current_points)))
    return groups


def find_lidar_port(
    explicit_port: Optional[str] = None,
    ports: Optional[Sequence[Any]] = None,
    exists: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Find the RPLidar serial port, preferring macOS callout devices."""

    exists = exists or (lambda value: Path(value).exists())
    requested = (
        None
        if explicit_port is None or explicit_port.strip().lower() in ("", "auto")
        else explicit_port
    )
    if requested is not None:
        # Windows serial endpoints such as COM7 are valid device names but are
        # not filesystem paths, so Path("COM7").exists() is normally false.
        upper = requested.upper()
        if exists(requested) or (
            upper.startswith("COM") and upper[3:].isdigit()
        ):
            return requested

    if ports is None:
        try:
            from serial.tools import list_ports
        except ImportError:
            return None
        ports = tuple(list_ports.comports())

    candidates = tuple(ports)
    if sys.platform == "darwin":
        callout_ports = [
            port for port in candidates
            if str(getattr(port, "device", "")).startswith("/dev/cu.")
        ]
        if callout_ports:
            candidates = tuple(callout_ports)

    scored = []
    for port in candidates:
        score = _score_lidar_port(port)
        if score > 0:
            scored.append((score, str(getattr(port, "device", ""))))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _score_lidar_port(port: Any) -> int:
    device = str(getattr(port, "device", ""))
    text = " ".join(
        str(value).lower()
        for value in (
            device,
            getattr(port, "description", ""),
            getattr(port, "hwid", ""),
            getattr(port, "manufacturer", ""),
        )
    )
    if "debug-console" in text or "bluetooth" in text:
        return 0
    if "arduino" in text or "usbmodem" in text:
        return 0

    score = 0
    if "usbserial" in text:
        score += 30
    if "ch340" in text or "1a86:7523" in text:
        score += 20
    if "cp210" in text or "silicon labs" in text:
        score += 15
    if "usb serial" in text:
        score += 10
    if "ttyusb" in text:
        score += 5
    if "vid:pid" in text:
        score += 2
    if sys.platform == "darwin" and device.startswith("/dev/cu."):
        score += 3
    return score


class LidarCsvReplay:
    """Select the nearest recorded scan using time relative to CSV start."""

    def __init__(self, path: str):
        self.scans = load_lidar_csv(path)
        if not self.scans:
            raise ValueError("LiDAR CSV contains no scans: %s" % path)
        self._relative_times = [scan.timestamp - self.scans[0].timestamp for scan in self.scans]

    @property
    def duration_s(self) -> float:
        return self._relative_times[-1]

    def scan_at_elapsed(self, elapsed_s: float) -> LidarScan:
        index = bisect.bisect_left(self._relative_times, max(0.0, elapsed_s))
        if index <= 0:
            return self.scans[0]
        if index >= len(self.scans):
            return self.scans[-1]
        before = self._relative_times[index - 1]
        after = self._relative_times[index]
        if elapsed_s - before <= after - elapsed_s:
            return self.scans[index - 1]
        return self.scans[index]


class RplidarScanner:
    """Background reader exposing only the latest complete RPLidar scan."""

    def __init__(self, port: str, max_buf_meas: int = 500):
        self.port = port
        self.max_buf_meas = max_buf_meas
        self._lock = threading.Lock()
        self._latest: Optional[LidarScan] = None
        self._error: Optional[Exception] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="rplidar", daemon=True)
        self._thread.start()

    def latest(self) -> Optional[LidarScan]:
        with self._lock:
            return self._latest

    @property
    def error(self) -> Optional[Exception]:
        with self._lock:
            return self._error

    def close(self) -> None:
        self._stop_event.set()
        device = self._device
        if device is not None:
            for method_name in ("stop", "stop_motor", "disconnect"):
                try:
                    getattr(device, method_name)()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._device = None

    def _run(self) -> None:
        from rplidar import RPLidar

        reset_on_connect = False
        while not self._stop_event.is_set():
            device = None
            try:
                device = RPLidar(self.port)
                self._device = device

                # A previous process or an unplug/replug can leave measurement
                # bytes in the serial input buffer.  If those bytes are read
                # as the next command descriptor, rplidar raises
                # "Incorrect descriptor starting bytes" and no scan is ever
                # produced.  Put the sensor in a known idle state before each
                # scan session and reset it after a failed session.
                try:
                    device.stop()
                except Exception:
                    device.clean_input()
                if reset_on_connect:
                    device.reset()
                device.clean_input()

                for raw_scan in device.iter_scans(
                    max_buf_meas=self.max_buf_meas
                ):
                    if self._stop_event.is_set():
                        break
                    points = tuple(
                        LidarPoint(
                            int(point[0]),
                            float(point[1]),
                            float(point[2]),
                        )
                        for point in raw_scan
                        if len(point) >= 3
                    )
                    scan = LidarScan(time.time(), points)
                    with self._lock:
                        self._latest = scan
                        self._error = None
                    reset_on_connect = False
                if not self._stop_event.is_set():
                    raise RuntimeError("RPLidar scan stream ended")
            except Exception as exc:
                with self._lock:
                    self._latest = None
                    self._error = exc
                reset_on_connect = True
            finally:
                if device is not None:
                    for method_name in (
                        "stop",
                        "stop_motor",
                        "disconnect",
                    ):
                        try:
                            getattr(device, method_name)()
                        except Exception:
                            pass
                if self._device is device:
                    self._device = None

            # Retry automatically instead of permanently killing the reader
            # thread after one malformed serial descriptor.
            self._stop_event.wait(0.5)
