from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Dict, Iterable, Optional, Tuple


SENSOR_KEYS = ("FC", "FR", "FL", "SR", "SL")
FRONT_KEYS = ("FC", "FR", "FL")


@dataclass(frozen=True)
class UltrasonicConfig:
    min_valid_mm: int = 50
    max_valid_mm: int = 3200
    median_window: int = 3
    max_age_seconds: float = 0.6


@dataclass(frozen=True)
class UltrasonicSnapshot:
    fc: int = 0
    fr: int = 0
    fl: int = 0
    sr: int = 0
    sl: int = 0
    fresh_keys: Tuple[str, ...] = ()
    age_seconds: float = float("inf")

    @property
    def front_fresh(self) -> bool:
        return all(key in self.fresh_keys for key in FRONT_KEYS)

    @property
    def front_fresh_count(self) -> int:
        return sum(key in self.fresh_keys for key in FRONT_KEYS)

    def front_ready(self, minimum_sensors: int = 2) -> bool:
        required = max(1, min(len(FRONT_KEYS), int(minimum_sensors)))
        return self.front_fresh_count >= required

    @property
    def fresh_front_values(self) -> Tuple[int, ...]:
        values = {"FC": self.fc, "FR": self.fr, "FL": self.fl}
        return tuple(
            values[key]
            for key in FRONT_KEYS
            if key in self.fresh_keys and values[key] > 0
        )

    @property
    def front_min_mm(self) -> Optional[int]:
        values = self.fresh_front_values
        return min(values) if values else None

    def front_close_count(self, threshold_mm: float) -> int:
        limit = max(0.0, float(threshold_mm))
        return sum(value <= limit for value in self.fresh_front_values)


class UltrasonicFilter:
    """Parse Arduino US lines and expose fresh median-filtered distances.

    Zero is retained as the firmware's explicit no-echo value. Readings below
    min_valid_mm (other than zero) and above max_valid_mm are rejected without
    refreshing that sensor, so stuck/noisy channels eventually become stale.
    """

    def __init__(self, config: UltrasonicConfig = UltrasonicConfig()):
        self.config = config
        size = max(1, int(config.median_window))
        self._samples: Dict[str, Deque[int]] = {
            key: deque(maxlen=size) for key in SENSOR_KEYS
        }
        self._values: Dict[str, int] = {key: 0 for key in SENSOR_KEYS}
        self._updated_at: Dict[str, Optional[float]] = {
            key: None for key in SENSOR_KEYS
        }

    def reset(self) -> None:
        for samples in self._samples.values():
            samples.clear()
        for key in SENSOR_KEYS:
            self._values[key] = 0
            self._updated_at[key] = None

    def update_lines(self, lines: Iterable[str], now: float) -> bool:
        updated = False
        for line in lines:
            parsed = parse_ultrasonic_line(line)
            for key, raw_value in parsed.items():
                if not self._valid_raw(raw_value):
                    continue
                samples = self._samples[key]
                samples.append(raw_value)
                filtered = int(median(samples))
                previous = self._values[key]
                # Collision avoidance needs a fast attack and can tolerate a
                # slower release. A new positive echo after several zero
                # no-echo samples must not wait for the median window to fill,
                # and a closing object must never be reported farther away than
                # the newest valid measurement.
                if raw_value > 0 and (previous <= 0 or raw_value < previous):
                    filtered = raw_value
                self._values[key] = filtered
                self._updated_at[key] = now
                updated = True
        return updated

    def snapshot(self, now: float) -> UltrasonicSnapshot:
        max_age = max(0.0, float(self.config.max_age_seconds))
        fresh = tuple(
            key
            for key in SENSOR_KEYS
            if self._updated_at[key] is not None
            and now - float(self._updated_at[key]) <= max_age
        )
        timestamps = [
            float(value) for value in self._updated_at.values() if value is not None
        ]
        age = now - max(timestamps) if timestamps else float("inf")
        return UltrasonicSnapshot(
            fc=self._values["FC"],
            fr=self._values["FR"],
            fl=self._values["FL"],
            sr=self._values["SR"],
            sl=self._values["SL"],
            fresh_keys=fresh,
            age_seconds=max(0.0, age),
        )

    def _valid_raw(self, value: int) -> bool:
        if value == 0:
            return True
        return self.config.min_valid_mm <= value <= self.config.max_valid_mm


def parse_ultrasonic_line(line: str) -> Dict[str, int]:
    if not line.startswith("US "):
        return {}
    values: Dict[str, int] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key not in SENSOR_KEYS:
            continue
        try:
            values[key] = int(value)
        except ValueError:
            continue
    return values
