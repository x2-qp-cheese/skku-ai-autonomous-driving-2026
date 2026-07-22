from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple


@dataclass(frozen=True)
class LocalOccupancyConfig:
    """Short-horizon ego-centric occupancy-grid parameters.

    The camera has no odometry, so this map deliberately remembers only a
    fraction of a second. That bridges segmentation dropouts without pretending
    to be a globally consistent SLAM map.
    """

    enabled: bool = True
    decay_seconds: float = 0.65
    hit_probability: float = 0.85
    occupied_probability: float = 0.65
    inflation_radius_px: int = 8
    max_log_odds: float = 4.0


@dataclass(frozen=True)
class LocalOccupancySnapshot:
    mask: Any
    instances: Tuple[Any, ...] = ()
    confidence: float = 0.0
    occupied_ratio: float = 0.0
    age_seconds: float = float("inf")

    @property
    def found(self) -> bool:
        return bool(self.instances)


class LocalOccupancyGrid:
    """Bayesian BEV occupancy grid built from YOLO obstacle masks.

    Ultrasonic measurements are intentionally not projected into arbitrary BEV
    cells because these sensors provide range but no reliable bearing. They are
    fused later by ``ObstacleFusionPlanner`` as physical range and side-clearance
    confirmation.
    """

    def __init__(
        self,
        config: LocalOccupancyConfig = LocalOccupancyConfig(),
    ) -> None:
        self.config = config
        self._log_odds = None
        self._shape: Tuple[int, int] = (0, 0)
        self._last_update_at = None
        self._last_hit_at = None

    def reset(self, shape: Tuple[int, int] = (0, 0)) -> None:
        import numpy as np

        height, width = shape
        self._shape = (max(0, int(height)), max(0, int(width)))
        self._log_odds = (
            np.zeros(self._shape, dtype=np.float32)
            if self._shape[0] > 0 and self._shape[1] > 0
            else None
        )
        self._last_update_at = None
        self._last_hit_at = None

    def update(
        self,
        obstacle_masks: Sequence[Any],
        shape: Tuple[int, int],
        confidence: float,
        now: float,
        running: bool,
    ) -> LocalOccupancySnapshot:
        import numpy as np

        height, width = max(0, int(shape[0])), max(0, int(shape[1]))
        if not self.config.enabled or height <= 0 or width <= 0:
            self.reset()
            return self._empty_snapshot((height, width))
        if not running:
            self.reset((height, width))
            return self._empty_snapshot((height, width))
        if self._log_odds is None or self._shape != (height, width):
            self.reset((height, width))

        self._decay(now)
        measured = np.zeros((height, width), dtype=np.uint8)
        for mask in obstacle_masks:
            array = np.asarray(mask)
            if array.shape[:2] != (height, width):
                continue
            measured[array > 0] = 255

        if measured.any():
            measured = self._inflate(measured)
            strength = max(0.0, min(1.0, float(confidence)))
            hit_probability = self._clip_probability(
                self.config.hit_probability
            )
            hit_log_odds = math.log(hit_probability / (1.0 - hit_probability))
            self._log_odds[measured > 0] += hit_log_odds * strength
            np.clip(
                self._log_odds,
                -abs(float(self.config.max_log_odds)),
                abs(float(self.config.max_log_odds)),
                out=self._log_odds,
            )
            self._last_hit_at = float(now)

        self._last_update_at = float(now)
        probability = 1.0 / (1.0 + np.exp(-self._log_odds))
        occupied = probability >= self._clip_probability(
            self.config.occupied_probability
        )
        output = np.zeros((height, width), dtype=np.uint8)
        output[occupied] = 255
        instances = self._extract_instances(output)
        occupied_probability = probability[occupied]
        map_confidence = (
            float(occupied_probability.max())
            if occupied_probability.size
            else 0.0
        )
        occupied_ratio = float(occupied.mean())
        age = (
            max(0.0, float(now) - float(self._last_hit_at))
            if self._last_hit_at is not None
            else float("inf")
        )
        return LocalOccupancySnapshot(
            mask=output,
            instances=instances,
            confidence=map_confidence,
            occupied_ratio=occupied_ratio,
            age_seconds=age,
        )

    def _decay(self, now: float) -> None:
        if self._log_odds is None or self._last_update_at is None:
            return
        dt = max(0.0, float(now) - float(self._last_update_at))
        decay_seconds = max(1e-3, float(self.config.decay_seconds))
        self._log_odds *= math.exp(-dt / decay_seconds)

    def _inflate(self, mask: Any) -> Any:
        radius = max(0, int(self.config.inflation_radius_px))
        if radius <= 0:
            return mask
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for occupancy-grid inflation"
            ) from exc
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(mask, kernel, iterations=1)

    @staticmethod
    def _extract_instances(mask: Any) -> Tuple[Any, ...]:
        import cv2
        import numpy as np

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        return tuple(
            np.where(labels == label, 255, 0).astype(np.uint8)
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= 4
        )

    @staticmethod
    def _clip_probability(value: float) -> float:
        return max(1e-4, min(1.0 - 1e-4, float(value)))

    @staticmethod
    def _empty_snapshot(shape: Tuple[int, int]) -> LocalOccupancySnapshot:
        import numpy as np

        return LocalOccupancySnapshot(
            mask=np.zeros(shape, dtype=np.uint8),
        )
