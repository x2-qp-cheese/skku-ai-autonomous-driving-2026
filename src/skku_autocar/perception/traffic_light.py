from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..types import ControlCommand


@dataclass(frozen=True)
class TrafficLightConfig:
    """HSV color comparison inside YOLO's ``light`` segmentation mask."""

    min_saturation: int = 90
    min_value: int = 90
    min_color_pixels: int = 8
    min_color_ratio: float = 0.01
    dominance_ratio: float = 1.35
    confirm_frames: int = 3
    # Virtual front-contact line in frame coordinates. A confirmed RED is remembered
    # as soon as it is seen, but braking starts only when the bottom of the light
    # mask reaches this y ratio. This lets the car enter the approach section and
    # stop at its near boundary instead of stopping at the first distant detection.
    stop_line_y_ratio: float = 0.82


@dataclass(frozen=True)
class TrafficLightObservation:
    detected: bool = False
    candidate: str = "unknown"
    state: str = "unknown"
    red_pixels: int = 0
    green_pixels: int = 0
    mask_pixels: int = 0
    mask_bottom_y_ratio: float = 0.0
    contact: bool = False
    stop_latched: bool = False


class TrafficLightController:
    """Latch RED until GREEN is confirmed over consecutive frames.

    An absent or ambiguous light never releases a confirmed red. This avoids the
    car starting because YOLO briefly loses the signal while stopped. Before any
    red has been confirmed, unknown/no-light leaves normal lane following alone.
    """

    def __init__(self, config: TrafficLightConfig = TrafficLightConfig()):
        self.config = config
        self.state = "unknown"
        self._pending = "unknown"
        self._pending_frames = 0
        self._stop_latched = False

    def update(self, frame: Any, masks: Iterable[Any]) -> TrafficLightObservation:
        import cv2
        import numpy as np

        union = None
        for mask in masks:
            layer = np.asarray(mask)
            if layer.ndim == 3:
                layer = layer[:, :, 0]
            layer = layer > 0
            union = layer if union is None else (union | layer)

        if union is None or not bool(union.any()):
            self._reset_pending()
            return TrafficLightObservation(
                state=self.state,
                stop_latched=self._stop_latched,
            )

        rows = np.flatnonzero(union.any(axis=1))
        frame_height = int(union.shape[0])
        denominator = max(1, frame_height - 1)
        mask_bottom_y_ratio = float(rows[-1]) / float(denominator)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        vivid = union & (saturation >= self.config.min_saturation) & (value >= self.config.min_value)

        # OpenCV hue is 0..179. Red wraps around zero; green occupies roughly
        # 35..95. Yellow is intentionally neither, so it cannot release RED.
        red = vivid & ((hue <= 12) | (hue >= 168))
        green = vivid & (hue >= 35) & (hue <= 95)
        red_pixels = int(red.sum())
        green_pixels = int(green.sum())
        mask_pixels = int(union.sum())
        candidate = self._candidate(red_pixels, green_pixels, mask_pixels)
        self._advance(candidate)
        contact = mask_bottom_y_ratio >= min(1.0, max(0.0, self.config.stop_line_y_ratio))
        if self.state == "green":
            self._stop_latched = False
        elif self.state == "red" and contact:
            self._stop_latched = True
        return TrafficLightObservation(
            detected=True,
            candidate=candidate,
            state=self.state,
            red_pixels=red_pixels,
            green_pixels=green_pixels,
            mask_pixels=mask_pixels,
            mask_bottom_y_ratio=mask_bottom_y_ratio,
            contact=contact,
            stop_latched=self._stop_latched,
        )

    def apply(self, command: ControlCommand, running: bool) -> ControlCommand:
        if running and self.state == "red" and self._stop_latched:
            return ControlCommand.stop("traffic_light:red_contact")
        return command

    def _candidate(self, red: int, green: int, total: int) -> str:
        minimum = max(self.config.min_color_pixels, int(total * self.config.min_color_ratio))
        dominance = max(1.0, self.config.dominance_ratio)
        if red >= minimum and red >= green * dominance:
            return "red"
        if green >= minimum and green >= red * dominance:
            return "green"
        return "unknown"

    def _advance(self, candidate: str) -> None:
        if candidate == "unknown":
            self._reset_pending()
            return
        if candidate == self.state:
            self._reset_pending()
            return
        if candidate != self._pending:
            self._pending = candidate
            self._pending_frames = 1
        else:
            self._pending_frames += 1
        if self._pending_frames >= max(1, self.config.confirm_frames):
            self.state = candidate
            self._reset_pending()

    def _reset_pending(self) -> None:
        self._pending = "unknown"
        self._pending_frames = 0
