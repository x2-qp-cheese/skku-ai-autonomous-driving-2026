from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

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
    red_confirm_frames: Optional[int] = None
    contact_confirm_frames: int = 2
    contact_hold_frames: int = 5
    # Virtual front-contact line in frame coordinates. A confirmed RED is remembered
    # as soon as it is seen, but braking starts only when the bottom of the supplied
    # contact mask reaches this y ratio. The drive runtime supplies crosswalk masks.
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
        self._previous_contact_bottom: Optional[float] = None
        self._contact_crossing_armed = False
        self._contact_above_frames = 0
        self._contact_hold_frames = 0

    def update(
        self,
        frame: Any,
        masks: Iterable[Any],
        contact_masks: Optional[Iterable[Any]] = None,
    ) -> TrafficLightObservation:
        import cv2
        import numpy as np

        union = self._union_masks(masks)
        contact_union = union if contact_masks is None else self._union_masks(contact_masks)
        detected = union is not None and bool(union.any())
        candidate = "unknown"
        red_pixels = 0
        green_pixels = 0
        mask_pixels = 0

        if not detected:
            self._reset_pending()
        else:
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

        contact_detected = contact_union is not None and bool(contact_union.any())
        mask_bottom_y_ratio = 0.0
        threshold = min(1.0, max(0.0, self.config.stop_line_y_ratio))
        crossed_contact_line = False
        if contact_detected:
            rows = np.flatnonzero(contact_union.any(axis=1))
            denominator = max(1, int(contact_union.shape[0]) - 1)
            mask_bottom_y_ratio = float(rows[-1]) / float(denominator)
            if self._previous_contact_bottom is not None:
                crossed_contact_line = self._previous_contact_bottom < threshold <= mask_bottom_y_ratio
            if crossed_contact_line:
                self._contact_crossing_armed = True
                self._contact_above_frames = 1
            elif self._contact_crossing_armed and mask_bottom_y_ratio >= threshold:
                self._contact_above_frames += 1
            elif mask_bottom_y_ratio < threshold:
                self._contact_crossing_armed = False
                self._contact_above_frames = 0
            self._previous_contact_bottom = mask_bottom_y_ratio
        else:
            self._previous_contact_bottom = None
            self._contact_crossing_armed = False
            self._contact_above_frames = 0

        if (
            self._contact_crossing_armed
            and self._contact_above_frames >= max(1, self.config.contact_confirm_frames)
        ):
            self._contact_hold_frames = max(1, self.config.contact_hold_frames)
            self._contact_crossing_armed = False
            self._contact_above_frames = 0
        contact = self._contact_hold_frames > 0
        if self._contact_hold_frames > 0:
            self._contact_hold_frames -= 1
        if self.state == "green":
            self._stop_latched = False
        elif self.state == "red" and contact:
            self._stop_latched = True
        return TrafficLightObservation(
            detected=detected,
            candidate=candidate,
            state=self.state,
            red_pixels=red_pixels,
            green_pixels=green_pixels,
            mask_pixels=mask_pixels,
            mask_bottom_y_ratio=mask_bottom_y_ratio,
            contact=contact,
            stop_latched=self._stop_latched,
        )

    @staticmethod
    def _union_masks(masks: Iterable[Any]) -> Any:
        import numpy as np

        union = None
        for mask in masks:
            layer = np.asarray(mask)
            if layer.ndim == 3:
                layer = layer[:, :, 0]
            layer = layer > 0
            union = layer if union is None else (union | layer)
        return union

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
        required_frames = self.config.confirm_frames
        if candidate == "red" and self.config.red_confirm_frames is not None:
            required_frames = self.config.red_confirm_frames
        if self._pending_frames >= max(1, required_frames):
            self.state = candidate
            self._reset_pending()

    def _reset_pending(self) -> None:
        self._pending = "unknown"
        self._pending_frames = 0
