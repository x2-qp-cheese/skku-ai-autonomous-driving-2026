from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class YoloLaneConfig:
    model_path: Path = Path("trained_model/best.pt")
    confidence: float = 0.35
    image_size: int = 640
    device: str = "auto"
    center_classes: Tuple[str, ...] = ("lane-center", "center")
    side_classes: Tuple[str, ...] = ("lane-side", "side")
    crosswalk_classes: Tuple[str, ...] = ("crosswalk", "cross-walk", "zebra")
    lane_classes: Tuple[str, ...] = (
        "lane",
        "line",
        "road",
        "track",
        "drivable",
        "driveable",
    )
    min_mask_area_ratio: float = 0.001
    fallback_lane_width_ratio: float = 0.24
    min_lane_width_ratio: float = 0.12
    max_lane_width_ratio: float = 0.58
    lane_width_smooth_alpha: float = 0.35


@dataclass(frozen=True)
class YoloLaneMask:
    mask: Any
    confidence: float
    class_id: int
    class_name: str
    device: str
    inference_ms: float


@dataclass(frozen=True)
class YoloClassMasks:
    """Per-class frame-space masks WITHOUT any corridor geometry applied.

    Plan A warps these into BEV first and builds the corridor there, so the raw
    class masks are all this stage produces. Side instances are kept separate
    (a tuple, not ORed) so left/right side lines can be fitted independently.
    """

    center: Tuple[Any, ...] = ()
    side: Tuple[Any, ...] = ()
    lane: Tuple[Any, ...] = ()
    crosswalk: Tuple[Any, ...] = ()
    center_conf: float = 0.0
    side_conf: float = 0.0
    lane_conf: float = 0.0
    crosswalk_conf: float = 0.0
    device: str = "cpu"
    inference_ms: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.center or self.side or self.lane or self.crosswalk)


def select_yolo_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


class YoloLaneSegmenter:
    def __init__(self, config: YoloLaneConfig):
        self.config = config
        self.model_path = Path(config.model_path).expanduser()
        _configure_runtime_dirs()
        if not self.model_path.exists():
            raise FileNotFoundError(
                "YOLO model not found: %s. Put your trained model under trained_model/ "
                "or pass --model." % self.model_path
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO driving. Run: pip install ultralytics"
            ) from exc

        self.device = select_yolo_device(config.device)
        self.model = YOLO(str(self.model_path))
        self.names = self._read_model_names()
        self._lane_width_px: Optional[float] = None

    def segment(self, frame: Any) -> Optional[YoloLaneMask]:
        try:
            result = self._predict(frame)
        except Exception:
            if self.device != "mps":
                raise
            # Apple MPS occasionally lacks an op for a given torch/ultralytics build.
            # Keep the car runnable by falling back to CPU instead of crashing.
            self.device = "cpu"
            result = self._predict(frame)
        return self._best_mask(result, frame.shape[:2])

    def segment_class_masks(self, frame: Any) -> YoloClassMasks:
        """Plan A entry point: return per-class masks with no corridor applied.

        Same MPS->CPU fallback as segment(); the caller warps these into BEV and
        runs BevCorridorLaneEstimator.
        """
        try:
            result = self._predict(frame)
        except Exception:
            if self.device != "mps":
                raise
            self.device = "cpu"
            result = self._predict(frame)
        return self._class_masks(result, frame.shape[:2])

    def _predict(self, frame: Any) -> Any:
        results = self.model.predict(
            source=frame,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def _build_candidates(self, result: Any, frame_hw: Tuple[int, int]) -> Tuple[list, Any]:
        """Extract per-detection candidates shared by the frame-space corridor
        path (_best_mask) and the Plan A per-class path (_class_masks).

        Each candidate tuple is:
          (kind, score, index, mask, area, class_id, class_name, confidence)
        """
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return [], None

        mask_tensors = masks.data
        count = len(mask_tensors)
        if count == 0:
            return [], None

        classes, confidences = self._box_metadata(result, count)
        total_area = float(frame_hw[0] * frame_hw[1])
        min_area = total_area * self.config.min_mask_area_ratio

        candidates = []
        for index in range(count):
            class_id = int(classes[index])
            class_name = self.names.get(class_id, str(class_id))
            mask = self._to_frame_mask(mask_tensors[index], frame_hw)
            kind = self._class_kind(class_name)
            if kind is None:
                continue
            area = float((mask > 0).sum())
            if area < min_area:
                continue
            confidence = float(confidences[index])
            score = confidence * (area / total_area)
            candidates.append((kind, score, index, mask, area, class_id, class_name, confidence))

        return candidates, confidences

    def _best_mask(self, result: Any, frame_hw: Tuple[int, int]) -> Optional[YoloLaneMask]:
        candidates, confidences = self._build_candidates(result, frame_hw)
        if not candidates:
            return None

        selected = self._select_group(candidates, frame_hw)
        if selected is None:
            return None
        index = selected["index"]
        return YoloLaneMask(
            mask=selected["mask"],
            confidence=float(confidences[index]),
            class_id=selected["class_id"],
            class_name=selected["class_name"],
            device=self.device,
            inference_ms=self._inference_ms(result),
        )

    def _class_masks(self, result: Any, frame_hw: Tuple[int, int]) -> YoloClassMasks:
        candidates, _ = self._build_candidates(result, frame_hw)

        def group(kind: str) -> Tuple[Tuple[Any, ...], float]:
            items = [c for c in candidates if c[0] == kind]
            masks = tuple(c[3] for c in items)
            conf = max((c[7] for c in items), default=0.0)
            return masks, conf

        center, center_conf = group("center")
        side, side_conf = group("side")
        lane, lane_conf = group("lane")
        crosswalk, crosswalk_conf = group("crosswalk")
        return YoloClassMasks(
            center=center,
            side=side,
            lane=lane,
            crosswalk=crosswalk,
            center_conf=center_conf,
            side_conf=side_conf,
            lane_conf=lane_conf,
            crosswalk_conf=crosswalk_conf,
            device=self.device,
            inference_ms=self._inference_ms(result),
        )

    @staticmethod
    def _inference_ms(result: Any) -> float:
        speed = getattr(result, "speed", {}) or {}
        return float(speed.get("inference", 0.0))

    def _to_frame_mask(self, mask_tensor: Any, frame_hw: Tuple[int, int]) -> Any:
        import cv2
        import numpy as np

        mask = mask_tensor.detach().float().cpu().numpy()
        mask = (mask >= 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != frame_hw:
            mask = cv2.resize(mask, (frame_hw[1], frame_hw[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _box_metadata(self, result: Any, count: int) -> Tuple[Any, Any]:
        import numpy as np

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return np.full(count, -1), np.ones(count)

        if getattr(boxes, "cls", None) is None:
            classes = np.full(count, -1)
        else:
            classes = boxes.cls.detach().cpu().numpy().astype(int)
        if getattr(boxes, "conf", None) is None:
            confidences = np.ones(count)
        else:
            confidences = boxes.conf.detach().cpu().numpy()

        if len(classes) < count:
            classes = np.pad(classes, (0, count - len(classes)), constant_values=-1)
        if len(confidences) < count:
            confidences = np.pad(confidences, (0, count - len(confidences)), constant_values=1.0)
        return classes[:count], confidences[:count]

    def _read_model_names(self) -> Dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def _class_kind(self, class_name: str) -> Optional[str]:
        lowered = class_name.lower()
        # Crosswalk is checked first so it is never absorbed by a lane class.
        if any(token in lowered for token in self.config.crosswalk_classes):
            return "crosswalk"
        if any(token in lowered for token in self.config.center_classes):
            return "center"
        if any(token in lowered for token in self.config.side_classes):
            return "side"
        if any(token in lowered for token in self.config.lane_classes):
            return "lane"
        return None

    def _select_group(self, candidates: list, frame_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        import cv2

        corridor = self._center_right_corridor(candidates, frame_hw)
        if corridor is not None:
            return corridor

        center_only = self._center_only_corridor(candidates, frame_hw)
        if center_only is not None:
            return center_only

        side_only = self._single_side_corridor(candidates, frame_hw)
        if side_only is not None:
            return side_only

        priority = ("center", "side", "lane")
        for kind in priority:
            group = [item for item in candidates if item[0] == kind]
            if not group:
                continue

            mask = group[0][3].copy()
            for item in group[1:]:
                mask = cv2.bitwise_or(mask, item[3])

            best = max(group, key=lambda item: item[1])
            class_names = sorted({item[6] for item in group})
            return {
                "index": best[2],
                "mask": mask,
                "class_id": best[5],
                "class_name": "+".join(class_names),
            }

        return None

    def _center_right_corridor(self, candidates: list, frame_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        import cv2

        center_group = [item for item in candidates if item[0] == "center"]
        side_group = [item for item in candidates if item[0] == "side"]
        if not center_group or not side_group:
            return None

        center_mask = center_group[0][3].copy()
        for item in center_group[1:]:
            center_mask = cv2.bitwise_or(center_mask, item[3])

        center_fit = self._fit_mask_centerline(center_mask, frame_hw)
        if center_fit is None:
            return None

        height, width = frame_hw
        target_y = height * 0.72
        center_x = self._x_at(center_fit, target_y)
        right_side = self._select_right_side(side_group, center_x, target_y, width, frame_hw)
        if right_side is None:
            return None

        right_fit = right_side["fit"]
        measured_width = right_side["distance"]
        if not self._valid_lane_width(measured_width, width):
            return None
        lane_width = self._update_lane_width(measured_width)

        y_start = int(max(center_fit["min_y"], right_fit["min_y"]))
        y_end = int(min(center_fit["max_y"], right_fit["max_y"]))
        if y_end <= y_start:
            return None

        corridor_mask = self._build_corridor_mask(center_fit, right_fit, frame_hw, y_start, y_end)
        if int((corridor_mask > 0).sum()) == 0:
            return None

        best_center = max(center_group, key=lambda item: item[1])
        best_side = right_side["candidate"]
        return {
            "index": best_center[2],
            "mask": corridor_mask,
            "class_id": best_center[5],
            "class_name": "%s+right-%s" % (best_center[6], best_side[6]),
            "lane_width": lane_width,
        }

    def _center_only_corridor(self, candidates: list, frame_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        import cv2

        center_group = [item for item in candidates if item[0] == "center"]
        if not center_group:
            return None

        center_mask = center_group[0][3].copy()
        for item in center_group[1:]:
            center_mask = cv2.bitwise_or(center_mask, item[3])

        center_fit = self._fit_mask_centerline(center_mask, frame_hw)
        if center_fit is None:
            return None

        height, width = frame_hw
        lane_width = self._lane_width(width)
        virtual_right_fit = self._offset_fit(center_fit, lane_width)
        y_start = int(center_fit["min_y"])
        y_end = int(center_fit["max_y"])
        corridor_mask = self._build_corridor_mask(center_fit, virtual_right_fit, frame_hw, y_start, y_end)
        if int((corridor_mask > 0).sum()) == 0:
            return None

        best_center = max(center_group, key=lambda item: item[1])
        return {
            "index": best_center[2],
            "mask": corridor_mask,
            "class_id": best_center[5],
            "class_name": "%s+virtual-right-side" % best_center[6],
            "lane_width": lane_width,
        }

    def _single_side_corridor(self, candidates: list, frame_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        height, width = frame_hw
        side_group = [item for item in candidates if item[0] == "side"]
        if not side_group:
            return None

        target_y = height * 0.72
        fitted_sides = []
        for item in side_group:
            fit = self._fit_mask_centerline(item[3], frame_hw)
            if fit is None:
                continue
            x = self._x_at(fit, target_y)
            fitted_sides.append({"candidate": item, "fit": fit, "x": x, "distance_to_vehicle": abs(x - width / 2.0)})
        if not fitted_sides:
            return None

        selected = min(fitted_sides, key=lambda item: item["distance_to_vehicle"])
        side_fit = selected["fit"]
        lane_width = self._lane_width(width)
        virtual_fit = self._offset_fit(side_fit, -lane_width)
        class_name = "virtual-lane-center+right-%s" % selected["candidate"][6]

        y_start = int(side_fit["min_y"])
        y_end = int(side_fit["max_y"])
        corridor_mask = self._build_corridor_mask(virtual_fit, side_fit, frame_hw, y_start, y_end)
        if int((corridor_mask > 0).sum()) == 0:
            return None

        candidate = selected["candidate"]
        return {
            "index": candidate[2],
            "mask": corridor_mask,
            "class_id": candidate[5],
            "class_name": class_name,
            "lane_width": lane_width,
        }

    def _select_right_side(
        self,
        side_group: list,
        center_x: float,
        target_y: float,
        width: int,
        frame_hw: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        right_candidates = []
        for item in side_group:
            fit = self._fit_mask_centerline(item[3], frame_hw)
            if fit is None:
                continue
            side_x = self._x_at(fit, target_y)
            if side_x <= center_x + width * 0.03:
                continue
            right_candidates.append(
                {
                    "candidate": item,
                    "fit": fit,
                    "x": side_x,
                    "distance": side_x - center_x,
                }
            )
        if not right_candidates:
            return None
        return min(right_candidates, key=lambda item: item["distance"])

    def _lane_width(self, image_width: int) -> float:
        remembered = getattr(self, "_lane_width_px", None)
        if remembered is not None and self._valid_lane_width(remembered, image_width):
            return remembered
        return image_width * self.config.fallback_lane_width_ratio

    def _update_lane_width(self, measured_width: float) -> float:
        remembered = getattr(self, "_lane_width_px", None)
        if remembered is None:
            self._lane_width_px = measured_width
        else:
            alpha = self.config.lane_width_smooth_alpha
            self._lane_width_px = alpha * measured_width + (1.0 - alpha) * remembered
        return self._lane_width_px

    def _valid_lane_width(self, lane_width: float, image_width: int) -> bool:
        return (
            image_width * self.config.min_lane_width_ratio
            <= lane_width
            <= image_width * self.config.max_lane_width_ratio
        )

    def _fit_mask_centerline(self, mask: Any, frame_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        import numpy as np

        height, _ = frame_hw
        binary = mask > 0
        samples = []
        band_half = max(2, int(height * 0.01))
        for y in np.linspace(0, height - 1, 28).astype(int):
            y0 = max(0, y - band_half)
            y1 = min(height, y + band_half + 1)
            columns = np.where(binary[y0:y1, :].any(axis=0))[0]
            if len(columns) == 0:
                continue
            samples.append((float(y), float((columns[0] + columns[-1]) / 2.0)))

        if len(samples) < 2:
            return None

        ys = np.array([item[0] for item in samples], dtype=float)
        xs = np.array([item[1] for item in samples], dtype=float)
        degree = min(2, len(samples) - 1)
        fit = np.polyfit(ys, xs, degree)
        return {
            "fit": fit,
            "min_y": float(np.min(ys)),
            "max_y": float(np.max(ys)),
        }

    def _offset_fit(self, fit_info: Dict[str, Any], dx: float) -> Dict[str, Any]:
        import numpy as np

        fit = np.array(fit_info["fit"], dtype=float).copy()
        fit[-1] += dx
        return {
            "fit": fit,
            "min_y": fit_info["min_y"],
            "max_y": fit_info["max_y"],
        }

    def _build_corridor_mask(
        self,
        center_fit: Dict[str, Any],
        right_fit: Dict[str, Any],
        frame_hw: Tuple[int, int],
        y_start: int,
        y_end: int,
    ) -> Any:
        import cv2
        import numpy as np

        height, width = frame_hw
        mask = np.zeros((height, width), dtype=np.uint8)
        for y in range(max(0, y_start), min(height, y_end + 1)):
            x_center = int(round(self._x_at(center_fit, float(y))))
            x_right = int(round(self._x_at(right_fit, float(y))))
            left = max(0, min(width - 1, min(x_center, x_right)))
            right = max(0, min(width - 1, max(x_center, x_right)))
            if right - left < width * 0.03:
                continue
            mask[y, left:right + 1] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _x_at(self, fit_info: Dict[str, Any], y: float) -> float:
        import numpy as np

        return float(np.polyval(fit_info["fit"], y))


def _configure_runtime_dirs() -> None:
    tmp = Path.cwd() / "tmp"
    yolo_dir = tmp / "ultralytics"
    matplotlib_dir = tmp / "matplotlib"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
