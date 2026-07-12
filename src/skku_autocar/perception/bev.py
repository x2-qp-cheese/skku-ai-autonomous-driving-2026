from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class BevConfig:
    """Bird's-eye-view homography defined by 4 source points on the ground plane.

    Points are stored as ratios of the input frame size so the same config works
    across camera resolutions. Order is fixed: top-left, top-right,
    bottom-right, bottom-left (clockwise from the far side).

    The camera is fixed on the vehicle, so these ratios encode the camera height
    and angle implicitly; you never measure them directly. Tune them once with
    scripts/bev_tune.py and paste the result here (or into configs).
    """

    # Tuned on data/raw/20260709_120821_raw.mp4 (1280x720, low car-mounted cam).
    # Symmetric about x=0.5 so the vehicle forward axis maps to the BEV center;
    # top edge sits just below the road vanishing point (~y=0.45) for lookahead,
    # bottom edge spans the full near-field ground just above the car body.
    src_top_left: Tuple[float, float] = (0.20, 0.40)
    src_top_right: Tuple[float, float] = (0.80, 0.40)
    src_bottom_right: Tuple[float, float] = (1.00, 0.88)
    src_bottom_left: Tuple[float, float] = (0.00, 0.88)

    # Output BEV canvas size in pixels.
    out_width: int = 480
    out_height: int = 640

    # Horizontal margin (ratio of out_width) that the src trapezoid maps to,
    # leaving room on both sides for lanes that bend outward.
    dst_x_margin: float = 0.25


class BevTransformer:
    def __init__(self, config: BevConfig = BevConfig()):
        self.config = config
        self._frame_hw: Optional[Tuple[int, int]] = None
        self._matrix: Optional[Any] = None
        self._inverse: Optional[Any] = None

    @property
    def out_size(self) -> Tuple[int, int]:
        """(width, height) of the BEV canvas."""
        return (self.config.out_width, self.config.out_height)

    def _ensure_matrix(self, frame_hw: Tuple[int, int]) -> None:
        if self._matrix is not None and self._frame_hw == frame_hw:
            return

        import cv2
        import numpy as np

        height, width = frame_hw
        cfg = self.config
        src = np.array(
            [
                (cfg.src_top_left[0] * width, cfg.src_top_left[1] * height),
                (cfg.src_top_right[0] * width, cfg.src_top_right[1] * height),
                (cfg.src_bottom_right[0] * width, cfg.src_bottom_right[1] * height),
                (cfg.src_bottom_left[0] * width, cfg.src_bottom_left[1] * height),
            ],
            dtype=np.float32,
        )

        out_w, out_h = cfg.out_width, cfg.out_height
        left = cfg.dst_x_margin * out_w
        right = (1.0 - cfg.dst_x_margin) * out_w
        dst = np.array(
            [
                (left, 0.0),
                (right, 0.0),
                (right, out_h),
                (left, out_h),
            ],
            dtype=np.float32,
        )

        self._matrix = cv2.getPerspectiveTransform(src, dst)
        self._inverse = cv2.getPerspectiveTransform(dst, src)
        self._frame_hw = frame_hw

    def warp_mask(self, mask: Any) -> Any:
        """Warp a binary YOLO mask into BEV. Nearest-neighbour keeps it binary."""
        import cv2

        self._ensure_matrix(mask.shape[:2])
        return cv2.warpPerspective(
            mask,
            self._matrix,
            self.out_size,
            flags=cv2.INTER_NEAREST,
        )

    def warp_frame(self, frame: Any) -> Any:
        """Warp a color frame into BEV (for debug / tuning only)."""
        import cv2

        self._ensure_matrix(frame.shape[:2])
        return cv2.warpPerspective(
            frame,
            self._matrix,
            self.out_size,
            flags=cv2.INTER_LINEAR,
        )

    def bev_to_frame(self, points: Any, frame_hw: Tuple[int, int]) -> Any:
        """Map BEV pixel points back to the original frame for overlay drawing.

        points: array-like of shape (N, 2) in BEV pixel coordinates.
        returns: numpy array of shape (N, 2) in frame pixel coordinates.
        """
        import cv2
        import numpy as np

        self._ensure_matrix(frame_hw)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self._inverse)
        return mapped.reshape(-1, 2)

    def src_polygon(self, frame_hw: Tuple[int, int]) -> Any:
        """Return the 4 src points in frame pixel coordinates for drawing."""
        import numpy as np

        height, width = frame_hw
        cfg = self.config
        return np.array(
            [
                (cfg.src_top_left[0] * width, cfg.src_top_left[1] * height),
                (cfg.src_top_right[0] * width, cfg.src_top_right[1] * height),
                (cfg.src_bottom_right[0] * width, cfg.src_bottom_right[1] * height),
                (cfg.src_bottom_left[0] * width, cfg.src_bottom_left[1] * height),
            ],
            dtype=np.float32,
        )
