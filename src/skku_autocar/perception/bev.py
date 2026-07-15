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

    # Tuned on data/raw/20260714.mp4 (1280x720, final low/forward car-mounted cam).
    # Symmetric about x=0.5 so the vehicle forward axis maps to the BEV center.
    #
    # WHY THESE VALUES (and what to change if the BEV mask looks wrong):
    #  - src_top_y (0.52): how far ahead the top edge reaches. The closer it is to
    #    the road's vanishing point (smaller y, higher up), the more the far field
    #    is magnified -- i.e. "the top of the BEV balloons / gets wider". It was
    #    0.45 (right at the track's far edge) which over-magnified the far slab;
    #    0.52 stops mapping that extreme-far strip. RAISE this value to shrink the
    #    ballooning (costs forward lookahead); LOWER it to see farther.
    #  - top width vs bottom width sets where the trapezoid's left/right sides meet
    #    (the vanishing point). If lane lines FAN OUT toward the top in the BEV
    #    (width not uniform), the top edge is too narrow for the true perspective ->
    #    widen it (raise src_top_left toward 0.5 less / spread the top x's); if they
    #    PINCH inward at the top, narrow the top edge. Goal: a straight lane's two
    #    lines are vertical and equal-width top-to-bottom.
    #  - bottom edge kept inside the frame (0.0 .. 1.0): sampling past the frame
    #    (was -0.10 .. 1.10) only pulls in out-of-frame black.
    #  - The black side wedges are dst_x_margin (curve room), not a mapping error;
    #    lower dst_x_margin to shrink them at the cost of less room for curves.
    #
    # Re-tune per camera with scripts/bev_tune.py or scripts/bev_replay.py --bev-*
    # (press 'p' in bev_replay to print ratios), then paste the result here.
    #
    # Previous values: this (low) cam earlier used top y=0.45 w=0.36, bottom
    # -0.10/1.10; the higher 20260709 cam used top (0.20,0.40)/(0.80,0.40), bottom
    # (0.00,0.88)/(1.00,0.88).
    src_top_left: Tuple[float, float] = (0.30, 0.52)
    src_top_right: Tuple[float, float] = (0.70, 0.52)
    src_bottom_right: Tuple[float, float] = (1.00, 1.00)
    src_bottom_left: Tuple[float, float] = (0.00, 1.00)

    # Output BEV canvas size in pixels. Kept square (was 480x640): a canvas taller
    # than it is wide over-stretches the forward (y) axis, so a curve's dx/dy --
    # which drives the heading/curvature estimate -- reads weaker and the car
    # under-reacts to curves. Lateral error and forward lookahead coverage do NOT
    # depend on height (only on out_width / dst_x_margin), and shortening the canvas
    # does not clip curves (clipping is horizontal), so a shorter canvas simply
    # strengthens the curve signal. Tune per taste with --bev-out-height; if curves
    # over-steer after this, soften --bev-heading-gain.
    out_width: int = 480
    out_height: int = 480

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
