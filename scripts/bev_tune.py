"""Interactive BEV homography tuner.

Adjust the 4 source points of the ground-plane trapezoid with trackbars while
watching the warped bird's-eye view live. Press 'p' to print the ratios to paste
into BevConfig (or configs), 's' to save a snapshot, 'q'/esc to quit.

Usage:
    python3 scripts/bev_tune.py                 # camera 0
    python3 scripts/bev_tune.py --source 1      # camera index 1
    python3 scripts/bev_tune.py --source clip.mp4
    python3 scripts/bev_tune.py --source frame.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.perception.bev import BevConfig, BevTransformer  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOW = "BEV Tune (original)"
BEV_WINDOW = "BEV Tune (warped)"


def main(argv=None) -> int:
    import cv2

    args = parse_args(argv)
    frame_provider, release = open_source(cv2, args)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(BEV_WINDOW, cv2.WINDOW_NORMAL)
    create_trackbars(cv2, args)

    print("keys: p=print ratios  s=save snapshot  q/esc=quit")
    try:
        while True:
            frame = frame_provider()
            if frame is None:
                break

            config = config_from_trackbars(cv2, args)
            transformer = BevTransformer(config)
            frame_hw = frame.shape[:2]

            overlay = draw_src(cv2, frame.copy(), transformer, frame_hw)
            bev = transformer.warp_frame(frame)
            draw_bev_grid(cv2, bev)

            cv2.imshow(WINDOW, overlay)
            cv2.imshow(BEV_WINDOW, bev)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                print_ratios(config)
            if key == ord("s"):
                save_snapshot(cv2, overlay, bev)
    finally:
        release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Interactive BEV homography tuner")
    parser.add_argument("--source", default="0", help="camera index, video path, or image path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out-width", type=int, default=BevConfig.out_width)
    parser.add_argument("--out-height", type=int, default=BevConfig.out_height)
    parser.add_argument("--dst-margin", type=float, default=BevConfig.dst_x_margin)
    return parser.parse_args(argv)


def open_source(cv2, args):
    source = args.source
    ext = Path(source).suffix.lower()

    if ext in IMAGE_EXTS:
        image = cv2.imread(source)
        if image is None:
            raise RuntimeError("could not read image: %s" % source)
        return (lambda: image.copy()), (lambda: None)

    cap_source = int(source) if source.isdigit() else source
    if isinstance(cap_source, int) and sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("could not open source: %s" % source)

    is_video_file = not isinstance(cap_source, int)

    def read():
        ok, frame = cap.read()
        if not ok:
            if is_video_file:  # loop the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                return None
        return frame

    return read, cap.release


# Trackbars are stored as integers 0..1000 mapped to ratios 0..1.
_BARS = [
    ("top_y", 620),
    ("top_left_x", 420),
    ("top_right_x", 580),
    ("bottom_y", 950),
    ("bottom_left_x", 50),
    ("bottom_right_x", 950),
]


def create_trackbars(cv2, args):
    for name, default in _BARS:
        cv2.createTrackbar(name, WINDOW, default, 1000, lambda _v: None)


def config_from_trackbars(cv2, args) -> BevConfig:
    def r(name):
        return cv2.getTrackbarPos(name, WINDOW) / 1000.0

    top_y = r("top_y")
    bottom_y = r("bottom_y")
    return BevConfig(
        src_top_left=(r("top_left_x"), top_y),
        src_top_right=(r("top_right_x"), top_y),
        src_bottom_right=(r("bottom_right_x"), bottom_y),
        src_bottom_left=(r("bottom_left_x"), bottom_y),
        out_width=args.out_width,
        out_height=args.out_height,
        dst_x_margin=args.dst_margin,
    )


def draw_src(cv2, frame, transformer, frame_hw):
    import numpy as np

    pts = transformer.src_polygon(frame_hw).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 255), thickness=2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def draw_bev_grid(cv2, bev):
    h, w = bev.shape[:2]
    for gx in range(1, 4):
        x = int(w * gx / 4)
        cv2.line(bev, (x, 0), (x, h), (60, 60, 60), 1)
    for gy in range(1, 8):
        y = int(h * gy / 8)
        cv2.line(bev, (0, y), (w, y), (60, 60, 60), 1)
    cv2.line(bev, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)


def print_ratios(config: BevConfig) -> None:
    print("\n--- paste into BevConfig ---")
    print("    src_top_left=(%.3f, %.3f)," % config.src_top_left)
    print("    src_top_right=(%.3f, %.3f)," % config.src_top_right)
    print("    src_bottom_right=(%.3f, %.3f)," % config.src_bottom_right)
    print("    src_bottom_left=(%.3f, %.3f)," % config.src_bottom_left)
    print("----------------------------\n")


def save_snapshot(cv2, overlay, bev) -> None:
    out_dir = ROOT / "data" / "bev_tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "original.png"), overlay)
    cv2.imwrite(str(out_dir / "bev.png"), bev)
    print("saved snapshot to %s" % out_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
