"""Interactive BEV homography tuner.

Adjust the 4 source points of the ground-plane trapezoid with trackbars while
watching the warped bird's-eye view live. Press 'p' to print the ratios to paste
into BevConfig (or configs), 's' to save a snapshot, 'q'/esc to quit.

Usage:
    python3 scripts/bev_tune.py                 # camera 0
    python3 scripts/bev_tune.py --source 1      # camera index 1
    python3 scripts/bev_tune.py --source clip.mp4
    python3 scripts/bev_tune.py --source frame.png

Non-interactive export (no GUI/trackbars, uses BevConfig defaults) that writes a
side-by-side "original + trapezoid | warped BEV" debug video to data/processed
for offline debugging:

    python3 scripts/bev_tune.py --source clip.mp4 --save
    python3 scripts/bev_tune.py --source clip.mp4 --save --out-dir data/processed
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
from skku_autocar.estimation.bev_lane import BevLaneConfig, BevLaneGeometryEstimator  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOW = "BEV Tune (original)"
BEV_WINDOW = "BEV Tune (warped)"


def main(argv=None) -> int:
    import cv2

    args = parse_args(argv)
    if args.save:
        return export_video(cv2, args)

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
    parser.add_argument(
        "--save",
        action="store_true",
        help="non-interactive: warp the whole source video and write a debug clip (no GUI)",
    )
    parser.add_argument("--out-dir", default="data/processed", help="output directory for --save")
    parser.add_argument("--fps", type=float, default=0.0, help="output fps for --save (0 = copy source fps)")
    # [--save] src trapezoid overrides (ratios of frame size). Default to BevConfig.
    parser.add_argument("--top-y", type=float, default=BevConfig.src_top_left[1])
    parser.add_argument("--top-left-x", type=float, default=BevConfig.src_top_left[0])
    parser.add_argument("--top-right-x", type=float, default=BevConfig.src_top_right[0])
    parser.add_argument("--bottom-y", type=float, default=BevConfig.src_bottom_left[1])
    parser.add_argument("--bottom-left-x", type=float, default=BevConfig.src_bottom_left[0])
    parser.add_argument("--bottom-right-x", type=float, default=BevConfig.src_bottom_right[0])
    # Road centerline overlay (YOLO mask -> BEV -> curved centerline fit).
    parser.add_argument(
        "--centerline",
        dest="centerline",
        action="store_true",
        default=True,
        help="[--save] overlay the road-following centerline (default on)",
    )
    parser.add_argument(
        "--no-centerline",
        dest="centerline",
        action="store_false",
        help="[--save] disable the centerline overlay (BEV warp only)",
    )
    parser.add_argument("--model", default="trained_model/best.pt", help="YOLO seg model for the centerline")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, cuda, ...")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--lookahead", type=float, default=0.45, help="BEV row ratio for lateral error")
    parser.add_argument("--start-frame", type=int, default=0, help="[--save] first frame to process")
    parser.add_argument("--max-frames", type=int, default=0, help="[--save] process at most N frames (0 = all)")
    return parser.parse_args(argv)


def export_video(cv2, args) -> int:
    source = args.source
    if Path(source).suffix.lower() in IMAGE_EXTS:
        raise RuntimeError("--save expects a video source, not an image: %s" % source)

    cap_source = int(source) if source.isdigit() else source
    if isinstance(cap_source, int):
        raise RuntimeError("--save expects a video file, not a camera index: %s" % source)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError("could not open source: %s" % source)

    config = BevConfig(
        src_top_left=(args.top_left_x, args.top_y),
        src_top_right=(args.top_right_x, args.top_y),
        src_bottom_right=(args.bottom_right_x, args.bottom_y),
        src_bottom_left=(args.bottom_left_x, args.bottom_y),
        out_width=args.out_width,
        out_height=args.out_height,
        dst_x_margin=args.dst_margin,
    )
    transformer = BevTransformer(config)

    segmenter = None
    estimator = None
    if args.centerline:
        segmenter, estimator = build_centerline(args)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_fps = args.fps if args.fps > 0 else (src_fps if src_fps > 0 else 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames > 0:
        total = min(total or args.max_frames, args.max_frames)

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_centerline" if args.centerline else "_bev"
    out_path = out_dir / ("%s%s.mp4" % (Path(source).stem, suffix))

    writer = None
    index = 0
    try:
        while True:
            if args.max_frames > 0 and index >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_hw = frame.shape[:2]
            overlay = draw_src(cv2, frame.copy(), transformer, frame_hw)
            bev = transformer.warp_frame(frame)
            draw_bev_grid(cv2, bev)

            if segmenter is not None:
                draw_centerline(cv2, frame, overlay, bev, transformer, segmenter, estimator)

            combined = compose_side_by_side(cv2, overlay, bev, config)

            if writer is None:
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("failed to open video writer: %s" % out_path)

            writer.write(combined)
            index += 1
            if index % 30 == 0:
                pct = (" (%d%%)" % int(100 * index / total)) if total else ""
                print("processed %d/%s frames%s" % (index, total or "?", pct))
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print("saved %d frames -> %s (%.1f fps)" % (index, out_path, out_fps))
    return 0


def build_centerline(args):
    """Build the YOLO segmenter + BEV centerline estimator used for the overlay."""
    from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter

    model_path = resolve_model_path(args.model)
    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(model_path=model_path, confidence=args.conf, image_size=args.imgsz, device=args.device)
    )
    estimator = BevLaneGeometryEstimator(BevLaneConfig(lookahead_y_ratio=args.lookahead))
    print("centerline model=%s device=%s" % (model_path, segmenter.device))
    return segmenter, estimator


def resolve_model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), ROOT):
        candidate = base / value
        if candidate.exists():
            return candidate
    trained = ROOT / "trained_model"
    pts = sorted(trained.glob("*.pt")) if trained.exists() else []
    if value == "trained_model/best.pt" and len(pts) == 1:
        return pts[0]
    return ROOT / value


def draw_centerline(cv2, frame, overlay, bev, transformer, segmenter, estimator) -> None:
    """Run YOLO -> BEV -> centerline and draw the curved centerline on both panels."""
    import numpy as np

    frame_hw = frame.shape[:2]
    mask_result = segmenter.segment(frame)
    bev_mask = transformer.warp_mask(mask_result.mask) if mask_result is not None else None
    lane = estimator.estimate(bev_mask)

    # Corridor mask tint on the BEV panel so the drivable area is visible.
    if bev_mask is not None:
        bev[bev_mask > 0] = (0.35 * np.array((0, 200, 90)) + 0.65 * bev[bev_mask > 0]).astype(np.uint8)

    # Fitted (curved) centerline in BEV coords.
    pts_bev = [(int(x), int(y)) for x, y in estimator.last_centerline_bev]
    for i in range(1, len(pts_bev)):
        cv2.line(bev, pts_bev[i - 1], pts_bev[i], (0, 0, 255), 2)
    if lane.found:
        cv2.circle(bev, (int(lane.center_x), int(lane.target_y)), 6, (0, 0, 255), -1)

    # Same centerline mapped back onto the original frame.
    if estimator.last_centerline_bev:
        pts_frame = transformer.bev_to_frame(estimator.last_centerline_bev, frame_hw).astype(np.int32)
        for i in range(1, len(pts_frame)):
            cv2.line(overlay, tuple(pts_frame[i - 1]), tuple(pts_frame[i]), (0, 0, 255), 2)

    text = "lane=%s err=%.3f head=%.3f conf=%.2f" % (
        lane.reason, lane.lateral_error_norm, lane.heading_error, lane.confidence,
    )
    cv2.putText(overlay, text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def compose_side_by_side(cv2, overlay, bev, config: BevConfig):
    import numpy as np

    fh, fw = overlay.shape[:2]
    bh, bw = bev.shape[:2]
    canvas = np.zeros((max(fh, bh), fw + bw, 3), dtype=np.uint8)
    canvas[:fh, :fw] = overlay
    canvas[:bh, fw:fw + bw] = bev
    return canvas


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
# Defaults mirror the tuned BevConfig (see src/skku_autocar/perception/bev.py).
_BARS = [
    ("top_y", 400),
    ("top_left_x", 200),
    ("top_right_x", 800),
    ("bottom_y", 880),
    ("bottom_left_x", 0),
    ("bottom_right_x", 1000),
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
