"""Offline BEV pipeline viewer for recorded car-view video.

Runs the full lane pipeline on a recorded clip (or image / camera) WITHOUT any
serial output, so you can check on a laptop how the YOLO mask warps into BEV and
how the centerline is extracted:

    raw frame -> YOLO mask -> warp_mask -> BevLaneGeometryEstimator

Two windows:
  - "BEV Replay"      : original frame + mask overlay + src trapezoid +
                        centerline mapped back + vehicle line + error text
  - "BEV Replay (top)": warped BEV mask + fitted centerline + vehicle center +
                        lookahead marker

Keys: space=pause/play  n=step one frame (when paused)  r=restart  q/esc=quit

Usage:
    python3 scripts/bev_replay.py --source clip.mp4
    python3 scripts/bev_replay.py --source clip.mp4 --device cpu --start-frame 300
    python3 scripts/bev_replay.py --source frame.png --show-bev-mask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.estimation.bev_lane import BevLaneConfig, BevLaneGeometryEstimator  # noqa: E402
from skku_autocar.perception.bev import BevConfig, BevTransformer  # noqa: E402
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOW = "BEV Replay"
BEV_WINDOW = "BEV Replay (top)"


def main(argv=None) -> int:
    import cv2

    args = parse_args(argv)
    model_path = resolve_model_path(args.model)

    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(model_path=model_path, confidence=args.conf, image_size=args.imgsz, device=args.device)
    )
    transformer = BevTransformer(BevConfig())
    estimator = BevLaneGeometryEstimator(BevLaneConfig(lookahead_y_ratio=args.lookahead))

    if args.save:
        return export_video(cv2, args, segmenter, transformer, estimator)

    frames, release, is_static = open_source(cv2, args)
    print("model=%s device=%s" % (model_path, segmenter.device))
    print("keys: space=pause/play  n=step  r=restart  q/esc=quit")

    paused = False
    frame = None
    try:
        while True:
            if not paused or frame is None:
                new_frame = frames()
                if new_frame is None:
                    if is_static:
                        break
                    frames_restart(cv2, frames)
                    continue
                frame = new_frame

            mask_result = segmenter.segment(frame)
            bev_mask = transformer.warp_mask(mask_result.mask) if mask_result is not None else None
            lane = estimator.estimate(bev_mask)

            display = draw_original(cv2, frame, transformer, mask_result, lane, estimator)
            cv2.imshow(WINDOW, display)
            cv2.imshow(BEV_WINDOW, draw_bev(cv2, transformer, bev_mask, lane, estimator, args.show_bev_mask))

            key = cv2.waitKey(0 if (paused or is_static) else max(1, args.delay)) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("n"):
                paused = True
                frame = None  # force reading the next frame once
            if key == ord("r"):
                frames_restart(cv2, frames)
                frame = None
    finally:
        release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Offline BEV pipeline viewer for recorded video")
    parser.add_argument("--source", required=True, help="video path, image path, or camera index")
    parser.add_argument("--model", default="trained_model/best.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, cuda, ...")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lookahead", type=float, default=BevLaneConfig.lookahead_y_ratio)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--delay", type=int, default=30, help="ms between frames during playback")
    parser.add_argument("--show-bev-mask", action="store_true", help="show the raw warped mask instead of a black canvas")
    parser.add_argument("--save", action="store_true", help="non-interactive: run the whole clip and write a debug video (no GUI)")
    parser.add_argument("--out-dir", default="data/processed", help="output directory for --save")
    parser.add_argument("--fps", type=float, default=0.0, help="output fps for --save (0 = copy source fps)")
    return parser.parse_args(argv)


def export_video(cv2, args, segmenter, transformer, estimator) -> int:
    ext = Path(args.source).suffix.lower()
    if ext in IMAGE_EXTS or args.source.isdigit():
        raise RuntimeError("--save expects a video file source: %s" % args.source)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("could not open source: %s" % args.source)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_fps = args.fps if args.fps > 0 else (src_fps if src_fps > 0 else 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("%s_centerline.mp4" % Path(args.source).stem)

    out_w, out_h = transformer.out_size
    writer = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask_result = segmenter.segment(frame)
            bev_mask = transformer.warp_mask(mask_result.mask) if mask_result is not None else None
            lane = estimator.estimate(bev_mask)

            left = draw_original(cv2, frame, transformer, mask_result, lane, estimator)
            right = draw_bev(cv2, transformer, bev_mask, lane, estimator, args.show_bev_mask)
            h, w = left.shape[:2]
            scale = out_h / h
            left = cv2.resize(left, (int(w * scale), out_h))
            combined = cv2.hconcat([left, right])

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (combined.shape[1], combined.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("failed to open video writer: %s" % out_path)
            writer.write(combined)
            index += 1
            if index % 60 == 0:
                pct = (" (%d%%)" % int(100 * index / total)) if total else ""
                print("processed %d/%s frames%s" % (index, total or "?", pct))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print("saved %d frames -> %s (%.1f fps)" % (index, out_path, out_fps))
    return 0


def open_source(cv2, args):
    source = args.source
    ext = Path(source).suffix.lower()

    if ext in IMAGE_EXTS:
        image = cv2.imread(source)
        if image is None:
            raise RuntimeError("could not read image: %s" % source)
        return (lambda: image.copy()), (lambda: None), True

    cap_source = int(source) if source.isdigit() else source
    if isinstance(cap_source, int) and sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError("could not open source: %s" % source)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    def read():
        ok, frame = cap.read()
        return frame if ok else None

    read._cap = cap  # type: ignore[attr-defined]
    return read, cap.release, False


def frames_restart(cv2, frames) -> None:
    cap = getattr(frames, "_cap", None)
    if cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


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


def draw_original(cv2, frame, transformer, mask_result, lane, estimator):
    import numpy as np

    display = frame.copy()
    frame_hw = frame.shape[:2]

    if mask_result is not None:
        overlay = display.copy()
        overlay[mask_result.mask > 0] = (0, 220, 80)
        display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0)

    # src trapezoid used for the homography
    src = transformer.src_polygon(frame_hw).astype(np.int32)
    cv2.polylines(display, [src], isClosed=True, color=(0, 220, 255), thickness=2)

    # centerline mapped from BEV back onto the original frame
    if estimator.last_centerline_bev:
        pts = transformer.bev_to_frame(estimator.last_centerline_bev, frame_hw).astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(display, tuple(pts[i - 1]), tuple(pts[i]), (0, 0, 255), 2)

    height, width = frame_hw
    cv2.line(display, (width // 2, height), (width // 2, int(height * 0.5)), (255, 255, 0), 1)

    mask_name = mask_result.class_name if mask_result else "none"
    lines = [
        "mask=%s lane=%s conf=%.2f" % (mask_name, lane.reason, lane.confidence),
        "err=%.3f head=%.3f" % (lane.lateral_error_norm, lane.heading_error),
    ]
    for i, text in enumerate(lines):
        cv2.putText(display, text, (24, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return display


def draw_bev(cv2, transformer, bev_mask, lane, estimator, show_mask):
    import numpy as np

    out_w, out_h = transformer.out_size
    if bev_mask is not None and show_mask:
        canvas = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if bev_mask is not None:
            canvas[bev_mask > 0] = (0, 120, 40)

    # vehicle centerline (forward is up)
    cv2.line(canvas, (out_w // 2, 0), (out_w // 2, out_h), (0, 255, 0), 1)

    # fitted centerline in BEV coords
    pts = [(int(x), int(y)) for x, y in estimator.last_centerline_bev]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 0, 255), 2)

    if lane.found:
        cv2.circle(canvas, (int(lane.center_x), int(lane.target_y)), 6, (0, 0, 255), -1)
    return canvas


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
