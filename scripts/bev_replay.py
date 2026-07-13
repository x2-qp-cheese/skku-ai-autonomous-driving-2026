"""Offline BEV pipeline viewer + tuning harness for recorded car-view video.

Runs the full lane pipeline on a recorded clip (or image / camera) WITHOUT any
serial output, so you can check on a laptop how the YOLO mask warps into BEV and
how the centerline is extracted, and (Plan D) overlay what the controller would
command frame by frame.

Two pipelines:
  legacy (default): YOLO fused corridor mask -> warp_mask -> BevLaneGeometryEstimator
  --corridor (Plan A): YOLO per-class masks -> warp EACH -> BevCorridorLaneEstimator
                       (corridor built in BEV, so a virtual lane is always a
                       parallel curve of the real one)

Plan D tuning aids:
  --follower       run YoloLaneFollower on the lane and overlay speed/steering
  --dump-csv PATH  write per-frame telemetry to CSV (implies --follower)

Two windows:
  - "BEV Replay"      : original frame + mask overlay + src trapezoid +
                        centerline mapped back + vehicle line + error/command text
  - "BEV Replay (top)": warped BEV mask + fitted centerline + lane boundaries +
                        vehicle center + lookahead marker

Keys: space=pause/play  n=step one frame (when paused)  r=restart  q/esc=quit

Usage:
    python3 scripts/bev_replay.py --source clip.mp4
    python3 scripts/bev_replay.py --source clip.mp4 --corridor --follower
    python3 scripts/bev_replay.py --source clip.mp4 --corridor --dump-csv run.csv --save
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.estimation.bev_corridor import (  # noqa: E402
    BevClassMasks,
    BevCorridorConfig,
    BevCorridorLaneEstimator,
    warp_class_masks,
)
from skku_autocar.estimation.bev_lane import BevLaneConfig, BevLaneGeometryEstimator  # noqa: E402
from skku_autocar.estimation.lane_geometry import LaneGeometry  # noqa: E402
from skku_autocar.perception.bev import BevConfig, BevTransformer  # noqa: E402
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter  # noqa: E402
from skku_autocar.planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig  # noqa: E402
from skku_autocar.types import ControlCommand  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOW = "BEV Replay"
BEV_WINDOW = "BEV Replay (top)"


@dataclass
class FrameResult:
    frame_mask: Any            # fused frame-space mask (for the green overlay), or None
    bev_mask: Any             # fused BEV mask (for the top view), or None
    lane: LaneGeometry
    command: Optional[ControlCommand]
    mask_name: str
    tier: int
    lane_width_px: float
    crosswalk_mask: Any = None      # frame-space crosswalk mask (magenta), or None
    crosswalk_bev: Any = None       # BEV crosswalk mask (magenta), or None
    crosswalk_visible: bool = False


class ReplayPipeline:
    """Runs one pipeline (legacy or corridor) + optional follower per frame."""

    def __init__(self, segmenter: YoloLaneSegmenter, transformer: BevTransformer, args: argparse.Namespace):
        self.segmenter = segmenter
        self.transformer = transformer
        self.corridor = args.corridor
        if args.corridor:
            self.estimator: Any = BevCorridorLaneEstimator(
                BevCorridorConfig(lookahead_y_ratio=args.lookahead, lane_width_px=args.lane_width_px)
            )
        else:
            self.estimator = BevLaneGeometryEstimator(BevLaneConfig(lookahead_y_ratio=args.lookahead))
        self.follower: Optional[YoloLaneFollower] = (
            YoloLaneFollower(YoloLaneFollowerConfig(base_speed=args.speed)) if (args.follower or args.dump_csv) else None
        )

    def process(self, frame: Any) -> FrameResult:
        crosswalk_mask = None
        crosswalk_bev = None
        crosswalk_visible = False
        if self.corridor:
            class_masks = self.segmenter.segment_class_masks(frame)
            bev = warp_class_masks(self.transformer, class_masks)
            lane = self.estimator.estimate(bev)
            frame_mask = fuse_masks([*class_masks.center, *class_masks.side, *class_masks.lane])
            bev_mask = fuse_masks([*bev.center, *bev.side, *bev.lane])
            crosswalk_mask = fuse_masks(list(class_masks.crosswalk))
            crosswalk_bev = fuse_masks(list(bev.crosswalk))
            mask_name = self.estimator.last_class_name if class_masks.found else "none"
            tier = self.estimator.last_tier
            lane_width_px = self.estimator.last_lane_width_px
            crosswalk_visible = self.estimator.last_crosswalk_visible
        else:
            mask_result = self.segmenter.segment(frame)
            bev_mask = self.transformer.warp_mask(mask_result.mask) if mask_result is not None else None
            lane = self.estimator.estimate(bev_mask)
            frame_mask = mask_result.mask if mask_result is not None else None
            mask_name = mask_result.class_name if mask_result is not None else "none"
            tier = 0
            lane_width_px = 0.0

        command = None
        if self.follower is not None:
            # Always plan (as if driving) so the overlay/CSV reflect real output.
            command = self.follower.plan(lane)
        return FrameResult(
            frame_mask, bev_mask, lane, command, mask_name, tier, lane_width_px,
            crosswalk_mask=crosswalk_mask, crosswalk_bev=crosswalk_bev,
            crosswalk_visible=crosswalk_visible,
        )


def fuse_masks(masks: List[Any]) -> Any:
    import numpy as np

    fused = None
    for mask in masks:
        arr = np.asarray(mask)
        fused = arr.copy() if fused is None else np.maximum(fused, arr)
    return fused


def main(argv=None) -> int:
    import cv2

    args = parse_args(argv)
    model_path = resolve_model_path(args.model)

    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(model_path=model_path, confidence=args.conf, image_size=args.imgsz, device=args.device)
    )
    transformer = BevTransformer(BevConfig())
    pipeline = ReplayPipeline(segmenter, transformer, args)

    if args.save:
        return export_video(cv2, args, pipeline, transformer)

    frames, release, is_static = open_source(cv2, args)
    csv_writer = CsvDumper(args.dump_csv)
    recorder = DebugRecorder(cv2, args.record_debug, args.out_dir, args.source, args.fps)
    _, out_h = transformer.out_size
    print("model=%s device=%s pipeline=%s" % (model_path, segmenter.device, "corridor" if args.corridor else "legacy"))
    if recorder.enabled:
        print("recording debug view -> %s" % recorder.path)
    print("keys: space=pause/play  n=step  r=restart  q/esc=quit")

    paused = False
    frame = None
    index = 0
    try:
        while True:
            is_new = not paused or frame is None
            if is_new:
                new_frame = frames()
                if new_frame is None:
                    if is_static:
                        break
                    frames_restart(cv2, frames)
                    continue
                frame = new_frame

            result = pipeline.process(frame)

            display = draw_original(cv2, frame, transformer, pipeline.estimator, result)
            bev_display = draw_bev(cv2, transformer, pipeline.estimator, result, args.show_bev_mask)
            cv2.imshow(WINDOW, display)
            cv2.imshow(BEV_WINDOW, bev_display)
            # Record/dump only newly read frames so pausing does not duplicate.
            if is_new:
                csv_writer.write(index, result)
                if recorder.enabled:
                    recorder.write(combine_views(cv2, display, bev_display, out_h))
                index += 1

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
        csv_writer.close()
        recorder.close()
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
    # Plan A: build the corridor in BEV from per-class masks.
    parser.add_argument("--corridor", action="store_true", help="use the BEV corridor pipeline (Plan A)")
    parser.add_argument(
        "--lane-width-px",
        type=float,
        default=BevCorridorConfig.lane_width_px,
        help="[--corridor] BEV lane width (px) between the center line and the outer side line; read the live measured value off the overlay to tune",
    )
    # Plan D: run the controller offline.
    parser.add_argument("--follower", action="store_true", help="run the lane follower and overlay speed/steering")
    parser.add_argument("--speed", type=int, default=YoloLaneFollowerConfig.base_speed, help="[--follower] base speed passed to the follower")
    parser.add_argument("--dump-csv", default=None, help="write per-frame telemetry CSV to this path (implies --follower)")
    parser.add_argument("--save", action="store_true", help="non-interactive: run the whole clip and write a debug video (no GUI)")
    parser.add_argument(
        "--record-debug",
        nargs="?",
        const="",
        default=None,
        help="record the interactive debug view (original+BEV) to a video while playing; optional PATH, else auto-named in --out-dir",
    )
    parser.add_argument("--out-dir", default="data/processed", help="output directory for --save / --record-debug")
    parser.add_argument("--fps", type=float, default=0.0, help="output fps for --save (0 = copy source fps)")
    return parser.parse_args(argv)


def combine_views(cv2, left, right, out_h):
    """Resize the original panel to the BEV height and place them side by side."""
    h, w = left.shape[:2]
    scale = out_h / h
    left = cv2.resize(left, (int(w * scale), out_h))
    return cv2.hconcat([left, right])


class DebugRecorder:
    """Writes the combined interactive debug view to a video, opened lazily on
    the first frame so it matches the real frame size."""

    def __init__(self, cv2, path: Optional[str], out_dir: str, source: str, fps: float):
        self.cv2 = cv2
        self.writer = None
        self.path: Optional[Path] = None
        self.fps = fps if fps > 0 else 30.0
        self.enabled = path is not None
        if not self.enabled:
            return
        directory = ROOT / out_dir if not Path(out_dir).is_absolute() else Path(out_dir)
        if path:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / ("%s_replay.mp4" % Path(source).stem)

    def write(self, combined) -> None:
        if not self.enabled:
            return
        if self.writer is None:
            fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
            h, w = combined.shape[:2]
            self.writer = self.cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError("failed to open debug recorder: %s" % self.path)
        self.writer.write(combined)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("saved debug video -> %s" % self.path)


class CsvDumper:
    COLUMNS = [
        "frame", "pipeline_mask", "tier", "lane_width_px", "crosswalk",
        "found", "confidence", "err_norm", "heading", "target_y", "center_x",
        "speed", "steering", "brake", "reason",
    ]

    def __init__(self, path: Optional[str]):
        self._file = None
        self._writer = None
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._file = out.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def write(self, index: int, result: FrameResult) -> None:
        if self._writer is None:
            return
        lane = result.lane
        cmd = result.command
        self._writer.writerow([
            index, result.mask_name, result.tier, "%.1f" % result.lane_width_px, int(result.crosswalk_visible),
            int(lane.found), "%.3f" % lane.confidence, "%.4f" % lane.lateral_error_norm,
            "%.4f" % lane.heading_error, "%.1f" % lane.target_y, "%.1f" % lane.center_x,
            cmd.speed if cmd else "", cmd.steering if cmd else "",
            int(cmd.brake) if cmd else "", cmd.reason if cmd else lane.reason,
        ])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def export_video(cv2, args, pipeline: ReplayPipeline, transformer: BevTransformer) -> int:
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
    suffix = "_corridor" if args.corridor else "_centerline"
    out_path = out_dir / ("%s%s.mp4" % (Path(args.source).stem, suffix))

    csv_writer = CsvDumper(args.dump_csv)
    out_w, out_h = transformer.out_size
    writer = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame)
            csv_writer.write(index, result)

            left = draw_original(cv2, frame, transformer, pipeline.estimator, result)
            right = draw_bev(cv2, transformer, pipeline.estimator, result, args.show_bev_mask)
            combined = combine_views(cv2, left, right, out_h)

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
        csv_writer.close()
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


def draw_original(cv2, frame, transformer, estimator, result: FrameResult):
    import numpy as np

    display = frame.copy()
    frame_hw = frame.shape[:2]
    lane = result.lane

    if result.frame_mask is not None:
        overlay = display.copy()
        overlay[result.frame_mask > 0] = (0, 220, 80)
        display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0)

    # crosswalk in magenta so it reads clearly against the green lane tint
    if result.crosswalk_mask is not None:
        overlay = display.copy()
        overlay[result.crosswalk_mask > 0] = (200, 0, 200)
        display = cv2.addWeighted(overlay, 0.45, display, 0.55, 0)

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

    lines = [
        "mask=%s lane=%s conf=%.2f" % (result.mask_name, lane.reason, lane.confidence),
        "err=%.3f head=%.3f" % (lane.lateral_error_norm, lane.heading_error),
    ]
    if result.tier:
        lines.append("tier=%d lane_w=%.0fpx" % (result.tier, result.lane_width_px))
    if result.command is not None:
        c = result.command
        lines.append("speed=%d steer=%d %s" % (c.speed, c.steering, "BRAKE" if c.brake else ""))
    for i, text in enumerate(lines):
        cv2.putText(display, text, (24, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    if lane.reason == "crosswalk":
        cv2.putText(display, "CROSSWALK - HOLD", (24, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 0, 200), 3, cv2.LINE_AA)
    elif result.crosswalk_visible:
        cv2.putText(display, "crosswalk (following lanes)", (24, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 0, 200), 2, cv2.LINE_AA)
    if result.command is not None:
        draw_steering_bar(cv2, display, result.command)
    return display


def draw_steering_bar(cv2, display, command: ControlCommand) -> None:
    h, w = display.shape[:2]
    cx, base_y = w // 2, h - 30
    half = w // 4
    cv2.line(display, (cx - half, base_y), (cx + half, base_y), (120, 120, 120), 2)
    cv2.line(display, (cx, base_y - 10), (cx, base_y + 10), (200, 200, 200), 1)
    # steering assumed within [-max_steering, max_steering]; clamp to bar width.
    frac = max(-1.0, min(1.0, command.steering / 120.0))
    tip = int(cx + frac * half)
    color = (0, 0, 255) if command.brake else (0, 255, 0)
    cv2.arrowedLine(display, (cx, base_y), (tip, base_y), color, 4, tipLength=0.3)


def draw_bev(cv2, transformer, estimator, result: FrameResult, show_mask):
    import numpy as np

    bev_mask = result.bev_mask
    out_w, out_h = transformer.out_size
    if bev_mask is not None and show_mask:
        canvas = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if bev_mask is not None:
            canvas[bev_mask > 0] = (0, 120, 40)

    # crosswalk in magenta
    if result.crosswalk_bev is not None:
        canvas[result.crosswalk_bev > 0] = (200, 0, 200)

    # vehicle centerline (forward is up)
    cv2.line(canvas, (out_w // 2, 0), (out_w // 2, out_h), (0, 255, 0), 1)

    # lane boundaries (corridor pipeline only), yellow/cyan
    _draw_polyline(cv2, canvas, getattr(estimator, "last_center_line_bev", []), (0, 200, 255))
    _draw_polyline(cv2, canvas, getattr(estimator, "last_right_line_bev", []), (255, 200, 0))

    # fitted centerline in BEV coords, red
    _draw_polyline(cv2, canvas, estimator.last_centerline_bev, (0, 0, 255))

    if result.lane.found:
        cv2.circle(canvas, (int(result.lane.center_x), int(result.lane.target_y)), 6, (0, 0, 255), -1)
    return canvas


def _draw_polyline(cv2, canvas, points, color) -> None:
    pts = [(int(x), int(y)) for x, y in points]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], color, 2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
