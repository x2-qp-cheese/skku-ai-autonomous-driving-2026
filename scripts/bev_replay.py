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
                BevCorridorConfig(
                    lookahead_y_ratio=args.lookahead,
                    vehicle_center_x_offset_ratio=args.vehicle_center_offset,
                    poly_degree=args.poly_degree,
                    centerline_bias=args.centerline_bias,
                    lane_width_px=args.lane_width_px,
                    center_anchor=args.center_anchor == "on",
                    center_smooth_alpha=args.center_smooth,
                    max_center_jump_px=args.max_center_jump,
                    max_coast_frames=args.max_coast_frames,
                    max_width_jump_px=args.max_width_jump,
                    crosswalk_lane_width_px=args.crosswalk_lane_width_px,
                    crosswalk_center_smooth_alpha=args.crosswalk_center_smooth,
                    crosswalk_max_center_jump_px=args.crosswalk_max_center_jump,
                    crosswalk_option=args.crosswalk_option,
                    crosswalk_right_offset_px=args.crosswalk_right_offset_px,
                    virtual_hold=args.virtual_hold == "on",
                    vehicle_width_px=args.vehicle_width_px,
                    virtual_hold_max_frames=args.virtual_hold_max_frames,
                    virtual_hold_confidence=args.virtual_hold_conf,
                    virtual_hold_recenter_alpha=args.virtual_hold_recenter,
                )
            )
        else:
            self.estimator = BevLaneGeometryEstimator(
                BevLaneConfig(
                    lookahead_y_ratio=args.lookahead,
                    vehicle_center_x_offset_ratio=args.vehicle_center_offset,
                )
            )
        self.follower: Optional[YoloLaneFollower] = (
            YoloLaneFollower(
                YoloLaneFollowerConfig(
                    base_speed=args.speed,
                    max_speed=args.max_speed,
                    min_curve_speed=args.min_curve_speed,
                    max_steering=args.max_steering,
                    steering_rate_limit=args.steering_rate_limit,
                    min_steering_rate_limit=args.min_steering_rate_limit,
                    steering_release_rate_limit=args.steering_release_rate_limit,
                    kp_lateral=args.kp_lateral,
                    kd_lateral=args.kd_lateral,
                    kp_heading=args.kp_heading,
                    kd_heading=args.kd_heading,
                    speed_curve_slowdown=args.speed_curve_slowdown,
                    lateral_priority_threshold=args.lateral_priority_threshold,
                    curve_strength_alpha=args.curve_strength_alpha,
                    straight_steering_scale=args.straight_steering_scale,
                    curve_steering_scale=args.curve_steering_scale,
                    center_recovery_error_threshold=args.center_recovery_error_threshold,
                    center_recovery_steering_boost=args.center_recovery_steering_boost,
                    center_recovery_min_steering=args.center_recovery_min_steering,
                    center_recovery_rate_limit=args.center_recovery_rate_limit,
                    center_recovery_max_speed=args.center_recovery_max_speed,
                    lane_lost_hold_frames=args.lane_lost_hold_frames,
                    pure_pursuit=args.pure_pursuit,
                    pure_pursuit_gain=args.pp_gain,
                    pure_pursuit_full_angle=args.pp_full_angle,
                )
            )
            if (args.follower or args.dump_csv)
            else None
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
            # The estimator's last_class_name is authoritative across all paths
            # (tierN / coast / virtual-hold / none), so trust it even when YOLO
            # returned nothing this frame (virtual-hold still drives a lane).
            mask_name = self.estimator.last_class_name
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
    bev_config = build_bev_config(args)
    transformer = BevTransformer(bev_config)
    pipeline = ReplayPipeline(segmenter, transformer, args)
    print_bev_ratios(bev_config)

    if args.save:
        return export_video(cv2, args, pipeline, transformer)

    frames, release, is_static = open_source(cv2, args)
    csv_writer = CsvDumper(args.dump_csv)
    recorder = DebugRecorder(cv2, args.record_debug, args.out_dir, args.source, args.fps)
    _, out_h = transformer.out_size
    print("model=%s device=%s pipeline=%s" % (model_path, segmenter.device, "corridor" if args.corridor else "legacy"))
    if recorder.enabled:
        print("recording debug view -> %s" % recorder.path)
    print("keys: space=pause/play  n=step  r=restart  p=print bev ratios  q/esc=quit")

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
            if key == ord("p"):
                print_bev_ratios(bev_config)
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
    parser.add_argument(
        "--vehicle-center-offset",
        type=float,
        default=BevCorridorConfig.vehicle_center_x_offset_ratio,
        help="vehicle center x offset as width ratio; positive shifts the vehicle reference axis right (camera mounted left of the car centerline). Default matches the last good run (~0.585 frame x)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--delay", type=int, default=30, help="ms between frames during playback")
    parser.add_argument("--show-bev-mask", action="store_true", help="show the raw warped mask instead of a black canvas")
    # BEV homography (ground-plane trapezoid) overrides. Default to the tuned
    # BevConfig; pass these to re-fit the mapping to a new camera angle without
    # editing source, then press 'p' to print the ratios to paste into BevConfig.
    parser.add_argument("--bev-top-y", type=float, default=BevConfig.src_top_left[1],
                        help="src trapezoid TOP edge y (ratio of frame height)")
    parser.add_argument("--bev-bottom-y", type=float, default=BevConfig.src_bottom_left[1],
                        help="src trapezoid BOTTOM edge y (ratio of frame height)")
    parser.add_argument("--bev-top-left-x", type=float, default=BevConfig.src_top_left[0],
                        help="src trapezoid top-left x (ratio of frame width)")
    parser.add_argument("--bev-top-right-x", type=float, default=BevConfig.src_top_right[0],
                        help="src trapezoid top-right x (ratio of frame width)")
    parser.add_argument("--bev-bottom-left-x", type=float, default=BevConfig.src_bottom_left[0],
                        help="src trapezoid bottom-left x (ratio of frame width; may be <0)")
    parser.add_argument("--bev-bottom-right-x", type=float, default=BevConfig.src_bottom_right[0],
                        help="src trapezoid bottom-right x (ratio of frame width; may be >1)")
    parser.add_argument("--bev-out-width", type=int, default=BevConfig.out_width)
    parser.add_argument("--bev-out-height", type=int, default=BevConfig.out_height)
    parser.add_argument("--bev-dst-margin", type=float, default=BevConfig.dst_x_margin,
                        help="horizontal margin (ratio) the src trapezoid maps to inside the BEV canvas")
    # Plan A: build the corridor in BEV from per-class masks.
    parser.add_argument("--corridor", action="store_true", help="use the BEV corridor pipeline (Plan A)")
    parser.add_argument(
        "--lane-width-px",
        type=float,
        default=BevCorridorConfig.lane_width_px,
        help="[--corridor] BEV lane width (px) between the center line and the outer side line; read the live measured value off the overlay to tune",
    )
    parser.add_argument("--center-anchor", choices=("on", "off"), default="on",
                        help="[--corridor] anchor centerline on center line + half smoothed width (less jitter); off = raw midpoint")
    parser.add_argument("--center-smooth", type=float, default=BevCorridorConfig.center_smooth_alpha,
                        help="[--corridor] center-x EMA factor (lower = smoother/laggier)")
    parser.add_argument("--max-center-jump", type=float, default=BevCorridorConfig.max_center_jump_px,
                        help="[--corridor] reject+coast a frame whose center_x jumps more than this many BEV px")
    parser.add_argument("--max-coast-frames", type=int, default=BevCorridorConfig.max_coast_frames,
                        help="[--corridor] max rejected frames to coast before lost")
    parser.add_argument("--max-width-jump", type=float, default=BevCorridorConfig.max_width_jump_px,
                        help="[--corridor] reject a lane-width measurement jumping more than this many px")
    parser.add_argument("--crosswalk-lane-width-px", type=float, default=BevCorridorConfig.crosswalk_lane_width_px,
                        help="[--corridor] fixed BEV lane width used to build the virtual centerline while a crosswalk is in view (zebra makes the measured width unreliable)")
    parser.add_argument("--crosswalk-center-smooth", type=float,
                        default=BevCorridorConfig.crosswalk_center_smooth_alpha,
                        help="[--corridor] center-x EMA factor while a crosswalk is visible (lower = steadier)")
    parser.add_argument("--crosswalk-max-center-jump", type=float,
                        default=BevCorridorConfig.crosswalk_max_center_jump_px,
                        help="[--corridor] reject+coast a crosswalk frame whose center_x jumps more than this many BEV px")
    parser.add_argument("--crosswalk-option", choices=("a", "b"),
                        default=BevCorridorConfig.crosswalk_option,
                        help="[--corridor] a=center-line virtual corridor, b=right-boundary fixed offset")
    parser.add_argument("--crosswalk-right-offset-px", type=float,
                        default=BevCorridorConfig.crosswalk_right_offset_px,
                        help="[--corridor] option B target distance left of the detected right boundary")
    # [--corridor] vehicle-width virtual-lane hold: when all lane evidence is gone
    # and coasting is exhausted, hold a straight vehicle-width virtual lane and
    # keep the car centered instead of braking.
    parser.add_argument("--virtual-hold", choices=("on", "off"), default="on",
                        help="[--corridor] hold a vehicle-width virtual lane to keep centered when the lane is lost")
    parser.add_argument("--vehicle-width-px", type=float, default=BevCorridorConfig.vehicle_width_px,
                        help="[--corridor] BEV pixel width of the car (virtual-lane width when holding)")
    parser.add_argument("--virtual-hold-max-frames", type=int, default=BevCorridorConfig.virtual_hold_max_frames,
                        help="[--corridor] hold the virtual lane at most this many frames before declaring lost")
    parser.add_argument("--virtual-hold-conf", type=float, default=BevCorridorConfig.virtual_hold_confidence,
                        help="[--corridor] confidence reported while holding the virtual lane")
    parser.add_argument("--virtual-hold-recenter", type=float, default=BevCorridorConfig.virtual_hold_recenter_alpha,
                        help="[--corridor] per-frame easing of the held center toward the vehicle axis (0=freeze, 1=snap straight)")
    parser.add_argument("--poly-degree", type=int, default=BevCorridorConfig.poly_degree,
                        help="[--corridor] polynomial degree for the BEV line/centerline fit (3 = follows S-curves; 2 = smoother on straights)")
    parser.add_argument("--centerline-bias", type=float, default=BevCorridorConfig.centerline_bias,
                        help="[--corridor] driving line position between boundaries: 0=center line, 0.5=midpoint, 1=outer side line. Raise if the car rides too far inside")
    # Plan D: run the controller offline.
    parser.add_argument(
        "--control-preset",
        choices=("legacy-frame",),
        default=None,
        help="apply a saved controller/perception parameter set; legacy-frame = pre-BEV 20260714 tuning",
    )
    parser.add_argument("--follower", action="store_true", help="run the lane follower and overlay speed/steering")
    parser.add_argument("--speed", type=int, default=YoloLaneFollowerConfig.base_speed, help="[--follower] base speed passed to the follower")
    parser.add_argument("--max-speed", type=int, default=YoloLaneFollowerConfig.max_speed)
    parser.add_argument("--min-curve-speed", type=int, default=YoloLaneFollowerConfig.min_curve_speed)
    parser.add_argument("--max-steering", type=int, default=YoloLaneFollowerConfig.max_steering)
    parser.add_argument("--steering-rate-limit", type=int, default=YoloLaneFollowerConfig.steering_rate_limit)
    parser.add_argument("--min-steering-rate-limit", type=int, default=YoloLaneFollowerConfig.min_steering_rate_limit)
    parser.add_argument("--steering-release-rate-limit", type=int, default=YoloLaneFollowerConfig.steering_release_rate_limit)
    parser.add_argument("--kp-lateral", type=float, default=YoloLaneFollowerConfig.kp_lateral)
    parser.add_argument("--kd-lateral", type=float, default=YoloLaneFollowerConfig.kd_lateral)
    parser.add_argument("--kp-heading", type=float, default=YoloLaneFollowerConfig.kp_heading)
    parser.add_argument("--kd-heading", type=float, default=YoloLaneFollowerConfig.kd_heading)
    parser.add_argument("--speed-curve-slowdown", type=int, default=YoloLaneFollowerConfig.speed_curve_slowdown)
    parser.add_argument("--lateral-priority-threshold", type=float, default=YoloLaneFollowerConfig.lateral_priority_threshold)
    parser.add_argument("--curve-strength-alpha", type=float, default=YoloLaneFollowerConfig.curve_strength_alpha)
    parser.add_argument("--straight-steering-scale", type=float, default=YoloLaneFollowerConfig.straight_steering_scale)
    parser.add_argument("--curve-steering-scale", type=float, default=YoloLaneFollowerConfig.curve_steering_scale)
    parser.add_argument("--center-recovery-error-threshold", type=float, default=YoloLaneFollowerConfig.center_recovery_error_threshold)
    parser.add_argument("--center-recovery-steering-boost", type=float, default=YoloLaneFollowerConfig.center_recovery_steering_boost)
    parser.add_argument("--center-recovery-min-steering", type=int, default=YoloLaneFollowerConfig.center_recovery_min_steering)
    parser.add_argument("--center-recovery-rate-limit", type=int, default=YoloLaneFollowerConfig.center_recovery_rate_limit)
    parser.add_argument("--center-recovery-max-speed", type=int, default=YoloLaneFollowerConfig.center_recovery_max_speed)
    parser.add_argument("--lane-lost-hold-frames", type=int, default=YoloLaneFollowerConfig.lane_lost_hold_frames)
    parser.add_argument("--pure-pursuit", action="store_true",
                        help="[--follower] steer via pure pursuit toward the BEV lookahead point (better on curves) instead of the lateral+heading PID")
    parser.add_argument("--pp-gain", type=float, default=YoloLaneFollowerConfig.pure_pursuit_gain,
                        help="[--pure-pursuit] steering units per radian of lookahead angle (larger = sharper)")
    parser.add_argument("--pp-full-angle", type=float, default=YoloLaneFollowerConfig.pure_pursuit_full_angle,
                        help="[--pure-pursuit] lookahead angle (rad) at which curve-speed slowdown saturates")
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
    args = parser.parse_args(argv)
    if args.control_preset == "legacy-frame":
        apply_legacy_frame_preset(args)
    return args


def apply_legacy_frame_preset(args) -> None:
    """Saved pre-BEV parameters from the 2026-07-14 camera run."""
    values = {
        "speed": 255,
        "min_curve_speed": 255,
        "max_speed": 255,
        "max_steering": 155,
        "kp_lateral": 188.0,
        "kd_lateral": 100.0,
        "kp_heading": 1.5,
        "kd_heading": 0.3,
        "speed_curve_slowdown": 255,
        "min_steering_rate_limit": 85,
        "steering_rate_limit": 165,
        "steering_release_rate_limit": 2,
        "lateral_priority_threshold": 0.16,
        "curve_strength_alpha": 0.18,
        "straight_steering_scale": 0.46,
        "curve_steering_scale": 1.18,
        "center_recovery_error_threshold": 0.26,
        "center_recovery_steering_boost": 1.10,
        "center_recovery_min_steering": 40,
        "center_recovery_rate_limit": 115,
        "center_recovery_max_speed": 255,
        "lane_lost_hold_frames": 10,
        "vehicle_center_offset": 0.085,
        "conf": 0.45,
        "lookahead": 0.65,
    }
    for name, value in values.items():
        setattr(args, name, value)


def build_bev_config(args) -> BevConfig:
    """Assemble the BEV homography from the --bev-* overrides (default = BevConfig)."""
    return BevConfig(
        src_top_left=(args.bev_top_left_x, args.bev_top_y),
        src_top_right=(args.bev_top_right_x, args.bev_top_y),
        src_bottom_right=(args.bev_bottom_right_x, args.bev_bottom_y),
        src_bottom_left=(args.bev_bottom_left_x, args.bev_bottom_y),
        out_width=args.bev_out_width,
        out_height=args.bev_out_height,
        dst_x_margin=args.bev_dst_margin,
    )


def print_bev_ratios(config: BevConfig) -> None:
    print("\n--- paste into BevConfig (src/skku_autocar/perception/bev.py) ---")
    print("    src_top_left=(%.3f, %.3f)," % config.src_top_left)
    print("    src_top_right=(%.3f, %.3f)," % config.src_top_right)
    print("    src_bottom_right=(%.3f, %.3f)," % config.src_bottom_right)
    print("    src_bottom_left=(%.3f, %.3f)," % config.src_bottom_left)
    print("----------------------------------------------------------------\n")


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
    # Vehicle reference axis: the camera sits left of the car centerline, so the
    # true vehicle center is shifted right by vehicle_center_x_offset_ratio. Draw
    # it (and the car origin) there so the overlay matches the lateral-error ref.
    offset = getattr(estimator.config, "vehicle_center_x_offset_ratio", 0.0)
    vehicle_x = int(width * (0.5 + offset))
    cv2.line(display, (vehicle_x, height), (vehicle_x, int(height * 0.5)), (255, 255, 0), 1)

    # red target point, back-projected from BEV onto the real frame (same dot as
    # the top view), with a line from the car to it
    if lane.found:
        target = transformer.bev_to_frame([(lane.center_x, lane.target_y)], frame_hw)[0]
        tp = (int(target[0]), int(target[1]))
        cv2.line(display, (vehicle_x, height - 1), tp, (0, 0, 255), 2)
        cv2.circle(display, tp, 9, (255, 255, 255), -1)
        cv2.circle(display, tp, 6, (0, 0, 255), -1)

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
        draw_steering_bar(cv2, display, result.command, vehicle_x)
    return display


def draw_steering_bar(cv2, display, command: ControlCommand, vehicle_center_x: int) -> None:
    h, w = display.shape[:2]
    # Use the same calibrated vehicle axis as lateral-error calculation and the
    # cyan center marker. The camera is not mounted at the vehicle's exact center,
    # so anchoring this arrow at w/2 made it visually disagree with steering.
    cx = max(0, min(w - 1, int(vehicle_center_x)))
    base_y = h - 30
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

    # vehicle centerline (forward is up), shifted right by the camera-mount offset
    offset = getattr(estimator.config, "vehicle_center_x_offset_ratio", 0.0)
    vehicle_x = int(out_w * (0.5 + offset))
    cv2.line(canvas, (vehicle_x, 0), (vehicle_x, out_h), (0, 255, 0), 1)

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
