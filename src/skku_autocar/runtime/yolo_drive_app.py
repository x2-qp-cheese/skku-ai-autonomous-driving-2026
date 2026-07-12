from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..control.serial_vehicle import SerialVehicleClient, SerialVehicleConfig
from ..estimation.bev_lane import BevLaneConfig, BevLaneGeometryEstimator
from ..estimation.lane_geometry import LaneGeometry, LaneGeometryConfig, MaskLaneGeometryEstimator
from ..perception.bev import BevConfig, BevTransformer
from ..perception.yolo_lane import YoloLaneConfig, YoloLaneMask, YoloLaneSegmenter
from ..planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig
from ..types import ControlCommand


LOG = logging.getLogger("skku_autocar.yolo_drive")
DEFAULT_MODEL = "trained_model/best.pt"


def main(argv: Optional[list] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.info("interrupted")
        return 130
    except Exception as exc:
        LOG.error("%s", exc)
        return 1


def run(args: argparse.Namespace) -> int:
    cv2 = load_cv2()
    model_path = resolve_model_path(args.model)

    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(
            model_path=model_path,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
        )
    )
    estimator = MaskLaneGeometryEstimator(
        LaneGeometryConfig(
            lookahead_y_ratio=args.lookahead,
            sample_top_y_ratio=args.sample_top,
            sample_bottom_y_ratio=args.sample_bottom,
            vehicle_center_x_offset_ratio=args.vehicle_center_offset,
        )
    )
    # BEV lane pipeline (same logic as scripts/bev_tune.py / bev_replay.py):
    # warp the YOLO mask to bird's-eye view, then fit the road-following
    # centerline there so the target direction respects road curvature.
    transformer = BevTransformer(BevConfig()) if args.bev else None
    bev_estimator = (
        BevLaneGeometryEstimator(
            BevLaneConfig(
                lookahead_y_ratio=args.bev_lookahead,
                vehicle_center_x_offset_ratio=args.vehicle_center_offset,
                center_smooth_alpha=args.bev_center_smooth,
                heading_smooth_alpha=args.bev_heading_smooth,
            )
        )
        if args.bev
        else None
    )
    follower = YoloLaneFollower(
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
            straight_steering_scale=args.straight_steering_scale,
            curve_steering_scale=args.curve_steering_scale,
            center_recovery_error_threshold=args.center_recovery_error_threshold,
            center_recovery_steering_boost=args.center_recovery_steering_boost,
            center_recovery_min_steering=args.center_recovery_min_steering,
            center_recovery_rate_limit=args.center_recovery_rate_limit,
            center_recovery_max_speed=args.center_recovery_max_speed,
            lane_lost_hold_frames=args.lane_lost_hold_frames,
        )
    )

    vehicle = None
    if not args.no_serial:
        vehicle = SerialVehicleClient(
            SerialVehicleConfig(
                port=args.serial_port,
                baudrate=args.baudrate,
                ready_timeout_s=args.ready_timeout,
            ),
            max_speed=args.max_speed,
            max_steering=args.max_steering,
        )
        vehicle.connect()
        LOG.info("serial connected: %s", vehicle.port)
    else:
        LOG.info("serial disabled: dry video/control preview mode")

    cap = open_camera(cv2, args.camera, args.width, args.height, args.fourcc)
    # A video-file source (not a live camera index) should loop for review instead
    # of erroring out at the end of the clip.
    is_video_file = not str(args.camera).isdigit()
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("camera frame read failed")

    recorder = DriveRecorder(cv2, args, first_frame.shape)
    running = False
    last_command_at = 0.0
    last_log_at = 0.0
    last_frame_at = time.monotonic()
    fps = 0.0
    command = ControlCommand.stop("paused")

    LOG.info("model=%s device=%s camera=%s", model_path, segmenter.device, args.camera)
    if recorder.enabled:
        LOG.info("recording raw video: %s", recorder.raw_video_path)
        if recorder.debug_video_path is not None:
            LOG.info("recording debug video: %s", recorder.debug_video_path)
    LOG.info("space: start/stop toggle | q or esc: quit")

    pending_frame = first_frame
    try:
        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if not ok and is_video_file:  # loop the clip for review
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("camera frame read failed")
            else:
                frame = pending_frame
                pending_frame = None

            now = time.monotonic()
            dt = max(1e-6, now - last_frame_at)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            last_frame_at = now

            mask_result = segmenter.segment(frame)
            bev_mask = None
            if transformer is not None:
                bev_mask = transformer.warp_mask(mask_result.mask) if mask_result else None
                lane = bev_estimator.estimate(bev_mask)
            else:
                lane = estimator.estimate(mask_result.mask if mask_result else None, frame.shape)
            command = follower.plan(lane) if running else ControlCommand.stop("paused")

            if vehicle is not None and now - last_command_at >= 1.0 / args.command_rate:
                vehicle.send(command)
                last_command_at = now

            display = draw_debug(cv2, frame, mask_result, lane, command, running, fps, transformer, bev_estimator)
            recorder.write(frame, display)
            cv2.imshow("YOLO Drive", display)
            if args.show_mask:
                # In BEV mode show the warped (bird's-eye) road mask so the road is
                # separated in top-down view; otherwise fall back to the frame mask.
                if bev_mask is not None:
                    cv2.imshow("BEV Lane Mask", draw_bev_mask_debug(cv2, bev_mask, lane))
                elif mask_result is not None:
                    cv2.imshow("YOLO Lane Mask", mask_result.mask)

            if now - last_log_at >= args.log_interval:
                log_status(mask_result, lane, command, running, fps, segmenter.device)
                last_log_at = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                running = not running
                LOG.info("driving: %s", "ON" if running else "OFF")
                if not running and vehicle is not None:
                    vehicle.stop("paused")
    finally:
        recorder.close()
        if vehicle is not None:
            try:
                try:
                    vehicle.stop("shutdown")
                except Exception as exc:
                    LOG.warning("serial stop failed during shutdown: %s", exc)
            finally:
                vehicle.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv: Optional[list]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 segmentation autonomous driving runtime")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="trained YOLO segmentation model path")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, 0, cuda, ...")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--camera", default="0", help="camera index or video path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--serial-port", default=None, help="Arduino port, e.g. COM3 or /dev/cu.usbmodemXXXX")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--ready-timeout", type=float, default=3.0)
    parser.add_argument("--no-serial", action="store_true", help="run without Arduino output")
    parser.add_argument("--speed", type=int, default=105)
    parser.add_argument("--max-speed", type=int, default=170)
    parser.add_argument("--min-curve-speed", type=int, default=60)
    parser.add_argument("--max-steering", type=int, default=120)
    parser.add_argument("--steering-rate-limit", type=int, default=110)
    parser.add_argument("--min-steering-rate-limit", type=int, default=40)
    parser.add_argument("--steering-release-rate-limit", type=int, default=22)
    parser.add_argument("--kp-lateral", type=float, default=190.0)
    parser.add_argument("--kd-lateral", type=float, default=45.0)
    parser.add_argument("--kp-heading", type=float, default=12.0)
    parser.add_argument("--kd-heading", type=float, default=4.0)
    parser.add_argument("--speed-curve-slowdown", type=int, default=70)
    parser.add_argument("--straight-steering-scale", type=float, default=0.45)
    parser.add_argument("--curve-steering-scale", type=float, default=1.45)
    parser.add_argument("--center-recovery-error-threshold", type=float, default=0.14)
    parser.add_argument("--center-recovery-steering-boost", type=float, default=2.0)
    parser.add_argument("--center-recovery-min-steering", type=int, default=85)
    parser.add_argument("--center-recovery-rate-limit", type=int, default=120)
    parser.add_argument("--center-recovery-max-speed", type=int, default=50)
    parser.add_argument(
        "--lane-lost-hold-frames",
        type=int,
        default=20,
        help="keep the last steering/speed for up to this many frames when the lane is not detected (e.g. crosswalks) before stopping",
    )
    # BEV lane pipeline (default on): warp the mask to bird's-eye view and fit the
    # road-following centerline there, same as scripts/bev_tune.py.
    parser.add_argument("--bev", dest="bev", action="store_true", default=True, help="use the BEV centerline pipeline (default)")
    parser.add_argument("--no-bev", dest="bev", action="store_false", help="use the legacy frame-plane lane estimator instead")
    parser.add_argument("--bev-lookahead", type=float, default=BevLaneConfig.lookahead_y_ratio, help="BEV row ratio where lateral error is measured")
    parser.add_argument(
        "--bev-center-smooth",
        type=float,
        default=BevLaneConfig.center_smooth_alpha,
        help="BEV lane-center EMA factor (0..1). 1.0 = no smoothing (target dot sits on the raw line, fast/jittery), lower = smoother/laggier",
    )
    parser.add_argument(
        "--bev-heading-smooth",
        type=float,
        default=BevLaneConfig.heading_smooth_alpha,
        help="BEV heading EMA factor (0..1). 1.0 = no smoothing, lower = smoother",
    )
    # Legacy frame-plane estimator params (used only with --no-bev).
    parser.add_argument("--lookahead", type=float, default=0.72)
    parser.add_argument("--sample-top", type=float, default=0.45)
    parser.add_argument("--sample-bottom", type=float, default=0.92)
    parser.add_argument(
        "--vehicle-center-offset",
        type=float,
        default=0.0,
        help="vehicle center x offset as frame width ratio; positive makes centered targets steer left",
    )
    parser.add_argument("--command-rate", type=float, default=20.0)
    parser.add_argument("--log-interval", type=float, default=0.5)
    parser.add_argument("--show-mask", action="store_true")
    parser.add_argument("--record", choices=("on", "off"), default="off")
    parser.add_argument("--record-dir", default="data/raw/drive_recordings")
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--record-fourcc", default="mp4v")
    parser.add_argument(
        "--record-debug",
        choices=("on", "off", "auto"),
        default="auto",
        help="save the annotated debug screen. auto=follow --record (on when --record on), on=always, off=never",
    )
    parser.add_argument("--debug-dir", default="data/processed", help="output dir for the debug screen video")
    return parser.parse_args(argv)


def resolve_model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    root_path = project_root() / path
    if root_path.exists():
        return root_path
    if value == DEFAULT_MODEL:
        trained_dir = project_root() / "trained_model"
        model_files = sorted(trained_dir.glob("*.pt")) if trained_dir.exists() else []
        if len(model_files) == 1:
            return model_files[0]
    return root_path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class DriveRecorder:
    def __init__(self, cv2: Any, args: argparse.Namespace, frame_shape: tuple):
        # Raw and debug (annotated) recording are independent so the debug overlay
        # with the on-screen parameter values can be captured to data/processed for
        # real-time tuning even without saving the large raw clip.
        # --record on saves the raw clip AND auto-saves the annotated debug screen
        # to data/processed (for reviewing/tuning). --record-debug on saves only the
        # debug screen. --record-debug off force-disables it even with --record on.
        self.raw_enabled = args.record == "on"
        self.debug_enabled = args.record_debug == "on" or (self.raw_enabled and args.record_debug == "auto")
        self.enabled = self.raw_enabled or self.debug_enabled
        self.raw_writer = None
        self.debug_writer = None
        self.raw_video_path: Optional[Path] = None
        self.debug_video_path: Optional[Path] = None
        self.frames = 0

        if not self.enabled:
            return

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        height, width = frame_shape[:2]
        fps = args.record_fps
        fourcc = cv2.VideoWriter_fourcc(*args.record_fourcc)

        if self.raw_enabled:
            raw_dir = self._resolve(args.record_dir) / session_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            self.raw_video_path = raw_dir / ("%s_raw.mp4" % session_id)
            self.raw_writer = cv2.VideoWriter(str(self.raw_video_path), fourcc, fps, (width, height))
            if not self.raw_writer.isOpened():
                raise RuntimeError("failed to open raw recorder: %s" % self.raw_video_path)

        if self.debug_enabled:
            debug_dir = self._resolve(args.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_video_path = debug_dir / ("%s_debug.mp4" % session_id)
            self.debug_writer = cv2.VideoWriter(str(self.debug_video_path), fourcc, fps, (width, height))
            if not self.debug_writer.isOpened():
                if self.raw_writer is not None:
                    self.raw_writer.release()
                raise RuntimeError("failed to open debug recorder: %s" % self.debug_video_path)

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else project_root() / path

    def write(self, raw_frame: Any, debug_frame: Any) -> None:
        if not self.enabled:
            return
        if self.raw_writer is not None:
            self.raw_writer.write(raw_frame)
        if self.debug_writer is not None:
            self.debug_writer.write(debug_frame)
        self.frames += 1

    def close(self) -> None:
        if self.raw_writer is not None:
            self.raw_writer.release()
            self.raw_writer = None
        if self.debug_writer is not None:
            self.debug_writer.release()
            self.debug_writer = None
        if self.enabled:
            if self.raw_video_path is not None:
                LOG.info("recorded frames=%d raw=%s", self.frames, self.raw_video_path)
            if self.debug_video_path is not None:
                LOG.info("recorded frames=%d debug=%s", self.frames, self.debug_video_path)


def open_camera(cv2: Any, camera: str, width: int, height: int, fourcc: str) -> Any:
    source: Any = int(camera) if str(camera).isdigit() else camera
    if isinstance(source, int) and sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif isinstance(source, int) and sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if not cap.isOpened():
        raise RuntimeError("camera could not be opened: %s" % camera)
    return cap


def draw_debug(
    cv2: Any,
    frame: Any,
    mask_result: Optional[YoloLaneMask],
    lane: LaneGeometry,
    command: ControlCommand,
    running: bool,
    fps: float,
    transformer: Any = None,
    bev_estimator: Any = None,
) -> Any:
    display = frame.copy()
    if mask_result is not None:
        overlay = display.copy()
        overlay[mask_result.mask > 0] = (0, 220, 80)
        display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0)

    height, width = display.shape[:2]
    if transformer is not None and bev_estimator is not None:
        # BEV mode: lane geometry is in BEV pixel coords. The camera is centered,
        # so the vehicle axis is the frame midline. Draw a single line joining the
        # center axis (car, bottom) to the target point mapped back from BEV.
        cv2.line(display, (width // 2, height), (width // 2, 0), (255, 255, 0), 1)
        if lane.found:
            target = transformer.bev_to_frame([(lane.center_x, lane.target_y)], (height, width))[0]
            tp = (int(target[0]), int(target[1]))
            cv2.line(display, (width // 2, height - 1), tp, (0, 0, 255), 2)
            cv2.circle(display, tp, 9, (255, 255, 255), -1)
            cv2.circle(display, tp, 6, (0, 0, 255), -1)
    else:
        cv2.line(display, (int(lane.vehicle_center_x), height), (int(lane.vehicle_center_x), 0), (255, 255, 0), 1)
        if lane.found:
            target = (int(lane.center_x), int(lane.target_y))
            cv2.circle(display, target, 7, (0, 0, 255), -1)
            cv2.line(display, (int(lane.vehicle_center_x), height - 1), target, (0, 0, 255), 2)

    status = "RUN" if running else "PAUSE"
    color = (0, 255, 0) if running else (0, 180, 255)
    mask_name = mask_result.class_name if mask_result else "none"
    lines = [
        status,
        "mask=%s lane=%s conf=%.2f" % (mask_name, lane.reason, lane.confidence),
        "err=%.3f head=%.3f speed=%d steer=%d" % (
            lane.lateral_error_norm,
            lane.heading_error,
            command.speed,
            command.steering,
        ),
        "fps=%.1f" % fps,
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (24, 42 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color if index == 0 else (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def draw_bev_mask_debug(cv2: Any, bev_mask: Any, lane: LaneGeometry) -> Any:
    """Binary BEV road mask (white=road) with the vehicle center axis, a single
    line joining it to the target point, and the tracked target dot."""
    canvas = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]

    # vehicle center axis (forward is up), cyan
    cx = int(lane.vehicle_center_x) if lane is not None else w // 2
    cv2.line(canvas, (cx, 0), (cx, h), (255, 255, 0), 1)

    if lane is not None and lane.found:
        target = (int(lane.center_x), int(lane.target_y))
        # single line connecting the center axis (car, bottom) to the target point
        cv2.line(canvas, (cx, h - 1), target, (0, 0, 255), 2)
        # tracked target point: red dot with a white outline for visibility
        cv2.circle(canvas, target, 7, (255, 255, 255), -1)
        cv2.circle(canvas, target, 5, (0, 0, 255), -1)
    return canvas


def log_status(
    mask_result: Optional[YoloLaneMask],
    lane: LaneGeometry,
    command: ControlCommand,
    running: bool,
    fps: float,
    device: str,
) -> None:
    mask_name = mask_result.class_name if mask_result else "none"
    mask_conf = mask_result.confidence if mask_result else 0.0
    inference_ms = mask_result.inference_ms if mask_result else 0.0
    LOG.info(
        "run=%s device=%s fps=%.1f infer=%.1fms mask=%s %.2f lane=%s err=%.3f head=%.3f speed=%d steer=%d",
        "on" if running else "off",
        device,
        fps,
        inference_ms,
        mask_name,
        mask_conf,
        lane.reason,
        lane.lateral_error_norm,
        lane.heading_error,
        command.speed,
        command.steering,
    )


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for camera capture/display") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
