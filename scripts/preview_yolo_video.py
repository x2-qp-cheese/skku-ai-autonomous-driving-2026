import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.estimation.lane_geometry import LaneGeometryConfig, MaskLaneGeometryEstimator
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from skku_autocar.planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig
from skku_autocar.runtime.yolo_drive_app import draw_debug, resolve_model_path


def main() -> int:
    args = parse_args()
    cv2 = load_cv2()

    input_path = Path(args.video).expanduser()
    if not input_path.exists():
        print("video not found: %s" % input_path)
        return 1

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        )
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("could not open video: %s" % input_path)
        return 1

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok:
        print("could not read first frame: %s" % input_path)
        cap.release()
        return 1

    if args.width:
        frame = resize_to_width(cv2, frame, args.width)

    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.fourcc),
        source_fps / max(1, args.stride),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print("could not open output writer: %s" % output_path)
        return 1

    processed = 0
    written = 0
    lost = 0
    started = time.monotonic()
    first_frame = frame
    while True:
        if processed == 0:
            frame = first_frame
        else:
            ok, frame = cap.read()
            if not ok:
                break
            if args.width:
                frame = resize_to_width(cv2, frame, args.width)

        if processed % args.stride == 0:
            mask_result = segmenter.segment(frame)
            lane = estimator.estimate(mask_result.mask if mask_result else None, frame.shape)
            command = follower.plan(lane)
            if not lane.found:
                lost += 1
            display = draw_debug(cv2, frame, mask_result, lane, command, True, source_fps)
            writer.write(display)
            written += 1

        processed += 1
        if args.max_frames and processed >= args.max_frames:
            break
        if args.progress and written > 0 and written % args.progress == 0:
            elapsed = time.monotonic() - started
            print("written=%d processed=%d/%s elapsed=%.1fs" % (written, processed, total_frames or "?", elapsed))

    cap.release()
    writer.release()
    elapsed = time.monotonic() - started
    print("input=%s" % input_path)
    print("model=%s" % model_path)
    print("device=%s" % segmenter.device)
    print("output=%s" % output_path)
    print("processed=%d written=%d lost=%d elapsed=%.1fs" % (processed, written, lost, elapsed))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a YOLO lane preview video")
    parser.add_argument("video")
    parser.add_argument("--model", default="trained_model/best.pt")
    parser.add_argument("--output", default="tmp/yolo_preview.mp4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--progress", type=int, default=30)
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
    parser.add_argument("--lookahead", type=float, default=0.72)
    parser.add_argument("--sample-top", type=float, default=0.45)
    parser.add_argument("--sample-bottom", type=float, default=0.92)
    parser.add_argument(
        "--vehicle-center-offset",
        type=float,
        default=0.0,
        help="vehicle center x offset as frame width ratio; positive makes centered targets steer left",
    )
    return parser.parse_args()


def resize_to_width(cv2, frame, width):
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    scale = width / float(current_width)
    return cv2.resize(frame, (width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
