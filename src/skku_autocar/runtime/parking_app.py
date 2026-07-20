from __future__ import annotations

import argparse
import logging
import tempfile
import time
import zipfile
from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path
from typing import Any, Optional, Tuple

from ..control.serial_vehicle import (
    SerialVehicleClient,
    SerialVehicleConfig,
    UltrasonicReadings,
    parse_ultrasonic_line,
)
from ..estimation.parking_geometry import ParkingGeometry, ParkingGeometryEstimator
from ..estimation.parking_lidar import (
    LidarParkingObservation,
    LidarParkingSpaceEstimator,
    infer_dynamic_slot_polygon,
)
from ..parking_config import ParkingAppConfig, load_parking_config
from ..perception.bev import BevTransformer
from ..perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from ..planning.t_parking_planner import ParkingState, TParkingPlanner
from ..sensors.lidar import LidarCsvReplay, RplidarScanner


LOG = logging.getLogger("skku_autocar.parking")
ROOT = Path(__file__).resolve().parents[3]


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        LOG.exception("parking runtime failed: %s", exc)
        return 1


def run(args: argparse.Namespace) -> int:
    recording_temp = None
    if args.recording_zip:
        if args.serial:
            raise RuntimeError("--serial is forbidden with --recording-zip replay")
        recording_temp = tempfile.TemporaryDirectory(prefix="skku_parking_")
        video_path, lidar_path = extract_recording_zip(
            resolve_path(args.recording_zip),
            Path(recording_temp.name),
        )
        if args.source is None:
            args.source = str(video_path)
        if args.lidar_csv is None:
            args.lidar_csv = str(lidar_path)
    try:
        return run_prepared(args)
    finally:
        if recording_temp is not None:
            recording_temp.cleanup()


def run_prepared(args: argparse.Namespace) -> int:
    cv2 = load_cv2()
    import numpy as np

    config = apply_cli_overrides(
        load_parking_config(str(resolve_path(args.config))),
        args,
    )
    source = str(args.source) if args.source is not None else str(config.rear_camera.index)
    is_video = not source.isdigit()
    if is_video and args.serial:
        raise RuntimeError("--serial is forbidden for video replay; use a numeric live camera source")
    model_path = resolve_path(args.model or config.yolo.model_path)

    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(
            model_path=model_path,
            confidence=args.conf if args.conf is not None else config.yolo.confidence,
            image_size=args.imgsz if args.imgsz is not None else config.yolo.image_size,
            device=args.device or config.yolo.device,
            min_mask_area_ratio=config.yolo.min_mask_area_ratio,
        )
    )
    transformer = BevTransformer(config.bev)
    geometry_estimator = ParkingGeometryEstimator(config.geometry)
    lidar_estimator = LidarParkingSpaceEstimator(config.lidar)
    planner = TParkingPlanner(config.planner, config.path)

    lidar_replay = LidarCsvReplay(str(resolve_path(args.lidar_csv))) if args.lidar_csv else None
    lidar_scanner = None
    lidar_port = args.lidar_port
    if lidar_port is not None:
        lidar_scanner = RplidarScanner(lidar_port)
        lidar_scanner.start()
    if config.runtime.require_lidar and lidar_replay is None and lidar_scanner is None and not args.allow_no_lidar:
        raise RuntimeError("LiDAR is required: pass --lidar-csv, --lidar-port, or explicitly --allow-no-lidar")

    try:
        cap = open_capture(cv2, source, config)
    except Exception:
        if lidar_scanner is not None:
            lidar_scanner.close()
        raise
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    try:
        vehicle = open_vehicle(args, config) if args.serial else None
    except Exception:
        cap.release()
        if lidar_scanner is not None:
            lidar_scanner.close()
        raise
    last_command_at = 0.0
    last_ultrasonic: Optional[UltrasonicReadings] = None
    last_ultrasonic_at = float("-inf")
    run_started_at = time.monotonic()
    last_frame_at = run_started_at
    fps = 0.0
    frame_index = args.start_frame
    last_state = planner.state

    if args.auto_start:
        if not is_video:
            raise RuntimeError("--auto-start is allowed only for serial-disabled video replay")
        planner.start(video_elapsed_s(cv2, cap, frame_index, run_started_at, is_video))
        LOG.info("parking mission auto-started for replay")

    LOG.info("source=%s model=%s device=%s", source, model_path, segmenter.device)
    LOG.info("controls: SPACE=start/cancel | R=reset | Q/ESC=quit")
    if not args.serial:
        LOG.info("serial output disabled; pass --serial only after replay/calibration checks")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    break
                raise RuntimeError("rear camera frame read failed")

            monotonic_now = time.monotonic()
            if vehicle is not None:
                sample = newest_ultrasonic_sample(vehicle.read_lines())
                if sample is not None:
                    last_ultrasonic = sample
                    last_ultrasonic_at = monotonic_now
            dt = max(1e-6, monotonic_now - last_frame_at)
            fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            last_frame_at = monotonic_now
            elapsed = video_elapsed_s(cv2, cap, frame_index, run_started_at, is_video)

            class_masks = segmenter.segment_class_masks(frame)
            parking_masks = list(class_masks.lane)
            bev_masks = [transformer.warp_mask(mask) for mask in parking_masks]
            geometry = geometry_estimator.estimate(bev_masks, class_masks.lane_conf)

            lidar_observation, lidar_scan = current_lidar_observation(
                lidar_estimator,
                lidar_replay,
                lidar_scanner,
                elapsed + args.lidar_offset + config.runtime.lidar_video_offset_s,
                args.allow_no_lidar,
            )
            ultrasonic_fresh = (
                last_ultrasonic is not None
                and monotonic_now - last_ultrasonic_at
                <= config.planner.ultrasonic_stale_after_s
            )
            left_ultrasonic_mm = (
                last_ultrasonic.side_left_mm if ultrasonic_fresh else None
            )
            right_ultrasonic_mm = (
                last_ultrasonic.side_right_mm if ultrasonic_fresh else None
            )
            plan = planner.update(
                geometry,
                lidar_observation,
                elapsed,
                enabled=True,
                left_ultrasonic_mm=left_ultrasonic_mm,
                right_ultrasonic_mm=right_ultrasonic_mm,
            )
            if planner.state != last_state:
                LOG.info("parking state: %s -> %s (%s)", last_state.value, planner.state.value, plan.reason)
                last_state = planner.state

            if (
                vehicle is not None
                and monotonic_now - last_command_at
                >= 1.0 / max(1.0, config.runtime.command_rate_hz)
            ):
                sample = newest_ultrasonic_sample(vehicle.send(plan.command))
                if sample is not None:
                    last_ultrasonic = sample
                    last_ultrasonic_at = monotonic_now
                last_command_at = monotonic_now

            display, bev_display = draw_debug(
                cv2,
                np,
                frame,
                parking_masks,
                bev_masks,
                transformer,
                geometry,
                lidar_observation,
                plan,
                fps,
                left_ultrasonic_mm,
                right_ultrasonic_mm,
            )
            cv2.imshow("T Parking - Rear", display)
            cv2.imshow("T Parking - BEV", bev_display)
            cv2.imshow("T Parking - LiDAR", draw_lidar_debug(
                cv2,
                np,
                lidar_estimator.vehicle_points(lidar_scan),
                config,
                lidar_observation,
            ))

            key = cv2.waitKey(1 if not is_video else max(1, args.replay_delay_ms)) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if planner.state == ParkingState.IDLE:
                    planner.start(elapsed)
                    geometry_estimator.reset()
                    lidar_estimator.reset()
                    LOG.info("parking mission started")
                else:
                    planner.reset(elapsed)
                    if vehicle is not None:
                        vehicle.stop("operator_cancel")
                    LOG.info("parking mission cancelled")
            elif key == ord("r"):
                planner.reset(elapsed)
                geometry_estimator.reset()
                lidar_estimator.reset()
                if vehicle is not None:
                    vehicle.stop("operator_reset")
                LOG.info("parking mission reset")
            skipped = 0
            if is_video:
                for _ in range(max(1, args.frame_stride) - 1):
                    if not cap.grab():
                        break
                    skipped += 1
            frame_index += 1 + skipped
    finally:
        if vehicle is not None:
            try:
                vehicle.stop("parking_shutdown")
            finally:
                try:
                    vehicle.write_line("USOFF")
                finally:
                    vehicle.close()
        if lidar_scanner is not None:
            lidar_scanner.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def current_lidar_observation(
    estimator: LidarParkingSpaceEstimator,
    replay: Optional[LidarCsvReplay],
    scanner: Optional[RplidarScanner],
    elapsed_s: float,
    allow_no_lidar: bool,
) -> Tuple[LidarParkingObservation, Any]:
    if replay is not None:
        scan = replay.scan_at_elapsed(elapsed_s)
        return estimator.estimate(scan, now=scan.timestamp), scan
    if scanner is not None:
        scan = scanner.latest()
        return estimator.estimate(scan, now=time.time()), scan
    if allow_no_lidar:
        target = estimator.config.sensor_to_rear_axle_y_back_mm
        return LidarParkingObservation(
            timestamp=time.time(),
            valid=True,
            unsafe=False,
            observed_points=1,
            car_count=2,
            first_car_seen=True,
            second_car_seen=True,
            gap_found=True,
            gap_confirmed=True,
            gap_width_mm=estimator.config.expected_observed_gap_mm,
            gap_center_y_back_mm=target,
            entry_target_y_back_mm=target,
            entry_error_mm=0.0,
            entry_reached=True,
            reason="lidar_explicitly_bypassed",
        ), None
    return estimator.estimate(None), None


def draw_debug(
    cv2: Any,
    np: Any,
    frame: Any,
    frame_masks: list,
    bev_masks: list,
    transformer: BevTransformer,
    geometry: ParkingGeometry,
    lidar: LidarParkingObservation,
    plan: Any,
    fps: float,
    left_ultrasonic_mm: Optional[float] = None,
    right_ultrasonic_mm: Optional[float] = None,
) -> Tuple[Any, Any]:
    display = frame.copy()
    for index, mask in enumerate(frame_masks):
        color = np.asarray(parking_mask_color(index, geometry), dtype=np.float32)
        selected = mask > 0
        display[selected] = (0.55 * display[selected] + 0.45 * color).astype(np.uint8)

    bev_display = transformer.warp_frame(frame)
    for index, mask in enumerate(bev_masks):
        color = np.asarray(parking_mask_color(index, geometry), dtype=np.float32)
        selected = mask > 0
        bev_display[selected] = (0.45 * bev_display[selected] + 0.55 * color).astype(np.uint8)

    height, width = bev_display.shape[:2]
    vehicle_x = int(width / 2)
    cv2.line(bev_display, (vehicle_x, 0), (vehicle_x, height - 1), (90, 90, 90), 1)
    cv2.circle(bev_display, (vehicle_x, int(height * 0.95)), 7, (255, 255, 255), -1)
    draw_parking_line(cv2, bev_display, geometry.left, (255, 255, 0))
    draw_parking_line(cv2, bev_display, geometry.right, (0, 255, 0))
    draw_parking_line(cv2, bev_display, geometry.back, (0, 0, 255))
    draw_reverse_path(cv2, np, bev_display, plan.path)

    depth = "-" if geometry.depth_remaining_px is None else "%.1fpx" % geometry.depth_remaining_px
    safety = "-" if lidar.nearest_safety_mm is None else "%.0fmm" % lidar.nearest_safety_mm
    lines = (
        "state=%s speed=%d steer=%d" % (plan.state.value, plan.command.speed, plan.command.steering),
        "geom=%s n=%d conf=%.2f lat=%.2f head=%.1f depth=%s" % (
            geometry.reason,
            geometry.observed_line_count,
            geometry.confidence,
            geometry.lateral_error_norm,
            geometry.heading_error_deg,
            depth,
        ),
        "lidar=%s points=%d cars=%d gap=%s err=%s safety=%s" % (
            lidar.reason,
            lidar.observed_points,
            lidar.car_count,
            "-" if lidar.gap_width_mm is None else "%.0fmm" % lidar.gap_width_mm,
            "-" if lidar.entry_error_mm is None else "%+.0fmm" % lidar.entry_error_mm,
            safety,
        ),
        "first-car confirmed=%s edge=%s turnErr=%s trigger=%s" % (
            "Y" if lidar.first_car_confirmed else "N",
            (
                "-"
                if lidar.first_car_slot_edge_y_back_mm is None
                else "%+.0fmm" % lidar.first_car_slot_edge_y_back_mm
            ),
            (
                "-"
                if lidar.first_car_turn_error_mm is None
                else "%+.0fmm" % lidar.first_car_turn_error_mm
            ),
            "Y" if lidar.first_car_turn_reached else "N",
        ),
        "plan=%s" % plan.reason,
        "ultrasonic left=%s right=%s" % (
            "-" if left_ultrasonic_mm is None else "%.0fmm" % left_ultrasonic_mm,
            "-" if right_ultrasonic_mm is None else "%.0fmm" % right_ultrasonic_mm,
        ),
        "fps=%.1f | SPACE start/cancel | R reset | Q quit" % fps,
    )
    for index, text in enumerate(lines):
        cv2.putText(
            display,
            text,
            (18, 32 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    legend = "CYAN=LEFT  GREEN=RIGHT  RED=BACK  MAGENTA=UNCLASSIFIED"
    cv2.putText(
        display,
        legend,
        (18, max(24, display.shape[0] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bev_display,
        legend,
        (10, max(22, bev_display.shape[0] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display, bev_display


def parking_mask_color(index: int, geometry: ParkingGeometry) -> Tuple[int, int, int]:
    """Stable OpenCV BGR colors after the parking topology is classified."""

    if geometry.left is not None and geometry.left.mask_index == index:
        return (255, 255, 0)  # cyan
    if geometry.right is not None and geometry.right.mask_index == index:
        return (0, 255, 0)  # green
    if geometry.back is not None and geometry.back.mask_index == index:
        return (0, 0, 255)  # red
    return (255, 0, 255)  # magenta


def draw_lidar_debug(
    cv2: Any,
    np: Any,
    points: list,
    config: ParkingAppConfig,
    observation: LidarParkingObservation,
) -> Any:
    size = 600
    scale = 0.10  # 6 m across the full canvas.
    origin = (size // 2, size // 2)
    rotation_deg = config.runtime.lidar_display_rotation_deg
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    draw_world_segment(
        cv2, canvas, (-3000.0, 0.0), (3000.0, 0.0),
        origin, scale, rotation_deg, (55, 55, 55), 1,
    )
    draw_world_segment(
        cv2, canvas, (0.0, -3000.0), (0.0, 3000.0),
        origin, scale, rotation_deg, (55, 55, 55), 1,
    )
    draw_roi(
        cv2, canvas, config.lidar.safety_roi,
        origin, scale, rotation_deg, (0, 0, 255),
    )
    for x_right, y_back in points:
        px, py = world_to_lidar_pixel(
            x_right, y_back, origin, scale, rotation_deg
        )
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(canvas, (px, py), 2, (180, 220, 180), -1)
    for cluster in observation.car_clusters:
        draw_cluster(
            cv2, canvas, cluster, origin, scale, rotation_deg, (255, 180, 0)
        )
    if (
        observation.first_car_slot_edge_x_right_mm is not None
        and observation.first_car_slot_edge_y_back_mm is not None
    ):
        first_edge = (
            observation.first_car_slot_edge_x_right_mm,
            observation.first_car_slot_edge_y_back_mm,
        )
        turn_target = (
            observation.first_car_slot_edge_x_right_mm,
            config.lidar.first_car_turn_target_y_back_mm,
        )
        draw_world_segment(
            cv2,
            canvas,
            first_edge,
            turn_target,
            origin,
            scale,
            rotation_deg,
            (0, 255, 255),
            2,
        )
        cv2.circle(
            canvas,
            world_to_lidar_pixel(
                first_edge[0],
                first_edge[1],
                origin,
                scale,
                rotation_deg,
            ),
            6,
            (0, 255, 255),
            -1,
        )
    dynamic_slot = infer_dynamic_slot_polygon(
        observation,
        config.lidar.parking_space_depth_mm,
    )
    if dynamic_slot is not None:
        slot_color = (0, 120, 200) if observation.coasted else (0, 180, 255)
        draw_world_polygon(
            cv2,
            canvas,
            dynamic_slot,
            origin,
            scale,
            rotation_deg,
            slot_color,
            3,
        )
        # The first two corners are the detected, sensor-facing obstacle
        # surfaces where the parking-space entrance begins.
        for edge in dynamic_slot[:2]:
            cv2.circle(
                canvas,
                world_to_lidar_pixel(
                    edge[0], edge[1], origin, scale, rotation_deg
                ),
                5,
                slot_color,
                -1,
            )
        slot_depth_first = (
            (dynamic_slot[0][0] + dynamic_slot[1][0]) / 2.0,
            (dynamic_slot[0][1] + dynamic_slot[1][1]) / 2.0,
        )
        slot_depth_second = (
            (dynamic_slot[3][0] + dynamic_slot[2][0]) / 2.0,
            (dynamic_slot[3][1] + dynamic_slot[2][1]) / 2.0,
        )
        draw_world_segment(
            cv2,
            canvas,
            slot_depth_first,
            slot_depth_second,
            origin,
            scale,
            rotation_deg,
            (0, 255, 0),
            3,
        )
        slot_center = (
            (slot_depth_first[0] + slot_depth_second[0]) / 2.0,
            (slot_depth_first[1] + slot_depth_second[1]) / 2.0,
        )
        cv2.circle(
            canvas,
            world_to_lidar_pixel(
                slot_center[0], slot_center[1], origin, scale, rotation_deg
            ),
            6,
            (0, 255, 0),
            -1,
        )
    draw_vehicle_outline(
        cv2,
        canvas,
        config.runtime.lidar_debug_vehicle_width_mm,
        config.runtime.lidar_debug_vehicle_length_mm,
        config.runtime.lidar_debug_sensor_behind_vehicle_rear_mm,
        origin,
        scale,
        rotation_deg,
    )
    lidar_pixel = world_to_lidar_pixel(0.0, 0.0, origin, scale, rotation_deg)
    cv2.drawMarker(
        canvas,
        lidar_pixel,
        (255, 0, 255),
        cv2.MARKER_CROSS,
        12,
        2,
        cv2.LINE_AA,
    )
    rear_reference_y = config.lidar.sensor_to_rear_axle_y_back_mm
    half_vehicle_width = config.runtime.lidar_debug_vehicle_width_mm / 2.0
    draw_world_segment(
        cv2,
        canvas,
        (-half_vehicle_width, rear_reference_y),
        (half_vehicle_width, rear_reference_y),
        origin,
        scale,
        rotation_deg,
        (255, 255, 0),
        4,
    )
    cv2.circle(
        canvas,
        world_to_lidar_pixel(
            0.0, rear_reference_y, origin, scale, rotation_deg
        ),
        5,
        (255, 255, 0),
        -1,
    )
    draw_vehicle_direction_labels(cv2, canvas, origin, scale, rotation_deg)
    cv2.putText(
        canvas,
        "%s%s cars=%d first=%s turnErr=%s gap=%s err=%s" % (
            observation.reason,
            " HOLD" if observation.coasted else "",
            observation.car_count,
            "Y" if observation.first_car_confirmed else "N",
            (
                "-"
                if observation.first_car_turn_error_mm is None
                else "%+.0f" % observation.first_car_turn_error_mm
            ),
            "Y" if observation.gap_confirmed else "N",
            "-" if observation.entry_error_mm is None else "%+.0f" % observation.entry_error_mm,
        ),
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "yellow=first-car trigger | orange=slot | green=center | cyan=axle | blue=cars",
        (12, size - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "display rotation=%+.0f deg (perception offset=%+.0f deg)" % (
            rotation_deg,
            config.lidar.angle_offset_deg,
        ),
        (12, size - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "vehicle=%.0fx%.0fcm | LiDAR behind rear=%.0fcm | axle=%+.0fcm" % (
            config.runtime.lidar_debug_vehicle_width_mm / 10.0,
            config.runtime.lidar_debug_vehicle_length_mm / 10.0,
            config.runtime.lidar_debug_sensor_behind_vehicle_rear_mm / 10.0,
            config.lidar.sensor_to_rear_axle_y_back_mm / 10.0,
        ),
        (12, size - 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_roi(
    cv2: Any,
    image: Any,
    roi: Any,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    color: Tuple[int, int, int],
) -> None:
    draw_world_polygon(
        cv2,
        image,
        (
            (roi.x_min_mm, roi.y_back_min_mm),
            (roi.x_max_mm, roi.y_back_min_mm),
            (roi.x_max_mm, roi.y_back_max_mm),
            (roi.x_min_mm, roi.y_back_max_mm),
        ),
        origin,
        scale,
        rotation_deg,
        color,
        2,
    )


def draw_cluster(
    cv2: Any,
    image: Any,
    cluster: Any,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    color: Tuple[int, int, int],
) -> None:
    draw_world_polygon(
        cv2,
        image,
        (
            (cluster.x_min_mm, cluster.y_back_min_mm),
            (cluster.x_max_mm, cluster.y_back_min_mm),
            (cluster.x_max_mm, cluster.y_back_max_mm),
            (cluster.x_min_mm, cluster.y_back_max_mm),
        ),
        origin,
        scale,
        rotation_deg,
        color,
        2,
    )


def draw_world_y(
    cv2: Any,
    image: Any,
    y_back_mm: float,
    roi: Any,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    draw_world_segment(
        cv2,
        image,
        (roi.x_min_mm, y_back_mm),
        (roi.x_max_mm, y_back_mm),
        origin,
        scale,
        rotation_deg,
        color,
        thickness,
    )


def world_to_lidar_pixel(
    x_right_mm: float,
    y_back_mm: float,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
) -> Tuple[int, int]:
    """Map vehicle coordinates to the debug canvas.

    Positive display rotation is clockwise. This rotates only visualization;
    LiDAR perception uses ``LidarParkingConfig.angle_offset_deg`` separately.
    """

    angle = radians(rotation_deg)
    cosine = cos(angle)
    sine = sin(angle)
    display_x = cosine * x_right_mm - sine * y_back_mm
    display_y = sine * x_right_mm + cosine * y_back_mm
    return (
        int(round(origin[0] + display_x * scale)),
        int(round(origin[1] + display_y * scale)),
    )


def draw_world_segment(
    cv2: Any,
    image: Any,
    first: Tuple[float, float],
    second: Tuple[float, float],
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.line(
        image,
        world_to_lidar_pixel(first[0], first[1], origin, scale, rotation_deg),
        world_to_lidar_pixel(second[0], second[1], origin, scale, rotation_deg),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_world_polygon(
    cv2: Any,
    image: Any,
    points: Tuple[Tuple[float, float], ...],
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    pixels = tuple(
        world_to_lidar_pixel(x, y, origin, scale, rotation_deg)
        for x, y in points
    )
    for first, second in zip(pixels, pixels[1:] + pixels[:1]):
        cv2.line(image, first, second, color, thickness, cv2.LINE_AA)


def draw_vehicle_outline(
    cv2: Any,
    image: Any,
    width_mm: float,
    length_mm: float,
    sensor_behind_vehicle_rear_mm: float,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
) -> None:
    half_width = max(1.0, width_mm) / 2.0
    rear_y = -max(0.0, sensor_behind_vehicle_rear_mm)
    front_y = rear_y - max(1.0, length_mm)
    center_y = (front_y + rear_y) / 2.0
    draw_world_polygon(
        cv2,
        image,
        (
            (-half_width, front_y),
            (half_width, front_y),
            (half_width, rear_y),
            (-half_width, rear_y),
        ),
        origin,
        scale,
        rotation_deg,
        (255, 255, 255),
        2,
    )
    start = world_to_lidar_pixel(0.0, center_y, origin, scale, rotation_deg)
    front = world_to_lidar_pixel(0.0, front_y, origin, scale, rotation_deg)
    cv2.arrowedLine(
        image,
        start,
        front,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def draw_vehicle_direction_labels(
    cv2: Any,
    image: Any,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
) -> None:
    labels = (
        ("FRONT", 0.0, -2200.0),
        ("REAR", 0.0, 2200.0),
        ("LEFT", -2200.0, 0.0),
        ("RIGHT", 2200.0, 0.0),
    )
    for text, x_right, y_back in labels:
        x, y = world_to_lidar_pixel(
            x_right, y_back, origin, scale, rotation_deg
        )
        width, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
        )[0]
        cv2.putText(
            image,
            text,
            (x - width // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def draw_parking_line(cv2: Any, image: Any, line: Any, color: Tuple[int, int, int]) -> None:
    if line is None:
        return
    half = line.length_px / 2.0
    first = (
        int(round(line.center_x - line.direction_x * half)),
        int(round(line.center_y - line.direction_y * half)),
    )
    second = (
        int(round(line.center_x + line.direction_x * half)),
        int(round(line.center_y + line.direction_y * half)),
    )
    cv2.line(image, first, second, color, 3, cv2.LINE_AA)


def draw_reverse_path(cv2: Any, np: Any, image: Any, path: Any) -> None:
    if path is None or not path.points:
        return
    points = np.asarray(
        [[int(round(point[0])), int(round(point[1]))] for point in path.points],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(image, [points], False, (255, 120, 0), 3, cv2.LINE_AA)
    target = path.points[-1]
    cv2.circle(image, (int(round(target[0])), int(round(target[1]))), 7, (255, 255, 255), 2)
    if path.lookahead_point is not None:
        cv2.circle(
            image,
            (int(round(path.lookahead_point[0])), int(round(path.lookahead_point[1]))),
            6,
            (0, 255, 255),
            -1,
        )


def open_vehicle(args: argparse.Namespace, config: ParkingAppConfig) -> SerialVehicleClient:
    client = SerialVehicleClient(
        SerialVehicleConfig(
            port=args.serial_port or config.serial.arduino_port,
            baudrate=config.serial.baudrate,
            timeout_s=config.serial.timeout_s,
            ready_timeout_s=3.0,
        ),
        max_speed=max(
            abs(config.planner.search_speed),
            abs(config.planner.gap_tracking_speed),
            abs(config.planner.position_speed),
            abs(config.planner.prealign_speed),
            abs(config.planner.reverse_entry_speed),
            abs(config.planner.reverse_center_speed),
        ),
        max_steering=max(
            abs(config.planner.max_steering),
            abs(config.planner.prealign_steering),
        ),
    )
    client.connect()
    client.write_line("USON")
    LOG.info("serial connected: %s", client.port)
    return client


def newest_ultrasonic_sample(lines: list[str]) -> Optional[UltrasonicReadings]:
    newest = None
    for line in lines:
        parsed = parse_ultrasonic_line(line)
        if parsed is not None:
            newest = parsed
    return newest


def open_capture(cv2: Any, source: str, config: ParkingAppConfig) -> Any:
    value: Any = int(source) if source.isdigit() else str(resolve_path(source))
    if isinstance(value, int) and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(value, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(value)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.rear_camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.rear_camera.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.rear_camera.fourcc))
    if not cap.isOpened():
        raise RuntimeError("rear camera/video could not be opened: %s" % source)
    return cap


def video_elapsed_s(cv2: Any, cap: Any, frame_index: int, started_at: float, is_video: bool) -> float:
    if not is_video:
        return time.monotonic() - started_at
    milliseconds = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if milliseconds > 0.0:
        return milliseconds / 1000.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return frame_index / fps if fps > 0.0 else time.monotonic() - started_at


def apply_cli_overrides(config: ParkingAppConfig, args: argparse.Namespace) -> ParkingAppConfig:
    """Apply replay-time calibration without editing ``parking.json``."""

    bev = config.bev
    top_y = args.bev_top_y
    bottom_y = args.bev_bottom_y
    bev = replace(
        bev,
        src_top_left=(
            bev.src_top_left[0] if args.bev_top_left_x is None else args.bev_top_left_x,
            bev.src_top_left[1] if top_y is None else top_y,
        ),
        src_top_right=(
            bev.src_top_right[0] if args.bev_top_right_x is None else args.bev_top_right_x,
            bev.src_top_right[1] if top_y is None else top_y,
        ),
        src_bottom_left=(
            bev.src_bottom_left[0] if args.bev_bottom_left_x is None else args.bev_bottom_left_x,
            bev.src_bottom_left[1] if bottom_y is None else bottom_y,
        ),
        src_bottom_right=(
            bev.src_bottom_right[0] if args.bev_bottom_right_x is None else args.bev_bottom_right_x,
            bev.src_bottom_right[1] if bottom_y is None else bottom_y,
        ),
        dst_x_margin=(
            bev.dst_x_margin if args.bev_dst_margin is None else args.bev_dst_margin
        ),
        out_width=bev.out_width if args.bev_out_width is None else args.bev_out_width,
        out_height=bev.out_height if args.bev_out_height is None else args.bev_out_height,
    )
    lidar = config.lidar
    if args.lidar_angle_offset is not None:
        lidar = replace(lidar, angle_offset_deg=args.lidar_angle_offset)
    if args.lidar_to_rear_axle_cm is not None:
        lidar = replace(
            lidar,
            sensor_to_rear_axle_y_back_mm=args.lidar_to_rear_axle_cm * 10.0,
        )
    if args.first_car_turn_target_cm is not None:
        lidar = replace(
            lidar,
            first_car_turn_target_y_back_mm=args.first_car_turn_target_cm * 10.0,
        )
    runtime = config.runtime
    if args.lidar_display_rotation is not None:
        runtime = replace(
            runtime,
            lidar_display_rotation_deg=args.lidar_display_rotation,
        )
    if args.lidar_behind_vehicle_rear_cm is not None:
        runtime = replace(
            runtime,
            lidar_debug_sensor_behind_vehicle_rear_mm=(
                args.lidar_behind_vehicle_rear_cm * 10.0
            ),
        )
    planner = config.planner
    if args.prealign_speed is not None:
        planner = replace(planner, prealign_speed=args.prealign_speed)
    if args.prealign_steering is not None:
        planner = replace(planner, prealign_steering=args.prealign_steering)
    if args.prealign_timeout_s is not None:
        planner = replace(planner, prealign_timeout_s=args.prealign_timeout_s)
    if bev.src_top_left[0] >= bev.src_top_right[0]:
        raise ValueError("BEV top-left x must be smaller than top-right x")
    if bev.src_bottom_left[0] >= bev.src_bottom_right[0]:
        raise ValueError("BEV bottom-left x must be smaller than bottom-right x")
    if bev.src_top_left[1] >= bev.src_bottom_left[1]:
        raise ValueError("BEV top y must be smaller than bottom y")
    if not 0.0 <= bev.dst_x_margin < 0.5:
        raise ValueError("BEV destination margin must be at least 0 and below 0.5")
    return replace(config, bev=bev, lidar=lidar, planner=planner, runtime=runtime)


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def extract_recording_zip(zip_path: Path, destination: Path) -> Tuple[Path, Path]:
    if not zip_path.exists():
        raise FileNotFoundError("recording ZIP not found: %s" % zip_path)
    videos = []
    lidar_csvs = []
    with zipfile.ZipFile(str(zip_path), "r") as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path in recording ZIP: %s" % member.filename)
            suffix = relative.suffix.lower()
            if suffix not in (".mp4", ".csv"):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if suffix == ".mp4":
                videos.append(target)
            elif target.name.lower().endswith("_lidar.csv"):
                lidar_csvs.append(target)
    if len(videos) != 1 or len(lidar_csvs) != 1:
        raise ValueError(
            "recording ZIP must contain exactly one MP4 and one *_lidar.csv "
            "(found videos=%d lidar_csv=%d)" % (len(videos), len(lidar_csvs))
        )
    return videos[0], lidar_csvs[0]


def parse_args(argv: Optional[list]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rear-camera YOLO + LiDAR T-parking runtime")
    parser.add_argument("--config", default="configs/parking.json")
    parser.add_argument("--source", default=None, help="rear camera index or recorded video")
    parser.add_argument("--recording-zip", default=None, help="ZIP containing one MP4 and one *_lidar.csv")
    parser.add_argument("--model", default=None, help="parking YOLO segmentation model")
    parser.add_argument("--device", default=None, help="auto, cpu, mps, 0, cuda, ...")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size; CPU replay can use 512")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence override")
    parser.add_argument("--lidar-csv", default=None, help="recorded LiDAR CSV synchronized by relative time")
    parser.add_argument("--lidar-port", default=None, help="live RPLidar serial port")
    parser.add_argument("--lidar-offset", type=float, default=0.0, help="seconds added to video time for CSV lookup")
    parser.add_argument("--allow-no-lidar", action="store_true", help="explicit unsafe test bypass; preview only")
    parser.add_argument("--serial", action="store_true", help="enable Arduino output (disabled by default)")
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="video replay only: infer every Nth frame")
    parser.add_argument("--auto-start", action="store_true", help="start state machine at the beginning of video replay")
    parser.add_argument("--replay-delay-ms", type=int, default=1)
    parser.add_argument("--bev-top-y", type=float, default=None, help="BEV source top y ratio")
    parser.add_argument("--bev-top-left-x", type=float, default=None, help="BEV source top-left x ratio")
    parser.add_argument("--bev-top-right-x", type=float, default=None, help="BEV source top-right x ratio")
    parser.add_argument("--bev-bottom-y", type=float, default=None, help="BEV source bottom y ratio")
    parser.add_argument("--bev-bottom-left-x", type=float, default=None, help="BEV source bottom-left x ratio")
    parser.add_argument("--bev-bottom-right-x", type=float, default=None, help="BEV source bottom-right x ratio")
    parser.add_argument("--bev-dst-margin", type=float, default=None, help="BEV output side margin ratio")
    parser.add_argument("--bev-out-width", type=int, default=None)
    parser.add_argument("--bev-out-height", type=int, default=None)
    parser.add_argument(
        "--lidar-display-rotation",
        type=float,
        default=None,
        help="clockwise LiDAR debug-view rotation in degrees; perception is unchanged",
    )
    parser.add_argument(
        "--lidar-angle-offset",
        type=float,
        default=None,
        help="LiDAR perception bearing offset in degrees; calibrate separately from display rotation",
    )
    parser.add_argument(
        "--lidar-behind-vehicle-rear-cm",
        type=float,
        default=None,
        help="debug view: distance the LiDAR sits behind the vehicle rear bumper",
    )
    parser.add_argument(
        "--lidar-to-rear-axle-cm",
        type=float,
        default=None,
        help="signed rear-axle coordinate from LiDAR (negative=ahead of rear-mounted LiDAR)",
    )
    parser.add_argument(
        "--prealign-speed",
        type=int,
        default=None,
        help="forward speed while swinging the rear toward the parking slot",
    )
    parser.add_argument(
        "--first-car-turn-target-cm",
        type=float,
        default=None,
        help="vehicle-frame yBack trigger for the first car; negative is ahead",
    )
    parser.add_argument(
        "--prealign-steering",
        type=int,
        default=None,
        help="signed configured maximum steering command for pre-alignment (left is negative)",
    )
    parser.add_argument(
        "--prealign-timeout-s",
        type=float,
        default=None,
        help="maximum seconds allowed for the LiDAR-closed-loop pre-alignment arc",
    )
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.imgsz is not None and args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.conf is not None and not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if (
        args.lidar_behind_vehicle_rear_cm is not None
        and args.lidar_behind_vehicle_rear_cm < 0.0
    ):
        parser.error("--lidar-behind-vehicle-rear-cm cannot be negative")
    unit_ratio_options = (
        "bev_top_y",
        "bev_bottom_y",
        "bev_dst_margin",
    )
    for name in unit_ratio_options:
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error("--%s must be between 0 and 1" % name.replace("_", "-"))
    x_ratio_options = (
        "bev_top_left_x",
        "bev_top_right_x",
        "bev_bottom_left_x",
        "bev_bottom_right_x",
    )
    for name in x_ratio_options:
        value = getattr(args, name)
        if value is not None and not -1.0 <= value <= 2.0:
            parser.error("--%s must be between -1 and 2" % name.replace("_", "-"))
    if args.bev_out_width is not None and args.bev_out_width < 32:
        parser.error("--bev-out-width must be at least 32")
    if args.bev_out_height is not None and args.bev_out_height < 32:
        parser.error("--bev-out-height must be at least 32")
    if args.prealign_speed is not None and args.prealign_speed <= 0:
        parser.error("--prealign-speed must be positive")
    if args.prealign_steering == 0:
        parser.error("--prealign-steering cannot be zero")
    if args.prealign_steering is not None and abs(args.prealign_steering) > 150:
        parser.error("--prealign-steering must be between -150 and 150")
    if args.prealign_timeout_s is not None and args.prealign_timeout_s <= 0.0:
        parser.error("--prealign-timeout-s must be positive")
    if (
        args.first_car_turn_target_cm is not None
        and not -250.0 <= args.first_car_turn_target_cm <= 250.0
    ):
        parser.error("--first-car-turn-target-cm must be between -250 and 250")
    return args


def load_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for parking runtime") from exc
