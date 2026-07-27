from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from datetime import datetime
from math import cos, isfinite, radians, sin
from pathlib import Path
from typing import Any, Optional, Tuple

from ..control.serial_vehicle import (
    SerialVehicleClient,
    SerialVehicleConfig,
    UltrasonicReadings,
    parse_ultrasonic_line,
)
from ..estimation.lidar_slot_geometry import LidarSlotGeometryProjector
from ..estimation.parking_geometry import ParkingGeometry, ParkingGeometryEstimator
from ..estimation.parking_lidar import (
    LidarParkingObservation,
    LidarParkingSpaceEstimator,
    infer_dynamic_slot_polygon,
)
from ..parking_config import (
    ParkingAppConfig,
    load_parking_config,
    validate_model_based_parking_config,
)
from ..perception.bev import BevTransformer
from ..perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from ..planning.model_based_parking import ModelBasedTParkingPlanner
from ..planning.t_parking_planner import ParkingState
from ..sensors.lidar import (
    LidarCsvRecorder,
    LidarCsvReplay,
    RplidarScanner,
    find_lidar_port,
)


LOG = logging.getLogger("skku_autocar.parking")
ROOT = Path(__file__).resolve().parents[3]
class DashboardVideoRecorder:
    """Record the exact 1280x720 dashboard shown by the live runtime."""

    def __init__(self, cv2: Any, path: Path, fps: float) -> None:
        self.cv2 = cv2
        self.path = path
        self.fps = fps
        self.writer: Any = None
        self.frame_size: Optional[Tuple[int, int]] = None
        self.next_frame_s: Optional[float] = None
        self.frames_written = 0

    def write(self, frame: Any, elapsed_s: float) -> None:
        height, width = frame.shape[:2]
        frame_size = (width, height)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = self.cv2.VideoWriter(
                str(self.path), fourcc, self.fps, frame_size
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(
                    "dashboard recording could not be opened: %s" % self.path
                )
            self.frame_size = frame_size
        elif frame_size != self.frame_size:
            raise ValueError(
                "dashboard frame size changed from %s to %s"
                % (self.frame_size, frame_size)
            )

        # Keep video duration close to wall-clock time even when YOLO inference
        # produces frames slower than the recording FPS. Missing instants repeat
        # the most recent dashboard, exactly as it appeared on screen.
        interval_s = 1.0 / self.fps
        target_s = max(0.0, elapsed_s)
        if self.next_frame_s is None:
            # Model/device warm-up happens before the first visible dashboard;
            # do not block live control by encoding that startup delay.
            self.next_frame_s = target_s
        writes_this_update = 0
        while self.next_frame_s <= target_s + 1e-9:
            self.writer.write(frame)
            self.frames_written += 1
            self.next_frame_s += interval_s
            writes_this_update += 1
            if writes_this_update >= 3 and self.next_frame_s <= target_s:
                # Recording must never stall steering/motor updates after an
                # unusually slow inference or a debugger pause.
                self.next_frame_s = target_s + interval_s
                break

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class TriangulationCsvRecorder:
    """Crash-safe per-scan record of every LiDAR steering decision."""

    FIELDS = (
        "elapsed_s",
        "lidar_timestamp",
        "state",
        "reason",
        "speed",
        "steering",
        "pair_valid",
        "car_count",
        "gap_width_mm",
        "car1_x_mm",
        "car1_y_back_mm",
        "car2_x_mm",
        "car2_y_back_mm",
        "entrance_x_mm",
        "entrance_y_back_mm",
        "depth_x",
        "depth_y_back",
        "decision_angle_deg",
        "heading_error_deg",
        "lateral_error_mm",
        "depth_progress_mm",
        "depth_remaining_mm",
        "target_x_mm",
        "target_y_back_mm",
        "curvature_per_mm",
        "geometric_steering",
        "physical_steering",
        "clusters",
    )

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._last_timestamp: Optional[float] = None
        self.rows_written = 0

    def write(
        self,
        elapsed_s: float,
        lidar: LidarParkingObservation,
        plan: Any,
        debug: Any,
    ) -> None:
        timestamp = float(lidar.timestamp)
        if (
            self._last_timestamp is not None
            and timestamp <= self._last_timestamp + 1e-6
        ):
            return
        self._last_timestamp = timestamp
        clusters = "|".join(
            "%d:%.1f:%.1f:%.1f:%.1f"
            % (
                cluster.point_count,
                cluster.x_min_mm,
                cluster.x_max_mm,
                cluster.y_back_min_mm,
                cluster.y_back_max_mm,
            )
            for cluster in lidar.car_clusters
        )
        row = {
            "elapsed_s": "%.6f" % elapsed_s,
            "lidar_timestamp": "%.9f" % timestamp,
            "state": plan.state.value,
            "reason": plan.reason,
            "speed": plan.command.speed,
            "steering": plan.command.steering,
            "car_count": lidar.car_count,
            "clusters": clusters,
        }
        for field in self.FIELDS:
            if field in row:
                continue
            value = getattr(debug, field, None)
            if isinstance(value, bool):
                row[field] = int(value)
            elif isinstance(value, float):
                row[field] = "%.6f" % value
            elif value is not None:
                row[field] = value
            else:
                row[field] = ""
        self._writer.writerow(row)
        self._handle.flush()
        self.rows_written += 1

    def close(self) -> None:
        self._handle.close()


def timestamped_dashboard_path(directory: str, now: Optional[datetime] = None) -> Path:
    root = resolve_path(directory)
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    candidate = root / (timestamp + ".mp4")
    suffix = 1
    while candidate.exists():
        candidate = root / ("%s_%02d.mp4" % (timestamp, suffix))
        suffix += 1
    return candidate


def dashboard_recording_enabled(mode: str, is_video: bool) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return not is_video


def make_slot_geometry_projector(
    config: ParkingAppConfig,
) -> LidarSlotGeometryProjector:
    return LidarSlotGeometryProjector(
        config.lidar,
        config.geometry,
        config.bev.out_width,
        config.bev.out_height,
        vehicle_width_mm=config.runtime.lidar_debug_vehicle_width_mm,
        vehicle_length_mm=config.runtime.lidar_debug_vehicle_length_mm,
        rear_axle_to_rear_bumper_mm=(
            config.runtime.lidar_debug_rear_axle_to_rear_bumper_mm
        ),
    )
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
    camera_enabled = config.runtime.camera_enabled
    source = str(args.source) if args.source is not None else str(config.rear_camera.index)
    front_source = (
        str(args.front_source)
        if args.front_source is not None
        else str(config.front_camera.index)
    )
    is_video = camera_enabled and not source.isdigit() and not is_auto_camera_source(source)
    is_replay = is_video or (not camera_enabled and args.lidar_csv is not None)
    if args.serial is None:
        # Live parking is the primary entry point: enable the Arduino motor
        # output by default. Recorded video/CSV replay remains motor-safe.
        args.serial = not is_replay
    front_camera_enabled = (
        config.runtime.front_camera_enabled
        and camera_enabled
        and not is_video
    )
    if is_replay and args.serial:
        raise RuntimeError(
            "--serial is forbidden for recorded replay; use live LiDAR/camera input"
        )
    model_path = resolve_path(args.model or config.yolo.model_path)

    segmenter = None
    if camera_enabled and not config.runtime.camera_debug_only:
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
    lidar_geometry_projector = make_slot_geometry_projector(config)
    planner = ModelBasedTParkingPlanner(
        config.model_planner,
        config.vehicle,
        config.hybrid_path,
        sensor_to_rear_axle_y_back_mm=(
            config.lidar.sensor_to_rear_axle_y_back_mm
        ),
    )

    lidar_replay = LidarCsvReplay(str(resolve_path(args.lidar_csv))) if args.lidar_csv else None
    lidar_scanner = None
    lidar_port = None if lidar_replay is not None else find_lidar_port(args.lidar_port)
    if (
        args.lidar_port
        and args.lidar_port.strip().lower() not in ("", "auto")
        and lidar_port is not None
        and lidar_port != args.lidar_port
    ):
        LOG.warning(
            "requested LiDAR port %s is unavailable; using detected port %s",
            args.lidar_port,
            lidar_port,
        )
    if args.lidar_port and lidar_port is None:
        raise RuntimeError(
            "LiDAR serial port was not found for %s. Run "
            "PYTHONPATH=src venv/bin/python scripts/list_serial_ports.py "
            "and use the /dev/cu.usbserial-* port."
            % args.lidar_port
        )
    if lidar_port is not None:
        lidar_scanner = RplidarScanner(lidar_port)
        lidar_scanner.start()
        LOG.info("lidar scanner started: %s", lidar_port)
    if config.runtime.require_lidar and lidar_replay is None and lidar_scanner is None and not args.allow_no_lidar:
        raise RuntimeError("LiDAR is required: pass --lidar-csv, --lidar-port, or explicitly --allow-no-lidar")

    cap = None
    front_cap = None
    try:
        if camera_enabled:
            cap = open_capture(
                cv2,
                source,
                config,
                segmenter=segmenter,
                transformer=transformer,
            )
            if front_camera_enabled:
                front_cap = open_front_capture(
                    cv2,
                    front_source,
                    config,
                    source,
                )
    except Exception:
        if cap is not None:
            cap.release()
        if lidar_scanner is not None:
            lidar_scanner.close()
        raise
    if cap is not None and args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    try:
        vehicle = open_vehicle(args, config) if args.serial else None
    except Exception:
        if cap is not None:
            cap.release()
        if front_cap is not None:
            front_cap.release()
        if lidar_scanner is not None:
            lidar_scanner.close()
        raise
    last_command_at = 0.0
    run_started_at = time.monotonic()
    last_frame_at = run_started_at
    fps = 0.0
    frame_index = args.start_frame
    last_state = planner.state
    ultrasonic_readings = UltrasonicReadings()
    ultrasonic_received_at: Optional[float] = None

    if args.auto_start or (config.runtime.auto_start and not args.manual_start):
        planner.start(video_elapsed_s(cv2, cap, frame_index, run_started_at, is_video))
        LOG.info("parking mission auto-started")

    if segmenter is not None:
        LOG.info(
            "rear_source=%s front_source=%s front_enabled=%s model=%s device=%s",
            source,
            front_source,
            front_camera_enabled,
            model_path,
            segmenter.device,
        )
    elif camera_enabled:
        LOG.info(
            "rear camera debug display enabled; camera control and YOLO disabled"
        )
    else:
        LOG.info("camera=disabled; LiDAR-only parking runtime")
    LOG.info("controls: SPACE=start/resume | R=stop/reset | Q/ESC=quit")
    if not args.serial:
        LOG.info("serial output disabled; pass --serial only after replay/calibration checks")

    dashboard_recorder: Optional[DashboardVideoRecorder] = None
    dashboard_record_path: Optional[Path] = None
    lidar_recorder: Optional[LidarCsvRecorder] = None
    if dashboard_recording_enabled(args.record_dashboard, is_replay):
        dashboard_record_path = timestamped_dashboard_path(args.parking_record_dir)
        dashboard_recorder = DashboardVideoRecorder(
            cv2,
            dashboard_record_path,
            args.dashboard_record_fps,
        )
        LOG.info("dashboard recording enabled: %s", dashboard_record_path)
        if lidar_scanner is not None:
            lidar_record_path = dashboard_record_path.with_name(
                dashboard_record_path.stem + "_lidar.csv"
            )
            lidar_recorder = LidarCsvRecorder(lidar_record_path)
            LOG.info("raw LiDAR recording enabled: %s", lidar_record_path)
    telemetry_base = (
        dashboard_record_path
        if dashboard_record_path is not None
        else timestamped_dashboard_path(args.parking_record_dir)
    )
    triangulation_record_path = telemetry_base.with_name(
        telemetry_base.stem + "_triangulation.csv"
    )
    triangulation_recorder = TriangulationCsvRecorder(
        triangulation_record_path
    )
    LOG.info(
        "triangulation telemetry enabled: %s",
        triangulation_record_path,
    )

    try:
        while True:
            monotonic_now = time.monotonic()
            elapsed = video_elapsed_s(
                cv2, cap, frame_index, run_started_at, is_video
            )
            if (
                not camera_enabled
                and lidar_replay is not None
                and elapsed > lidar_replay.duration_s
            ):
                break
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    if is_video:
                        break
                    raise RuntimeError("rear camera frame read failed")
            else:
                frame = np.zeros(
                    (config.rear_camera.height, config.rear_camera.width, 3),
                    dtype=np.uint8,
                )
                cv2.putText(
                    frame,
                    "CAMERA DISABLED - LIDAR ONLY",
                    (30, max(45, config.rear_camera.height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            front_frame = None
            if front_cap is not None:
                front_ok, front_candidate = front_cap.read()
                if front_ok and front_candidate is not None:
                    front_frame = front_candidate
                else:
                    front_frame = camera_placeholder_frame(
                        cv2,
                        np,
                        config.front_camera,
                        "FRONT CAMERA READ FAILED",
                    )
            if vehicle is not None:
                for line in vehicle.read_lines():
                    report = parse_ultrasonic_line(line)
                    if report is not None:
                        ultrasonic_readings = report
                        ultrasonic_received_at = monotonic_now
            dt = max(1e-6, monotonic_now - last_frame_at)
            fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            last_frame_at = monotonic_now

            if segmenter is not None:
                class_masks = segmenter.segment_class_masks(frame)
                parking_masks = list(class_masks.lane)
                bev_masks = [transformer.warp_mask(mask) for mask in parking_masks]
                camera_geometry = geometry_estimator.estimate(
                    bev_masks,
                    class_masks.lane_conf,
                )
            else:
                parking_masks = []
                bev_masks = []
                camera_geometry = ParkingGeometry(
                    reason=(
                        "camera_debug_only"
                        if camera_enabled
                        else "camera_disabled"
                    )
                )

            lidar_observation, lidar_scan = current_lidar_observation(
                lidar_estimator,
                lidar_replay,
                lidar_scanner,
                elapsed + args.lidar_offset + config.runtime.lidar_video_offset_s,
                args.allow_no_lidar,
            )
            if lidar_recorder is not None:
                lidar_recorder.write(lidar_scan)
            lidar_points = lidar_estimator.vehicle_points(lidar_scan)
            # Debug projection only. The controller consumes the fresh
            # two-car LiDAR triangle directly; no slot pose is locked.
            geometry = lidar_geometry_projector.project(lidar_observation)
            ultrasonic_fresh = (
                ultrasonic_received_at is not None
                and monotonic_now - ultrasonic_received_at
                <= config.model_planner.ultrasonic_stale_after_s
            )
            left_ultrasonic_mm = (
                ultrasonic_readings.side_left_mm
                if ultrasonic_fresh
                else None
            )
            right_ultrasonic_mm = (
                ultrasonic_readings.side_right_mm
                if ultrasonic_fresh
                else None
            )
            front_left_ultrasonic_mm = (
                ultrasonic_readings.front_left_mm
                if ultrasonic_fresh
                else None
            )
            front_center_ultrasonic_mm = (
                ultrasonic_readings.front_center_mm
                if ultrasonic_fresh
                else None
            )
            front_right_ultrasonic_mm = (
                ultrasonic_readings.front_right_mm
                if ultrasonic_fresh
                else None
            )
            plan = planner.update(
                geometry,
                lidar_observation,
                None,
                elapsed,
                enabled=True,
                left_ultrasonic_mm=left_ultrasonic_mm,
                right_ultrasonic_mm=right_ultrasonic_mm,
                front_left_ultrasonic_mm=front_left_ultrasonic_mm,
                front_center_ultrasonic_mm=front_center_ultrasonic_mm,
                front_right_ultrasonic_mm=front_right_ultrasonic_mm,
                right_ultrasonic_reported=ultrasonic_fresh,
                right_ultrasonic_timestamp=ultrasonic_received_at,
            )
            triangulation_recorder.write(
                elapsed,
                lidar_observation,
                plan,
                planner.debug_snapshot,
            )
            if planner.consume_lidar_reset_request():
                # Discard every pre-turn cluster. Only post-turn scans may
                # become the live two-car decision triangle.
                lidar_estimator.reset()
                lidar_observation = LidarParkingObservation(
                    timestamp=lidar_observation.timestamp,
                    reason="lidar_reset_by_entry_workflow",
                )
                geometry = ParkingGeometry(reason="lidar_reset_by_entry_workflow")
                LOG.info(
                    "LiDAR slot estimator reset by entry phase: %s",
                    planner.entry_phase,
                )
            if planner.state != last_state:
                LOG.info("parking state: %s -> %s (%s)", last_state.value, planner.state.value, plan.reason)
                last_state = planner.state

            if (
                vehicle is not None
                and monotonic_now - last_command_at
                >= 1.0 / max(1.0, config.runtime.command_rate_hz)
            ):
                serial_lines = vehicle.send(plan.command)
                for line in serial_lines:
                    report = parse_ultrasonic_line(line)
                    if report is not None:
                        ultrasonic_readings = report
                        ultrasonic_received_at = monotonic_now
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
                front_left_ultrasonic_mm,
                front_center_ultrasonic_mm,
                front_right_ultrasonic_mm,
                show_status=False,
                mask_geometry=camera_geometry,
            )
            lidar_display = draw_lidar_debug(
                cv2,
                np,
                lidar_points,
                config,
                lidar_observation,
                geometry,
                plan,
                slot_polygon=None,
                slot_status="live_triangulation",
            )
            dashboard = draw_live_dashboard(
                cv2,
                np,
                display,
                bev_display,
                lidar_display,
                geometry,
                lidar_observation,
                plan,
                elapsed,
                fps,
                left_ultrasonic_mm,
                right_ultrasonic_mm,
                vehicle is not None,
                dashboard_record_path,
                camera_enabled=camera_enabled,
                front_display=front_frame,
                front_camera_enabled=front_camera_enabled,
                front_left_ultrasonic_mm=front_left_ultrasonic_mm,
                front_center_ultrasonic_mm=front_center_ultrasonic_mm,
                front_right_ultrasonic_mm=front_right_ultrasonic_mm,
            )
            cv2.imshow("T Parking - Live Dashboard", dashboard)
            if dashboard_recorder is not None:
                dashboard_recorder.write(
                    dashboard,
                    time.monotonic() - run_started_at,
                )
            if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE"):
                try:
                    if cv2.getWindowProperty(
                        "T Parking - Live Dashboard",
                        cv2.WND_PROP_VISIBLE,
                    ) < 1:
                        LOG.info("dashboard window closed; stopping parking runtime")
                        break
                except cv2.error:
                    # Some macOS OpenCV backends do not implement this query.
                    pass

            delay_ms = 1
            if is_replay:
                delay_ms = max(1, args.replay_delay_ms)
            elif not camera_enabled:
                delay_ms = max(
                    1,
                    int(round(1000.0 / max(1.0, config.runtime.command_rate_hz))),
                )
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if planner.state in (
                    ParkingState.IDLE,
                    ParkingState.ABORTED,
                    ParkingState.EMERGENCY_STOP,
                    ParkingState.PARKED,
                    ParkingState.EXIT_DONE,
                ):
                    planner.start(elapsed)
                    geometry_estimator.reset()
                    lidar_estimator.reset()
                    LOG.info("parking mission started")
                else:
                    LOG.info("parking mission already running; SPACE ignored, press R to stop/reset")
            elif key == ord("r"):
                planner.reset(elapsed)
                geometry_estimator.reset()
                lidar_estimator.reset()
                if vehicle is not None:
                    vehicle.stop("operator_reset")
                LOG.info("parking mission reset")
            skipped = 0
            if is_video and cap is not None:
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
                vehicle.close()
        if dashboard_recorder is not None:
            dashboard_recorder.close()
            if dashboard_recorder.frames_written > 0:
                LOG.info(
                    "dashboard recording saved: %s (%d frames)",
                    dashboard_recorder.path,
                    dashboard_recorder.frames_written,
                )
        if lidar_recorder is not None:
            lidar_recorder.close()
            LOG.info(
                "raw LiDAR recording saved: %s (%d scans, %d points)",
                lidar_recorder.path,
                lidar_recorder.scans_written,
                lidar_recorder.points_written,
            )
        triangulation_recorder.close()
        LOG.info(
            "triangulation telemetry saved: %s (%d scans)",
            triangulation_recorder.path,
            triangulation_recorder.rows_written,
        )
        if lidar_scanner is not None:
            lidar_scanner.close()
        if cap is not None:
            cap.release()
        if front_cap is not None:
            front_cap.release()
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
        if scan is None and scanner.error is not None:
            return LidarParkingObservation(
                timestamp=time.time(),
                reason="lidar_error:%s" % scanner.error,
            ), None
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


def camera_placeholder_frame(
    cv2: Any,
    np: Any,
    camera_config: Any,
    message: str,
) -> Any:
    frame = np.zeros(
        (camera_config.height, camera_config.width, 3),
        dtype=np.uint8,
    )
    cv2.putText(
        frame,
        message,
        (30, max(45, camera_config.height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def compose_parking_dashboard(
    cv2: Any,
    np: Any,
    rear_display: Any,
    bev_display: Any,
    lidar_display: Any,
    header_text: str,
    header_color: Tuple[int, int, int],
    status_lines: Tuple[str, ...],
    front_display: Optional[Any] = None,
    front_label: str = "FRONT",
) -> Any:
    """Build the one dashboard layout shared by live and offline replay."""

    rear_panel = rear_display.copy()
    cv2.rectangle(
        rear_panel,
        (0, 0),
        (rear_panel.shape[1], 58),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        rear_panel,
        header_text,
        (18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        header_color,
        2,
        cv2.LINE_AA,
    )

    dashboard = np.zeros((720, 1280, 3), dtype=np.uint8)
    dashboard[0:495, 0:880] = cv2.resize(rear_panel, (880, 495))
    if front_display is not None:
        front_panel = cv2.resize(front_display, (312, 176))
        cv2.rectangle(front_panel, (0, 0), (312, 28), (0, 0, 0), -1)
        cv2.putText(
            front_panel,
            front_label,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x0, y0 = 552, 304
        dashboard[y0 : y0 + 176, x0 : x0 + 312] = front_panel
        cv2.rectangle(
            dashboard,
            (x0, y0),
            (x0 + 311, y0 + 175),
            (0, 255, 255),
            2,
        )
    dashboard[0:360, 900:1260] = cv2.resize(bev_display, (360, 360))
    dashboard[360:720, 900:1260] = cv2.resize(lidar_display, (360, 360))
    cv2.putText(
        dashboard,
        "REAR + YOLO",
        (12, 487),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        dashboard,
        "BEV",
        (905, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        dashboard,
        "LiDAR",
        (905, 384),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, text in enumerate(status_lines[:8]):
        color = (0, 255, 255) if index == 0 else (220, 220, 220)
        cv2.putText(
            dashboard,
            text,
            (18, 510 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return dashboard


def draw_live_dashboard(
    cv2: Any,
    np: Any,
    rear_display: Any,
    bev_display: Any,
    lidar_display: Any,
    geometry: ParkingGeometry,
    lidar: LidarParkingObservation,
    plan: Any,
    elapsed_s: float,
    fps: float,
    left_ultrasonic_mm: Optional[float],
    right_ultrasonic_mm: Optional[float],
    motor_output_enabled: bool,
    recording_path: Optional[Path],
    camera_enabled: bool = True,
    front_display: Optional[Any] = None,
    front_camera_enabled: bool = False,
    front_left_ultrasonic_mm: Optional[float] = None,
    front_center_ultrasonic_mm: Optional[float] = None,
    front_right_ultrasonic_mm: Optional[float] = None,
) -> Any:
    state_color = parking_state_color(plan.state)
    wall_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    recording_name = "OFF" if recording_path is None else recording_path.name
    center_cm = (
        None
        if lidar.gap_center_y_back_mm is None
        else lidar.gap_center_y_back_mm / 10.0
    )
    width_cm = None if lidar.gap_width_mm is None else lidar.gap_width_mm / 10.0
    depth = (
        "-"
        if geometry.depth_remaining_px is None
        else "%.1fpx" % geometry.depth_remaining_px
    )
    status_lines = (
        "%s | MOTOR=%s | REC=%s" % (
            (
                "LIVE REAR+FRONT"
                if camera_enabled and front_camera_enabled
                else ("LIVE CAMERA" if camera_enabled else "LIDAR ONLY")
            ),
            "ENABLED" if motor_output_enabled else "DISABLED",
            recording_name,
        ),
        "STATE %-22s drive=%+3d steer=%+4d reason=%s" % (
            plan.state.value,
            plan.command.speed,
            plan.command.steering,
            plan.reason,
        ),
        "LiDAR %s raw=%d valid=%d R=%d/%d L=%d cars=%d gap=%s centerY=%s cm width=%s cm" % (
            lidar.reason,
            lidar.raw_points,
            lidar.observed_points,
            lidar.car_roi_points,
            lidar.accumulated_car_roi_points,
            lidar.left_roi_points,
            lidar.car_count,
            "CONFIRMED" if lidar.gap_confirmed else (
                "candidate" if lidar.gap_found else "no"
            ),
            format_dashboard_value(center_cm, signed=True),
            format_dashboard_value(width_cm),
        ),
        "TRIANGLE pair=%s coast=%s entryErr=%s cm safety=%s cm" % (
            "Y" if lidar.gap_pair_observed else "N",
            "Y" if lidar.coasted else "N",
            format_dashboard_value(
                None
                if lidar.entry_error_mm is None
                else lidar.entry_error_mm / 10.0,
                signed=True,
            ),
            format_dashboard_value(
                None
                if lidar.nearest_safety_mm is None
                else lidar.nearest_safety_mm / 10.0
            ),
        ),
        "LIVE LIDAR GEOM inside=%.0f%% full=%s conf=%.2f lat=%+.2f head=%+.1f depth=%s (%s)" % (
            geometry.vehicle_inside_ratio * 100.0,
            "Y" if geometry.vehicle_fully_inside else "N",
            geometry.confidence,
            geometry.lateral_error_norm,
            geometry.heading_error_deg,
            depth,
            geometry.reason,
        ),
        "ULTRASONIC FL=%s FC=%s FR=%s | SL=%s SR=%s cm" % (
            format_dashboard_value(
                None
                if front_left_ultrasonic_mm is None
                else front_left_ultrasonic_mm / 10.0
            ),
            format_dashboard_value(
                None
                if front_center_ultrasonic_mm is None
                else front_center_ultrasonic_mm / 10.0
            ),
            format_dashboard_value(
                None
                if front_right_ultrasonic_mm is None
                else front_right_ultrasonic_mm / 10.0
            ),
            format_dashboard_value(
                None if left_ultrasonic_mm is None else left_ultrasonic_mm / 10.0
            ),
            format_dashboard_value(
                None if right_ultrasonic_mm is None else right_ultrasonic_mm / 10.0
            ),
        ),
        "FPS=%.1f | elapsed=%.2fs | %s" % (fps, elapsed_s, wall_time),
        "Colors: CYAN=left GREEN=right RED=back MAGENTA=unclassified | SPACE start/resume R stop/reset Q quit",
    )
    return compose_parking_dashboard(
        cv2,
        np,
        rear_display,
        bev_display,
        lidar_display,
        "LIVE %s | drive=%+d steer=%+d | t=%.2fs" % (
            plan.state.value,
            plan.command.speed,
            plan.command.steering,
            elapsed_s,
        ),
        state_color,
        status_lines,
        front_display=front_display,
        front_label="FRONT CAMERA",
    )


def format_dashboard_value(value: Optional[float], signed: bool = False) -> str:
    if value is None or not isfinite(value):
        return "-"
    return ("%+.1f" if signed else "%.1f") % value


def parking_state_color(state: ParkingState) -> Tuple[int, int, int]:
    if state in (ParkingState.EMERGENCY_STOP, ParkingState.ABORTED):
        return (0, 0, 255)
    if state in (ParkingState.PARKED, ParkingState.EXIT_DONE):
        return (255, 255, 255)
    if state in (
        ParkingState.FOLLOW_ENTRY_CURVE,
        ParkingState.FOLLOW_SLOT_CENTER,
        ParkingState.EXIT_RIGHT,
        ParkingState.EXIT_STRAIGHT,
    ):
        return (0, 255, 0)
    if state in (
        ParkingState.TRACK_GAP,
        ParkingState.POSITION_REAR_AXLE,
        ParkingState.PREALIGN_LEFT,
        ParkingState.VERIFY_SLOT_BOX,
        ParkingState.ENTRY_SETUP,
        ParkingState.PLAN_REVERSE_PATH,
    ):
        return (0, 165, 255)
    if state == ParkingState.SEARCH_CARS:
        return (0, 220, 255)
    return (180, 180, 180)


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
    front_left_ultrasonic_mm: Optional[float] = None,
    front_center_ultrasonic_mm: Optional[float] = None,
    front_right_ultrasonic_mm: Optional[float] = None,
    show_status: bool = True,
    mask_geometry: Optional[ParkingGeometry] = None,
) -> Tuple[Any, Any]:
    mask_roles = mask_geometry if mask_geometry is not None else geometry
    display = frame.copy()
    for index, mask in enumerate(frame_masks):
        color = np.asarray(parking_mask_color(index, mask_roles), dtype=np.float32)
        selected = mask > 0
        display[selected] = (0.55 * display[selected] + 0.45 * color).astype(np.uint8)
    polygon = (
        transformer.src_polygon(frame.shape[:2])
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )
    cv2.polylines(display, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
    draw_camera_alignment_line(cv2, display)

    bev_display = transformer.warp_frame(frame)
    for index, mask in enumerate(bev_masks):
        color = np.asarray(parking_mask_color(index, mask_roles), dtype=np.float32)
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
        "slot pair=%s coast=%s entryErr=%s centerX=%s" % (
            "Y" if lidar.gap_pair_observed else "N",
            "Y" if lidar.coasted else "N",
            "-" if lidar.entry_error_mm is None else "%+.0fmm" % lidar.entry_error_mm,
            (
                "-"
                if lidar.gap_center_x_right_mm is None
                else "%+.0fmm" % lidar.gap_center_x_right_mm
            ),
        ),
        "plan=%s" % plan.reason,
        "ultrasonic FL=%s FC=%s FR=%s SL=%s SR=%s" % (
            "-" if front_left_ultrasonic_mm is None else "%.0fmm" % front_left_ultrasonic_mm,
            "-" if front_center_ultrasonic_mm is None else "%.0fmm" % front_center_ultrasonic_mm,
            "-" if front_right_ultrasonic_mm is None else "%.0fmm" % front_right_ultrasonic_mm,
            "-" if left_ultrasonic_mm is None else "%.0fmm" % left_ultrasonic_mm,
            "-" if right_ultrasonic_mm is None else "%.0fmm" % right_ultrasonic_mm,
        ),
        "fps=%.1f | SPACE start/resume | R stop/reset | Q quit" % fps,
    )
    if show_status:
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


def draw_camera_alignment_line(cv2: Any, image: Any) -> None:
    height, width = image.shape[:2]
    x = width // 2
    cv2.line(image, (x, 0), (x, height - 1), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(image, (x, 0), (x, height - 1), (255, 255, 255), 2, cv2.LINE_AA)


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
    geometry: ParkingGeometry,
    plan: Any,
    slot_polygon: Optional[tuple] = None,
    slot_status: str = "",
) -> Any:
    size = 600
    scale = 0.065  # About 9.2 m across, including the expanded right ROI.
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
        cv2, canvas, config.lidar.car_detection_roi,
        origin, scale, rotation_deg, (120, 80, 0),
    )
    if observation.gap_confirmed and config.lidar.slot_tracking_roi is not None:
        draw_roi(
            cv2, canvas, config.lidar.slot_tracking_roi,
            origin, scale, rotation_deg, (120, 0, 120),
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
    dynamic_slot = slot_polygon
    if dynamic_slot is None:
        dynamic_slot = infer_dynamic_slot_polygon(
            observation,
            config.lidar.parking_space_depth_mm,
            config.lidar.parking_space_width_mm,
        )
    if dynamic_slot is not None:
        slot_color = (
            (0, 120, 200)
            if observation.coasted or "hold" in slot_status
            else (0, 180, 255)
        )
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
    draw_reverse_path_on_lidar(
        cv2,
        np,
        canvas,
        config,
        geometry,
        getattr(plan, "path", None),
        origin,
        scale,
        rotation_deg,
    )
    draw_hybrid_path_on_lidar(
        cv2,
        np,
        canvas,
        config,
        getattr(plan, "world_path", None),
        origin,
        scale,
        rotation_deg,
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
        "%s%s cars=%d gap=%s pair=%s err=%s centerX=%s" % (
            observation.reason,
            " HOLD" if observation.coasted else "",
            observation.car_count,
            "Y" if observation.gap_confirmed else "N",
            "Y" if observation.gap_pair_observed else "N",
            "-" if observation.entry_error_mm is None else "%+.0f" % observation.entry_error_mm,
            (
                "-"
                if observation.gap_center_x_right_mm is None
                else "%+.0f" % observation.gap_center_x_right_mm
            ),
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
        "orange=live two-car corridor | green=live centerline | blue=cars",
        (12, size - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "%s | display=%+.0f deg (perception=%+.0f deg)" % (
            slot_status or "slot_unlocked",
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


def draw_world_polyline(
    cv2: Any,
    np: Any,
    image: Any,
    points: Tuple[Tuple[float, float], ...],
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    pixels = np.asarray(
        [
            world_to_lidar_pixel(point[0], point[1], origin, scale, rotation_deg)
            for point in points
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)


def draw_reverse_path_on_lidar(
    cv2: Any,
    np: Any,
    image: Any,
    config: ParkingAppConfig,
    geometry: ParkingGeometry,
    path: Any,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
) -> None:
    world_points = reverse_path_points_to_lidar_world(config, geometry, path)
    if not world_points:
        return
    draw_world_polyline(
        cv2,
        np,
        image,
        world_points,
        origin,
        scale,
        rotation_deg,
        (255, 255, 0),
        3,
    )
    target = world_points[-1]
    cv2.circle(
        image,
        world_to_lidar_pixel(target[0], target[1], origin, scale, rotation_deg),
        6,
        (255, 255, 255),
        2,
    )
    if path.lookahead_point is not None:
        lookahead = reverse_path_point_to_lidar_world(
            config,
            geometry,
            path.lookahead_point,
        )
        if lookahead is not None:
            cv2.circle(
                image,
                world_to_lidar_pixel(
                    lookahead[0], lookahead[1], origin, scale, rotation_deg
                ),
                5,
                (0, 255, 255),
                -1,
            )


def draw_hybrid_path_on_lidar(
    cv2: Any,
    np: Any,
    image: Any,
    config: ParkingAppConfig,
    path: Any,
    origin: Tuple[int, int],
    scale: float,
    rotation_deg: float,
) -> None:
    if path is None or not getattr(path, "poses", None):
        return
    sensor_to_axle = config.lidar.sensor_to_rear_axle_y_back_mm
    world_points = tuple(
        (
            float(pose.x_right_mm),
            sensor_to_axle - float(pose.y_forward_mm),
        )
        for pose in path.poses
    )
    if len(world_points) >= 2:
        draw_world_polyline(
            cv2,
            np,
            image,
            world_points,
            origin,
            scale,
            rotation_deg,
            (255, 255, 0),
            3,
        )
    if path.goal is not None:
        goal = (
            float(path.goal.x_right_mm),
            sensor_to_axle - float(path.goal.y_forward_mm),
        )
        cv2.drawMarker(
            image,
            world_to_lidar_pixel(
                goal[0],
                goal[1],
                origin,
                scale,
                rotation_deg,
            ),
            (255, 255, 255),
            cv2.MARKER_TILTED_CROSS,
            15,
            2,
            cv2.LINE_AA,
        )


def reverse_path_points_to_lidar_world(
    config: ParkingAppConfig,
    geometry: ParkingGeometry,
    path: Any,
) -> Tuple[Tuple[float, float], ...]:
    if path is None or not path.points:
        return ()
    result = []
    for point in path.points:
        world = reverse_path_point_to_lidar_world(config, geometry, point)
        if world is not None:
            result.append(world)
    return tuple(result)


def reverse_path_point_to_lidar_world(
    config: ParkingAppConfig,
    geometry: ParkingGeometry,
    point: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    if config.lidar.parking_space_width_mm <= 0.0:
        return None
    pixels_per_mm = (
        config.geometry.expected_slot_width_px
        / config.lidar.parking_space_width_mm
    )
    if pixels_per_mm <= 0.0:
        return None
    x_right = (point[0] - geometry.vehicle_x_px) / pixels_per_mm
    y_back = (
        config.lidar.sensor_to_rear_axle_y_back_mm
        + (geometry.vehicle_y_px - point[1]) / pixels_per_mm
    )
    return x_right, y_back


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
            startup_delay_s=2.0,
            ready_timeout_s=3.0,
        ),
        max_speed=max(
            abs(config.model_planner.search_speed),
            abs(config.model_planner.gap_tracking_speed),
            abs(config.model_planner.maneuver_forward_speed),
            abs(config.model_planner.maneuver_reverse_speed),
            abs(config.model_planner.final_reverse_speed),
            abs(config.model_planner.exit_speed),
        ),
        max_steering=abs(config.model_planner.max_steering_command),
    )
    client.connect()
    # The uploaded vehicle_controller firmware keeps ultrasonic streaming OFF
    # after reset. Without USON it only replies "OK DRIVE/STOP", so SR remains
    # None forever and the parking trigger can never fire.
    client.write_line("USON")
    LOG.info(
        "serial connected: %s; ultrasonic streaming enabled",
        client.port,
    )
    return client


def open_capture(
    cv2: Any,
    source: str,
    config: ParkingAppConfig,
    *,
    segmenter: Optional[YoloLaneSegmenter] = None,
    transformer: Optional[BevTransformer] = None,
) -> Any:
    if is_auto_camera_source(source):
        return open_auto_capture(cv2, config, segmenter, transformer)

    value: Any = int(source) if source.isdigit() else str(resolve_path(source))
    if isinstance(value, int):
        cap = open_camera_index(cv2, value, config.rear_camera)
    else:
        cap = cv2.VideoCapture(value)
        configure_capture(cv2, cap, config.rear_camera)
    if not cap.isOpened():
        raise RuntimeError("rear camera/video could not be opened: %s" % source)
    return cap


def open_front_capture(
    cv2: Any,
    source: str,
    config: ParkingAppConfig,
    rear_source: str,
) -> Any:
    if is_auto_camera_source(source):
        rear_index = int(rear_source) if rear_source.isdigit() else None
        candidates = camera_source_candidates(
            config.front_camera.index,
            max_index=2 if sys.platform == "darwin" else 5,
        )
        tried: list[int] = []
        for index in candidates:
            if rear_index is not None and index == rear_index:
                continue
            tried.append(index)
            try:
                cap = open_camera_index(cv2, index, config.front_camera)
            except RuntimeError:
                continue
            LOG.info("front camera auto-selected: index=%s", index)
            return cap
        raise RuntimeError(
            "front camera auto-detect failed; tried indices %s. "
            "Pass --front-source 0, --front-source 1, or --no-front-camera."
            % ",".join(str(index) for index in tried)
        )

    if source.isdigit():
        index = int(source)
        if rear_source.isdigit() and int(rear_source) == index:
            raise RuntimeError(
                "front camera index %s matches rear camera index; "
                "set --front-source to the other camera"
                % index
            )
        return open_camera_index(cv2, index, config.front_camera)

    cap = cv2.VideoCapture(str(resolve_path(source)))
    configure_capture(cv2, cap, config.front_camera)
    if not cap.isOpened():
        raise RuntimeError("front camera/video could not be opened: %s" % source)
    return cap


def is_auto_camera_source(source: str) -> bool:
    return source.strip().lower() == "auto"


def camera_source_candidates(preferred: Any, max_index: int = 5) -> list[int]:
    candidates: list[int] = []
    preferred_text = str(preferred).strip().lower()
    if preferred_text and preferred_text != "auto":
        try:
            preferred_index = int(preferred_text)
        except ValueError:
            preferred_index = None
        if preferred_index is not None and preferred_index >= 0:
            candidates.append(preferred_index)
    for index in range(max_index + 1):
        if index not in candidates:
            candidates.append(index)
    return candidates


def open_auto_capture(
    cv2: Any,
    config: ParkingAppConfig,
    segmenter: Optional[YoloLaneSegmenter],
    transformer: Optional[BevTransformer],
) -> Any:
    candidates = camera_source_candidates(
        config.rear_camera.index,
        max_index=2 if sys.platform == "darwin" else 5,
    )
    best_cap = None
    best_index = None
    best_score = float("-inf")
    best_reason = "not_tested"
    tried: list[int] = []
    for index in candidates:
        tried.append(index)
        try:
            cap = open_camera_index(cv2, index, config.rear_camera)
        except RuntimeError:
            continue
        score, reason = score_rear_camera_candidate(
            cap,
            config,
            segmenter,
            transformer,
        )
        if score > best_score:
            if best_cap is not None:
                best_cap.release()
            best_cap = cap
            best_index = index
            best_score = score
            best_reason = reason
        else:
            cap.release()

    if best_cap is None:
        raise RuntimeError(
            "rear camera auto-detect failed; tried indices %s. "
            "Pass --source 0 or --source 1 after checking the camera index."
            % ",".join(str(index) for index in tried)
        )
    LOG.info(
        "rear camera auto-selected: index=%s score=%.2f reason=%s",
        best_index,
        best_score,
        best_reason,
    )
    return best_cap


def open_camera_index(cv2: Any, index: int, camera_config: Any) -> Any:
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    configure_capture(cv2, cap, camera_config)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("camera index could not be opened: %s" % index)
    return cap


def configure_capture(cv2: Any, cap: Any, camera_config: Any) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_config.fourcc))


def score_rear_camera_candidate(
    cap: Any,
    config: ParkingAppConfig,
    segmenter: Optional[YoloLaneSegmenter],
    transformer: Optional[BevTransformer],
) -> Tuple[float, str]:
    frame = None
    for _ in range(5):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate
    if frame is None:
        return -1.0, "no_frame"
    if segmenter is None or transformer is None:
        return 0.0, "opened"

    try:
        class_masks = segmenter.segment_class_masks(frame)
        parking_masks = list(class_masks.lane)
        if not parking_masks:
            return 0.1, "no_parking_line_mask"
        estimator = ParkingGeometryEstimator(config.geometry)
        bev_masks = [transformer.warp_mask(mask) for mask in parking_masks]
        geometry = estimator.estimate(bev_masks, class_masks.lane_conf)
    except Exception as exc:
        LOG.warning("camera auto-score failed: %s", exc)
        return 0.0, "score_failed"

    score = geometry.confidence
    score += min(3, geometry.observed_line_count) * 0.25
    if geometry.has_side_pair:
        score += 1.0
    if geometry.has_back_line:
        score += 0.75
    if geometry.found:
        score += 1.0
    return score, geometry.reason


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
    if args.camera_enabled is not None:
        runtime = replace(runtime, camera_enabled=args.camera_enabled)
    if args.front_camera_enabled is not None:
        runtime = replace(runtime, front_camera_enabled=args.front_camera_enabled)
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
    model_planner = config.model_planner
    vehicle = config.vehicle
    if args.first_car_preemptive_turn is not None:
        planner = replace(
            planner,
            first_car_preemptive_turn_enabled=(
                args.first_car_preemptive_turn == "on"
            ),
        )
    if args.prealign_speed is not None:
        planner = replace(planner, prealign_speed=args.prealign_speed)
    if args.prealign_steering is not None:
        planner = replace(planner, prealign_steering=args.prealign_steering)
    if args.prealign_timeout_s is not None:
        planner = replace(planner, prealign_timeout_s=args.prealign_timeout_s)
    if args.straight_steering_trim is not None:
        planner = replace(
            planner,
            straight_steering_trim=args.straight_steering_trim,
        )
        model_planner = replace(
            model_planner,
            straight_steering_trim=args.straight_steering_trim,
        )
    if args.entry_setup_speed is not None:
        planner = replace(planner, entry_setup_speed=args.entry_setup_speed)
    if args.entry_setup_steering is not None:
        planner = replace(
            planner,
            entry_setup_steering=args.entry_setup_steering,
        )
    if args.entry_setup_min_s is not None:
        planner = replace(planner, entry_setup_min_s=args.entry_setup_min_s)
    if args.entry_setup_max_s is not None:
        planner = replace(planner, entry_setup_max_s=args.entry_setup_max_s)
    if args.entry_setup_target_heading_deg is not None:
        planner = replace(
            planner,
            entry_setup_target_heading_deg=args.entry_setup_target_heading_deg,
        )
    if args.park_hold_s is not None:
        planner = replace(planner, park_hold_s=args.park_hold_s)
        model_planner = replace(model_planner, park_hold_s=args.park_hold_s)
    if args.exit_speed is not None:
        planner = replace(planner, exit_speed=args.exit_speed)
        model_planner = replace(model_planner, exit_speed=args.exit_speed)
    if args.exit_turn_steering is not None:
        planner = replace(planner, exit_turn_steering=args.exit_turn_steering)
    if args.exit_turn_s is not None:
        planner = replace(planner, exit_turn_s=args.exit_turn_s)
    if args.exit_straight_s is not None:
        planner = replace(planner, exit_straight_s=args.exit_straight_s)
    if args.exit_right_min_clearance_cm is not None:
        planner = replace(
            planner,
            exit_right_min_clearance_mm=args.exit_right_min_clearance_cm * 10.0,
        )
    if args.wheelbase_mm is not None:
        vehicle = replace(vehicle, wheelbase_mm=args.wheelbase_mm)
    if args.max_steering_angle_deg is not None:
        vehicle = replace(
            vehicle,
            max_steering_angle_deg=args.max_steering_angle_deg,
        )
    if args.vehicle_width_mm is not None:
        vehicle = replace(vehicle, width_mm=args.vehicle_width_mm)
    if args.vehicle_length_mm is not None:
        vehicle = replace(vehicle, length_mm=args.vehicle_length_mm)
    if args.rear_axle_to_rear_bumper_mm is not None:
        vehicle = replace(
            vehicle,
            rear_axle_to_rear_bumper_mm=args.rear_axle_to_rear_bumper_mm,
        )
    if args.collision_clearance_mm is not None:
        vehicle = replace(
            vehicle,
            collision_clearance_mm=args.collision_clearance_mm,
        )
    if args.parking_back_clearance_mm is not None:
        model_planner = replace(
            model_planner,
            back_clearance_mm=args.parking_back_clearance_mm,
        )
    if args.forward_lookahead_mm is not None:
        model_planner = replace(
            model_planner,
            forward_lookahead_mm=args.forward_lookahead_mm,
        )
    if args.reverse_lookahead_mm is not None:
        model_planner = replace(
            model_planner,
            reverse_lookahead_mm=args.reverse_lookahead_mm,
        )
    if args.maneuver_forward_speed is not None:
        model_planner = replace(
            model_planner,
            maneuver_forward_speed=args.maneuver_forward_speed,
        )
    if args.maneuver_reverse_speed is not None:
        model_planner = replace(
            model_planner,
            maneuver_reverse_speed=args.maneuver_reverse_speed,
        )
    if args.final_reverse_speed is not None:
        model_planner = replace(
            model_planner,
            final_reverse_speed=args.final_reverse_speed,
        )
    if args.right_ultrasonic_first_car_max_cm is not None:
        model_planner = replace(
            model_planner,
            right_ultrasonic_first_car_max_mm=(
                args.right_ultrasonic_first_car_max_cm * 10.0
            ),
        )
    if args.right_ultrasonic_open_gap_min_cm is not None:
        model_planner = replace(
            model_planner,
            right_ultrasonic_open_gap_min_mm=(
                args.right_ultrasonic_open_gap_min_cm * 10.0
            ),
        )
    if args.right_ultrasonic_open_confirm_scans is not None:
        model_planner = replace(
            model_planner,
            right_ultrasonic_open_confirm_scans=(
                args.right_ultrasonic_open_confirm_scans
            ),
        )
    if args.auto_exit is not None:
        model_planner = replace(
            model_planner,
            auto_exit_enabled=args.auto_exit,
        )
    if bev.src_top_left[0] >= bev.src_top_right[0]:
        raise ValueError("BEV top-left x must be smaller than top-right x")
    if bev.src_bottom_left[0] >= bev.src_bottom_right[0]:
        raise ValueError("BEV bottom-left x must be smaller than bottom-right x")
    if bev.src_top_left[1] >= bev.src_bottom_left[1]:
        raise ValueError("BEV top y must be smaller than bottom y")
    if not 0.0 <= bev.dst_x_margin < 0.5:
        raise ValueError("BEV destination margin must be at least 0 and below 0.5")
    updated = replace(
        config,
        bev=bev,
        lidar=lidar,
        planner=planner,
        model_planner=model_planner,
        vehicle=vehicle,
        runtime=runtime,
    )
    validate_model_based_parking_config(updated)
    return updated


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
    parser = argparse.ArgumentParser(
        description="LiDAR-relative model-based T-parking runtime"
    )
    parser.add_argument("--config", default="configs/parking.json")
    parser.add_argument(
        "--source",
        default=None,
        help="rear camera index, auto, or recorded video; default uses config rear_camera.index",
    )
    parser.add_argument(
        "--front-source",
        default=None,
        help="front camera index, auto, or video; default uses config front_camera.index",
    )
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--camera",
        dest="camera_enabled",
        action="store_true",
        help="enable the optional rear-camera YOLO diagnostic panels",
    )
    camera_group.add_argument(
        "--no-camera",
        dest="camera_enabled",
        action="store_false",
        help="run LiDAR-only without opening a camera or loading YOLO",
    )
    parser.set_defaults(camera_enabled=None)
    front_camera_group = parser.add_mutually_exclusive_group()
    front_camera_group.add_argument(
        "--front-camera",
        dest="front_camera_enabled",
        action="store_true",
        help="show a front-camera inset on the live dashboard",
    )
    front_camera_group.add_argument(
        "--no-front-camera",
        dest="front_camera_enabled",
        action="store_false",
        help="disable the front-camera inset",
    )
    parser.set_defaults(front_camera_enabled=None)
    parser.add_argument("--recording-zip", default=None, help="ZIP containing one MP4 and one *_lidar.csv")
    parser.add_argument("--model", default=None, help="parking YOLO segmentation model")
    parser.add_argument("--device", default=None, help="auto, cpu, mps, 0, cuda, ...")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference size; CPU replay can use 512")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence override")
    parser.add_argument("--lidar-csv", default=None, help="recorded LiDAR CSV synchronized by relative time")
    parser.add_argument("--lidar-port", default=None, help="live RPLidar serial port")
    parser.add_argument("--lidar-offset", type=float, default=0.0, help="seconds added to video time for CSV lookup")
    parser.add_argument("--allow-no-lidar", action="store_true", help="explicit unsafe test bypass; preview only")
    serial_group = parser.add_mutually_exclusive_group()
    serial_group.add_argument(
        "--serial",
        dest="serial",
        action="store_true",
        help="enable Arduino output (default for live camera/LiDAR)",
    )
    serial_group.add_argument(
        "--no-serial",
        dest="serial",
        action="store_false",
        help="disable Arduino output for a live dry run",
    )
    parser.set_defaults(serial=None)
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="video replay only: infer every Nth frame")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="start the parking state machine immediately",
    )
    parser.add_argument(
        "--manual-start",
        action="store_true",
        help="wait for SPACE instead of using the config auto_start default",
    )
    parser.add_argument("--replay-delay-ms", type=int, default=1)
    parser.add_argument(
        "--record-dashboard",
        choices=("auto", "on", "off"),
        default="auto",
        help="record the displayed dashboard; auto records numeric live-camera runs",
    )
    parser.add_argument(
        "--parking-record-dir",
        default="data/parking",
        help="directory for timestamped live dashboard MP4 files",
    )
    parser.add_argument(
        "--dashboard-record-fps",
        type=float,
        default=10.0,
        help="dashboard MP4 frame rate",
    )
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
        "--wheelbase-mm",
        type=float,
        default=None,
        help="measured rear-to-front axle distance for the bicycle model",
    )
    parser.add_argument(
        "--max-steering-angle-deg",
        type=float,
        default=None,
        help="measured physical road-wheel angle at maximum steering command",
    )
    parser.add_argument(
        "--vehicle-width-mm",
        type=float,
        default=None,
        help="widest body or wheel-envelope width used by collision checking",
    )
    parser.add_argument(
        "--vehicle-length-mm",
        type=float,
        default=None,
        help="front-to-rear body envelope length used by collision checking",
    )
    parser.add_argument(
        "--rear-axle-to-rear-bumper-mm",
        type=float,
        default=None,
        help="rear axle center to the rearmost body or sensor point",
    )
    parser.add_argument(
        "--collision-clearance-mm",
        type=float,
        default=None,
        help="extra safety envelope added around the vehicle footprint",
    )
    parser.add_argument(
        "--parking-back-clearance-mm",
        type=float,
        default=None,
        help="desired rear-bumper clearance from the slot back line",
    )
    parser.add_argument(
        "--forward-lookahead-mm",
        type=float,
        default=None,
        help="pure-pursuit lookahead while following forward path segments",
    )
    parser.add_argument(
        "--reverse-lookahead-mm",
        type=float,
        default=None,
        help="pure-pursuit lookahead while following reverse path segments",
    )
    parser.add_argument(
        "--maneuver-forward-speed",
        type=int,
        default=None,
        help="positive motor command on planned forward parking segments",
    )
    parser.add_argument(
        "--maneuver-reverse-speed",
        type=int,
        default=None,
        help="negative motor command on planned reverse parking segments",
    )
    parser.add_argument(
        "--final-reverse-speed",
        type=int,
        default=None,
        help="negative motor command inside the final slow-down distance",
    )
    parser.add_argument(
        "--right-ultrasonic-first-car-max-cm",
        type=float,
        default=None,
        help="right side ultrasonic distance treated as the first parked car",
    )
    parser.add_argument(
        "--right-ultrasonic-open-gap-min-cm",
        type=float,
        default=None,
        help="right side ultrasonic distance treated as an opened empty bay",
    )
    parser.add_argument(
        "--right-ultrasonic-open-confirm-scans",
        type=int,
        default=None,
        help="consecutive open ultrasonic readings before entry setup starts",
    )
    exit_mode = parser.add_mutually_exclusive_group()
    exit_mode.add_argument(
        "--auto-exit",
        dest="auto_exit",
        action="store_true",
        help="after the required hold, plan a forward path out of the bay",
    )
    exit_mode.add_argument(
        "--no-auto-exit",
        dest="auto_exit",
        action="store_false",
        help="remain parked after the required hold (recommended during tuning)",
    )
    parser.set_defaults(auto_exit=None)
    parser.add_argument(
        "--prealign-speed",
        type=int,
        default=None,
        help="legacy replay option; ignored by the model-based live planner",
    )
    parser.add_argument(
        "--first-car-turn-target-cm",
        type=float,
        default=None,
        help="legacy replay option; ignored by the model-based live planner",
    )
    parser.add_argument(
        "--first-car-preemptive-turn",
        choices=("on", "off"),
        default=None,
        help="legacy replay option; ignored by the model-based live planner",
    )
    parser.add_argument(
        "--prealign-steering",
        type=int,
        default=None,
        help="legacy replay option; ignored by the model-based live planner",
    )
    parser.add_argument(
        "--prealign-timeout-s",
        type=float,
        default=None,
        help="legacy replay option; ignored by the model-based live planner",
    )
    parser.add_argument(
        "--straight-steering-trim",
        type=int,
        default=None,
        help="signed steering trim while driving straight during slot search",
    )
    parser.add_argument(
        "--entry-setup-speed",
        type=int,
        default=None,
        help="legacy replay option; live setup speed is --maneuver-forward-speed",
    )
    parser.add_argument(
        "--entry-setup-steering",
        type=int,
        default=None,
        help="legacy replay option; live steering comes from the planned curvature",
    )
    parser.add_argument(
        "--entry-setup-min-s",
        type=float,
        default=None,
        help="legacy replay option; live movement never transitions by elapsed time",
    )
    parser.add_argument(
        "--entry-setup-max-s",
        type=float,
        default=None,
        help="legacy replay option; live movement never transitions by elapsed time",
    )
    parser.add_argument(
        "--entry-setup-target-heading-deg",
        type=float,
        default=None,
        help="legacy replay option; live gear changes come from Hybrid A*",
    )
    parser.add_argument(
        "--park-hold-s",
        type=float,
        default=None,
        help="seconds to stay stopped in the parking bay before exiting",
    )
    parser.add_argument(
        "--exit-speed",
        type=int,
        default=None,
        help="forward speed for the post-parking exit sequence",
    )
    parser.add_argument(
        "--exit-turn-steering",
        type=int,
        default=None,
        help="legacy replay option; live exit steering comes from its planned path",
    )
    parser.add_argument(
        "--exit-turn-s",
        type=float,
        default=None,
        help="legacy replay option; live exit never transitions by elapsed motion time",
    )
    parser.add_argument(
        "--exit-straight-s",
        type=float,
        default=None,
        help="legacy replay option; live exit ends at a relative lane pose",
    )
    parser.add_argument(
        "--exit-right-min-clearance-cm",
        type=float,
        default=None,
        help="legacy replay option; live emergency threshold is in model_planner",
    )
    args = parser.parse_args(argv)
    if args.auto_start and args.manual_start:
        parser.error("--auto-start and --manual-start cannot be used together")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.imgsz is not None and args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.dashboard_record_fps <= 0.0 or args.dashboard_record_fps > 60.0:
        parser.error("--dashboard-record-fps must be above 0 and at most 60")
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
        args.straight_steering_trim is not None
        and abs(args.straight_steering_trim) > 150
    ):
        parser.error("--straight-steering-trim must be between -150 and 150")
    if args.entry_setup_speed is not None and args.entry_setup_speed <= 0:
        parser.error("--entry-setup-speed must be positive")
    if (
        args.entry_setup_steering is not None
        and abs(args.entry_setup_steering) > 150
    ):
        parser.error("--entry-setup-steering must be between -150 and 150")
    if args.entry_setup_steering == 0:
        parser.error("--entry-setup-steering cannot be zero")
    if args.entry_setup_min_s is not None and args.entry_setup_min_s < 0.0:
        parser.error("--entry-setup-min-s cannot be negative")
    if args.entry_setup_max_s is not None and args.entry_setup_max_s <= 0.0:
        parser.error("--entry-setup-max-s must be positive")
    if (
        args.entry_setup_min_s is not None
        and args.entry_setup_max_s is not None
        and args.entry_setup_max_s < args.entry_setup_min_s
    ):
        parser.error("--entry-setup-max-s must be at least --entry-setup-min-s")
    if (
        args.entry_setup_target_heading_deg is not None
        and args.entry_setup_target_heading_deg <= 0.0
    ):
        parser.error("--entry-setup-target-heading-deg must be positive")
    if (
        args.park_hold_s is not None
        and not 3.0 <= args.park_hold_s <= 5.0
    ):
        parser.error("--park-hold-s must be between 3 and 5 seconds")
    if args.exit_speed is not None and args.exit_speed <= 0:
        parser.error("--exit-speed must be positive")
    if args.exit_turn_steering is not None and abs(args.exit_turn_steering) > 150:
        parser.error("--exit-turn-steering must be between -150 and 150")
    if args.exit_turn_s is not None and args.exit_turn_s < 0.0:
        parser.error("--exit-turn-s cannot be negative")
    if args.exit_straight_s is not None and args.exit_straight_s < 0.0:
        parser.error("--exit-straight-s cannot be negative")
    if (
        args.exit_right_min_clearance_cm is not None
        and args.exit_right_min_clearance_cm < 0.0
    ):
        parser.error("--exit-right-min-clearance-cm cannot be negative")
    if (
        args.first_car_turn_target_cm is not None
        and not -250.0 <= args.first_car_turn_target_cm <= 250.0
    ):
        parser.error("--first-car-turn-target-cm must be between -250 and 250")
    positive_model_values = (
        "wheelbase_mm",
        "max_steering_angle_deg",
        "vehicle_width_mm",
        "vehicle_length_mm",
        "rear_axle_to_rear_bumper_mm",
        "forward_lookahead_mm",
        "reverse_lookahead_mm",
    )
    for name in positive_model_values:
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    nonnegative_model_values = (
        "collision_clearance_mm",
        "parking_back_clearance_mm",
    )
    for name in nonnegative_model_values:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error("--%s cannot be negative" % name.replace("_", "-"))
    if args.maneuver_forward_speed is not None and args.maneuver_forward_speed <= 0:
        parser.error("--maneuver-forward-speed must be positive")
    if args.maneuver_reverse_speed is not None and args.maneuver_reverse_speed >= 0:
        parser.error("--maneuver-reverse-speed must be negative")
    if args.final_reverse_speed is not None and args.final_reverse_speed >= 0:
        parser.error("--final-reverse-speed must be negative")
    if (
        args.right_ultrasonic_first_car_max_cm is not None
        and args.right_ultrasonic_first_car_max_cm <= 0.0
    ):
        parser.error("--right-ultrasonic-first-car-max-cm must be positive")
    if (
        args.right_ultrasonic_open_gap_min_cm is not None
        and args.right_ultrasonic_open_gap_min_cm <= 0.0
    ):
        parser.error("--right-ultrasonic-open-gap-min-cm must be positive")
    if (
        args.right_ultrasonic_open_confirm_scans is not None
        and args.right_ultrasonic_open_confirm_scans < 1
    ):
        parser.error("--right-ultrasonic-open-confirm-scans must be at least 1")
    return args


def load_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for parking runtime") from exc
