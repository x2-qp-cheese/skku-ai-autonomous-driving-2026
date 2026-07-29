from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import replace
from math import atan2, cos, degrees, hypot, radians, sin
from pathlib import Path
from typing import Optional

from ..config import AppConfig, load_config
from ..control.serial_vehicle import SerialVehicle
from ..perception.rear_lidar import (
    RearLidarObservation,
    RearLidarPerception,
    TangentPair,
)
from ..planning.paper_controller import PaperParkingController
from ..sensors.lidar import (
    LidarCsvRecorder,
    LidarCsvReplay,
    RplidarScanner,
    find_lidar_ports,
)
from ..types import ControlCommand, ParkingState


LOG = logging.getLogger("skku_paper_parking")

# Figure 3 of the paper.  These are reference rays for the debug view,
# not extra perception thresholds.
PAPER_REFERENCE_ANGLES_DEG = (
    -110.0,
    -90.0,
    -60.0,
    -45.0,
    -30.0,
    0.0,
    30.0,
    45.0,
    60.0,
    90.0,
    110.0,
)


class TelemetryRecorder:
    """Record every value needed to inspect the paper equations."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=(
                "elapsed_s",
                "lidar_timestamp",
                "state",
                "speed",
                "steering",
                "steering_offset",
                "reason",
                "raw_points",
                "valid_points",
                "left_fov_points",
                "right_fov_points",
                "tangent_clusters",
                "tangent_candidates",
                "detected_vehicle_count",
                "right_vehicle",
                "right_vehicle_raw",
                "right_cluster_points",
                "right_vehicle_min_y_back_mm",
                "right_seen_scans",
                "right_missing_scans",
                "right_track_id",
                "is_prealign_near",
                "is_near",
                "pair_source_scans",
                "pair_valid",
                "angle_a_deg",
                "angle_b_deg",
                "angle_bisector_deg",
                "dist_a_mm",
                "dist_b_mm",
                "control_pair_valid",
                "control_angle_a_deg",
                "control_angle_b_deg",
                "control_angle_bisector_deg",
                "control_dist_a_mm",
                "control_dist_b_mm",
                "control_gap_width_mm",
                "control_gap_center_angle_deg",
                "control_pair_reason",
                "distance_bias_ab_mm",
                "angle_term",
                "distance_term",
                "paper_steering",
                "applied_paper_steering",
                "entry_center_angle_deg",
                "realigning_after_dropout",
                "prealign_near_seen",
                "prealign_ready_scans",
                "pair_hold_scans",
                "pair_reacquiring",
                "center_observation_scans",
                "dist_c_mm",
                "dist_d_mm",
                "c_y_back_mm",
                "d_y_back_mm",
                "distance_bias_cd_mm",
                "cd_balance_ratio",
                "cd_balance_error",
                "cd_balance_span",
                "cd_center_ready_scans",
                "left_side_heading_deg",
                "left_side_line_points",
                "left_side_line_extent_mm",
                "left_side_line_linearity",
                "right_side_heading_deg",
                "right_side_line_points",
                "right_side_line_extent_mm",
                "right_side_line_linearity",
                "observed_slot_heading_deg",
                "filtered_slot_heading_deg",
                "slot_heading_span_deg",
                "parallel_heading_ready_scans",
                "parallel_correction_cycles",
                "parallel_cd_missing_scans",
                "parallel_no_improvement_scans",
                "parallel_best_heading_deg",
            ),
        )
        self._writer.writeheader()

    def write(
        self,
        elapsed: float,
        observation: RearLidarObservation,
        controller: PaperParkingController,
        command: ControlCommand,
    ) -> None:
        pair = observation.pair
        control_pair = controller.control_pair
        gap_width, gap_center_angle = _pair_geometry(
            control_pair
        )
        debug = controller.debug
        self._writer.writerow(
            {
                "elapsed_s": "%.6f" % elapsed,
                "lidar_timestamp": "%.9f" % observation.timestamp,
                "state": controller.state.value,
                "speed": command.speed,
                "steering": command.steering,
                "steering_offset": (
                    controller.config.actuator_steering_offset
                ),
                "reason": command.reason,
                "raw_points": observation.raw_point_count,
                "valid_points": len(observation.points),
                "left_fov_points": (
                    observation.left_fov_point_count
                ),
                "right_fov_points": (
                    observation.right_fov_point_count
                ),
                "tangent_clusters": (
                    observation.tangent_cluster_count
                ),
                "tangent_candidates": (
                    observation.tangent_candidate_count
                ),
                "detected_vehicle_count": (
                    debug.detected_vehicle_count
                ),
                "right_vehicle": int(
                    observation.right_vehicle_present
                ),
                "right_vehicle_raw": int(
                    observation.right_vehicle_raw
                ),
                "right_cluster_points": (
                    observation.right_vehicle_cluster_points
                ),
                "right_vehicle_min_y_back_mm": _optional(
                    observation.right_vehicle_min_y_back_mm
                ),
                "right_seen_scans": (
                    observation.right_vehicle_seen_scans
                ),
                "right_missing_scans": (
                    observation.right_vehicle_missing_scans
                ),
                "right_track_id": (
                    observation.right_vehicle_track_id
                ),
                "is_prealign_near": int(
                    observation.prealign_near
                ),
                "is_near": int(observation.near),
                "pair_source_scans": (
                    observation.pair_source_scans
                ),
                "pair_valid": int(pair.valid),
                "angle_a_deg": _optional(
                    pair.angle_a_deg if pair.valid else None
                ),
                "angle_b_deg": _optional(
                    pair.angle_b_deg if pair.valid else None
                ),
                "angle_bisector_deg": _optional(
                    pair.angle_bisector_deg if pair.valid else None
                ),
                "dist_a_mm": _optional(
                    pair.dist_a_mm if pair.valid else None
                ),
                "dist_b_mm": _optional(
                    pair.dist_b_mm if pair.valid else None
                ),
                "control_pair_valid": int(control_pair.valid),
                "control_angle_a_deg": _optional(
                    (
                        control_pair.angle_a_deg
                        if control_pair.valid
                        else None
                    )
                ),
                "control_angle_b_deg": _optional(
                    (
                        control_pair.angle_b_deg
                        if control_pair.valid
                        else None
                    )
                ),
                "control_angle_bisector_deg": _optional(
                    (
                        control_pair.angle_bisector_deg
                        if control_pair.valid
                        else None
                    )
                ),
                "control_dist_a_mm": _optional(
                    (
                        control_pair.dist_a_mm
                        if control_pair.valid
                        else None
                    )
                ),
                "control_dist_b_mm": _optional(
                    (
                        control_pair.dist_b_mm
                        if control_pair.valid
                        else None
                    )
                ),
                "control_gap_width_mm": _optional(gap_width),
                "control_gap_center_angle_deg": _optional(
                    gap_center_angle
                ),
                "control_pair_reason": control_pair.reason,
                "distance_bias_ab_mm": _optional(
                    debug.distance_bias_ab_mm
                ),
                "angle_term": _optional(debug.angle_term),
                "distance_term": _optional(debug.distance_term),
                "paper_steering": _optional(debug.paper_steering),
                "applied_paper_steering": _optional(
                    debug.applied_paper_steering
                ),
                "entry_center_angle_deg": _optional(
                    debug.entry_center_angle_deg
                ),
                "realigning_after_dropout": int(
                    debug.realigning_after_dropout
                ),
                "prealign_near_seen": int(
                    debug.prealign_near_seen
                ),
                "prealign_ready_scans": (
                    debug.prealign_ready_scans
                ),
                "pair_hold_scans": debug.pair_hold_scans,
                "pair_reacquiring": int(
                    debug.pair_reacquiring
                ),
                "center_observation_scans": (
                    debug.center_observation_scans
                ),
                "dist_c_mm": _optional(observation.dist_c_mm),
                "dist_d_mm": _optional(observation.dist_d_mm),
                "c_y_back_mm": _optional(observation.c_y_back_mm),
                "d_y_back_mm": _optional(observation.d_y_back_mm),
                "distance_bias_cd_mm": _optional(
                    debug.distance_bias_cd_mm
                ),
                "cd_balance_ratio": _optional(
                    debug.cd_balance_ratio
                ),
                "cd_balance_error": _optional(
                    debug.cd_balance_error
                ),
                "cd_balance_span": _optional(
                    debug.cd_balance_span
                ),
                "cd_center_ready_scans": (
                    debug.cd_center_ready_scans
                ),
                "left_side_heading_deg": _optional(
                    (
                        observation.left_side_line.heading_deg
                        if observation.left_side_line.valid
                        else None
                    )
                ),
                "left_side_line_points": (
                    observation.left_side_line.point_count
                ),
                "left_side_line_extent_mm": _optional(
                    (
                        observation.left_side_line.extent_mm
                        if observation.left_side_line.valid
                        else None
                    )
                ),
                "left_side_line_linearity": _optional(
                    (
                        observation.left_side_line.linearity
                        if observation.left_side_line.valid
                        else None
                    )
                ),
                "right_side_heading_deg": _optional(
                    (
                        observation.right_side_line.heading_deg
                        if observation.right_side_line.valid
                        else None
                    )
                ),
                "right_side_line_points": (
                    observation.right_side_line.point_count
                ),
                "right_side_line_extent_mm": _optional(
                    (
                        observation.right_side_line.extent_mm
                        if observation.right_side_line.valid
                        else None
                    )
                ),
                "right_side_line_linearity": _optional(
                    (
                        observation.right_side_line.linearity
                        if observation.right_side_line.valid
                        else None
                    )
                ),
                "observed_slot_heading_deg": _optional(
                    observation.slot_heading_deg
                ),
                "filtered_slot_heading_deg": _optional(
                    debug.slot_heading_deg
                ),
                "slot_heading_span_deg": _optional(
                    debug.slot_heading_span_deg
                ),
                "parallel_heading_ready_scans": (
                    debug.parallel_heading_ready_scans
                ),
                "parallel_correction_cycles": (
                    debug.parallel_correction_cycles
                ),
                "parallel_cd_missing_scans": (
                    debug.parallel_cd_missing_scans
                ),
                "parallel_no_improvement_scans": (
                    debug.parallel_no_improvement_scans
                ),
                "parallel_best_heading_deg": _optional(
                    debug.parallel_best_heading_deg
                ),
            }
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def run(args: argparse.Namespace) -> int:
    config = _config_with_overrides(load_config(args.config), args)
    replay = LidarCsvReplay(args.lidar_csv) if args.lidar_csv else None
    scanner: Optional[RplidarScanner] = None
    vehicle: Optional[SerialVehicle] = None
    raw_recorder: Optional[LidarCsvRecorder] = None
    telemetry: Optional[TelemetryRecorder] = None
    debug_writer = None
    lidar_port: Optional[str] = None
    window_enabled = config.runtime.debug_window and not args.no_window
    motor_enabled = config.runtime.motor_enabled and not args.no_motor

    if replay is not None and motor_enabled:
        raise RuntimeError("replay is read-only; add --no-motor")

    if replay is None:
        lidar_candidates = find_lidar_ports(config.lidar.port)
        if not lidar_candidates:
            raise RuntimeError(
                "RPLidar port not found; pass --lidar-port explicitly"
            )
        lidar_failures = []
        for candidate in lidar_candidates:
            candidate_scanner = RplidarScanner(
                candidate,
                max_buf_meas=(
                    config.lidar.input_buffer_limit_bytes
                ),
                scan_type=config.lidar.scan_type,
            )
            candidate_scanner.start()
            if candidate_scanner.wait_for_scan(6.0):
                scanner = candidate_scanner
                lidar_port = candidate
                break
            lidar_failures.append(
                "%s: %s"
                % (
                    candidate,
                    candidate_scanner.error
                    or "no complete scan within 6 seconds",
                )
            )
            candidate_scanner.close()
        if scanner is None or lidar_port is None:
            raise RuntimeError(
                "No RPLidar scan from automatic candidates: %s"
                % " | ".join(lidar_failures)
            )
        LOG.info("rear LiDAR: %s", lidar_port)

    if motor_enabled:
        vehicle = SerialVehicle(
            config.serial,
            max_steering=config.controller.actuator_max_steering,
        )
        vehicle.connect(
            excluded_ports=(
                (lidar_port,) if lidar_port is not None else ()
            )
        )
        LOG.info("Arduino: %s", vehicle.port)

    perception = RearLidarPerception(config.lidar)
    controller = PaperParkingController(
        config.controller,
        steering_plus=args.steering_plus == "on",
    )
    if config.runtime.auto_start or args.auto_start:
        controller.start()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_dir = Path(
        args.record_dir or config.runtime.record_directory
    )
    telemetry = TelemetryRecorder(
        record_dir / (stamp + "_paper.csv")
    )
    if scanner is not None:
        raw_recorder = LidarCsvRecorder(
            record_dir / (stamp + "_lidar.csv")
        )
    if window_enabled and config.runtime.record_debug_video:
        import cv2

        debug_path = record_dir / (stamp + "_debug.mp4")
        debug_writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            config.controller.command_rate_hz,
            (1600, 900),
        )
        if not debug_writer.isOpened():
            LOG.warning(
                "debug video writer could not open: %s",
                debug_path,
            )
            debug_writer.release()
            debug_writer = None
        else:
            LOG.info("debug video: %s", debug_path)

    start = time.monotonic()
    last_command_at = 0.0
    period = 1.0 / config.controller.command_rate_hz
    last_command = ControlCommand.stop("startup")
    try:
        while True:
            loop_started = time.monotonic()
            elapsed = loop_started - start
            if replay is not None:
                if elapsed > replay.duration_s:
                    break
                scan = replay.scan_at_elapsed(elapsed)
            else:
                scan = scanner.latest() if scanner is not None else None

            if raw_recorder is not None:
                raw_recorder.write(scan)

            observation = perception.observe(scan)
            # This is input validity handling, not a parking transition.
            # It prevents an old frozen scan from being treated as new data.
            if (
                scan is not None
                and replay is None
                and time.time() - scan.timestamp
                > config.lidar.stale_after_s
            ):
                observation = replace(
                    observation,
                    valid=False,
                    reason="stale_lidar_scan",
                )

            if args.straight_only:
                if controller.state == ParkingState.IDLE:
                    command = ControlCommand.stop(
                        "press_SPACE_to_start_straight_only"
                    )
                else:
                    command = ControlCommand(
                        speed=config.controller.forward_speed,
                        steering=(
                            config.controller.actuator_steering_offset
                        ),
                        reason="straight_only_lidar_observation",
                    )
            else:
                command = controller.update(
                    observation,
                    loop_started,
                )
            last_command = command

            if loop_started - last_command_at >= period:
                if vehicle is not None:
                    vehicle.send(command)
                telemetry.write(
                    elapsed,
                    observation,
                    controller,
                    command,
                )
                last_command_at = loop_started

            if window_enabled:
                key = _show_debug(
                    config,
                    observation,
                    controller,
                    command,
                    motor_enabled,
                    scanner.error if scanner is not None else None,
                    debug_writer,
                    args.straight_only,
                )
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if controller.state == ParkingState.IDLE:
                        perception.reset()
                        controller.start()
                    else:
                        controller.reset()
                        perception.reset()
                        if vehicle is not None:
                            vehicle.stop()
                elif key in (ord("r"), ord("R")):
                    controller.reset()
                    perception.reset()
                    if vehicle is not None:
                        vehicle.stop()
            elif not (
                args.auto_start or config.runtime.auto_start
            ):
                raise RuntimeError(
                    "no debug window requires --auto-start"
                )

            remaining = period - (
                time.monotonic() - loop_started
            )
            if remaining > 0.0:
                time.sleep(min(remaining, 0.02))
    finally:
        if vehicle is not None:
            vehicle.close()
        if scanner is not None:
            scanner.close()
        if raw_recorder is not None:
            raw_recorder.close()
        if telemetry is not None:
            telemetry.close()
        if debug_writer is not None:
            debug_writer.release()
        if window_enabled:
            import cv2

            cv2.destroyAllWindows()

    LOG.info(
        "finished state=%s command=%s",
        controller.state.value,
        last_command.reason,
    )
    return 0


def _show_debug(
    config: AppConfig,
    observation: RearLidarObservation,
    controller: PaperParkingController,
    command: ControlCommand,
    motor_enabled: bool,
    lidar_error: Optional[Exception],
    debug_writer=None,
    straight_only: bool = False,
) -> int:
    import cv2
    import numpy as np

    # A large, separate LiDAR pane keeps individual raw returns visible in
    # the screen recording instead of hiding them under telemetry text.
    width, height = 1600, 900
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    origin = (570, 205)
    scale = 540.0 / max(
        500.0,
        config.runtime.display_range_mm,
    )

    def pixel(x_right_mm: float, y_back_mm: float):
        return (
            int(round(origin[0] + x_right_mm * scale)),
            int(round(origin[1] + y_back_mm * scale)),
        )

    vehicle_half_width = int(round(250.0 * scale))
    vehicle_length = int(round(1000.0 * scale))
    cv2.rectangle(
        canvas,
        (
            origin[0] - vehicle_half_width,
            origin[1] - vehicle_length - 30,
        ),
        (
            origin[0] + vehicle_half_width,
            origin[1] - 30,
        ),
        (210, 210, 210),
        2,
    )
    cv2.circle(canvas, origin, 7, (0, 0, 255), -1)
    cv2.putText(
        canvas,
        "LiDAR ORIGIN",
        (origin[0] - 75, origin[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        origin,
        int(round(config.lidar.near_distance_mm * scale)),
        (95, 30, 95),
        2,
    )
    cv2.circle(
        canvas,
        origin,
        int(round(config.lidar.side_distance_limit_mm * scale)),
        (55, 55, 55),
        1,
    )

    paper_angles = tuple(
        angle
        for angle in PAPER_REFERENCE_ANGLES_DEG
        if abs(angle) <= config.lidar.rear_fov_deg + 1e-6
    )
    for angle in paper_angles:
        endpoint = _polar_pixel(
            origin,
            config.runtime.display_range_mm,
            angle,
            scale,
        )
        if angle == 0.0:
            color = (0, 230, 255)
            thickness = 2
        elif abs(angle) == 90.0:
            color = (0, 190, 0)
            thickness = 2
        elif abs(angle) == 110.0:
            color = (255, 170, 45)
            thickness = 2
        else:
            color = (75, 75, 75)
            thickness = 1
        cv2.line(canvas, origin, endpoint, color, thickness)

    for angle in paper_angles:
        label_point = _polar_pixel(
            origin,
            config.runtime.display_range_mm * 0.94,
            angle,
            scale,
        )
        if angle == 0.0:
            text = "0 deg / REAR"
            color = (0, 230, 255)
            thickness = 2
        elif angle == -90.0:
            text = "-90 deg / LEFT"
            color = (0, 190, 0)
            thickness = 2
        elif angle == 90.0:
            text = "+90 deg / RIGHT"
            color = (0, 190, 0)
            thickness = 2
        elif abs(angle) == 110.0:
            text = "%+d deg / FOV EDGE" % int(angle)
            color = (255, 170, 45)
            thickness = 2
        else:
            text = "%+d deg" % int(angle)
            color = (165, 165, 165)
            thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            thickness,
        )
        label_x = max(
            8,
            min(
                1092 - text_width,
                label_point[0] - text_width // 2,
            ),
        )
        label_y = max(
            text_height + 6,
            min(
                height - 48,
                label_point[1] + text_height // 2,
            ),
        )
        cv2.putText(
            canvas,
            text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "FIG.3 ANGLES: 0, +/-30, +/-45, +/-60, +/-90, +/-110 deg",
        (35, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    c_points = []
    d_points = []
    for point in observation.points:
        in_c = (
            -config.lidar.side_angle_max_deg
            <= point.angle_deg
            <= -config.lidar.side_angle_min_deg
            and point.distance_mm
            < config.lidar.side_distance_limit_mm
        )
        in_d = (
            config.lidar.side_angle_min_deg
            <= point.angle_deg
            <= config.lidar.side_angle_max_deg
            and point.distance_mm
            < config.lidar.side_distance_limit_mm
        )
        if in_c:
            c_points.append(point)
        if in_d:
            d_points.append(point)
        if point.distance_mm < config.lidar.near_distance_mm:
            point_color = (220, 0, 220)
            point_radius = 4
        elif in_c:
            point_color = (255, 125, 0)
            point_radius = 3
        elif in_d:
            point_color = (0, 170, 255)
            point_radius = 3
        else:
            point_color = (185, 185, 185)
            point_radius = 3
        cv2.circle(
            canvas,
            pixel(point.x_right_mm, point.y_back_mm),
            point_radius,
            point_color,
            -1,
        )

    for label, candidates, color in (
        ("C", c_points, (255, 125, 0)),
        ("D", d_points, (0, 170, 255)),
    ):
        if candidates:
            closest = min(
                candidates,
                key=lambda item: item.distance_mm,
            )
            location = pixel(
                closest.x_right_mm,
                closest.y_back_mm,
            )
            cv2.circle(canvas, location, 9, color, 2)
            cv2.putText(
                canvas,
                label,
                (location[0] + 8, location[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    if observation.slot_heading_deg is not None:
        heading_endpoint = _polar_pixel(
            origin,
            min(1200.0, config.runtime.display_range_mm * 0.5),
            observation.slot_heading_deg,
            scale,
        )
        cv2.line(
            canvas,
            origin,
            heading_endpoint,
            (255, 80, 255),
            4,
        )
        cv2.putText(
            canvas,
            "SLOT HEADING %+.1f deg"
            % observation.slot_heading_deg,
            (heading_endpoint[0] + 8, heading_endpoint[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 80, 255),
            2,
            cv2.LINE_AA,
        )

    pair = observation.pair
    if pair.valid:
        point_a = _polar_pixel(
            origin,
            pair.dist_a_mm,
            pair.angle_a_deg,
            scale,
        )
        point_b = _polar_pixel(
            origin,
            pair.dist_b_mm,
            pair.angle_b_deg,
            scale,
        )
        bisector_point = _polar_pixel(
            origin,
            min(
                config.runtime.display_range_mm,
                max(pair.dist_a_mm, pair.dist_b_mm) + 400.0,
            ),
            pair.angle_bisector_deg,
            scale,
        )
        cv2.line(canvas, origin, point_a, (40, 40, 255), 4)
        cv2.line(canvas, origin, point_b, (0, 220, 255), 4)
        cv2.line(
            canvas,
            origin,
            bisector_point,
            (255, 255, 0),
            3,
        )
        cv2.putText(
            canvas,
            "A",
            point_a,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 255),
            2,
        )
        cv2.putText(
            canvas,
            "B",
            point_b,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
        )

    control_pair = controller.control_pair
    gap_width: Optional[float] = None
    gap_center_angle: Optional[float] = None
    if control_pair.valid:
        gap_width, gap_center_angle = _pair_geometry(
            control_pair
        )
        control_a = _polar_pixel(
            origin,
            control_pair.dist_a_mm,
            control_pair.angle_a_deg,
            scale,
        )
        control_b = _polar_pixel(
            origin,
            control_pair.dist_b_mm,
            control_pair.angle_b_deg,
            scale,
        )
        entrance_center = (
            int(round((control_a[0] + control_b[0]) / 2.0)),
            int(round((control_a[1] + control_b[1]) / 2.0)),
        )
        cv2.line(
            canvas,
            control_a,
            control_b,
            (0, 255, 90),
            4,
        )
        cv2.line(
            canvas,
            origin,
            entrance_center,
            (0, 255, 90),
            2,
        )
        cv2.circle(
            canvas,
            entrance_center,
            13,
            (0, 255, 90),
            3,
        )
        cv2.putText(
            canvas,
            "CONTROL GAP %.0fmm @ %.1fdeg"
            % (gap_width, gap_center_angle),
            (
                entrance_center[0] + 12,
                entrance_center[1] - 12,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 90),
            2,
            cv2.LINE_AA,
        )

    cv2.line(
        canvas,
        (1100, 0),
        (1100, height),
        (85, 85, 85),
        2,
    )
    cv2.putText(
        canvas,
        "RAW=WHITE  C=BLUE  D=ORANGE  <600mm=MAGENTA",
        (35, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )

    debug = controller.debug
    text_x = 1125
    min_distance = (
        min(point.distance_mm for point in observation.points)
        if observation.points
        else None
    )
    wall_now = time.time()
    wall_clock = "%s.%03d" % (
        time.strftime("%H:%M:%S", time.localtime(wall_now)),
        int((wall_now % 1.0) * 1000.0),
    )
    scan_age_ms = (
        max(0.0, (wall_now - observation.timestamp) * 1000.0)
        if observation.timestamp > 0.0
        else None
    )
    lines = [
        (
            "MODE: STRAIGHT ONLY - LiDAR observation"
            if straight_only
            else "PAPER ONLY: Rear 2D LiDAR ICCE-Asia 2023"
        ),
        "TIME   %s" % wall_clock,
        "STATE  %s" % controller.state.value,
        "MOTOR  %s" % (
            "ENABLED" if motor_enabled else "DISABLED"
        ),
        "CMD    speed=%+d steer=%+d offset=%+d"
        % (
            command.speed,
            command.steering,
            config.controller.actuator_steering_offset,
        ),
        "WHY    %s" % command.reason,
        "OBS    %s" % observation.reason,
        "scanAge=%sms raw=%d FOV=%d (L=%d R=%d)"
        % (
            _fmt(scan_age_ms, 0),
            observation.raw_point_count,
            len(observation.points),
            observation.left_fov_point_count,
            observation.right_fov_point_count,
        ),
        "minRange=%smm prealign(<%.0f)=%s near(<%.0f)=%s"
        % (
            _fmt(min_distance, 0),
            config.lidar.prealign_distance_mm,
            observation.prealign_near,
            config.lidar.near_distance_mm,
            observation.near,
        ),
        "countedCars=%d vehicleRight=%s raw=%s track=%d"
        % (
            debug.detected_vehicle_count,
            observation.right_vehicle_present,
            observation.right_vehicle_raw,
            observation.right_vehicle_track_id,
        ),
        "rightCluster points=%d seen=%d missing=%d minY=%smm"
        % (
            observation.right_vehicle_cluster_points,
            observation.right_vehicle_seen_scans,
            observation.right_vehicle_missing_scans,
            _fmt(observation.right_vehicle_min_y_back_mm, 0),
        ),
        "pair=%s fused=%d clusters=%d candidates=%d (%s)"
        % (
            pair.valid,
            observation.pair_source_scans,
            observation.tangent_cluster_count,
            observation.tangent_candidate_count,
            pair.reason,
        ),
        "A=%sdeg/%smm  B=%sdeg/%smm"
        % (
            _fmt(
                pair.angle_a_deg if pair.valid else None,
                1,
            ),
            _fmt(
                pair.dist_a_mm if pair.valid else None,
                0,
            ),
            _fmt(
                pair.angle_b_deg if pair.valid else None,
                1,
            ),
            _fmt(
                pair.dist_b_mm if pair.valid else None,
                0,
            ),
        ),
        "bisector=%sdeg biasAB=%smm"
        % (
            _fmt(
                pair.angle_bisector_deg
                if pair.valid
                else None,
                1,
            ),
            _fmt(debug.distance_bias_ab_mm, 0),
        ),
        "controlA=%s/%s controlB=%s/%s controlBis=%s"
        % (
            _fmt(
                (
                    control_pair.angle_a_deg
                    if control_pair.valid
                    else None
                ),
                1,
            ),
            _fmt(
                (
                    control_pair.dist_a_mm
                    if control_pair.valid
                    else None
                ),
                0,
            ),
            _fmt(
                (
                    control_pair.angle_b_deg
                    if control_pair.valid
                    else None
                ),
                1,
            ),
            _fmt(
                (
                    control_pair.dist_b_mm
                    if control_pair.valid
                    else None
                ),
                0,
            ),
            _fmt(
                (
                    control_pair.angle_bisector_deg
                    if control_pair.valid
                    else None
                ),
                1,
            ),
        ),
        "controlGap=%smm center=%sdeg source=%s"
        % (
            _fmt(gap_width, 0),
            _fmt(gap_center_angle, 1),
            control_pair.reason,
        ),
        "Eq5=%s entrySteer=%s entryCenter=%sdeg"
        % (
            _fmt(debug.paper_steering, 2),
            _fmt(debug.applied_paper_steering, 2),
            _fmt(debug.entry_center_angle_deg, 1),
        ),
        "ENTRY start<=%.1fdeg fullSteer@%.1fdeg"
        % (
            config.controller.entry_start_angle_deg,
            config.controller.entry_full_steer_angle_deg,
        ),
        "dropoutRealign=%s" % debug.realigning_after_dropout,
        "setupNearSeen=%s readyScans=%d pairHold=%d "
        "reacquire=%s centerScans=%d"
        % (
            debug.prealign_near_seen,
            debug.prealign_ready_scans,
            debug.pair_hold_scans,
            debug.pair_reacquiring,
            debug.center_observation_scans,
        ),
        "C=%s@y%s D=%s@y%s ratio=%s err=%s span=%s stable=%d/%d"
        % (
            _fmt(observation.dist_c_mm, 0),
            _fmt(observation.c_y_back_mm, 0),
            _fmt(observation.dist_d_mm, 0),
            _fmt(observation.d_y_back_mm, 0),
            _fmt(debug.cd_balance_ratio, 3),
            _fmt(debug.cd_balance_error, 3),
            _fmt(debug.cd_balance_span, 3),
            debug.cd_center_ready_scans,
            config.controller.cd_center_confirm_scans,
        ),
        "SIDE LINE L=%sdeg(%dpt) R=%sdeg(%dpt)"
        % (
            _fmt(
                (
                    observation.left_side_line.heading_deg
                    if observation.left_side_line.valid
                    else None
                ),
                1,
            ),
            observation.left_side_line.point_count,
            _fmt(
                (
                    observation.right_side_line.heading_deg
                    if observation.right_side_line.valid
                    else None
                ),
                1,
            ),
            observation.right_side_line.point_count,
        ),
        "PARALLEL obs=%s filtered=%s best=%s span=%s "
        "ready=%d/%d cycle=%d"
        % (
            _fmt(observation.slot_heading_deg, 1),
            _fmt(debug.slot_heading_deg, 1),
            _fmt(debug.parallel_best_heading_deg, 1),
            _fmt(debug.slot_heading_span_deg, 1),
            debug.parallel_heading_ready_scans,
            config.controller.parallel_heading_confirm_scans,
            debug.parallel_correction_cycles,
        ),
        "PARALLEL EXIT cdMissing=%d/%d noImprove=%d/%d"
        % (
            debug.parallel_cd_missing_scans,
            config.controller.parallel_cd_missing_exit_scans,
            debug.parallel_no_improvement_scans,
            config.controller.parallel_no_improvement_exit_scans,
        ),
        "FOV=+/-%.0fdeg C/D limit=<%.0fmm"
        % (
            config.lidar.rear_fov_deg,
            config.lidar.side_distance_limit_mm,
        ),
        "LiDAR=%s buffer=%dB | A/B <%.0fmm fusion=%d gap>=%.0fdeg"
        % (
            config.lidar.scan_type,
            config.lidar.input_buffer_limit_bytes,
            config.lidar.tangent_distance_limit_mm,
            config.lidar.tangent_fusion_scans,
            config.lidar.tangent_min_angular_gap_deg,
        ),
        "display=%.0fmm (display only)"
        % config.runtime.display_range_mm,
        "SPACE start/pause | R reset | Q quit",
    ]
    if lidar_error is not None:
        lines.insert(
            5,
            "LIDAR ERROR %s" % str(lidar_error)[:50],
        )
    for index, text in enumerate(lines):
        color = (
            (0, 230, 255)
            if index < 5
            else (220, 220, 220)
        )
        cv2.putText(
            canvas,
            text,
            (text_x, 28 + 30 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )

    if debug_writer is not None:
        debug_writer.write(canvas)
    cv2.imshow("SKKU Paper-Only Rear LiDAR Parking", canvas)
    return cv2.waitKey(1) & 0xFF


def _polar_pixel(origin, distance_mm, angle_deg, scale):
    angle = radians(angle_deg)
    x_right = sin(angle) * distance_mm
    y_back = cos(angle) * distance_mm
    return (
        int(round(origin[0] + x_right * scale)),
        int(round(origin[1] + y_back * scale)),
    )


def _pair_geometry(
    pair: TangentPair,
) -> tuple[Optional[float], Optional[float]]:
    if not pair.valid:
        return None, None

    angle_a = radians(pair.angle_a_deg)
    angle_b = radians(pair.angle_b_deg)
    a_x = sin(angle_a) * pair.dist_a_mm
    a_y = cos(angle_a) * pair.dist_a_mm
    b_x = sin(angle_b) * pair.dist_b_mm
    b_y = cos(angle_b) * pair.dist_b_mm
    center_x = (a_x + b_x) / 2.0
    center_y = (a_y + b_y) / 2.0
    return (
        hypot(a_x - b_x, a_y - b_y),
        degrees(atan2(center_x, center_y)),
    )


def _optional(value: Optional[float]) -> str:
    return "" if value is None else "%.6f" % value


def _fmt(value: Optional[float], digits: int) -> str:
    if value is None:
        return "None"
    return ("%." + str(digits) + "f") % value


def _config_with_overrides(
    config: AppConfig,
    args: argparse.Namespace,
) -> AppConfig:
    serial = (
        replace(config.serial, port=args.serial_port)
        if args.serial_port
        else config.serial
    )
    lidar = (
        replace(config.lidar, port=args.lidar_port)
        if args.lidar_port
        else config.lidar
    )
    controller = config.controller
    if args.entry_start_angle is not None:
        if not 0.0 < args.entry_start_angle <= 90.0:
            raise ValueError(
                "--entry-start-angle must be in (0, 90]"
            )
        controller = replace(
            controller,
            entry_start_angle_deg=args.entry_start_angle,
        )
    return replace(
        config,
        serial=serial,
        lidar=lidar,
        controller=controller,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-only rear 2D LiDAR reverse parking",
    )
    parser.add_argument(
        "--config",
        default="configs/parking.json",
    )
    parser.add_argument("--serial-port")
    parser.add_argument("--lidar-port")
    parser.add_argument("--lidar-csv")
    parser.add_argument("--record-dir")
    parser.add_argument(
        "--entry-start-angle",
        type=float,
        help=(
            "A/B gap-center bearing that ends left-forward setup "
            "and starts reverse entry"
        ),
    )
    parser.add_argument(
        "--steering_plus",
        choices=("on", "off"),
        default="off",
        help="enable signed C/D recovery steering",
    )
    parser.add_argument("--no-motor", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument(
        "--straight-only",
        action="store_true",
        help=(
            "SPACE toggles fixed straight driving; LiDAR cannot change "
            "speed or steering"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return run(args)
