from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
import math
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..estimation.parking_geometry import ParkingGeometry, ParkingGeometryEstimator
from ..estimation.parking_lidar import (
    LidarParkingObservation,
    LidarParkingSpaceEstimator,
)
from ..parking_config import ParkingAppConfig, load_parking_config
from ..perception.bev import BevTransformer
from ..perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from ..planning.t_parking_planner import ParkingPlan, ParkingState, TParkingPlanner
from ..sensors.lidar import LidarScan, load_lidar_csv
from .parking_app import (
    compose_parking_dashboard,
    draw_lidar_debug,
    draw_parking_line,
    draw_reverse_path,
    extract_recording_zip,
    parking_mask_color,
)


LOG = logging.getLogger("skku_autocar.arduino_parking_replay")
ROOT = Path(__file__).resolve().parents[3]


class ReplayParkingState(str, Enum):
    SEARCHING = "SEARCHING"
    POSITIONING = "POSITIONING"
    REVERSING = "REVERSING"
    FINISHED = "FINISHED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    ABORTED = "ABORTED"


class ReplayPositioningPhase(str, Enum):
    TRACK_FIRST_CAR = "TRACK_FIRST_CAR"
    ALIGN_REAR_AXLE = "ALIGN_REAR_AXLE"
    PREALIGN_LEFT = "PREALIGN_LEFT"


@dataclass(frozen=True)
class ArduinoReplayConstants:
    search_speed: int = 28
    position_speed: int = 16
    reverse_speed: int = -22
    gap_confirm_cycles: int = 3
    gap_center_max_jump_cm: float = 40.0
    gap_confirm_filter_alpha: float = 0.40
    position_target_cm: float = 18.0
    position_tolerance_cm: float = 10.0
    position_confirm_cycles: int = 5
    position_filter_alpha: float = 0.45
    prealign_enabled: bool = True
    prealign_speed: int = 14
    prealign_steer_settle_s: float = 0.40
    prealign_timeout_s: float = 6.0
    prealign_slot_heading_tolerance_deg: float = 18.0
    prealign_entry_bearing_tolerance_deg: float = 25.0
    prealign_target_distance_min_cm: float = 25.0
    prealign_target_distance_max_cm: float = 220.0
    prealign_confirm_cycles: int = 3
    prealign_heading_overshoot_deg: float = 25.0
    complete_rear_cm: float = 20.0
    complete_confirm_cycles: int = 3
    emergency_cm: float = 10.0
    camera_kp_deg_per_pixel: float = 0.12
    ultrasonic_kp_deg_per_cm: float = 0.70
    steering_filter_alpha: float = 0.35
    max_camera_error_px: float = 250.0
    max_side_distance_cm: float = 250.0
    max_steer_deg: float = 45.0
    steer_deadband_deg: float = 1.0


@dataclass(frozen=True)
class CameraSample:
    elapsed_s: float
    line_error_px: Optional[float]
    found: bool
    reason: str
    inference_ms: float
    detected_line_masks: int = 0
    fitted_line_count: int = 0
    geometry: Optional[ParkingGeometry] = None


@dataclass(frozen=True)
class ReplayCommand:
    state: ReplayParkingState
    speed: int
    steering_deg: int
    event: str = ""


class ArduinoParkingControllerReplay:
    """Legacy standalone-Arduino controller kept for firmware unit tests.

    The actual replay command path below uses ``SharedParkingPlannerReplay`` so
    that video replay and ``parking.py`` execute the exact same Python planner.
    """

    def __init__(self, constants: ArduinoReplayConstants = ArduinoReplayConstants()):
        self.constants = constants
        self.state = ReplayParkingState.SEARCHING
        self.gap_confirm_count = 0
        self.position_confirm_count = 0
        self.finish_confirm_count = 0
        self.tracked_gap_center_y_back_cm = 0.0
        self.filtered_steering_deg = 0.0
        self.positioning_phase = ReplayPositioningPhase.ALIGN_REAR_AXLE
        self.phase_started_s = 0.0
        self.prealign_confirm_count = 0
        self._clock_s = 0.0

    def update(
        self,
        lidar: LidarParkingObservation,
        rear_lidar_cm: Optional[float],
        line_error_px: Optional[float],
        left_ultrasonic_cm: Optional[float] = None,
        right_ultrasonic_cm: Optional[float] = None,
        elapsed_s: Optional[float] = None,
    ) -> ReplayCommand:
        self._clock_s = (
            float(elapsed_s)
            if elapsed_s is not None
            else self._clock_s + 0.05
        )
        if self._is_emergency(left_ultrasonic_cm) or self._is_emergency(
            right_ultrasonic_cm
        ) or self._is_emergency(rear_lidar_cm):
            self.state = ReplayParkingState.EMERGENCY_STOP
            return ReplayCommand(self.state, 0, 0, "distance<=10cm")

        if self.state in (
            ReplayParkingState.FINISHED,
            ReplayParkingState.EMERGENCY_STOP,
        ):
            return ReplayCommand(self.state, 0, 0)

        if self.state == ReplayParkingState.SEARCHING:
            if self._confirm_gap(lidar):
                self.state = ReplayParkingState.POSITIONING
                self.positioning_phase = ReplayPositioningPhase.ALIGN_REAR_AXLE
                self.gap_confirm_count = 0
                return ReplayCommand(self.state, 0, 0, "gap_confirmed")
            return ReplayCommand(self.state, self.constants.search_speed, 0)

        if self.state == ReplayParkingState.POSITIONING:
            if self.positioning_phase == ReplayPositioningPhase.PREALIGN_LEFT:
                return self._update_prealign(lidar)
            center_cm = self._raw_gap_center_cm(lidar)
            if center_cm is None or lidar.coasted:
                self.position_confirm_count = 0
                return ReplayCommand(self.state, 0, 0, "gap_lost")

            alpha = self.constants.position_filter_alpha
            self.tracked_gap_center_y_back_cm = (
                alpha * center_cm
                + (1.0 - alpha) * self.tracked_gap_center_y_back_cm
            )
            error_cm = (
                self.tracked_gap_center_y_back_cm
                - self.constants.position_target_cm
            )
            if abs(error_cm) <= self.constants.position_tolerance_cm:
                self.position_confirm_count += 1
                if self.position_confirm_count >= self.constants.position_confirm_cycles:
                    if self.constants.prealign_enabled:
                        self.positioning_phase = ReplayPositioningPhase.PREALIGN_LEFT
                        self.phase_started_s = self._clock_s
                        self.prealign_confirm_count = 0
                        return ReplayCommand(
                            self.state,
                            0,
                            -int(round(self.constants.max_steer_deg)),
                            "entry_aligned:settling_max_left",
                        )
                    self.state = ReplayParkingState.REVERSING
                    return ReplayCommand(self.state, 0, 0, "entry_aligned")
                return ReplayCommand(self.state, 0, 0, "entry_confirming")

            self.position_confirm_count = 0
            # Negative yBack means the detected gap center is ahead.
            direction = 1 if error_cm < 0.0 else -1
            return ReplayCommand(
                self.state,
                direction * self.constants.position_speed,
                0,
            )

        if self._valid_distance(rear_lidar_cm) and rear_lidar_cm <= self.constants.complete_rear_cm:
            self.finish_confirm_count += 1
        else:
            self.finish_confirm_count = 0
        if self.finish_confirm_count >= self.constants.complete_confirm_cycles:
            self.state = ReplayParkingState.FINISHED
            return ReplayCommand(self.state, 0, 0, "rear_clearance_reached")

        steering = self._camera_correction(line_error_px)
        if self._usable_side(left_ultrasonic_cm) and self._usable_side(
            right_ultrasonic_cm
        ):
            steering += self.constants.ultrasonic_kp_deg_per_cm * (
                right_ultrasonic_cm - left_ultrasonic_cm
            )
        steering = max(
            -self.constants.max_steer_deg,
            min(self.constants.max_steer_deg, steering),
        )
        alpha = self.constants.steering_filter_alpha
        self.filtered_steering_deg = (
            alpha * steering + (1.0 - alpha) * self.filtered_steering_deg
        )
        if abs(self.filtered_steering_deg) < self.constants.steer_deadband_deg:
            self.filtered_steering_deg = 0.0
        return ReplayCommand(
            self.state,
            self.constants.reverse_speed,
            int(round(self.filtered_steering_deg)),
        )

    def _update_prealign(self, lidar: LidarParkingObservation) -> ReplayCommand:
        metrics = self._prealign_metrics(lidar)
        if metrics is None or lidar.coasted:
            self.prealign_confirm_count = 0
            return ReplayCommand(self.state, 0, 0, "prealign_waiting_for_slot")
        slot_heading, entry_bearing, distance_cm = metrics
        direct_ready = (
            abs(slot_heading)
            <= self.constants.prealign_slot_heading_tolerance_deg
            and abs(entry_bearing)
            <= self.constants.prealign_entry_bearing_tolerance_deg
            and self.constants.prealign_target_distance_min_cm
            <= distance_cm
            <= self.constants.prealign_target_distance_max_cm
        )
        self.prealign_confirm_count = (
            self.prealign_confirm_count + 1 if direct_ready else 0
        )
        if self.prealign_confirm_count >= max(
            1,
            self.constants.prealign_confirm_cycles,
        ):
            self.state = ReplayParkingState.REVERSING
            return ReplayCommand(self.state, 0, 0, "prealign_direct_reverse_ready")

        phase_elapsed = self._clock_s - self.phase_started_s
        overshot = slot_heading < -abs(
            self.constants.prealign_heading_overshoot_deg
        )
        if overshot or phase_elapsed >= self.constants.prealign_timeout_s:
            self.state = ReplayParkingState.REVERSING
            return ReplayCommand(
                self.state,
                0,
                0,
                "prealign_fallback:%s"
                % ("heading_overshoot" if overshot else "timeout"),
            )

        event = "prealign_left head=%+.1f bearing=%+.1f dist=%.0fcm" % (
            slot_heading,
            entry_bearing,
            distance_cm,
        )
        steering = -int(round(self.constants.max_steer_deg))
        if phase_elapsed < self.constants.prealign_steer_settle_s:
            return ReplayCommand(self.state, 0, steering, "steering_settle:" + event)
        return ReplayCommand(
            self.state,
            self.constants.prealign_speed,
            steering,
            event,
        )

    def _prealign_metrics(
        self,
        lidar: LidarParkingObservation,
    ) -> Optional[Tuple[float, float, float]]:
        if (
            not lidar.gap_confirmed
            or lidar.gap_center_x_right_mm is None
            or lidar.gap_center_y_back_mm is None
            or lidar.slot_depth_x_right is None
            or lidar.slot_depth_y_back is None
        ):
            return None
        rear_axle_y_mm = (
            lidar.entry_target_y_back_mm
            if lidar.entry_target_y_back_mm is not None
            else self.constants.position_target_cm * 10.0
        )
        target_x = lidar.gap_center_x_right_mm
        target_y = lidar.gap_center_y_back_mm - rear_axle_y_mm
        distance_mm = math.hypot(target_x, target_y)
        depth_length = math.hypot(
            lidar.slot_depth_x_right,
            lidar.slot_depth_y_back,
        )
        if distance_mm <= 1e-6 or depth_length <= 1e-6:
            return None
        slot_heading = math.degrees(
            math.atan2(
                lidar.slot_depth_x_right / depth_length,
                lidar.slot_depth_y_back / depth_length,
            )
        )
        entry_bearing = math.degrees(math.atan2(target_x, target_y))
        return slot_heading, entry_bearing, distance_mm / 10.0

    def _camera_correction(self, line_error_px: Optional[float]) -> float:
        if line_error_px is None or not math.isfinite(line_error_px):
            return 0.0
        error = max(
            -self.constants.max_camera_error_px,
            min(self.constants.max_camera_error_px, line_error_px),
        )
        return self.constants.camera_kp_deg_per_pixel * error

    def _confirm_gap(self, lidar: LidarParkingObservation) -> bool:
        center_cm = self._raw_gap_center_cm(lidar)
        if center_cm is None or lidar.coasted:
            self.gap_confirm_count = 0
            return False
        if (
            self.gap_confirm_count > 0
            and abs(center_cm - self.tracked_gap_center_y_back_cm)
            > self.constants.gap_center_max_jump_cm
        ):
            self.gap_confirm_count = 1
            self.tracked_gap_center_y_back_cm = center_cm
        else:
            alpha = self.constants.gap_confirm_filter_alpha
            self.tracked_gap_center_y_back_cm = (
                alpha * center_cm
                + (1.0 - alpha) * self.tracked_gap_center_y_back_cm
            )
            self.gap_confirm_count += 1
        return self.gap_confirm_count >= self.constants.gap_confirm_cycles

    @staticmethod
    def _raw_gap_center_cm(lidar: LidarParkingObservation) -> Optional[float]:
        if not lidar.gap_found:
            return None
        if (
            lidar.gap_near_edge_y_back_mm is not None
            and lidar.gap_far_edge_y_back_mm is not None
        ):
            return (
                lidar.gap_near_edge_y_back_mm
                + lidar.gap_far_edge_y_back_mm
            ) / 20.0
        if lidar.gap_center_y_back_mm is not None:
            return lidar.gap_center_y_back_mm / 10.0
        return None

    def _is_emergency(self, distance_cm: Optional[float]) -> bool:
        return self._valid_distance(distance_cm) and distance_cm <= self.constants.emergency_cm

    def _usable_side(self, distance_cm: Optional[float]) -> bool:
        return self._valid_distance(distance_cm) and distance_cm <= self.constants.max_side_distance_cm

    @staticmethod
    def _valid_distance(distance_cm: Optional[float]) -> bool:
        return distance_cm is not None and math.isfinite(distance_cm) and distance_cm >= 0.0


class SharedParkingPlannerReplay:
    """Thin replay adapter around the exact planner used by ``parking.py``."""

    def __init__(self, config: ParkingAppConfig):
        self.planner = TParkingPlanner(config.planner, config.path)
        self.planner.start(0.0)
        self.state = ReplayParkingState.SEARCHING
        self.positioning_phase = ReplayPositioningPhase.TRACK_FIRST_CAR
        self.prealign_confirm_count = 0
        self.tracked_gap_center_y_back_cm = 0.0
        self.gap_confirm_count = 0
        self.position_confirm_count = 0
        self.finish_confirm_count = 0
        self.last_plan: Optional[ParkingPlan] = None

    @property
    def planner_state(self) -> ParkingState:
        return self.planner.state

    def update(
        self,
        lidar: LidarParkingObservation,
        geometry: ParkingGeometry,
        elapsed_s: float,
        left_ultrasonic_cm: Optional[float] = None,
        right_ultrasonic_cm: Optional[float] = None,
    ) -> ReplayCommand:
        plan = self.planner.update(
            geometry,
            lidar,
            elapsed_s,
            enabled=True,
            left_ultrasonic_mm=(
                None
                if left_ultrasonic_cm is None
                else left_ultrasonic_cm * 10.0
            ),
            right_ultrasonic_mm=(
                None
                if right_ultrasonic_cm is None
                else right_ultrasonic_cm * 10.0
            ),
        )
        self.last_plan = plan
        self.state = replay_state_for_planner(plan.state)
        if plan.state == ParkingState.TRACK_GAP:
            self.positioning_phase = ReplayPositioningPhase.TRACK_FIRST_CAR
        elif plan.state == ParkingState.PREALIGN_LEFT:
            self.positioning_phase = ReplayPositioningPhase.PREALIGN_LEFT
        elif plan.state == ParkingState.POSITION_REAR_AXLE:
            self.positioning_phase = ReplayPositioningPhase.ALIGN_REAR_AXLE
        self.prealign_confirm_count = self.planner.prealign_confirmed_frames
        if lidar.gap_center_y_back_mm is not None:
            self.tracked_gap_center_y_back_cm = lidar.gap_center_y_back_mm / 10.0
        return ReplayCommand(
            self.state,
            plan.command.speed,
            plan.command.steering,
            plan.reason,
        )


def replay_state_for_planner(state: ParkingState) -> ReplayParkingState:
    if state in (ParkingState.SEARCH_CARS, ParkingState.TRACK_GAP):
        return ReplayParkingState.SEARCHING
    if state in (
        ParkingState.POSITION_REAR_AXLE,
        ParkingState.PREALIGN_LEFT,
        ParkingState.EXIT_RIGHT,
        ParkingState.EXIT_STRAIGHT,
    ):
        return ReplayParkingState.POSITIONING
    if state in (
        ParkingState.VERIFY_PARKING_LINES,
        ParkingState.PLAN_REVERSE_PATH,
        ParkingState.FOLLOW_ENTRY_CURVE,
        ParkingState.FOLLOW_SLOT_CENTER,
    ):
        return ReplayParkingState.REVERSING
    if state in (ParkingState.PARKED, ParkingState.EXIT_DONE):
        return ReplayParkingState.FINISHED
    if state == ParkingState.EMERGENCY_STOP:
        return ReplayParkingState.EMERGENCY_STOP
    return ReplayParkingState.ABORTED


def run_replay(args: argparse.Namespace) -> Dict[str, object]:
    config = load_parking_config(str(resolve_path(args.config)))
    config = replace(
        config,
        lidar=replace(
            config.lidar,
            entry_tolerance_mm=args.position_tolerance_cm * 10.0,
            first_car_turn_target_y_back_mm=(
                config.lidar.first_car_turn_target_y_back_mm
                if args.first_car_turn_target_cm is None
                else args.first_car_turn_target_cm * 10.0
            ),
        ),
        planner=replace(
            config.planner,
            prealign_speed=(
                config.planner.prealign_speed
                if args.prealign_speed is None
                else args.prealign_speed
            ),
            prealign_steering=(
                config.planner.prealign_steering
                if args.prealign_steering is None
                else args.prealign_steering
            ),
            prealign_timeout_s=(
                config.planner.prealign_timeout_s
                if args.prealign_timeout_s is None
                else args.prealign_timeout_s
            ),
        ),
    )
    rear_axle_offset_cm = (
        args.lidar_to_rear_axle_cm
        if args.lidar_to_rear_axle_cm is not None
        else config.lidar.sensor_to_rear_axle_y_back_mm / 10.0
    )
    if args.lidar_to_rear_axle_cm is not None:
        config = replace(
            config,
            lidar=replace(
                config.lidar,
                sensor_to_rear_axle_y_back_mm=rear_axle_offset_cm * 10.0,
            ),
        )
    if args.lidar_behind_vehicle_rear_cm is not None:
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                lidar_debug_sensor_behind_vehicle_rear_mm=(
                    args.lidar_behind_vehicle_rear_cm * 10.0
                ),
            ),
        )
    constants = ArduinoReplayConstants(
        position_target_cm=rear_axle_offset_cm,
        position_tolerance_cm=args.position_tolerance_cm,
        prealign_enabled=config.planner.prealign_enabled,
        prealign_speed=config.planner.prealign_speed,
        prealign_steer_settle_s=config.planner.prealign_steer_settle_s,
        prealign_timeout_s=config.planner.prealign_timeout_s,
        prealign_slot_heading_tolerance_deg=(
            config.planner.prealign_slot_heading_tolerance_deg
        ),
        prealign_entry_bearing_tolerance_deg=(
            config.planner.prealign_entry_bearing_tolerance_deg
        ),
        prealign_target_distance_min_cm=(
            config.planner.prealign_target_distance_min_mm / 10.0
        ),
        prealign_target_distance_max_cm=(
            config.planner.prealign_target_distance_max_mm / 10.0
        ),
        prealign_confirm_cycles=config.planner.prealign_confirm_frames,
        prealign_heading_overshoot_deg=(
            config.planner.prealign_heading_overshoot_deg
        ),
    )
    output_path = resolve_path(args.output)
    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="skku_arduino_replay_") as directory:
        video_path, lidar_path = extract_recording_zip(
            Path(args.recording_zip).expanduser().resolve(),
            Path(directory),
        )
        scans = load_lidar_csv(str(lidar_path))
        if not scans:
            raise ValueError("LiDAR CSV contains no scans")
        if args.show or args.save_video:
            rows, summary = run_visual_replay(
                video_path,
                scans,
                config,
                constants,
                args,
            )
        else:
            camera_samples = [] if args.no_camera else analyze_camera(
                video_path,
                config,
                args,
            )
            rows, summary = simulate_scans(scans, camera_samples, config, constants)
        summary.update(
            {
                "recording_zip": str(Path(args.recording_zip).expanduser().resolve()),
                "video": video_path.name,
                "lidar_csv": lidar_path.name,
                "output_csv": str(output_path),
                "camera_enabled": not args.no_camera,
                "camera_interval_s": args.camera_interval_s,
                "position_tolerance_cm": args.position_tolerance_cm,
                "lidar_to_rear_axle_cm": rear_axle_offset_cm,
                "planner_implementation": "TParkingPlanner(shared_with_parking.py)",
                "first_car_turn_target_cm": (
                    config.lidar.first_car_turn_target_y_back_mm / 10.0
                ),
                "prealign_speed": config.planner.prealign_speed,
                "prealign_steering": config.planner.prealign_steering,
                "prealign_timeout_s": config.planner.prealign_timeout_s,
                "visual_replay": bool(args.show or args.save_video),
                "saved_video": str(resolve_path(args.save_video)) if args.save_video else None,
                "limitations": [
                    "The ZIP has no LEFT/RIGHT ultrasonic samples; side-centering correction and side emergency stop are not replayed.",
                    "Virtual drive/steer commands cannot alter the prerecorded vehicle trajectory.",
                    "Rear clearance is approximated by the nearest valid LiDAR return around vehicle rear 180 degrees.",
                ],
            }
        )

    write_rows(output_path, rows)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


def simulate_scans(
    scans: Sequence[LidarScan],
    camera_samples: Sequence[CameraSample],
    config: ParkingAppConfig,
    constants: ArduinoReplayConstants,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    lidar_estimator = LidarParkingSpaceEstimator(config.lidar)
    controller = SharedParkingPlannerReplay(config)
    start = scans[0].timestamp
    rows: List[Dict[str, object]] = []
    transitions = []
    previous_state = controller.state
    previous_planner_state = controller.planner_state
    planner_transitions = []
    gap_confirmed_scans = 0
    gap_hold_scans = 0
    slot_depth_flip_count = 0
    minimum_slot_depth_dot: Optional[float] = None
    previous_slot_depth: Optional[Tuple[float, float]] = None
    closest_gap_center_cm: Optional[float] = None
    minimum_rear_cm: Optional[float] = None

    for scan in scans:
        elapsed = scan.timestamp - start
        lidar = lidar_estimator.estimate(scan, now=scan.timestamp)
        if lidar.gap_confirmed:
            gap_confirmed_scans += 1
        if lidar.coasted:
            gap_hold_scans += 1
        if (
            lidar.slot_depth_x_right is not None
            and lidar.slot_depth_y_back is not None
        ):
            current_slot_depth = (
                lidar.slot_depth_x_right,
                lidar.slot_depth_y_back,
            )
            if previous_slot_depth is not None:
                depth_dot = (
                    previous_slot_depth[0] * current_slot_depth[0]
                    + previous_slot_depth[1] * current_slot_depth[1]
                )
                if depth_dot < 0.0:
                    slot_depth_flip_count += 1
                minimum_slot_depth_dot = (
                    depth_dot
                    if minimum_slot_depth_dot is None
                    else min(minimum_slot_depth_dot, depth_dot)
                )
            previous_slot_depth = current_slot_depth
        if lidar.gap_center_y_back_mm is not None:
            center_abs_cm = abs(lidar.gap_center_y_back_mm / 10.0)
            closest_gap_center_cm = (
                center_abs_cm
                if closest_gap_center_cm is None
                else min(closest_gap_center_cm, center_abs_cm)
            )
        rear_cm = rear_lidar_distance_cm(scan, config)
        if rear_cm is not None:
            minimum_rear_cm = rear_cm if minimum_rear_cm is None else min(minimum_rear_cm, rear_cm)
        camera = latest_camera_sample(camera_samples, elapsed)
        line_error = camera.line_error_px if camera is not None and camera.found else None
        geometry = (
            camera.geometry
            if camera is not None and camera.geometry is not None
            else ParkingGeometry(reason="camera_not_sampled")
        )
        command = controller.update(lidar, geometry, elapsed)
        if command.state != previous_state:
            transitions.append(
                {
                    "elapsed_s": round(elapsed, 3),
                    "from": previous_state.value,
                    "to": command.state.value,
                    "event": command.event,
                }
            )
            previous_state = command.state
        if controller.planner_state != previous_planner_state:
            planner_transitions.append(
                {
                    "elapsed_s": round(elapsed, 3),
                    "from": previous_planner_state.value,
                    "to": controller.planner_state.value,
                    "event": command.event,
                }
            )
            previous_planner_state = controller.planner_state

        rows.append(
            {
                "elapsed_s": round(elapsed, 6),
                "state": command.state.value,
                "planner_state": controller.planner_state.value,
                "drive_speed": command.speed,
                "steer_deg": command.steering_deg,
                "event": command.event,
                "gap_found": int(lidar.gap_found),
                "gap_confirmed": int(lidar.gap_confirmed),
                "car_clusters": lidar.car_count,
                "first_car_confirmed": int(lidar.first_car_confirmed),
                "first_car_edge_y_back_cm": optional_round(
                    lidar.first_car_slot_edge_y_back_mm,
                    10.0,
                ),
                "first_car_turn_error_cm": optional_round(
                    lidar.first_car_turn_error_mm,
                    10.0,
                ),
                "first_car_turn_reached": int(lidar.first_car_turn_reached),
                "gap_center_y_back_cm": optional_round(lidar.gap_center_y_back_mm, 10.0),
                "gap_width_cm": optional_round(lidar.gap_width_mm, 10.0),
                "slot_depth_x": optional_round(lidar.slot_depth_x_right),
                "slot_depth_y": optional_round(lidar.slot_depth_y_back),
                "gap_coasted": int(lidar.coasted),
                "positioning_phase": controller.positioning_phase.value,
                "prealign_confirm_count": controller.prealign_confirm_count,
                "tracked_gap_center_cm": round(controller.tracked_gap_center_y_back_cm, 3),
                "gap_confirm_count": controller.gap_confirm_count,
                "position_confirm_count": controller.position_confirm_count,
                "finish_confirm_count": controller.finish_confirm_count,
                "rear_lidar_cm": optional_round(rear_cm),
                "line_error_px": optional_round(line_error),
                "camera_found": int(bool(camera and camera.found)),
                "camera_reason": camera.reason if camera else "not_sampled",
                "detected_line_masks": camera.detected_line_masks if camera else 0,
                "fitted_line_count": camera.fitted_line_count if camera else 0,
                "left_ultrasonic_cm": "",
                "right_ultrasonic_cm": "",
            }
        )

    valid_camera = [sample for sample in camera_samples if sample.found]
    summary: Dict[str, object] = {
        "lidar_scans": len(scans),
        "duration_s": round(scans[-1].timestamp - start, 3),
        "gap_confirmed_scans": gap_confirmed_scans,
        "gap_hold_scans": gap_hold_scans,
        "slot_depth_flip_count": slot_depth_flip_count,
        "minimum_consecutive_slot_depth_dot": optional_round(
            minimum_slot_depth_dot
        ),
        "closest_gap_center_abs_cm": optional_round(closest_gap_center_cm),
        "minimum_rear_lidar_cm": optional_round(minimum_rear_cm),
        "camera_samples": len(camera_samples),
        "camera_valid_samples": len(valid_camera),
        "camera_valid_ratio": round(len(valid_camera) / len(camera_samples), 4) if camera_samples else None,
        "camera_samples_with_line_mask": sum(
            sample.detected_line_masks >= 1 for sample in camera_samples
        ),
        "camera_samples_with_two_masks": sum(
            sample.detected_line_masks >= 2 for sample in camera_samples
        ),
        "camera_geometry_reason_counts": dict(
            Counter(sample.reason for sample in camera_samples)
        ),
        "final_state": controller.state.value,
        "final_planner_state": controller.planner_state.value,
        "transitions": transitions,
        "planner_transitions": planner_transitions,
        "ultrasonic_replayed": False,
    }
    return rows, summary


def run_visual_replay(
    video_path: Path,
    scans: Sequence[LidarScan],
    config: ParkingAppConfig,
    constants: ArduinoReplayConstants,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run the Arduino-equivalent controller while rendering recorded sensors."""

    import cv2
    import numpy as np

    segmenter = None
    if not args.no_camera:
        model_path = resolve_path(args.model or config.yolo.model_path)
        segmenter = YoloLaneSegmenter(
            YoloLaneConfig(
                model_path=model_path,
                confidence=args.conf if args.conf is not None else config.yolo.confidence,
                image_size=args.imgsz,
                device=args.device,
                min_mask_area_ratio=config.yolo.min_mask_area_ratio,
            )
        )
    transformer = BevTransformer(config.bev)
    geometry_estimator = ParkingGeometryEstimator(config.geometry)
    lidar_estimator = LidarParkingSpaceEstimator(config.lidar)
    controller = SharedParkingPlannerReplay(config)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cannot open recording video: %s" % video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    frame_stride = max(1, args.frame_stride)
    output_fps = max(1.0, fps / frame_stride)

    writer = None
    save_path = resolve_path(args.save_video) if args.save_video else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    camera_samples: List[CameraSample] = []
    transitions: List[Dict[str, object]] = []
    scan_start = scans[0].timestamp
    scan_relative_times = [scan.timestamp - scan_start for scan in scans]
    scan_index = 0
    frame_index = 0
    previous_state = controller.state
    previous_planner_state = controller.planner_state
    planner_transitions = []
    current_lidar = LidarParkingObservation(reason="waiting_for_first_scan")
    current_scan: Optional[LidarScan] = None
    current_rear_cm: Optional[float] = None
    current_command = ReplayCommand(
        controller.state,
        config.planner.search_speed,
        0,
        "searching_for_parked_cars",
    )
    stopped_by_user = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            elapsed = frame_index / fps
            if segmenter is None:
                frame_masks = []
                bev_masks = []
                geometry = ParkingGeometry(reason="camera_disabled")
                camera = CameraSample(
                    elapsed,
                    None,
                    False,
                    "camera_disabled",
                    0.0,
                    geometry=geometry,
                )
            else:
                class_masks = segmenter.segment_class_masks(frame)
                frame_masks = list(class_masks.lane)
                bev_masks = [transformer.warp_mask(mask) for mask in frame_masks]
                geometry = geometry_estimator.estimate(bev_masks, class_masks.lane_conf)
                found = geometry.found and geometry.has_side_pair
                camera = CameraSample(
                    elapsed_s=elapsed,
                    line_error_px=geometry.lateral_error_px if found else None,
                    found=found,
                    reason=geometry.reason,
                    inference_ms=class_masks.inference_ms,
                    detected_line_masks=len(frame_masks),
                    fitted_line_count=geometry.observed_line_count,
                    geometry=geometry,
                )
            camera_samples.append(camera)

            while (
                scan_index < len(scans)
                and scan_relative_times[scan_index] <= elapsed
            ):
                current_scan = scans[scan_index]
                current_lidar = lidar_estimator.estimate(
                    current_scan,
                    now=current_scan.timestamp,
                )
                current_rear_cm = rear_lidar_distance_cm(current_scan, config)
                old_state = controller.state
                current_command = controller.update(
                    current_lidar,
                    geometry,
                    scan_relative_times[scan_index],
                )
                if current_command.state != old_state:
                    transition = {
                        "elapsed_s": round(scan_relative_times[scan_index], 3),
                        "from": old_state.value,
                        "to": current_command.state.value,
                        "event": current_command.event,
                    }
                    transitions.append(transition)
                    LOG.info(
                        "Arduino replay state: %s -> %s at %.3fs (%s)",
                        old_state.value,
                        current_command.state.value,
                        scan_relative_times[scan_index],
                        current_command.event,
                    )
                if controller.planner_state != previous_planner_state:
                    planner_transition = {
                        "elapsed_s": round(scan_relative_times[scan_index], 3),
                        "from": previous_planner_state.value,
                        "to": controller.planner_state.value,
                        "event": current_command.event,
                    }
                    planner_transitions.append(planner_transition)
                    LOG.info(
                        "shared planner: %s -> %s at %.3fs (%s)",
                        previous_planner_state.value,
                        controller.planner_state.value,
                        scan_relative_times[scan_index],
                        current_command.event,
                    )
                    previous_planner_state = controller.planner_state
                previous_state = current_command.state
                rows.append(
                    replay_row(
                        scan_relative_times[scan_index],
                        current_command,
                        current_lidar,
                        current_rear_cm,
                        camera,
                        controller,
                    )
                )
                scan_index += 1

            lidar_for_display = replace(
                current_lidar,
                reason="arduino_%s" % current_command.state.value.lower(),
                entry_target_y_back_mm=constants.position_target_cm * 10.0,
                entry_error_mm=(
                    None
                    if current_lidar.gap_center_y_back_mm is None
                    else current_lidar.gap_center_y_back_mm
                    - constants.position_target_cm * 10.0
                ),
                entry_reached=(
                    current_lidar.gap_confirmed
                    and current_lidar.gap_center_y_back_mm is not None
                    and abs(
                        current_lidar.gap_center_y_back_mm
                        - constants.position_target_cm * 10.0
                    )
                    <= constants.position_tolerance_cm * 10.0
                ),
            )
            lidar_points = (
                lidar_estimator.vehicle_points(current_scan)
                if current_scan is not None
                else []
            )
            dashboard = draw_arduino_dashboard(
                cv2,
                np,
                frame,
                frame_masks,
                bev_masks,
                transformer,
                geometry,
                lidar_for_display,
                lidar_points,
                config,
                current_command,
                current_rear_cm,
                camera,
                elapsed,
                constants,
                controller,
            )

            if save_path is not None:
                if writer is None:
                    height, width = dashboard.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("cannot create replay video: %s" % save_path)
                writer.write(dashboard)

            if args.show:
                cv2.imshow("Arduino T Parking - Offline Replay", dashboard)
                key = cv2.waitKey(max(1, args.replay_delay_ms)) & 0xFF
                if key in (ord("q"), 27):
                    stopped_by_user = True
                    break
                if key == ord(" "):
                    while True:
                        pause_key = cv2.waitKey(50) & 0xFF
                        if pause_key == ord(" "):
                            break
                        if pause_key in (ord("q"), 27):
                            stopped_by_user = True
                            break
                    if stopped_by_user:
                        break

            if len(camera_samples) % 50 == 0:
                LOG.info(
                    "visual replay: frame=%d time=%.1fs state=%s",
                    frame_index,
                    elapsed,
                    current_command.state.value,
                )
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    valid_camera = [sample for sample in camera_samples if sample.found]
    gap_rows = [row for row in rows if row["gap_found"] == 1]
    rear_values = [
        float(row["rear_lidar_cm"])
        for row in rows
        if row["rear_lidar_cm"] != ""
    ]
    center_values = [
        abs(float(row["gap_center_y_back_cm"]))
        for row in gap_rows
        if row["gap_center_y_back_cm"] != ""
    ]
    depth_vectors = [
        (float(row["slot_depth_x"]), float(row["slot_depth_y"]))
        for row in gap_rows
        if row["slot_depth_x"] != "" and row["slot_depth_y"] != ""
    ]
    consecutive_depth_dots = [
        first[0] * second[0] + first[1] * second[1]
        for first, second in zip(depth_vectors, depth_vectors[1:])
    ]
    summary: Dict[str, object] = {
        "lidar_scans": len(rows),
        "duration_s": round(float(rows[-1]["elapsed_s"]), 3) if rows else 0.0,
        "gap_confirmed_scans": sum(row["gap_confirmed"] == 1 for row in rows),
        "gap_hold_scans": sum(row["gap_coasted"] == 1 for row in rows),
        "slot_depth_flip_count": sum(value < 0.0 for value in consecutive_depth_dots),
        "minimum_consecutive_slot_depth_dot": (
            round(min(consecutive_depth_dots), 6)
            if consecutive_depth_dots
            else None
        ),
        "closest_gap_center_abs_cm": round(min(center_values), 3) if center_values else None,
        "minimum_rear_lidar_cm": round(min(rear_values), 3) if rear_values else None,
        "camera_samples": len(camera_samples),
        "camera_valid_samples": len(valid_camera),
        "camera_valid_ratio": round(len(valid_camera) / len(camera_samples), 4) if camera_samples else None,
        "camera_samples_with_line_mask": sum(
            sample.detected_line_masks >= 1 for sample in camera_samples
        ),
        "camera_samples_with_two_masks": sum(
            sample.detected_line_masks >= 2 for sample in camera_samples
        ),
        "camera_geometry_reason_counts": dict(
            Counter(sample.reason for sample in camera_samples)
        ),
        "final_state": controller.state.value,
        "final_planner_state": controller.planner_state.value,
        "transitions": transitions,
        "planner_transitions": planner_transitions,
        "ultrasonic_replayed": False,
        "stopped_by_user": stopped_by_user,
        "processed_video_frames": len(camera_samples),
        "frame_stride": frame_stride,
    }
    return rows, summary


def replay_row(
    elapsed_s: float,
    command: ReplayCommand,
    lidar: LidarParkingObservation,
    rear_cm: Optional[float],
    camera: CameraSample,
    controller: SharedParkingPlannerReplay,
) -> Dict[str, object]:
    return {
        "elapsed_s": round(elapsed_s, 6),
        "state": command.state.value,
        "planner_state": controller.planner_state.value,
        "drive_speed": command.speed,
        "steer_deg": command.steering_deg,
        "event": command.event,
        "gap_found": int(lidar.gap_found),
        "gap_confirmed": int(lidar.gap_confirmed),
        "car_clusters": lidar.car_count,
        "first_car_confirmed": int(lidar.first_car_confirmed),
        "first_car_edge_y_back_cm": optional_round(
            lidar.first_car_slot_edge_y_back_mm,
            10.0,
        ),
        "first_car_turn_error_cm": optional_round(
            lidar.first_car_turn_error_mm,
            10.0,
        ),
        "first_car_turn_reached": int(lidar.first_car_turn_reached),
        "gap_center_y_back_cm": optional_round(lidar.gap_center_y_back_mm, 10.0),
        "gap_width_cm": optional_round(lidar.gap_width_mm, 10.0),
        "slot_depth_x": optional_round(lidar.slot_depth_x_right),
        "slot_depth_y": optional_round(lidar.slot_depth_y_back),
        "gap_coasted": int(lidar.coasted),
        "positioning_phase": controller.positioning_phase.value,
        "prealign_confirm_count": controller.prealign_confirm_count,
        "tracked_gap_center_cm": round(controller.tracked_gap_center_y_back_cm, 3),
        "gap_confirm_count": controller.gap_confirm_count,
        "position_confirm_count": controller.position_confirm_count,
        "finish_confirm_count": controller.finish_confirm_count,
        "rear_lidar_cm": optional_round(rear_cm),
        "line_error_px": optional_round(camera.line_error_px if camera.found else None),
        "camera_found": int(camera.found),
        "camera_reason": camera.reason,
        "detected_line_masks": camera.detected_line_masks,
        "fitted_line_count": camera.fitted_line_count,
        "left_ultrasonic_cm": "",
        "right_ultrasonic_cm": "",
    }


def draw_arduino_dashboard(
    cv2: object,
    np: object,
    frame: object,
    frame_masks: Sequence[object],
    bev_masks: Sequence[object],
    transformer: BevTransformer,
    geometry: ParkingGeometry,
    lidar: LidarParkingObservation,
    lidar_points: Sequence[Tuple[float, float]],
    config: ParkingAppConfig,
    command: ReplayCommand,
    rear_cm: Optional[float],
    camera: CameraSample,
    elapsed_s: float,
    constants: ArduinoReplayConstants,
    controller: SharedParkingPlannerReplay,
) -> object:
    rear_display = frame.copy()
    for index, mask in enumerate(frame_masks):
        color = np.asarray(parking_mask_color(index, geometry), dtype=np.float32)
        selected = mask > 0
        rear_display[selected] = (
            0.55 * rear_display[selected] + 0.45 * color
        ).astype(np.uint8)
    polygon = transformer.src_polygon(frame.shape[:2]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(rear_display, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)

    state_color = {
        ReplayParkingState.SEARCHING: (0, 220, 255),
        ReplayParkingState.POSITIONING: (0, 165, 255),
        ReplayParkingState.REVERSING: (0, 255, 0),
        ReplayParkingState.FINISHED: (255, 255, 255),
        ReplayParkingState.EMERGENCY_STOP: (0, 0, 255),
        ReplayParkingState.ABORTED: (0, 0, 255),
    }[command.state]

    bev_display = transformer.warp_frame(frame)
    for index, mask in enumerate(bev_masks):
        color = np.asarray(parking_mask_color(index, geometry), dtype=np.float32)
        selected = mask > 0
        bev_display[selected] = (
            0.45 * bev_display[selected] + 0.55 * color
        ).astype(np.uint8)
    draw_parking_line(cv2, bev_display, geometry.left, (255, 255, 0))
    draw_parking_line(cv2, bev_display, geometry.right, (0, 255, 0))
    draw_parking_line(cv2, bev_display, geometry.back, (0, 0, 255))
    draw_reverse_path(
        cv2,
        np,
        bev_display,
        controller.last_plan.path if controller.last_plan is not None else None,
    )
    cv2.line(
        bev_display,
        (bev_display.shape[1] // 2, 0),
        (bev_display.shape[1] // 2, bev_display.shape[0] - 1),
        (100, 100, 100),
        1,
    )

    lidar_display = draw_lidar_debug(
        cv2,
        np,
        list(lidar_points),
        config,
        lidar,
    )

    center_cm = (
        None if lidar.gap_center_y_back_mm is None
        else lidar.gap_center_y_back_mm / 10.0
    )
    width_cm = None if lidar.gap_width_mm is None else lidar.gap_width_mm / 10.0
    line_error = camera.line_error_px if camera.found else None
    status_lines = (
        "OFFLINE ZIP REPLAY - MOTOR OUTPUT DISABLED",
        "STATE %-22s drive=%+3d steer=%+4d event=%s" % (
            controller.planner_state.value,
            command.speed,
            command.steering_deg,
            command.event or "-",
        ),
        "LiDAR cars=%d gap=%s centerY=%s cm width=%s cm rear=%s cm" % (
            lidar.car_count,
            "CONFIRMED" if lidar.gap_confirmed else ("candidate" if lidar.gap_found else "no"),
            format_optional(center_cm, signed=True),
            format_optional(width_cm),
            format_optional(rear_cm),
        ),
        "FIRST CAR edgeY=%s cm target=%+.1f cm error=%s cm turn=%s" % (
            format_optional(
                None
                if lidar.first_car_slot_edge_y_back_mm is None
                else lidar.first_car_slot_edge_y_back_mm / 10.0,
                signed=True,
            ),
            config.lidar.first_car_turn_target_y_back_mm / 10.0,
            format_optional(
                None
                if lidar.first_car_turn_error_mm is None
                else lidar.first_car_turn_error_mm / 10.0,
                signed=True,
            ),
            "READY" if lidar.first_car_turn_reached else "waiting",
        ),
        "POSITION %s center=%s prealign=%d/%d" % (
            controller.positioning_phase.value,
            format_optional(controller.tracked_gap_center_y_back_cm, signed=True),
            controller.prealign_confirm_count,
            config.planner.prealign_confirm_frames,
        ),
        "CAM masks=%d fitted=%d validPair=%s lineError=%s px reason=%s" % (
            camera.detected_line_masks,
            camera.fitted_line_count,
            "Y" if camera.found else "N",
            format_optional(line_error, signed=True),
            camera.reason,
        ),
        "ULTRASONIC LEFT=N/A RIGHT=N/A (not present in this recording)",
        "Colors: CYAN=left GREEN=right RED=back MAGENTA=unclassified | SPACE pause Q quit",
    )
    return compose_parking_dashboard(
        cv2,
        np,
        rear_display,
        bev_display,
        lidar_display,
        "SHARED %s | drive=%+d steer=%+d | t=%.2fs" % (
            controller.planner_state.value,
            command.speed,
            command.steering_deg,
            elapsed_s,
        ),
        state_color,
        status_lines,
    )


def format_optional(value: Optional[float], signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return ("%+.1f" if signed else "%.1f") % value


def analyze_camera(
    video_path: Path,
    config: ParkingAppConfig,
    args: argparse.Namespace,
) -> List[CameraSample]:
    import cv2

    model_path = resolve_path(args.model or config.yolo.model_path)
    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(
            model_path=model_path,
            confidence=args.conf if args.conf is not None else config.yolo.confidence,
            image_size=args.imgsz,
            device=args.device,
            min_mask_area_ratio=config.yolo.min_mask_area_ratio,
        )
    )
    transformer = BevTransformer(config.bev)
    estimator = ParkingGeometryEstimator(config.geometry)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cannot open recording video: %s" % video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    interval_frames = max(1, int(round(args.camera_interval_s * fps)))
    samples: List[CameraSample] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval_frames != 0:
                frame_index += 1
                continue
            class_masks = segmenter.segment_class_masks(frame)
            bev_masks = [transformer.warp_mask(mask) for mask in class_masks.lane]
            geometry = estimator.estimate(bev_masks, class_masks.lane_conf)
            found = geometry.found and geometry.has_side_pair
            samples.append(
                CameraSample(
                    elapsed_s=frame_index / fps,
                    line_error_px=geometry.lateral_error_px if found else None,
                    found=found,
                    reason=geometry.reason,
                    inference_ms=class_masks.inference_ms,
                    detected_line_masks=len(class_masks.lane),
                    fitted_line_count=geometry.observed_line_count,
                    geometry=geometry,
                )
            )
            if len(samples) % 20 == 0:
                LOG.info("camera samples processed: %d", len(samples))
            frame_index += 1
    finally:
        capture.release()
    return samples


def rear_lidar_distance_cm(
    scan: LidarScan,
    config: ParkingAppConfig,
    window_deg: float = 4.0,
) -> Optional[float]:
    """Approximate Arduino vehicle-rear 180 degrees from the raw recording.

    The recording calibration maps raw bearing to the vehicle frame using the
    configured offset/direction. We transform each beam and select the nearest
    return within a small angular window around the rear axis.
    """

    distances = []
    for point in scan.points:
        if point.quality < config.lidar.quality_min:
            continue
        if not config.lidar.min_distance_mm <= point.distance_mm <= config.lidar.max_distance_mm:
            continue
        angle = point.angle_deg + config.lidar.angle_offset_deg
        if config.lidar.clockwise_angles:
            angle = -angle
        bearing = math.radians(angle)
        x_right = math.sin(bearing) * point.distance_mm
        y_back = -math.cos(bearing) * point.distance_mm
        vehicle_angle = math.degrees(math.atan2(-x_right, -y_back)) % 360.0
        if abs(signed_angle_difference(vehicle_angle, 180.0)) <= window_deg:
            distances.append(point.distance_mm / 10.0)
    return min(distances) if distances else None


def latest_camera_sample(
    samples: Sequence[CameraSample], elapsed_s: float
) -> Optional[CameraSample]:
    latest = None
    for sample in samples:
        if sample.elapsed_s > elapsed_s:
            break
        latest = sample
    return latest


def signed_angle_difference(angle_deg: float, reference_deg: float) -> float:
    return (angle_deg - reference_deg + 540.0) % 360.0 - 180.0


def optional_round(value: Optional[float], divisor: float = 1.0) -> object:
    if value is None:
        return ""
    return round(value / divisor, 3)


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("replay produced no rows")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an MP4+LiDAR ZIP through the Arduino T-parking logic"
    )
    parser.add_argument("--recording-zip", required=True, help="ZIP containing one MP4 and one *_lidar.csv")
    parser.add_argument("--config", default="configs/parking.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, ...")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--camera-interval-s", type=float, default=0.5)
    parser.add_argument("--no-camera", action="store_true", help="run a fast LiDAR/state-only replay")
    parser.add_argument("--position-tolerance-cm", type=float, default=10.0)
    parser.add_argument(
        "--first-car-turn-target-cm",
        type=float,
        default=None,
        help="vehicle-frame yBack trigger for the first car; negative is ahead",
    )
    parser.add_argument("--prealign-speed", type=int, default=None)
    parser.add_argument("--prealign-steering", type=int, default=None)
    parser.add_argument("--prealign-timeout-s", type=float, default=None)
    parser.add_argument(
        "--lidar-to-rear-axle-cm",
        type=float,
        default=None,
        help=(
            "signed longitudinal rear-axle coordinate from the LiDAR origin "
            "(negative=ahead of LiDAR, positive=behind); default comes from parking config"
        ),
    )
    parser.add_argument(
        "--lidar-behind-vehicle-rear-cm",
        type=float,
        default=None,
        help="debug view: distance the LiDAR sits behind the vehicle rear bumper",
    )
    parser.add_argument("--show", action="store_true", help="show rear+BEV+LiDAR Arduino-state dashboard")
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="data/processed/arduino_parking_replay_overlay.mp4",
        default=None,
        help="optionally save the rendered dashboard MP4",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="visual mode: infer every Nth video frame")
    parser.add_argument("--replay-delay-ms", type=int, default=1, help="visual mode: minimum GUI delay per rendered frame")
    parser.add_argument("--output", default="data/processed/arduino_parking_replay.csv")
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.camera_interval_s <= 0.0:
        parser.error("--camera-interval-s must be positive")
    if args.position_tolerance_cm < 0.0:
        parser.error("--position-tolerance-cm cannot be negative")
    if args.prealign_speed is not None and args.prealign_speed <= 0:
        parser.error("--prealign-speed must be positive")
    if args.prealign_steering == 0:
        parser.error("--prealign-steering cannot be zero")
    if args.prealign_steering is not None and abs(args.prealign_steering) > 150:
        parser.error("--prealign-steering must be between -150 and 150")
    if args.prealign_timeout_s is not None and args.prealign_timeout_s <= 0.0:
        parser.error("--prealign-timeout-s must be positive")
    if (
        args.lidar_behind_vehicle_rear_cm is not None
        and args.lidar_behind_vehicle_rear_cm < 0.0
    ):
        parser.error("--lidar-behind-vehicle-rear-cm cannot be negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        summary = run_replay(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        LOG.exception("Arduino parking replay failed: %s", exc)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
