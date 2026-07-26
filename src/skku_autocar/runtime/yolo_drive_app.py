from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..control.serial_vehicle import SerialVehicleClient, SerialVehicleConfig
from ..estimation.bev_corridor import BevCorridorConfig, BevCorridorLaneEstimator, warp_class_masks
from ..estimation.lane_geometry import LaneGeometry
from ..perception.bev import BevConfig, BevTransformer
from ..perception.traffic_light import TrafficLightConfig, TrafficLightController, TrafficLightObservation
from ..perception.yolo_lane import YoloClassMasks, YoloLaneConfig, YoloLaneMask, YoloLaneSegmenter
from ..planning.yolo_lane_follower import YoloLaneFollower, YoloLaneFollowerConfig
from ..types import ControlCommand
from .obstacle_mode import ObstacleDriveMode, add_obstacle_arguments


LOG = logging.getLogger("skku_autocar.yolo_drive")
DEFAULT_MODEL = "trained_model/skku_merged_yolov8n_seg_aug_best.pt"


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
    # The competition runtime has one lane path: per-class YOLO masks are warped
    # into BEV first, then the two-lane corridor is resolved there.
    transformer = BevTransformer(BevConfig())
    corridor_config = build_bev_corridor_config(args)
    corridor_estimator = BevCorridorLaneEstimator(corridor_config)
    follower_config = build_follower_config(args)
    follower = YoloLaneFollower(follower_config)
    command_filter = CommandSafetyFilter(args)
    drive_policy = DrivePriorityController(
        command_filter,
        traffic_light_enabled=args.traffic_light == "on",
    )
    obstacle_mode = ObstacleDriveMode(args, transformer, corridor_estimator)
    traffic_light = TrafficLightController(
        TrafficLightConfig(
            min_saturation=args.light_min_saturation,
            min_value=args.light_min_value,
            min_color_pixels=args.light_min_color_pixels,
            min_color_ratio=args.light_min_color_ratio,
            dominance_ratio=args.light_dominance_ratio,
            confirm_frames=args.light_confirm_frames,
            red_confirm_frames=args.light_red_confirm_frames,
            contact_confirm_frames=args.light_contact_confirm_frames,
            contact_hold_frames=args.light_contact_hold_frames,
            contact_min_row_width_ratio=args.light_contact_min_row_width_ratio,
            stop_line_y_ratio=args.light_stop_line_y,
        )
    )

    cap = open_camera(cv2, args.camera, args.width, args.height, args.fourcc)
    # A video-file source (not a live camera index) should loop for review instead
    # of erroring out at the end of the clip.
    is_video_file = not str(args.camera).isdigit()
    ok, first_frame = read_startup_frame(cap, 1 if is_video_file else 5)
    if not ok:
        cap.release()
        raise RuntimeError("camera frame read failed")
    try:
        enforce_camera_contract(
            first_frame.shape,
            args.width,
            args.height,
            live_camera=not is_video_file,
            policy=args.camera_resolution_policy,
        )
        if (
            not is_video_file
            and args.camera_resolution_policy == "allow"
            and not args.no_serial
        ):
            raise RuntimeError(
                "--camera-resolution-policy allow requires --no-serial; "
                "calibration-mismatched frames cannot control the vehicle"
            )
    except Exception:
        cap.release()
        raise
    actual_height, actual_width = first_frame.shape[:2]
    LOG.info(
        "camera=%s requested=%dx%d actual=%dx%d policy=%s",
        args.camera,
        args.width,
        args.height,
        actual_width,
        actual_height,
        args.camera_resolution_policy,
    )

    # Validate the calibrated image geometry before opening the motor serial
    # connection. A camera fallback must never leave the vehicle controllable.
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
        obstacle_mode.start_serial(vehicle)
        LOG.info("serial connected: %s", vehicle.port)
    else:
        LOG.info("serial disabled: dry video/control preview mode")

    recorder = DriveRecorder(cv2, args, first_frame.shape)
    running = bool(args.auto_start)
    last_command_at = 0.0
    last_log_at = 0.0
    last_frame_at = time.monotonic()
    fps = 0.0
    command = ControlCommand.stop("paused")

    LOG.info("model=%s device=%s camera=%s", model_path, segmenter.device, args.camera)
    obstacle_mode.validate_runtime(segmenter, args.no_serial)
    log_effective_config(args, corridor_config, follower_config)
    obstacle_mode.log_configuration(args)
    if recorder.enabled:
        LOG.info("recording raw video: %s", recorder.raw_video_path)
        if recorder.debug_video_path is not None:
            LOG.info("recording debug video: %s", recorder.debug_video_path)
    if obstacle_mode.enabled:
        LOG.info("space: start/stop | l: manual lane-change / return | q or esc: quit")
    else:
        LOG.info("space: start/stop | q or esc: quit")

    pending_frame = first_frame
    try:
        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if (
                    not ok
                    and is_video_file
                    and args.video_loop == "on"
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok and is_video_file and args.video_loop == "off":
                    break
                if not ok:
                    raise RuntimeError("camera frame read failed")
            else:
                frame = pending_frame
                pending_frame = None

            enforce_camera_contract(
                frame.shape,
                args.width,
                args.height,
                live_camera=not is_video_file,
                policy=args.camera_resolution_policy,
            )

            wall_now = time.monotonic()
            control_now = (
                max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                if is_video_file
                else wall_now
            )
            obstacle_mode.update_serial(vehicle, control_now)
            dt = max(1e-6, wall_now - last_frame_at)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            last_frame_at = wall_now

            bev_mask = None
            light_masks = ()
            class_masks = segmenter.segment_class_masks(frame)
            light_masks = class_masks.light
            bev = warp_class_masks(
                transformer,
                class_masks,
                include_obstacle=obstacle_mode.enabled,
            )
            lane = corridor_estimator.estimate(bev)
            bev_mask = fuse_masks([*bev.center, *bev.side, *bev.lane, *bev.crosswalk])
            mask_result = corridor_mask_result(class_masks, corridor_estimator, frame.shape)
            lane = obstacle_mode.update(
                class_masks,
                bev,
                lane,
                mask_result,
                frame.shape,
                control_now,
                running,
            )
            stop_crosswalk_masks = (
                class_masks.crosswalk
                if class_masks.crosswalk_conf >= args.light_crosswalk_min_conf
                else ()
            )
            if obstacle_mode.blocks_light_stop:
                stop_crosswalk_masks = ()
            light_observation = traffic_light.update(frame, light_masks, stop_crosswalk_masks)
            planned_command = follower.plan(lane) if running else ControlCommand.stop("paused")
            command = drive_policy.apply(
                planned_command,
                mask_result,
                lane,
                running,
                obstacle_mode,
                traffic_light,
            )

            if vehicle is not None and wall_now - last_command_at >= 1.0 / args.command_rate:
                serial_lines = vehicle.send(command)
                obstacle_mode.accept_serial_lines(serial_lines, control_now)
                last_command_at = wall_now

            display = draw_debug(
                cv2, frame, mask_result, lane, command, running, fps,
                transformer, corridor_estimator,
                light_masks, light_observation,
                args.light_stop_line_y,
                obstacle_mode.lane_change_state,
                obstacle_mode.status_text,
                obstacle_mode.frame_obstacle_masks,
            )
            recorder.write(frame, display)
            if not args.headless:
                cv2.imshow("YOLO Drive", display)
            if args.show_mask and not args.headless:
                # In BEV mode show the warped (bird's-eye) road mask so the road is
                # separated in top-down view; otherwise fall back to the frame mask.
                if bev_mask is not None:
                    cv2.imshow(
                        "BEV Lane Mask",
                        draw_bev_mask_debug(
                            cv2,
                            bev_mask,
                            lane,
                            fuse_masks(list(obstacle_mode.bev_obstacle_masks)),
                        ),
                    )
                elif mask_result is not None:
                    cv2.imshow("YOLO Lane Mask", mask_result.mask)

            if wall_now - last_log_at >= args.log_interval:
                log_status(mask_result, lane, command, running, fps, segmenter.device)
                last_log_at = wall_now

            key = -1 if args.headless else cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                running = not running
                LOG.info("driving: %s", "ON" if running else "OFF")
                if not running and vehicle is not None:
                    vehicle.stop("paused")
            if key == ord("l"):
                action, message = obstacle_mode.handle_key(running)
                if action.startswith("ignored"):
                    LOG.info(
                        "%s: mode=%s state=%s",
                        message,
                        args.lane_change_mode,
                        obstacle_mode.lane_change_state,
                    )
                else:
                    LOG.info("%s", message)
    finally:
        recorder.close()
        if vehicle is not None:
            try:
                try:
                    obstacle_mode.stop_serial(vehicle)
                    vehicle.stop("shutdown")
                except Exception as exc:
                    LOG.warning("serial stop failed during shutdown: %s", exc)
            finally:
                vehicle.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def build_bev_corridor_config(args: argparse.Namespace) -> BevCorridorConfig:
    defaults = BevCorridorConfig()
    return BevCorridorConfig(
        lane_width_px=args.corridor_lane_width_px,
        lookahead_y_ratio=_first_set(args.lookahead, args.bev_lookahead, defaults.lookahead_y_ratio),
        lane_change_near_y_ratio=args.bev_lane_change_near_y,
        vehicle_center_x_offset_ratio=args.vehicle_center_offset,
        heading_gain=args.bev_heading_gain,
        center_smooth_alpha=args.bev_center_smooth,
        heading_smooth_alpha=args.bev_heading_smooth,
        path_smooth_alpha=args.bev_path_smooth,
        path_max_step_px=args.bev_path_max_step,
        center_anchor=args.corridor_center_anchor == "on",
        max_center_jump_px=args.corridor_max_center_jump,
        max_heading_jump=args.corridor_max_heading_jump,
        max_coast_frames=args.corridor_max_coast_frames,
        max_width_jump_px=args.corridor_max_width_jump,
        crosswalk_halt=args.crosswalk_halt == "on",
        virtual_hold=args.corridor_virtual_hold == "on",
        vehicle_width_px=args.corridor_vehicle_width_px,
        poly_degree=args.corridor_poly_degree,
        centerline_bias=args.corridor_centerline_bias,
        crosswalk_lane_width_px=args.corridor_crosswalk_lane_width_px,
        crosswalk_center_smooth_alpha=args.corridor_crosswalk_center_smooth,
        crosswalk_max_center_jump_px=args.corridor_crosswalk_max_center_jump,
        crosswalk_option=args.corridor_crosswalk_option,
        crosswalk_right_offset_px=args.corridor_crosswalk_right_offset_px,
        crosswalk_transit_enabled=True,
    )


def fuse_masks(masks: list) -> Any:
    """OR a list of binary masks into one, for debug overlays. None if empty."""
    import numpy as np

    fused = None
    for mask in masks:
        arr = np.asarray(mask)
        fused = arr.copy() if fused is None else np.maximum(fused, arr)
    return fused


def corridor_mask_result(
    class_masks: YoloClassMasks,
    estimator: BevCorridorLaneEstimator,
    frame_shape: tuple,
) -> Optional[YoloLaneMask]:
    """Wrap the corridor result as a YoloLaneMask so the existing safety filter,
    overlay and logging keep working unchanged. The class_name carries the
    corridor tier's name (which contains "virtual" on tiers 2/3 and on the
    vehicle-width virtual-hold fallback), so CommandSafetyFilter's virtual-lane
    guard still triggers.

    When YOLO detects nothing this frame but the estimator is holding a
    virtual-width lane (found=True downstream), we still emit a wrapper carrying
    the "virtual" name over an all-zero mask -- otherwise mask_result would be None
    and the safety filter would treat the guessed lane as a fully reliable real
    lane (no speed/steering cap)."""
    import numpy as np

    name = estimator.last_class_name
    frame_mask = None
    if class_masks.found:
        frame_mask = fuse_masks([*class_masks.center, *class_masks.side, *class_masks.lane, *class_masks.crosswalk])
    if frame_mask is None:
        if "virtual" not in name:
            return None
        height, width = frame_shape[:2]
        frame_mask = np.zeros((height, width), dtype=np.uint8)
    conf = class_masks.center_conf or class_masks.side_conf or class_masks.lane_conf or class_masks.crosswalk_conf
    return YoloLaneMask(
        mask=frame_mask,
        confidence=float(conf),
        class_id=-1,
        class_name=estimator.last_class_name,
        device=class_masks.device,
        inference_ms=class_masks.inference_ms,
    )


def build_follower_config(args: argparse.Namespace) -> YoloLaneFollowerConfig:
    return YoloLaneFollowerConfig(
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
        curve_strength_release_alpha=args.curve_strength_release_alpha,
        straight_steering_scale=args.straight_steering_scale,
        curve_steering_scale=args.curve_steering_scale,
        center_recovery_error_threshold=args.center_recovery_error_threshold,
        center_recovery_steering_boost=args.center_recovery_steering_boost,
        center_recovery_min_steering=args.center_recovery_min_steering,
        center_recovery_rate_limit=args.center_recovery_rate_limit,
        center_recovery_max_speed=args.center_recovery_max_speed,
        center_lock_enabled=args.center_lock == "on",
        center_lock_error_threshold=args.center_lock_error_threshold,
        center_lock_min_steering=args.center_lock_min_steering,
        lane_lost_hold_frames=args.lane_lost_hold_frames,
        lane_lost_steering_release_rate_limit=args.lane_lost_steering_release_rate_limit,
        path_tracking=args.path_tracking,
        path_lateral_gain=args.path_lateral_gain,
        path_heading_gain=args.path_heading_gain,
        path_derivative_gain=args.path_derivative_gain,
        path_near_weight=args.path_near_weight,
        path_far_weight=args.path_far_weight,
        path_steering_rise_alpha=args.path_steering_rise_alpha,
        path_steering_release_alpha=args.path_steering_release_alpha,
        pure_pursuit=args.pure_pursuit,
        pure_pursuit_gain=args.pp_gain,
        pure_pursuit_full_angle=args.pp_full_angle,
    )


def log_effective_config(
    args: argparse.Namespace,
    corridor_config: BevCorridorConfig,
    follower_config: YoloLaneFollowerConfig,
) -> None:
    LOG.info(
        "lane pipeline=bev-corridor lookahead=%.2f lane_change_near=%.2f sample=%.2f..%.2f vehicle_offset=%.3f center_smooth=%.2f heading_smooth=%.2f path_smooth=%.2f path_max_step=%.1f heading_gain=%.2f",
        corridor_config.lookahead_y_ratio,
        corridor_config.lane_change_near_y_ratio,
        corridor_config.sample_top_y_ratio,
        corridor_config.sample_bottom_y_ratio,
        corridor_config.vehicle_center_x_offset_ratio,
        corridor_config.center_smooth_alpha,
        corridor_config.heading_smooth_alpha,
        corridor_config.path_smooth_alpha,
        corridor_config.path_max_step_px,
        corridor_config.heading_gain,
    )

    lane_lost_release = follower_config.lane_lost_steering_release_rate_limit
    if lane_lost_release is None:
        lane_lost_release = max(
            follower_config.min_steering_rate_limit,
            follower_config.steering_release_rate_limit,
        )
    LOG.info(
        "control mode=%s speed=%d min_curve=%d max=%d max_steer=%d path_gain=%.1f/%.1f/%.1f path_weight=%.2f..%.2f path_alpha=%.2f/%.2f center_lock=%s lane_lost_release=%d/frame",
        (
            "whole_path"
            if follower_config.path_tracking
            else ("pure_pursuit" if follower_config.pure_pursuit else "pd")
        ),
        follower_config.base_speed,
        follower_config.min_curve_speed,
        follower_config.max_speed,
        follower_config.max_steering,
        follower_config.path_lateral_gain,
        follower_config.path_heading_gain,
        follower_config.path_derivative_gain,
        follower_config.path_far_weight,
        follower_config.path_near_weight,
        follower_config.path_steering_rise_alpha,
        follower_config.path_steering_release_alpha,
        "on" if follower_config.center_lock_enabled else "off",
        lane_lost_release,
    )
    LOG.info(
        "safety fixed_speed=%s virtual_max_steer=%d virtual_speed_cap=%d virtual_warmup=%d virtual_blend=%.2f virtual_step=%d virtual_center_lock_scale=%.2f virtual_min_reliable=%d virtual_bootstrap_speed_cap=%d lane_lost_speed_cap=%d",
        args.fixed_speed,
        args.virtual_lane_max_steering,
        args.virtual_lane_speed_cap,
        args.virtual_lane_warmup_frames,
        args.virtual_lane_steering_blend,
        args.virtual_lane_max_steering_step,
        args.virtual_lane_center_lock_scale,
        args.virtual_lane_min_reliable_frames,
        args.virtual_lane_bootstrap_speed_cap,
        args.lane_lost_speed_cap,
    )
    LOG.info(
        "traffic_light=%s confirm_frames=%d stop_line_y=%.3f contact_min_row_width=%.2f",
        args.traffic_light,
        args.light_confirm_frames,
        args.light_stop_line_y,
        args.light_contact_min_row_width_ratio,
    )


def _first_set(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    raise ValueError("at least one default value is required")


class CommandSafetyFilter:
    def __init__(self, args: argparse.Namespace):
        self.fixed_speed = args.fixed_speed == "on"
        self.fixed_speed_value = args.speed
        self.virtual_lane_max_steering = args.virtual_lane_max_steering
        self.virtual_lane_speed_cap = args.virtual_lane_speed_cap
        self.virtual_lane_warmup_frames = args.virtual_lane_warmup_frames
        self.virtual_lane_steering_blend = args.virtual_lane_steering_blend
        self.virtual_lane_max_steering_step = args.virtual_lane_max_steering_step
        self.virtual_lane_center_lock_scale = args.virtual_lane_center_lock_scale
        self.virtual_lane_min_reliable_frames = args.virtual_lane_min_reliable_frames
        self.virtual_lane_bootstrap_speed_cap = args.virtual_lane_bootstrap_speed_cap
        self.lane_lost_speed_cap = args.lane_lost_speed_cap
        self._last_reliable_command: Optional[ControlCommand] = None
        self._last_filtered_command: Optional[ControlCommand] = None
        self._virtual_frames = 0
        self._reliable_frames = 0

    def apply(
        self,
        mask_result: Optional[YoloLaneMask],
        lane: LaneGeometry,
        command: ControlCommand,
        running: bool,
    ) -> ControlCommand:
        if not running:
            self.reset()
            return command

        virtual = self._is_virtual_mask(mask_result)
        if virtual:
            self._virtual_frames += 1
            guarded = self._guard_virtual_command(command)
            self._last_filtered_command = guarded
            return guarded

        self._virtual_frames = 0
        if not lane.found:
            guarded = self._guard_lane_lost_command(command)
            self._last_filtered_command = guarded
            return guarded

        guarded = self._force_fixed_speed(command)
        self._reliable_frames += 1
        self._last_reliable_command = guarded
        self._last_filtered_command = guarded
        return guarded

    def reset(self) -> None:
        self._last_reliable_command = None
        self._last_filtered_command = None
        self._virtual_frames = 0
        self._reliable_frames = 0

    def finalize(self, command: ControlCommand, running: bool) -> ControlCommand:
        if not running:
            return command
        return self._force_fixed_speed(command)

    def _guard_virtual_command(self, command: ControlCommand) -> ControlCommand:
        if self._reliable_frames < self.virtual_lane_min_reliable_frames:
            return self._guard_virtual_bootstrap_command(command)

        reason = command.reason
        command = self._scale_virtual_center_lock(command)
        steering = command.steering
        reason = command.reason
        if (
            self._last_reliable_command is not None
            and self._virtual_frames <= self.virtual_lane_warmup_frames
        ):
            steering = self._last_reliable_command.steering
            reason = self._append_reason(reason, "virtual_hold")
        elif self._last_reliable_command is not None:
            steering = self._blend_steering(self._last_reliable_command.steering, steering)
            reason = self._append_reason(reason, "virtual_blend")
        steering = self._limit_steering_step(steering)
        steering = self._clip_abs(steering, self.virtual_lane_max_steering)
        speed = self._cap_or_fix_speed(command.speed, self.virtual_lane_speed_cap)
        return ControlCommand(
            speed=speed,
            steering=steering,
            brake=command.brake,
            reason=self._append_reason(reason, "virtual_cap"),
        )

    def _guard_virtual_bootstrap_command(self, command: ControlCommand) -> ControlCommand:
        if self._last_reliable_command is None:
            return ControlCommand.stop("virtual_bootstrap:no_reliable")
        speed = self._cap_or_fix_speed(
            min(command.speed, self._last_reliable_command.speed),
            self.virtual_lane_bootstrap_speed_cap,
        )
        return ControlCommand(
            speed=speed,
            steering=self._last_reliable_command.steering,
            brake=False,
            reason=self._append_reason(command.reason, "virtual_bootstrap"),
        )

    def _scale_virtual_center_lock(self, command: ControlCommand) -> ControlCommand:
        if "center_lock" not in command.reason:
            return command
        scale = max(0.0, min(1.0, float(self.virtual_lane_center_lock_scale)))
        if scale >= 1.0:
            return command
        return ControlCommand(
            speed=command.speed,
            steering=int(round(command.steering * scale)),
            brake=command.brake,
            reason=self._append_reason(command.reason, "virtual_center_lock_scale"),
        )

    def _guard_lane_lost_command(self, command: ControlCommand) -> ControlCommand:
        if command.brake:
            return command
        return ControlCommand(
            speed=self._cap_or_fix_speed(command.speed, self.lane_lost_speed_cap),
            steering=command.steering,
            brake=command.brake,
            reason=self._append_reason(command.reason, "lane_lost_speed_cap"),
        )

    def _force_fixed_speed(self, command: ControlCommand) -> ControlCommand:
        if not self.fixed_speed or command.brake:
            return command
        if command.speed == self.fixed_speed_value and self._reason_has(command.reason, "fixed_speed"):
            return command
        reason = (
            command.reason
            if self._reason_has(command.reason, "fixed_speed")
            else self._append_reason(command.reason, "fixed_speed")
        )
        return ControlCommand(
            speed=self.fixed_speed_value,
            steering=command.steering,
            brake=command.brake,
            reason=reason,
        )

    def _cap_or_fix_speed(self, speed: int, cap: int) -> int:
        if self.fixed_speed:
            return self.fixed_speed_value
        return min(speed, cap)

    def _blend_steering(self, reference: int, target: int) -> int:
        blend = max(0.0, min(1.0, float(self.virtual_lane_steering_blend)))
        return int(round(reference * (1.0 - blend) + target * blend))

    def _limit_steering_step(self, steering: int) -> int:
        if self._last_filtered_command is None or self.virtual_lane_max_steering_step <= 0:
            return steering
        previous = self._last_filtered_command.steering
        delta = steering - previous
        limit = self.virtual_lane_max_steering_step
        if delta > limit:
            return previous + limit
        if delta < -limit:
            return previous - limit
        return steering

    @staticmethod
    def _is_virtual_mask(mask_result: Optional[YoloLaneMask]) -> bool:
        if mask_result is None:
            return False
        return "virtual" in mask_result.class_name

    @staticmethod
    def _clip_abs(value: int, limit: int) -> int:
        if limit <= 0:
            return 0
        return max(-limit, min(limit, int(value)))

    @staticmethod
    def _append_reason(reason: str, suffix: str) -> str:
        return "%s:%s" % (reason, suffix) if reason else suffix

    @staticmethod
    def _reason_has(reason: str, token: str) -> bool:
        return token in reason.split(":") if reason else False


class DrivePriorityController:
    """Single command-composition point for one runtime frame.

    Priority order:
    1. Lane follower provides the normal steering command.
    2. Obstacle lane-change logic may shift/assist steering.
    3. Geometry guards handle virtual or lost lane evidence.
    4. Mission stops, such as a red light at a crosswalk, may brake.
    5. Obstacle emergency safety may brake.
    6. Fixed-speed policy is finalized for every non-brake command.
    """

    def __init__(
        self,
        command_filter: CommandSafetyFilter,
        traffic_light_enabled: bool = True,
    ) -> None:
        self._command_filter = command_filter
        self._traffic_light_enabled = traffic_light_enabled

    def apply(
        self,
        base_command: ControlCommand,
        mask_result: Optional[YoloLaneMask],
        lane: LaneGeometry,
        running: bool,
        obstacle_mode: Any,
        traffic_light: TrafficLightController,
    ) -> ControlCommand:
        command = obstacle_mode.apply_steering(base_command)
        command = obstacle_mode.apply_speed_cap(command)
        command = self._command_filter.apply(mask_result, lane, command, running)
        if self._traffic_light_enabled and not obstacle_mode.blocks_light_stop:
            command = traffic_light.apply(command, running)
        command = obstacle_mode.apply_safety(command, running)
        return self._command_filter.finalize(command, running)


def parse_args(argv: Optional[list]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 segmentation autonomous driving runtime")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="trained YOLO segmentation model path")
    parser.add_argument("--device", default="mps", help="auto, mps, cpu, 0, cuda, ...")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--traffic-light", choices=("on", "off"), default="on",
        help="compare red/green pixels inside YOLO 'light' masks and latch stop on red until green is confirmed",
    )
    parser.add_argument("--light-confirm-frames", type=int, default=1)
    parser.add_argument(
        "--light-red-confirm-frames",
        type=int,
        default=TrafficLightConfig.red_confirm_frames,
        help="red-only confirmation frames; defaults to --light-confirm-frames when omitted",
    )
    parser.add_argument(
        "--light-crosswalk-min-conf",
        type=float,
        default=0.70,
        help="minimum crosswalk confidence used for traffic-light stop contact",
    )
    parser.add_argument(
        "--light-contact-hold-frames",
        type=int,
        default=TrafficLightConfig.contact_hold_frames,
        help="frames to retain a crosswalk lower-edge crossing while waiting for confirmed red",
    )
    parser.add_argument(
        "--light-contact-confirm-frames",
        type=int,
        default=TrafficLightConfig.contact_confirm_frames,
        help="consecutive frames required below the stop line after the crosswalk lower edge crosses it",
    )
    parser.add_argument(
        "--light-contact-min-row-width-ratio",
        type=float,
        default=TrafficLightConfig.contact_min_row_width_ratio,
        help="minimum frame-width ratio a crosswalk/contact mask row must occupy before it can trigger stop contact",
    )
    parser.add_argument("--light-min-saturation", type=int, default=TrafficLightConfig.min_saturation)
    parser.add_argument("--light-min-value", type=int, default=TrafficLightConfig.min_value)
    parser.add_argument("--light-min-color-pixels", type=int, default=TrafficLightConfig.min_color_pixels)
    parser.add_argument("--light-min-color-ratio", type=float, default=TrafficLightConfig.min_color_ratio)
    parser.add_argument("--light-dominance-ratio", type=float, default=TrafficLightConfig.dominance_ratio)
    parser.add_argument(
        "--light-stop-line-y",
        type=float,
        default=0.80,
        help="frame y ratio of the virtual crosswalk contact line; confirmed RED brakes when the crosswalk mask bottom reaches this line",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="front camera index or video path",
    )
    parser.add_argument(
        "--video-loop",
        choices=("on", "off"),
        default="on",
        help="loop a video-file camera source; off processes it exactly once",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="start driving immediately (intended for no-serial video replay)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable OpenCV windows while still producing debug recordings",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--camera-resolution-policy",
        choices=("strict", "allow"),
        default="strict",
        help="strict refuses a live-camera size that differs from --width/--height; allow is for calibration/debug only",
    )
    parser.add_argument("--serial-port", default=None, help="Arduino port, e.g. COM3 or /dev/cu.usbmodemXXXX")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--ready-timeout", type=float, default=3.0)
    parser.add_argument("--no-serial", action="store_true", help="run without Arduino output")
    parser.add_argument("--speed", type=int, default=255)
    parser.add_argument(
        "--fixed-speed",
        choices=("on", "off"),
        default="on",
        help="force every non-brake driving command to --speed after steering safety filters",
    )
    parser.add_argument("--max-speed", type=int, default=255)
    parser.add_argument("--min-curve-speed", type=int, default=255)
    add_obstacle_arguments(parser)
    parser.add_argument("--max-steering", type=int, default=150)
    parser.add_argument("--steering-rate-limit", type=int, default=150)
    parser.add_argument("--min-steering-rate-limit", type=int, default=35)
    parser.add_argument("--steering-release-rate-limit", type=int, default=6)
    parser.add_argument("--kp-lateral", type=float, default=205.0)
    parser.add_argument("--kd-lateral", type=float, default=75.0)
    parser.add_argument("--kp-heading", type=float, default=1.5)
    parser.add_argument("--kd-heading", type=float, default=0.3)
    parser.add_argument("--speed-curve-slowdown", type=int, default=0)
    parser.add_argument(
        "--lateral-priority-threshold",
        type=float,
        default=0.16,
        help="ignore conflicting heading only when lateral error is above this threshold",
    )
    parser.add_argument(
        "--curve-strength-alpha",
        type=float,
        default=0.60,
        help="curve strength rise smoothing alpha; higher enters curve steering faster",
    )
    parser.add_argument(
        "--curve-strength-release-alpha",
        type=float,
        default=0.18,
        help="curve strength fall smoothing alpha; lower holds curve steering longer at curve exit",
    )
    parser.add_argument("--straight-steering-scale", type=float, default=0.50)
    parser.add_argument("--curve-steering-scale", type=float, default=1.68)
    parser.add_argument("--center-recovery-error-threshold", type=float, default=0.08)
    parser.add_argument("--center-recovery-steering-boost", type=float, default=1.35)
    parser.add_argument("--center-recovery-min-steering", type=int, default=85)
    parser.add_argument("--center-recovery-rate-limit", type=int, default=150)
    parser.add_argument("--center-recovery-max-speed", type=int, default=255)
    parser.add_argument(
        "--center-lock",
        choices=("on", "off"),
        default="on",
        help="force at least --center-lock-min-steering toward lane center when lateral error exceeds --center-lock-error-threshold",
    )
    parser.add_argument(
        "--center-lock-error-threshold",
        type=float,
        default=0.055,
        help="normalized lateral error that activates center-lock steering",
    )
    parser.add_argument(
        "--center-lock-min-steering",
        type=int,
        default=75,
        help="minimum absolute steering while center-lock is active",
    )
    parser.add_argument(
        "--lane-lost-hold-frames",
        type=int,
        default=3,
        help="keep the last steering/speed for up to this many frames when the lane is not detected (e.g. crosswalks) before stopping",
    )
    parser.add_argument(
        "--lane-lost-steering-release-rate-limit",
        type=int,
        default=35,
        help="during lane-lost hold, release cached steering toward 0 by this many units/frame; default=max(min-steering-rate-limit, steering-release-rate-limit), 0 keeps the old cached steering",
    )
    parser.add_argument(
        "--lane-lost-speed-cap",
        type=int,
        default=255,
        help="cap speed while the planner is holding a lane-lost command",
    )
    parser.add_argument(
        "--virtual-lane-max-steering",
        type=int,
        default=150,
        help="cap absolute steering when YOLO uses a virtual lane fallback",
    )
    parser.add_argument(
        "--virtual-lane-speed-cap",
        type=int,
        default=255,
        help="cap speed when YOLO uses a virtual lane fallback",
    )
    parser.add_argument(
        "--virtual-lane-warmup-frames",
        type=int,
        default=0,
        help="for the first N consecutive virtual-lane frames, reuse the last reliable steering before applying the virtual cap",
    )
    parser.add_argument(
        "--virtual-lane-steering-blend",
        type=float,
        default=1.00,
        help="after warmup, blend virtual-lane steering with the last reliable steering. 0=ignore virtual steering, 1=trust raw virtual steering",
    )
    parser.add_argument(
        "--virtual-lane-max-steering-step",
        type=int,
        default=100,
        help="max steering change per frame while in virtual-lane fallback; 0 disables this guard",
    )
    parser.add_argument(
        "--virtual-lane-center-lock-scale",
        type=float,
        default=0.25,
        help="scale center-lock steering while using virtual-lane fallback; virtual lanes are guessed, so keep this below 1.0 for safety",
    )
    parser.add_argument(
        "--virtual-lane-min-reliable-frames",
        type=int,
        default=1,
        help="require this many real non-virtual lane frames before trusting virtual-lane steering; before that, hold the last reliable command",
    )
    parser.add_argument(
        "--virtual-lane-bootstrap-speed-cap",
        type=int,
        default=255,
        help="speed cap while virtual-lane steering is blocked because there are not enough reliable real-lane frames yet",
    )
    # Compatibility no-op for existing launch commands. BEV corridor is now the
    # only competition runtime pipeline and is always enabled.
    parser.add_argument(
        "--bev-corridor",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--corridor-lane-width-px",
        type=float,
        default=BevCorridorConfig.lane_width_px,
        help="[--bev-corridor] BEV lane width (px) between the center line and the outer side line; measure once with scripts/bev_replay.py --corridor",
    )
    parser.add_argument(
        "--corridor-center-anchor",
        choices=("on", "off"),
        default="on",
        help="[--bev-corridor] anchor the centerline on the center line + half smoothed width (less jitter, seamless tier1<->tier2); off = raw midpoint of center/side",
    )
    parser.add_argument(
        "--corridor-max-center-jump",
        type=float,
        default=150.0,
        help="[--bev-corridor] reject and coast a frame whose lookahead center_x jumps more than this many BEV px (lower = smoother, more likely to briefly coast on real fast curves)",
    )
    parser.add_argument(
        "--corridor-max-heading-jump",
        type=float,
        default=0.45,
        help="[--bev-corridor] reject and coast a frame whose heading jumps more than this normalized amount",
    )
    parser.add_argument(
        "--corridor-max-coast-frames",
        type=int,
        default=3,
        help="[--bev-corridor] hold the last good geometry for at most this many rejected frames before declaring the lane lost",
    )
    parser.add_argument(
        "--corridor-max-width-jump",
        type=float,
        default=40.0,
        help="[--bev-corridor] reject a measured lane width that jumps more than this many px from the current smoothed value",
    )
    parser.add_argument(
        "--crosswalk-halt",
        choices=("on", "off"),
        default="off",
        help="[--bev-corridor] stop at a crosswalk (traffic-light mission) instead of following the visible lanes through it",
    )
    parser.add_argument(
        "--corridor-virtual-hold",
        choices=("on", "off"),
        default="on",
        help="[--bev-corridor] when the lane is fully lost, hold a vehicle-width virtual lane and keep centered (guarded by the virtual-lane safety caps) instead of braking immediately",
    )
    parser.add_argument(
        "--corridor-vehicle-width-px",
        type=float,
        default=BevCorridorConfig.vehicle_width_px,
        help="[--bev-corridor] BEV pixel width of the car, used as the virtual-lane width while holding",
    )
    parser.add_argument(
        "--corridor-poly-degree",
        type=int,
        default=BevCorridorConfig.poly_degree,
        help="[--bev-corridor] polynomial degree for the BEV line/centerline fit (3 = follows S-curves; 2 = smoother on straights)",
    )
    parser.add_argument(
        "--corridor-centerline-bias",
        type=float,
        default=0.46,
        help="[--bev-corridor] driving line position between boundaries: 0=center line, 0.5=midpoint, 1=outer side line. Raise if the car rides too far inside",
    )
    parser.add_argument(
        "--corridor-crosswalk-lane-width-px",
        type=float,
        default=BevCorridorConfig.crosswalk_lane_width_px,
        help="[--bev-corridor] fixed BEV lane width used to build the virtual centerline while a crosswalk is in view (zebra makes the measured width unreliable)",
    )
    parser.add_argument(
        "--corridor-crosswalk-center-smooth",
        type=float,
        default=0.10,
        help="[--bev-corridor] center-x EMA factor while a crosswalk is visible (lower = steadier)",
    )
    parser.add_argument(
        "--corridor-crosswalk-max-center-jump",
        type=float,
        default=150.0,
        help="[--bev-corridor] reject and coast a crosswalk frame whose center_x jumps more than this many BEV px",
    )
    parser.add_argument(
        "--corridor-crosswalk-option",
        choices=("a", "b"),
        default="b",
        help="[--bev-corridor] a=center-line virtual corridor, b=detected right boundary with fixed inward offset",
    )
    parser.add_argument(
        "--corridor-crosswalk-right-offset-px",
        type=float,
        default=90.0,
        help="[--bev-corridor] option B target distance leftward from the detected right boundary in BEV pixels",
    )
    parser.add_argument(
        "--bev-lookahead",
        type=float,
        default=0.45,
        help="BEV row ratio where lateral error is measured; defaults to --lookahead when provided, otherwise BEV default",
    )
    parser.add_argument(
        "--bev-lane-change-near-y",
        type=float,
        default=BevCorridorConfig.lane_change_near_y_ratio,
        help="near-field BEV row used to confirm the whole vehicle has reached the next lane",
    )
    parser.add_argument(
        "--bev-center-smooth",
        type=float,
        default=0.60,
        help="BEV lane-center EMA factor (0..1). 1.0 = no smoothing (target dot sits on the raw line, fast/jittery), lower = smoother/laggier",
    )
    parser.add_argument(
        "--bev-heading-smooth",
        type=float,
        default=0.30,
        help="BEV heading EMA factor (0..1). 1.0 = no smoothing, lower = smoother",
    )
    parser.add_argument(
        "--bev-path-smooth",
        type=float,
        default=BevCorridorConfig.path_smooth_alpha,
        help="EMA factor applied to the complete fitted BEV center path",
    )
    parser.add_argument(
        "--bev-path-max-step",
        type=float,
        default=BevCorridorConfig.path_max_step_px,
        help="maximum accepted lateral movement of one BEV path anchor per frame",
    )
    parser.add_argument("--bev-heading-gain", type=float, default=BevCorridorConfig.heading_gain)
    parser.add_argument(
        "--lookahead",
        type=float,
        default=None,
        help="compatibility alias for --bev-lookahead",
    )
    parser.add_argument(
        "--vehicle-center-offset",
        type=float,
        default=BevCorridorConfig.vehicle_center_x_offset_ratio,
        help="vehicle center x offset as frame width ratio; positive shifts the vehicle center right (camera is mounted left of the car centerline), so centered targets steer left",
    )
    parser.add_argument(
        "--pure-pursuit",
        action="store_true",
        help="steer via pure pursuit toward the BEV lookahead point (better on curves/S/hairpins) instead of the lateral+heading PID",
    )
    parser.add_argument(
        "--path-tracking",
        action="store_true",
        help="track the complete fitted BEV center path instead of one lookahead point",
    )
    parser.add_argument(
        "--path-lateral-gain",
        type=float,
        default=YoloLaneFollowerConfig.path_lateral_gain,
    )
    parser.add_argument(
        "--path-heading-gain",
        type=float,
        default=YoloLaneFollowerConfig.path_heading_gain,
    )
    parser.add_argument(
        "--path-derivative-gain",
        type=float,
        default=YoloLaneFollowerConfig.path_derivative_gain,
    )
    parser.add_argument(
        "--path-near-weight",
        type=float,
        default=YoloLaneFollowerConfig.path_near_weight,
    )
    parser.add_argument(
        "--path-far-weight",
        type=float,
        default=YoloLaneFollowerConfig.path_far_weight,
    )
    parser.add_argument(
        "--path-steering-rise-alpha",
        type=float,
        default=YoloLaneFollowerConfig.path_steering_rise_alpha,
    )
    parser.add_argument(
        "--path-steering-release-alpha",
        type=float,
        default=YoloLaneFollowerConfig.path_steering_release_alpha,
    )
    parser.add_argument(
        "--pp-gain",
        type=float,
        default=YoloLaneFollowerConfig.pure_pursuit_gain,
        help="[--pure-pursuit] steering units per radian of lookahead angle (larger = sharper)",
    )
    parser.add_argument(
        "--pp-full-angle",
        type=float,
        default=YoloLaneFollowerConfig.pure_pursuit_full_angle,
        help="[--pure-pursuit] lookahead angle (rad) at which curve-speed slowdown saturates",
    )
    parser.add_argument("--command-rate", type=float, default=20.0)
    parser.add_argument("--log-interval", type=float, default=0.5)
    parser.add_argument("--show-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", choices=("on", "off"), default="on")
    parser.add_argument("--record-dir", default="data/raw/drive_recordings")
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--record-fourcc", default="mp4v")
    parser.add_argument(
        "--record-debug",
        choices=("on", "off", "auto"),
        default="on",
        help="save the annotated debug screen. auto=follow --record (on when --record on), on=always, off=never",
    )
    parser.add_argument("--debug-dir", default="data/processed", help="output dir for the debug screen video")
    parser.add_argument(
        "--debug-output",
        default=None,
        help="exact output path for the annotated debug video",
    )
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
            if args.debug_output:
                self.debug_video_path = self._resolve(args.debug_output)
                self.debug_video_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            else:
                debug_dir = self._resolve(args.debug_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                self.debug_video_path = debug_dir / (
                    "%s_debug.mp4" % session_id
                )
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
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("camera could not be opened: %s" % camera)
    return cap


def read_startup_frame(cap: Any, attempts: int) -> tuple:
    """Drain startup frames so a newly selected UVC mode can take effect."""
    ok = False
    frame = None
    for _ in range(max(1, int(attempts))):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    return frame is not None, frame


def enforce_camera_contract(
    frame_shape: tuple,
    expected_width: int,
    expected_height: int,
    live_camera: bool,
    policy: str,
) -> None:
    if not live_camera or policy == "allow":
        return
    actual_height, actual_width = frame_shape[:2]
    if actual_width == expected_width and actual_height == expected_height:
        return
    raise RuntimeError(
        "camera resolution mismatch: requested %dx%d but received %dx%d; "
        "BEV calibration is invalid for this frame, so driving was refused. "
        "Check --camera, close other camera apps, and verify the mode with "
        "scripts/camera_check.py."
        % (expected_width, expected_height, actual_width, actual_height)
    )


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
    light_masks: tuple = (),
    light_observation: Optional[TrafficLightObservation] = None,
    light_stop_line_y: float = TrafficLightConfig.stop_line_y_ratio,
    lane_change_status: str = "off",
    obstacle_status: str = "off",
    obstacle_masks: tuple = (),
) -> Any:
    display = frame.copy()
    if mask_result is not None:
        overlay = display.copy()
        overlay[mask_result.mask > 0] = (0, 220, 80)
        display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0)
    obstacle_mask = fuse_masks(list(obstacle_masks))
    if obstacle_mask is not None:
        overlay = display.copy()
        overlay[obstacle_mask > 0] = (0, 0, 255)
        display = cv2.addWeighted(overlay, 0.55, display, 0.45, 0)
    light_mask = fuse_masks(list(light_masks))
    if light_mask is not None:
        overlay = display.copy()
        overlay[light_mask > 0] = (0, 165, 255)
        display = cv2.addWeighted(overlay, 0.45, display, 0.55, 0)

    height, width = display.shape[:2]
    stop_y = int(round(height * min(1.0, max(0.0, light_stop_line_y))))
    cv2.line(display, (0, stop_y), (width - 1, stop_y), (0, 80, 255), 2)
    cv2.putText(
        display,
        "CROSSWALK STOP CONTACT",
        (max(8, width - 270), max(22, stop_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 80, 255),
        2,
        cv2.LINE_AA,
    )
    if transformer is not None and bev_estimator is not None:
        # BEV mode: lane geometry is in BEV pixel coords. The camera is centered,
        # so the vehicle axis is the frame midline. Draw a single line joining the
        # center axis (car, bottom) to the target point mapped back from BEV.
        offset = getattr(getattr(bev_estimator, "config", None), "vehicle_center_x_offset_ratio", 0.0)
        vehicle_x = int(width * (0.5 + offset))
        cv2.line(display, (vehicle_x, height), (vehicle_x, 0), (255, 255, 0), 1)
        if lane.found:
            boundary_specs = (
                (
                    getattr(bev_estimator, "last_center_line_bev", ()),
                    (0, 200, 255),
                ),
                (
                    getattr(bev_estimator, "last_right_line_bev", ()),
                    (255, 200, 0),
                ),
            )
            for boundary, boundary_color in boundary_specs:
                if len(boundary) < 2:
                    continue
                projected = transformer.bev_to_frame(
                    boundary,
                    (height, width),
                ).astype("int32")
                cv2.polylines(
                    display,
                    [projected],
                    isClosed=False,
                    color=boundary_color,
                    thickness=2,
                )
            if lane.path_points:
                path = transformer.bev_to_frame(
                    lane.path_points,
                    (height, width),
                ).astype("int32")
                if len(path) >= 2:
                    cv2.polylines(
                        display,
                        [path],
                        isClosed=False,
                        color=(255, 0, 255),
                        thickness=4,
                    )
            target = transformer.bev_to_frame([(lane.center_x, lane.target_y)], (height, width))[0]
            tp = (int(target[0]), int(target[1]))
            if not lane.path_points:
                cv2.line(display, (vehicle_x, height - 1), tp, (0, 0, 255), 2)
            cv2.circle(display, tp, 9, (255, 255, 255), -1)
            cv2.circle(display, tp, 6, (0, 0, 255), -1)
            if lane.near_center_x is not None and lane.near_target_y is not None:
                near = transformer.bev_to_frame(
                    [(lane.near_center_x, lane.near_target_y)],
                    (height, width),
                )[0]
                cv2.circle(display, (int(near[0]), int(near[1])), 5, (255, 0, 255), -1)
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
        "near_err=%s" % (
            "n/a"
            if lane.near_lateral_error_norm is None
            else "%.3f" % lane.near_lateral_error_norm
        ),
        "path_points=%d" % len(lane.path_points),
        "fps=%.1f" % fps,
    ]
    if light_observation is not None:
        lines.append(
            "light=%s candidate=%s red=%d green=%d crosswalk_bottom=%.3f contact=%s stop=%s" % (
                light_observation.state.upper(),
                light_observation.candidate,
                light_observation.red_pixels,
                light_observation.green_pixels,
                light_observation.mask_bottom_y_ratio,
                "Y" if light_observation.contact else "N",
                "Y" if light_observation.stop_latched else "N",
            )
        )
    if lane_change_status != "off":
        lines.append("lane_change=%s" % lane_change_status.upper())
    if obstacle_status != "off":
        lines.append("obstacle=%s" % obstacle_status)
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


def draw_bev_mask_debug(
    cv2: Any,
    bev_mask: Any,
    lane: LaneGeometry,
    obstacle_mask: Any = None,
) -> Any:
    """Binary BEV road mask (white=road) with the vehicle center axis, a single
    line joining it to the target point, and the tracked target dot."""
    canvas = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
    if obstacle_mask is not None:
        canvas[obstacle_mask > 0] = (0, 0, 255)
    h, w = canvas.shape[:2]

    # vehicle center axis (forward is up), cyan
    cx = int(lane.vehicle_center_x) if lane is not None else w // 2
    cv2.line(canvas, (cx, 0), (cx, h), (255, 255, 0), 1)

    if lane is not None and lane.found:
        if lane.path_points:
            points = [
                (int(round(x)), int(round(y)))
                for x, y in lane.path_points
            ]
            for index in range(1, len(points)):
                cv2.line(
                    canvas,
                    points[index - 1],
                    points[index],
                    (255, 0, 255),
                    4,
                )
        target = (int(lane.center_x), int(lane.target_y))
        if not lane.path_points:
            cv2.line(canvas, (cx, h - 1), target, (0, 0, 255), 2)
        # tracked target point: red dot with a white outline for visibility
        cv2.circle(canvas, target, 7, (255, 255, 255), -1)
        cv2.circle(canvas, target, 5, (0, 0, 255), -1)
        if lane.near_center_x is not None and lane.near_target_y is not None:
            near = (int(lane.near_center_x), int(lane.near_target_y))
            cv2.circle(canvas, near, 5, (255, 0, 255), -1)
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
