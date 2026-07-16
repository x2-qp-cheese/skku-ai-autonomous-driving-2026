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
from ..planning.lane_change_test import LaneChangeTestConfig, LaneChangeTestController
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
    # The competition runtime has one lane path: per-class YOLO masks are warped
    # into BEV first, then the two-lane corridor is resolved there.
    transformer = BevTransformer(BevConfig())
    corridor_config = build_bev_corridor_config(args)
    corridor_estimator = BevCorridorLaneEstimator(corridor_config)
    follower_config = build_follower_config(args)
    follower = YoloLaneFollower(follower_config)
    command_filter = CommandSafetyFilter(args)
    lane_change_test = LaneChangeTestController(build_lane_change_config(args))
    obstacle_lane_change = UltrasonicObstacleLaneChanger(args)
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
    log_effective_config(args, corridor_config, follower_config)
    if recorder.enabled:
        LOG.info("recording raw video: %s", recorder.raw_video_path)
        if recorder.debug_video_path is not None:
            LOG.info("recording debug video: %s", recorder.debug_video_path)
    LOG.info("space: start/stop toggle | l: obstacle lane-change / return | q or esc: quit")

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

            if vehicle is not None and obstacle_lane_change.should_poll(now):
                serial_lines = vehicle.write_line("USF")
                obstacle_lane_change.mark_polled(now)
                event = obstacle_lane_change.update(serial_lines, now, lane_change_test, running)
                if event:
                    LOG.info("%s", event)
            elif vehicle is not None:
                event = obstacle_lane_change.update(vehicle.read_lines(), now, lane_change_test, running)
                if event:
                    LOG.info("%s", event)

            bev_mask = None
            light_masks = ()
            class_masks = segmenter.segment_class_masks(frame)
            light_masks = class_masks.light
            bev = warp_class_masks(transformer, class_masks)
            lane = corridor_estimator.estimate(bev)
            bev_mask = fuse_masks([*bev.center, *bev.side, *bev.lane, *bev.crosswalk])
            mask_result = corridor_mask_result(class_masks, corridor_estimator, frame.shape)
            lane_reliable = (
                mask_result is not None
                and "virtual" not in mask_result.class_name
                and lane.found
            )
            lane_change = lane_change_test.update(
                lane,
                corridor_estimator.last_lane_width_px,
                bev.shape[1],
                now,
                running,
                lane_reliable=lane_reliable,
            )
            lane = lane_change.lane
            if lane_change.offset_px and corridor_estimator.last_centerline_bev:
                corridor_estimator.last_centerline_bev = [
                    (x + lane_change.offset_px, y)
                    for x, y in corridor_estimator.last_centerline_bev
                ]
            planned_command = follower.plan(lane) if running else ControlCommand.stop("paused")
            planned_command = lane_change_test.apply_steering_assist(planned_command, lane_change)
            command = command_filter.apply(mask_result, lane, planned_command, running)
            command = lane_change_test.apply_speed_cap(command, lane_change_test.speed_cap_active(lane_change))
            stop_crosswalk_masks = (
                class_masks.crosswalk
                if class_masks.crosswalk_conf >= args.light_crosswalk_min_conf
                else ()
            )
            lane_change_blocks_light_stop = (
                args.light_stop_during_lane_change == "off"
                and lane_change.state not in ("off", "lane2", "completed")
            )
            if lane_change_blocks_light_stop:
                stop_crosswalk_masks = ()
            light_observation = traffic_light.update(frame, light_masks, stop_crosswalk_masks)
            if args.traffic_light == "on" and not lane_change_blocks_light_stop:
                command = traffic_light.apply(command, running)
            command = obstacle_lane_change.apply_safety(command, lane_change.state, now, running)

            if vehicle is not None and now - last_command_at >= 1.0 / args.command_rate:
                event = obstacle_lane_change.update(vehicle.send(command), now, lane_change_test, running)
                if event:
                    LOG.info("%s", event)
                last_command_at = now

            display = draw_debug(
                cv2, frame, mask_result, lane, command, running, fps,
                transformer, corridor_estimator,
                light_masks, light_observation,
                args.light_stop_line_y,
                lane_change.state,
                obstacle_lane_change.status_text(now),
            )
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
            if key == ord("l"):
                action, message = handle_lane_change_key(lane_change_test, running)
                if action.startswith("ignored"):
                    LOG.info(
                        "%s: mode=%s state=%s",
                        message,
                        args.lane_change_test,
                        lane_change_test.state,
                    )
                else:
                    LOG.info("%s", message)
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


def build_bev_corridor_config(args: argparse.Namespace) -> BevCorridorConfig:
    defaults = BevCorridorConfig()
    return BevCorridorConfig(
        lane_width_px=args.corridor_lane_width_px,
        lookahead_y_ratio=_first_set(args.bev_lookahead, args.lookahead, defaults.lookahead_y_ratio),
        vehicle_center_x_offset_ratio=args.vehicle_center_offset,
        heading_gain=args.bev_heading_gain,
        center_smooth_alpha=args.bev_center_smooth,
        heading_smooth_alpha=args.bev_heading_smooth,
        center_anchor=args.corridor_center_anchor == "on",
        max_center_jump_px=args.corridor_max_center_jump,
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
        pure_pursuit=args.pure_pursuit,
        pure_pursuit_gain=args.pp_gain,
        pure_pursuit_full_angle=args.pp_full_angle,
    )


def build_lane_change_config(args: argparse.Namespace) -> LaneChangeTestConfig:
    steering_cap = args.lane_change_steering_cap
    if steering_cap is None:
        steering_cap = args.max_steering
    return LaneChangeTestConfig(
        mode=args.lane_change_test,
        trigger_seconds=args.lane_change_trigger_seconds,
        transition_seconds=args.lane_change_transition_seconds,
        hold_seconds=args.lane_change_hold_seconds,
        max_straight_heading=args.lane_change_max_heading,
        speed_cap=args.lane_change_speed_cap,
        steering_min=args.lane_change_steering_min,
        steering_boost=args.lane_change_steering_boost,
        steering_cap=steering_cap,
        settle_seconds=args.lane_change_settle_seconds,
        steering_override=args.lane_change_steering_override == "on",
        stable_lateral_error=args.lane_change_stable_lateral_error,
        stable_heading_error=args.lane_change_stable_heading_error,
        stable_required_frames=args.lane_change_stable_frames,
        stabilize_timeout_seconds=args.lane_change_stabilize_timeout,
        allow_virtual_stabilize=args.lane_change_allow_virtual_stabilize == "on",
    )


def handle_lane_change_key(
    lane_change_test: LaneChangeTestController,
    running: bool,
) -> tuple:
    if not running:
        return "ignored_not_running", "lane-change key ignored while paused"

    if lane_change_test.state in ("lane2", "completed"):
        if lane_change_test.request("keyboard_obstacle"):
            return "request", "lane-change request armed: waiting for a straight frame"
        return "ignored_request", "lane-change request ignored"

    if lane_change_test.state in ("changing_to_lane1", "lane1"):
        if lane_change_test.request_return("keyboard_clear"):
            return "return", "lane-change return requested: waiting for a straight frame"
        return "ignored_return", "lane-change return ignored"

    return "ignored_busy", "lane-change key ignored while transition is busy"


def log_effective_config(
    args: argparse.Namespace,
    corridor_config: BevCorridorConfig,
    follower_config: YoloLaneFollowerConfig,
) -> None:
    LOG.info(
        "lane pipeline=bev-corridor lookahead=%.2f sample=%.2f..%.2f vehicle_offset=%.3f center_smooth=%.2f heading_smooth=%.2f heading_gain=%.2f",
        corridor_config.lookahead_y_ratio,
        corridor_config.sample_top_y_ratio,
        corridor_config.sample_bottom_y_ratio,
        corridor_config.vehicle_center_x_offset_ratio,
        corridor_config.center_smooth_alpha,
        corridor_config.heading_smooth_alpha,
        corridor_config.heading_gain,
    )

    lane_lost_release = follower_config.lane_lost_steering_release_rate_limit
    if lane_lost_release is None:
        lane_lost_release = max(
            follower_config.min_steering_rate_limit,
            follower_config.steering_release_rate_limit,
        )
    LOG.info(
        "control speed=%d min_curve=%d max=%d max_steer=%d kp_lat=%.1f kd_lat=%.1f kp_head=%.1f kd_head=%.1f scale=%.2f..%.2f lateral_priority=%.2f curve_alpha=%.2f center_lock=%s threshold=%.2f min=%d lane_lost_release=%d/frame",
        follower_config.base_speed,
        follower_config.min_curve_speed,
        follower_config.max_speed,
        follower_config.max_steering,
        follower_config.kp_lateral,
        follower_config.kd_lateral,
        follower_config.kp_heading,
        follower_config.kd_heading,
        follower_config.straight_steering_scale,
        follower_config.curve_steering_scale,
        follower_config.lateral_priority_threshold,
        follower_config.curve_strength_alpha,
        "on" if follower_config.center_lock_enabled else "off",
        follower_config.center_lock_error_threshold,
        follower_config.center_lock_min_steering,
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
        "traffic_light=%s confirm_frames=%d stop_line_y=%.3f contact_min_row_width=%.2f stop_during_lane_change=%s",
        args.traffic_light,
        args.light_confirm_frames,
        args.light_stop_line_y,
        args.light_contact_min_row_width_ratio,
        args.light_stop_during_lane_change,
    )
    LOG.info(
        "lane_change_test=%s trigger=%.1fs transition=%.1fs settle=%.1fs stable_err=%.2f stable_head=%.2f stable_frames=%d stabilize_timeout=%.1fs hold=%.1fs max_heading=%.3f speed_cap=%d steer_min=%d steer_boost=%d steer_cap=%s override=%s",
        args.lane_change_test,
        args.lane_change_trigger_seconds,
        args.lane_change_transition_seconds,
        args.lane_change_settle_seconds,
        args.lane_change_stable_lateral_error,
        args.lane_change_stable_heading_error,
        args.lane_change_stable_frames,
        args.lane_change_stabilize_timeout,
        args.lane_change_hold_seconds,
        args.lane_change_max_heading,
        args.lane_change_speed_cap,
        args.lane_change_steering_min,
        args.lane_change_steering_boost,
        args.lane_change_steering_cap if args.lane_change_steering_cap is not None else args.max_steering,
        args.lane_change_steering_override,
    )
    LOG.info(
        "obstacle_lane_change=%s trigger=%.0fmm clear=%.0fmm min_valid=%.0fmm speed_cap=%d stop=%.0fmm poll=%.2fs max_age=%.2fs cooldown=%.1fs front_mode=%s",
        args.obstacle_lane_change,
        args.obstacle_trigger_mm,
        args.obstacle_clear_mm,
        args.obstacle_min_mm,
        args.obstacle_speed_cap,
        args.obstacle_stop_mm,
        args.obstacle_poll_interval,
        args.obstacle_max_age,
        args.obstacle_cooldown_seconds,
        args.obstacle_front_mode,
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
        return ControlCommand(
            speed=self.fixed_speed_value,
            steering=command.steering,
            brake=command.brake,
            reason=self._append_reason(command.reason, "fixed_speed"),
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


class UltrasonicObstacleLaneChanger:
    def __init__(self, args: argparse.Namespace):
        self.enabled = args.obstacle_lane_change == "on"
        self.trigger_mm = float(args.obstacle_trigger_mm)
        self.clear_mm = float(args.obstacle_clear_mm)
        self.min_mm = float(args.obstacle_min_mm)
        self.speed_cap = int(args.obstacle_speed_cap)
        self.stop_mm = float(args.obstacle_stop_mm)
        self.slow_hold_seconds = float(args.obstacle_slow_hold_seconds)
        self.poll_interval = float(args.obstacle_poll_interval)
        self.max_age = float(args.obstacle_max_age)
        self.cooldown_seconds = float(args.obstacle_cooldown_seconds)
        self.front_mode = args.obstacle_front_mode
        self.values = {"FR": 0, "FL": 0, "SR": 0, "SL": 0}
        self.last_read_at: Optional[float] = None
        self._next_poll_at = 0.0
        self._was_obstacle = False
        self._last_trigger_at = -1e9
        self._last_obstacle_at = -1e9

    def should_poll(self, now: float) -> bool:
        return self.enabled and now >= self._next_poll_at

    def mark_polled(self, now: float) -> None:
        self._next_poll_at = now + max(0.05, self.poll_interval)

    def update(
        self,
        lines: list,
        now: float,
        lane_change_test: LaneChangeTestController,
        running: bool,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        for line in lines:
            parsed = parse_ultrasonic_line(line)
            if parsed:
                self.values.update(parsed)
                self.last_read_at = now

        if not running:
            self._was_obstacle = False
            return None

        obstacle = self._obstacle_present(now)
        clear = self._obstacle_clear(now)
        event = None
        if clear:
            self._was_obstacle = False
        if (
            obstacle
            and not self._was_obstacle
            and now - self._last_trigger_at >= max(0.0, self.cooldown_seconds)
        ):
            event = self._trigger_lane_change(lane_change_test, now)
        if obstacle and (event is not None or self._state_can_trigger(lane_change_test.state)):
            self._was_obstacle = True
            self._last_obstacle_at = now
        elif obstacle:
            self._last_obstacle_at = now
        return event

    def apply_safety(
        self,
        command: ControlCommand,
        lane_change_state: str,
        now: float,
        running: bool,
    ) -> ControlCommand:
        if not self.enabled or not running or command.brake:
            return command
        if self._emergency_close(now):
            return ControlCommand.stop(self._append_reason(command.reason, "obstacle_stop"))
        if not self._slow_active(lane_change_state, now):
            return command
        cap = max(0, int(self.speed_cap))
        speed = self._clip_abs(command.speed, cap)
        if speed == command.speed:
            return command
        return ControlCommand(
            speed=speed,
            steering=command.steering,
            brake=False,
            reason=self._append_reason(command.reason, "obstacle_cap"),
        )

    def status_text(self, now: float) -> str:
        if not self.enabled:
            return "off"
        age = 0.0 if self.last_read_at is None else max(0.0, now - self.last_read_at)
        if self._emergency_close(now):
            state = "STOP"
        elif self._obstacle_present(now):
            state = "DETECT"
        else:
            state = "clear"
        return "FR=%d FL=%d %s age=%.1fs" % (
            self.values.get("FR", 0),
            self.values.get("FL", 0),
            state,
            age,
        )

    def _trigger_lane_change(self, lane_change_test: LaneChangeTestController, now: float) -> Optional[str]:
        state = lane_change_test.state
        source = "ultrasonic_obstacle"
        if state in ("lane2", "completed"):
            if lane_change_test.request(source):
                self._last_trigger_at = now
                return "ultrasonic obstacle: lane2 -> lane1 requested (%s)" % self.status_text(now)
        if state == "lane1":
            if lane_change_test.request_return(source):
                self._last_trigger_at = now
                return "ultrasonic obstacle: lane1 -> lane2 requested (%s)" % self.status_text(now)
        return None

    @staticmethod
    def _state_can_trigger(state: str) -> bool:
        return state in ("lane2", "completed", "lane1")

    def _front_values(self) -> list:
        return [int(self.values.get("FR", 0)), int(self.values.get("FL", 0))]

    def _fresh(self, now: float) -> bool:
        if self.last_read_at is None:
            return False
        return now - self.last_read_at <= max(0.05, self.max_age)

    def _valid_detection(self, value: int) -> bool:
        return self.min_mm <= value <= self.trigger_mm

    def _valid_front_value(self, value: int) -> bool:
        return value >= self.min_mm

    def _clear_value(self, value: int) -> bool:
        return value == 0 or value < self.min_mm or value >= self.clear_mm

    def _obstacle_present(self, now: float) -> bool:
        if not self._fresh(now):
            return False
        values = self._front_values()
        detections = [self._valid_detection(value) for value in values]
        if self.front_mode == "both":
            return all(detections)
        return any(detections)

    def _obstacle_clear(self, now: float) -> bool:
        if not self._fresh(now):
            return True
        return all(self._clear_value(value) for value in self._front_values())

    def _emergency_close(self, now: float) -> bool:
        if not self._fresh(now) or self.stop_mm <= 0:
            return False
        return any(
            self._valid_front_value(value) and value <= self.stop_mm
            for value in self._front_values()
        )

    def _slow_active(self, lane_change_state: str, now: float) -> bool:
        if lane_change_state in (
            "armed",
            "changing_to_lane1",
            "stabilizing_lane1",
            "changing_to_lane2",
            "stabilizing_lane2",
        ):
            return True
        if self._obstacle_present(now):
            return True
        return now - self._last_obstacle_at <= max(0.0, self.slow_hold_seconds)

    @staticmethod
    def _clip_abs(value: int, limit: int) -> int:
        if limit <= 0:
            return 0
        return max(-limit, min(limit, int(value)))

    @staticmethod
    def _append_reason(reason: str, suffix: str) -> str:
        return "%s:%s" % (reason, suffix) if reason else suffix


def parse_ultrasonic_line(line: str) -> dict:
    if not line.startswith("US "):
        return {}
    values = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key not in ("FR", "FL", "SR", "SL"):
            continue
        try:
            values[key] = int(value)
        except ValueError:
            continue
    return values


def parse_args(argv: Optional[list]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 segmentation autonomous driving runtime")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="trained YOLO segmentation model path")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, 0, cuda, ...")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--traffic-light", choices=("on", "off"), default="on",
        help="compare red/green pixels inside YOLO 'light' masks and latch stop on red until green is confirmed",
    )
    parser.add_argument("--light-confirm-frames", type=int, default=TrafficLightConfig.confirm_frames)
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
    parser.add_argument(
        "--light-stop-during-lane-change",
        choices=("on", "off"),
        default="off",
        help="allow traffic-light braking while a keyboard/manual lane-change is active; off prevents far-red false stops during obstacle bypass",
    )
    parser.add_argument("--light-min-saturation", type=int, default=TrafficLightConfig.min_saturation)
    parser.add_argument("--light-min-value", type=int, default=TrafficLightConfig.min_value)
    parser.add_argument("--light-min-color-pixels", type=int, default=TrafficLightConfig.min_color_pixels)
    parser.add_argument("--light-min-color-ratio", type=float, default=TrafficLightConfig.min_color_ratio)
    parser.add_argument("--light-dominance-ratio", type=float, default=TrafficLightConfig.dominance_ratio)
    parser.add_argument(
        "--light-stop-line-y",
        type=float,
        default=TrafficLightConfig.stop_line_y_ratio,
        help="frame y ratio of the virtual crosswalk contact line; confirmed RED brakes when the crosswalk mask bottom reaches this line",
    )
    parser.add_argument("--camera", default="0", help="camera index or video path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--serial-port", default=None, help="Arduino port, e.g. COM3 or /dev/cu.usbmodemXXXX")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--ready-timeout", type=float, default=3.0)
    parser.add_argument("--no-serial", action="store_true", help="run without Arduino output")
    parser.add_argument("--speed", type=int, default=105)
    parser.add_argument(
        "--fixed-speed",
        choices=("on", "off"),
        default="off",
        help="force every non-brake driving command to --speed after steering safety filters",
    )
    parser.add_argument("--max-speed", type=int, default=170)
    parser.add_argument("--min-curve-speed", type=int, default=60)
    parser.add_argument(
        "--lane-change-test",
        choices=("off", "keyboard", "manual", "timed", "external"),
        default="keyboard",
        help="straight-section 2->1->2 controller; keyboard uses L for obstacle/clear, manual uses L plus timed return, timed uses a delay, external waits for detector requests",
    )
    parser.add_argument("--lane-change-trigger-seconds", type=float, default=LaneChangeTestConfig.trigger_seconds)
    parser.add_argument("--lane-change-transition-seconds", type=float, default=LaneChangeTestConfig.transition_seconds)
    parser.add_argument("--lane-change-hold-seconds", type=float, default=LaneChangeTestConfig.hold_seconds)
    parser.add_argument(
        "--lane-change-max-heading",
        type=float,
        default=LaneChangeTestConfig.max_straight_heading,
        help="start each transition only when absolute BEV heading error is below this straightness threshold",
    )
    parser.add_argument("--lane-change-speed-cap", type=int, default=LaneChangeTestConfig.speed_cap)
    parser.add_argument(
        "--lane-change-steering-min",
        type=int,
        default=LaneChangeTestConfig.steering_min,
        help="minimum absolute steering forced while actively changing lanes; 0 disables the minimum",
    )
    parser.add_argument(
        "--lane-change-steering-boost",
        type=int,
        default=LaneChangeTestConfig.steering_boost,
        help="extra steering added in the lane-change direction while actively changing lanes",
    )
    parser.add_argument(
        "--lane-change-steering-cap",
        type=int,
        default=None,
        help="absolute steering cap for lane-change assist; default follows --max-steering",
    )
    parser.add_argument(
        "--lane-change-settle-seconds",
        type=float,
        default=LaneChangeTestConfig.settle_seconds,
        help="keep lane-change steering priority for this long after the timed lane shift finishes, before resuming normal lane following",
    )
    parser.add_argument(
        "--lane-change-steering-override",
        choices=("on", "off"),
        default="off",
        help="on = ignore normal lane-follow steering and command the lane-change direction at the lane-change steering cap while changing/settling",
    )
    parser.add_argument(
        "--lane-change-stable-lateral-error",
        type=float,
        default=LaneChangeTestConfig.stable_lateral_error,
        help="after a lane change, require absolute lateral error below this value before accepting the new lane as stable",
    )
    parser.add_argument(
        "--lane-change-stable-heading-error",
        type=float,
        default=LaneChangeTestConfig.stable_heading_error,
        help="after a lane change, require absolute heading error below this value before accepting the new lane as stable",
    )
    parser.add_argument(
        "--lane-change-stable-frames",
        type=int,
        default=LaneChangeTestConfig.stable_required_frames,
        help="consecutive stable frames required before leaving stabilizing_lane1/stabilizing_lane2",
    )
    parser.add_argument(
        "--lane-change-stabilize-timeout",
        type=float,
        default=LaneChangeTestConfig.stabilize_timeout_seconds,
        help="maximum seconds to stay in stabilizing state before falling back to the target lane state; 0 disables timeout",
    )
    parser.add_argument(
        "--lane-change-allow-virtual-stabilize",
        choices=("on", "off"),
        default="off",
        help="off requires a non-virtual YOLO mask before a new lane can be marked stable",
    )
    parser.add_argument(
        "--obstacle-lane-change",
        choices=("on", "off"),
        default="on",
        help="poll front ultrasonic sensors and request lane changes automatically when an obstacle is within --obstacle-trigger-mm",
    )
    parser.add_argument(
        "--obstacle-trigger-mm",
        type=float,
        default=850.0,
        help="front ultrasonic distance at or below this value requests the next lane change",
    )
    parser.add_argument(
        "--obstacle-clear-mm",
        type=float,
        default=950.0,
        help="front ultrasonic distance at or above this value arms the detector for the next obstacle",
    )
    parser.add_argument(
        "--obstacle-min-mm",
        type=float,
        default=50.0,
        help="ignore front ultrasonic readings below this value as timeout/noise, e.g. 0mm or a stuck 9mm sensor",
    )
    parser.add_argument(
        "--obstacle-speed-cap",
        type=int,
        default=90,
        help="speed cap while an obstacle is detected, lane-change is armed, or lane-change is actively transitioning",
    )
    parser.add_argument(
        "--obstacle-stop-mm",
        type=float,
        default=300.0,
        help="force STOP if any valid front ultrasonic reading is at or below this distance; 0 disables the emergency stop",
    )
    parser.add_argument(
        "--obstacle-slow-hold-seconds",
        type=float,
        default=0.8,
        help="keep --obstacle-speed-cap for this long after the last front obstacle detection",
    )
    parser.add_argument(
        "--obstacle-poll-interval",
        type=float,
        default=0.20,
        help="seconds between USF front-ultrasonic polls while driving runtime is connected to Arduino",
    )
    parser.add_argument(
        "--obstacle-max-age",
        type=float,
        default=0.70,
        help="maximum age in seconds of ultrasonic readings before they are treated as stale/clear",
    )
    parser.add_argument(
        "--obstacle-cooldown-seconds",
        type=float,
        default=1.5,
        help="minimum time between automatic obstacle lane-change requests",
    )
    parser.add_argument(
        "--obstacle-front-mode",
        choices=("any", "both"),
        default="any",
        help="any=FR or FL can trigger; both=FR and FL must both be valid obstacle readings",
    )
    parser.add_argument("--max-steering", type=int, default=120)
    parser.add_argument("--steering-rate-limit", type=int, default=110)
    parser.add_argument("--min-steering-rate-limit", type=int, default=40)
    parser.add_argument("--steering-release-rate-limit", type=int, default=22)
    parser.add_argument("--kp-lateral", type=float, default=190.0)
    parser.add_argument("--kd-lateral", type=float, default=45.0)
    parser.add_argument("--kp-heading", type=float, default=12.0)
    parser.add_argument("--kd-heading", type=float, default=4.0)
    parser.add_argument("--speed-curve-slowdown", type=int, default=70)
    parser.add_argument(
        "--lateral-priority-threshold",
        type=float,
        default=0.10,
        help="ignore conflicting heading only when lateral error is above this threshold",
    )
    parser.add_argument(
        "--curve-strength-alpha",
        type=float,
        default=0.35,
        help="curve strength smoothing alpha; lower keeps curve state longer",
    )
    parser.add_argument("--straight-steering-scale", type=float, default=0.45)
    parser.add_argument("--curve-steering-scale", type=float, default=1.45)
    parser.add_argument("--center-recovery-error-threshold", type=float, default=0.14)
    parser.add_argument("--center-recovery-steering-boost", type=float, default=2.0)
    parser.add_argument("--center-recovery-min-steering", type=int, default=85)
    parser.add_argument("--center-recovery-rate-limit", type=int, default=120)
    parser.add_argument("--center-recovery-max-speed", type=int, default=50)
    parser.add_argument(
        "--center-lock",
        choices=("on", "off"),
        default="off",
        help="force at least --center-lock-min-steering toward lane center when lateral error exceeds --center-lock-error-threshold",
    )
    parser.add_argument(
        "--center-lock-error-threshold",
        type=float,
        default=0.05,
        help="normalized lateral error that activates center-lock steering",
    )
    parser.add_argument(
        "--center-lock-min-steering",
        type=int,
        default=90,
        help="minimum absolute steering while center-lock is active",
    )
    parser.add_argument(
        "--lane-lost-hold-frames",
        type=int,
        default=20,
        help="keep the last steering/speed for up to this many frames when the lane is not detected (e.g. crosswalks) before stopping",
    )
    parser.add_argument(
        "--lane-lost-steering-release-rate-limit",
        type=int,
        default=None,
        help="during lane-lost hold, release cached steering toward 0 by this many units/frame; default=max(min-steering-rate-limit, steering-release-rate-limit), 0 keeps the old cached steering",
    )
    parser.add_argument(
        "--lane-lost-speed-cap",
        type=int,
        default=180,
        help="cap speed while the planner is holding a lane-lost command",
    )
    parser.add_argument(
        "--virtual-lane-max-steering",
        type=int,
        default=90,
        help="cap absolute steering when YOLO uses a virtual lane fallback",
    )
    parser.add_argument(
        "--virtual-lane-speed-cap",
        type=int,
        default=180,
        help="cap speed when YOLO uses a virtual lane fallback",
    )
    parser.add_argument(
        "--virtual-lane-warmup-frames",
        type=int,
        default=3,
        help="for the first N consecutive virtual-lane frames, reuse the last reliable steering before applying the virtual cap",
    )
    parser.add_argument(
        "--virtual-lane-steering-blend",
        type=float,
        default=0.20,
        help="after warmup, blend virtual-lane steering with the last reliable steering. 0=ignore virtual steering, 1=trust raw virtual steering",
    )
    parser.add_argument(
        "--virtual-lane-max-steering-step",
        type=int,
        default=20,
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
        default=3,
        help="require this many real non-virtual lane frames before trusting virtual-lane steering; before that, hold the last reliable command",
    )
    parser.add_argument(
        "--virtual-lane-bootstrap-speed-cap",
        type=int,
        default=150,
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
        default=BevCorridorConfig.max_center_jump_px,
        help="[--bev-corridor] reject and coast a frame whose lookahead center_x jumps more than this many BEV px (lower = smoother, more likely to briefly coast on real fast curves)",
    )
    parser.add_argument(
        "--corridor-max-coast-frames",
        type=int,
        default=BevCorridorConfig.max_coast_frames,
        help="[--bev-corridor] hold the last good geometry for at most this many rejected frames before declaring the lane lost",
    )
    parser.add_argument(
        "--corridor-max-width-jump",
        type=float,
        default=BevCorridorConfig.max_width_jump_px,
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
        default=BevCorridorConfig.centerline_bias,
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
        default=BevCorridorConfig.crosswalk_center_smooth_alpha,
        help="[--bev-corridor] center-x EMA factor while a crosswalk is visible (lower = steadier)",
    )
    parser.add_argument(
        "--corridor-crosswalk-max-center-jump",
        type=float,
        default=BevCorridorConfig.crosswalk_max_center_jump_px,
        help="[--bev-corridor] reject and coast a crosswalk frame whose center_x jumps more than this many BEV px",
    )
    parser.add_argument(
        "--corridor-crosswalk-option",
        choices=("a", "b"),
        default=BevCorridorConfig.crosswalk_option,
        help="[--bev-corridor] a=center-line virtual corridor, b=detected right boundary with fixed inward offset",
    )
    parser.add_argument(
        "--corridor-crosswalk-right-offset-px",
        type=float,
        default=BevCorridorConfig.crosswalk_right_offset_px,
        help="[--bev-corridor] option B target distance leftward from the detected right boundary in BEV pixels",
    )
    parser.add_argument(
        "--bev-lookahead",
        type=float,
        default=None,
        help="BEV row ratio where lateral error is measured; defaults to --lookahead when provided, otherwise BEV default",
    )
    parser.add_argument(
        "--bev-center-smooth",
        type=float,
        default=BevCorridorConfig.center_smooth_alpha,
        help="BEV lane-center EMA factor (0..1). 1.0 = no smoothing (target dot sits on the raw line, fast/jittery), lower = smoother/laggier",
    )
    parser.add_argument(
        "--bev-heading-smooth",
        type=float,
        default=BevCorridorConfig.heading_smooth_alpha,
        help="BEV heading EMA factor (0..1). 1.0 = no smoothing, lower = smoother",
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
        help="vehicle center x offset as frame width ratio; positive shifts the vehicle center right (camera is mounted left of the car centerline), so centered targets steer left. Default matches the last good run (~0.585 frame x)",
    )
    parser.add_argument(
        "--pure-pursuit",
        action="store_true",
        help="steer via pure pursuit toward the BEV lookahead point (better on curves/S/hairpins) instead of the lateral+heading PID",
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
    light_masks: tuple = (),
    light_observation: Optional[TrafficLightObservation] = None,
    light_stop_line_y: float = TrafficLightConfig.stop_line_y_ratio,
    lane_change_status: str = "off",
    obstacle_status: str = "off",
) -> Any:
    display = frame.copy()
    if mask_result is not None:
        overlay = display.copy()
        overlay[mask_result.mask > 0] = (0, 220, 80)
        display = cv2.addWeighted(overlay, 0.28, display, 0.72, 0)
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
            target = transformer.bev_to_frame([(lane.center_x, lane.target_y)], (height, width))[0]
            tp = (int(target[0]), int(target[1]))
            cv2.line(display, (vehicle_x, height - 1), tp, (0, 0, 255), 2)
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
