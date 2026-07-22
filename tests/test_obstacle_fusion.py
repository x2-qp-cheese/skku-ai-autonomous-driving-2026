import unittest
from dataclasses import replace

import numpy as np

from skku_autocar.estimation.lane_geometry import LaneGeometry
from skku_autocar.perception.yolo_lane import (
    YoloLaneConfig,
    YoloLaneMask,
    YoloLaneSegmenter,
)
from skku_autocar.planning.lane_change import LaneChangeConfig, LaneChangeController
from skku_autocar.planning.obstacle_fusion import (
    FramePathGeometry,
    ObstacleFusionConfig,
    ObstacleFusionPlanner,
)
from skku_autocar.runtime.obstacle_mode import (
    ObstacleDriveMode,
    build_lane_change_config,
    build_obstacle_fusion_config,
    lane_change_geometry_reliable,
)
from skku_autocar.runtime.yolo_drive_app import (
    parse_args,
)
from skku_autocar.sensors.ultrasonic import SENSOR_KEYS, UltrasonicSnapshot
from skku_autocar.types import ControlCommand


SHAPE = (100, 200)
CENTERLINE = [(140.0, float(y)) for y in range(0, 100, 5)]


def obstacle_mask(x0, x1, y0, y1):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def projected_paths():
    return FramePathGeometry(
        lane1=tuple((80.0, float(y)) for y in range(0, 100, 5)),
        lane2=tuple((140.0, float(y)) for y in range(0, 100, 5)),
    )


def lane():
    return LaneGeometry(
        found=True,
        center_x=140.0,
        vehicle_center_x=100.0,
        target_y=55.0,
        lateral_error_px=40.0,
        lateral_error_norm=0.4,
        heading_error=0.0,
        confidence=1.0,
        reason="corridor",
        height=100.0,
    )


def controller():
    return LaneChangeController(
        LaneChangeConfig(
            mode="external",
            target_lane_width_px=60.0,
            stable_required_frames=0,
        )
    )


def ultrasound(fc=800, fr=900, fl=900, sr=500, sl=500):
    return UltrasonicSnapshot(
        fc=fc,
        fr=fr,
        fl=fl,
        sr=sr,
        sl=sl,
        fresh_keys=SENSOR_KEYS,
        age_seconds=0.0,
    )


def planner(**overrides):
    values = dict(
        lane_width_px=60.0,
        visual_trigger_y_ratio=0.55,
        target_block_y_ratio=0.55,
        visual_emergency_y_ratio=0.88,
        path_half_width_px=24.0,
        min_path_overlap_ratio=0.2,
        visual_confirm_frames=2,
        visual_clear_frames=2,
        ultrasonic_trigger_mm=1000.0,
        ultrasonic_clear_mm=1150.0,
        ultrasonic_stop_mm=300.0,
        side_clearance_mm=300.0,
        cooldown_seconds=0.0,
        speed_cap=70,
    )
    values.update(overrides)
    return ObstacleFusionPlanner(ObstacleFusionConfig(**values))


class ObstacleFusionPlannerTest(unittest.TestCase):
    def test_obstacle_class_is_kept_separate_from_lane_classes(self):
        segmenter = object.__new__(YoloLaneSegmenter)
        segmenter.config = YoloLaneConfig()
        segmenter.names = {0: "lane-center", 1: "obstacle"}

        self.assertEqual(segmenter._class_kind("obstacle"), "obstacle")
        self.assertTrue(segmenter.has_obstacle_class)

    def test_visual_and_ultrasonic_agreement_requests_lane_change(self):
        fusion = planner()
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)

        first = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True
        )
        second = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.1, True
        )

        self.assertIsNone(first)
        self.assertIn("lane2 -> lane1", second)
        self.assertEqual(change.state, "armed")
        self.assertEqual(change.request_source, "obstacle_fusion")

    def test_frame_obstacle_caps_speed_before_it_enters_bev_roi(self):
        fusion = planner(
            visual_trigger_y_ratio=0.80,
            frame_visual_trigger_y_ratio=0.15,
            visual_slowdown_enabled=True,
            approach_speed_cap=90,
        )
        change = controller()
        frame_mask = obstacle_mask(132, 149, 16, 30)

        fusion.update(
            [],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            UltrasonicSnapshot(),
            1.0,
            True,
            frame_obstacle_masks=[frame_mask],
            frame_paths=projected_paths(),
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"),
            change.state,
            True,
        )

        self.assertTrue(fusion.observation.visual_detected)
        self.assertEqual(guarded.speed, 90)
        self.assertFalse(guarded.brake)

    def test_visual_tracks_at_full_speed_then_slows_and_changes_at_two_meters(self):
        fusion = planner(
            visual_confirm_frames=1,
            ultrasonic_trigger_mm=2000.0,
            ultrasonic_clear_mm=2300.0,
            ttc_trigger_seconds=0.0,
            speed_cap=135,
            visual_slowdown_enabled=False,
        )
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)

        far_event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=2500, fr=2520, fl=2540), 1.0, True
        )
        far_command = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )
        near_event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=2000, fr=2020, fl=2040), 1.1, True
        )
        near_command = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertIsNone(far_event)
        self.assertTrue(fusion.observation.plan_ready)
        self.assertEqual(fusion.observation.planned_target_lane, 1)
        self.assertEqual(far_command.speed, 255)
        self.assertIn("lane2 -> lane1", near_event)
        self.assertEqual(change.state, "armed")
        self.assertEqual(near_command.speed, 135)

    def test_frame_visual_and_early_range_commit_before_old_one_meter_gate(self):
        fusion = planner(
            frame_visual_trigger_y_ratio=0.15,
            ultrasonic_trigger_mm=1600.0,
            ultrasonic_clear_mm=1800.0,
        )
        change = controller()
        frame_mask = obstacle_mask(132, 149, 16, 30)

        fusion.update(
            [],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(fc=1500, fr=1540, fl=1560),
            1.0,
            True,
            frame_obstacle_masks=[frame_mask],
            frame_paths=projected_paths(),
        )
        event = fusion.update(
            [],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(fc=1480, fr=1520, fl=1540),
            1.1,
            True,
            frame_obstacle_masks=[frame_mask],
            frame_paths=projected_paths(),
        )

        self.assertIn("lane2 -> lane1", event)
        self.assertEqual(change.state, "armed")

    def test_two_fresh_front_sensors_are_enough_for_range_quorum(self):
        fusion = planner(
            ultrasonic_trigger_mm=1600.0,
            ultrasonic_clear_mm=1800.0,
            min_front_sensors=2,
        )
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)
        partial = UltrasonicSnapshot(
            fc=1400,
            fr=1450,
            fl=200,
            sl=500,
            fresh_keys=("FC", "FR", "SL"),
            age_seconds=0.0,
        )

        fusion.update([mask], SHAPE, CENTERLINE, lane(), change, partial, 1.0, True)
        event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, partial, 1.1, True
        )

        self.assertIn("lane2 -> lane1", event)
        self.assertEqual(fusion.observation.front_sensor_count, 2)
        self.assertEqual(fusion.observation.front_mm, 1400)

    def test_ttc_can_confirm_before_distance_threshold(self):
        fusion = planner(
            ultrasonic_trigger_mm=1600.0,
            ultrasonic_clear_mm=1800.0,
            ttc_trigger_seconds=1.8,
            min_closing_rate_mm_s=120.0,
        )
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)

        fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=2200, fr=2220, fl=2240), 1.0, True
        )
        event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=1800, fr=1820, fl=1840), 1.2, True
        )
        fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=1700, fr=1720, fl=1740), 1.3, True
        )

        self.assertIn("lane2 -> lane1", event)
        self.assertLessEqual(fusion.observation.ttc_seconds, 1.8)

    def test_range_confirmation_requires_consecutive_distance_frames(self):
        fusion = planner(range_confirm_frames=2)
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)
        stale = UltrasonicSnapshot()

        fusion.update([mask], SHAPE, CENTERLINE, lane(), change, stale, 1.0, True)
        fusion.update([mask], SHAPE, CENTERLINE, lane(), change, stale, 1.1, True)
        first_range = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.2, True
        )
        second_range = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.3, True
        )

        self.assertIsNone(first_range)
        self.assertIn("lane2 -> lane1", second_range)

    def test_low_confidence_visual_only_tracks_at_full_speed(self):
        fusion = planner(visual_action_confidence=0.75)
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)

        fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True,
            obstacle_confidence=0.69,
        )
        event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.1, True,
            obstacle_confidence=0.69,
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertIsNone(event)
        self.assertTrue(fusion.observation.visual_detected)
        self.assertFalse(fusion.observation.visual_actionable)
        self.assertFalse(fusion.observation.visual_confirmed)
        self.assertEqual(guarded.speed, 255)

    def test_close_low_confidence_visual_stops_instead_of_changing_lane(self):
        fusion = planner(
            visual_action_confidence=0.75,
            frame_visual_trigger_y_ratio=0.15,
        )
        change = controller()
        close_mask = obstacle_mask(132, 149, 75, 96)

        fusion.update(
            [], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True,
            frame_obstacle_masks=[close_mask],
            frame_paths=projected_paths(),
            obstacle_confidence=0.69,
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertFalse(fusion.observation.visual_actionable)
        self.assertTrue(guarded.brake)
        self.assertEqual(change.state, "lane2")

    def test_ultrasonic_without_visual_occupancy_does_not_change_lane(self):
        fusion = planner()
        change = controller()

        fusion.update([], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True)
        event = fusion.update(
            [], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.1, True
        )

        self.assertIsNone(event)
        self.assertEqual(change.state, "lane2")

    def test_visual_without_fresh_ultrasonic_does_not_change_lane_in_fused_mode(self):
        fusion = planner()
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)
        stale = UltrasonicSnapshot()

        fusion.update([mask], SHAPE, CENTERLINE, lane(), change, stale, 1.0, True)
        event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, stale, 1.1, True
        )

        self.assertIsNone(event)
        self.assertTrue(fusion.observation.visual_confirmed)
        self.assertFalse(fusion.observation.range_confirmed)

    def test_destination_visual_occupancy_blocks_lane_change(self):
        fusion = planner()
        change = controller()
        current = obstacle_mask(130, 151, 45, 70)
        destination = obstacle_mask(70, 91, 72, 94)

        fusion.update(
            [current, destination],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(),
            1.0,
            True,
        )
        event = fusion.update(
            [current, destination],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(),
            1.1,
            True,
        )

        self.assertIsNone(event)
        self.assertTrue(fusion.observation.target_blocked)

    def test_frame_destination_occupancy_blocks_early_change(self):
        fusion = planner(
            frame_visual_trigger_y_ratio=0.15,
            frame_target_block_y_ratio=0.15,
            ultrasonic_trigger_mm=1600.0,
        )
        change = controller()
        current = obstacle_mask(132, 149, 16, 30)
        destination = obstacle_mask(72, 89, 16, 30)
        kwargs = dict(
            frame_obstacle_masks=[current, destination],
            frame_paths=projected_paths(),
        )

        fusion.update(
            [], SHAPE, CENTERLINE, lane(), change, ultrasound(fc=1400),
            1.0, True, **kwargs
        )
        event = fusion.update(
            [], SHAPE, CENTERLINE, lane(), change, ultrasound(fc=1380),
            1.1, True, **kwargs
        )

        self.assertIsNone(event)
        self.assertTrue(fusion.observation.target_blocked)

    def test_lane_side_solid_inside_swept_corridor_blocks_change(self):
        fusion = planner(
            ultrasonic_trigger_mm=1600.0,
            solid_crossing_margin_px=2.0,
            solid_min_overlap_ratio=0.05,
        )
        change = controller()
        current = obstacle_mask(130, 151, 45, 70)
        solid = obstacle_mask(108, 113, 20, 100)

        fusion.update(
            [current], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=1400), 1.0, True, solid_masks=[solid]
        )
        event = fusion.update(
            [current], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=1380), 1.1, True, solid_masks=[solid]
        )

        self.assertIsNone(event)
        self.assertTrue(fusion.observation.solid_blocked)
        self.assertEqual(change.state, "lane2")

    def test_destination_outer_solid_boundary_does_not_block_change(self):
        fusion = planner(
            ultrasonic_trigger_mm=1600.0,
            solid_crossing_margin_px=2.0,
        )
        change = controller()
        current = obstacle_mask(130, 151, 45, 70)
        destination_outer = obstacle_mask(15, 25, 20, 100)

        fusion.update(
            [current], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=1400), 1.0, True, solid_masks=[destination_outer]
        )
        event = fusion.update(
            [current], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=1380), 1.1, True, solid_masks=[destination_outer]
        )

        self.assertFalse(fusion.observation.solid_blocked)
        self.assertIn("lane2 -> lane1", event)

    def test_blocked_destination_stops_before_collision(self):
        fusion = planner(
            ultrasonic_trigger_mm=1600.0,
            blocked_stop_mm=650.0,
        )
        change = controller()
        current = obstacle_mask(130, 151, 45, 70)
        destination = obstacle_mask(70, 91, 45, 70)

        fusion.update(
            [current, destination], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=600, fr=620, fl=640), 1.0, True
        )
        fusion.update(
            [current, destination], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=590, fr=610, fl=630), 1.1, True
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertTrue(fusion.observation.target_blocked)
        self.assertTrue(guarded.brake)

    def test_stable_destination_lane_restores_normal_speed(self):
        fusion = planner(approach_speed_cap=90, speed_cap=120)
        change = controller()
        change.state = "lane1"
        old_lane2_obstacle = obstacle_mask(130, 151, 45, 70)

        fusion.update(
            [old_lane2_obstacle], SHAPE, CENTERLINE, lane(), change,
            UltrasonicSnapshot(), 2.0, True
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertFalse(fusion.observation.visual_detected)
        self.assertEqual(guarded.speed, 255)
        self.assertFalse(guarded.brake)

    def test_destination_side_ultrasonic_blocks_lane_change(self):
        fusion = planner()
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)
        blocked_left = ultrasound(sl=150)

        fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, blocked_left, 1.0, True
        )
        event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, blocked_left, 1.1, True
        )

        self.assertIsNone(event)
        self.assertFalse(fusion.observation.side_clear)

    def test_active_change_keeps_checking_committed_destination_side(self):
        fusion = planner()
        change = controller()
        change.state = "changing_to_lane1"
        lane1_obstacle = obstacle_mask(70, 91, 45, 70)

        fusion.update(
            [lane1_obstacle],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(sl=150, sr=500),
            1.0,
            True,
        )

        self.assertFalse(fusion.observation.side_clear)

    def test_one_close_sensor_stops_when_visual_obstacle_agrees(self):
        fusion = planner()
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)
        fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, ultrasound(fc=200), 1.0, True
        )

        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertTrue(guarded.brake)
        self.assertIn("obstacle_fusion_stop", guarded.reason)

    def test_two_close_front_sensors_provide_independent_emergency_stop(self):
        fusion = planner()
        change = controller()
        fusion.update(
            [], SHAPE, CENTERLINE, lane(), change, ultrasound(fc=200, fr=220), 1.0, True
        )

        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertTrue(guarded.brake)

    def test_source_lane_close_echo_does_not_stop_clear_active_change(self):
        fusion = planner()
        change = controller()
        change.state = "changing_to_lane1"
        source_lane_obstacle = obstacle_mask(130, 151, 45, 70)

        fusion.update(
            [source_lane_obstacle],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(fc=200, fr=220, fl=240),
            1.0,
            True,
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, -150, reason="lane"), change.state, True
        )

        self.assertFalse(fusion.observation.visual_detected)
        self.assertFalse(fusion.observation.emergency)
        self.assertFalse(guarded.brake)
        self.assertEqual(guarded.speed, 70)

    def test_committed_change_does_not_cross_associate_source_range(self):
        fusion = planner(blocked_stop_mm=650.0)
        change = controller()
        change.state = "changing_to_lane1"
        destination_obstacle = obstacle_mask(70, 91, 45, 70)
        source_obstacle = obstacle_mask(130, 151, 45, 70)

        for now, distance in ((1.0, 684), (1.1, 498)):
            fusion.update(
                [destination_obstacle, source_obstacle],
                SHAPE,
                CENTERLINE,
                lane(),
                change,
                ultrasound(fc=distance, fr=distance + 20, fl=distance + 40),
                now,
                True,
            )
        guarded = fusion.apply_safety(
            ControlCommand(255, -150, reason="lane"), change.state, True
        )

        self.assertTrue(fusion.observation.fused_hazard)
        self.assertTrue(fusion.observation.target_blocked)
        self.assertTrue(fusion.observation.maneuver_active)
        self.assertFalse(fusion.observation.emergency)
        self.assertFalse(guarded.brake)
        self.assertEqual(guarded.speed, 70)
        self.assertIn("COMMITTED", fusion.status_text())

    def test_visual_contact_does_not_interrupt_committed_evasion(self):
        fusion = planner()
        change = controller()
        change.state = "changing_to_lane1"
        destination_contact = obstacle_mask(70, 91, 82, 96)

        fusion.update(
            [destination_contact],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(fc=500, fr=520, fl=540),
            1.0,
            True,
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, -150, reason="lane"), change.state, True
        )

        self.assertFalse(fusion.observation.emergency)
        self.assertFalse(guarded.brake)
        self.assertEqual(guarded.speed, 70)

    def test_source_overlap_does_not_stop_immediately_after_completion(self):
        fusion = planner(blocked_stop_mm=650.0)
        change = controller()
        change.state = "lane1"
        lane1_obstacle = obstacle_mask(70, 91, 45, 70)

        fusion.update(
            [lane1_obstacle], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=800, fr=820, fl=840), 1.0, True
        )
        fusion.update(
            [lane1_obstacle], SHAPE, CENTERLINE, lane(), change,
            ultrasound(fc=780, fr=800, fl=820), 1.1, True
        )
        change.state = "completed"
        destination_contact = obstacle_mask(130, 151, 82, 96)
        source_contact = obstacle_mask(70, 91, 82, 96)

        fusion.update(
            [destination_contact, source_contact],
            SHAPE,
            CENTERLINE,
            lane(),
            change,
            ultrasound(fc=250, fr=270, fl=290),
            1.2,
            True,
        )
        guarded = fusion.apply_safety(
            ControlCommand(255, 20, reason="lane"), change.state, True
        )

        self.assertTrue(fusion.observation.clearing_source)
        self.assertFalse(fusion.observation.emergency)
        self.assertFalse(guarded.brake)
        self.assertIn("CLEARING_SOURCE", fusion.status_text())

    def test_single_ultrasonic_echo_without_visual_support_does_not_stop(self):
        fusion = planner()
        change = controller()
        fusion.update(
            [], SHAPE, CENTERLINE, lane(), change, ultrasound(fc=200), 1.0, True
        )

        guarded = fusion.apply_safety(
            ControlCommand(255, 0, reason="lane"), change.state, True
        )

        self.assertFalse(guarded.brake)

    def test_lane1_obstacle_requests_stable_return_to_lane2(self):
        fusion = planner()
        change = controller()
        change.state = "lane1"
        lane1_mask = obstacle_mask(70, 91, 45, 70)

        fusion.update(
            [lane1_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 2.0, True
        )
        event = fusion.update(
            [lane1_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 2.1, True
        )

        self.assertIn("lane1 -> lane2", event)
        self.assertEqual(change.return_source, "obstacle_fusion")

    def test_old_source_lane_obstacle_cannot_request_immediate_return(self):
        fusion = planner(rearm_clear_frames=3)
        change = controller()
        lane2_mask = obstacle_mask(130, 151, 45, 70)

        fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True
        )
        fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.1, True
        )
        change.state = "lane1"
        first = fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.2, True
        )
        second = fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.3, True
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(change.state, "lane1")

    def test_new_path_obstacle_replans_without_a_clear_gap(self):
        fusion = planner(rearm_clear_frames=3)
        change = controller()
        lane2_mask = obstacle_mask(130, 151, 45, 70)
        lane1_mask = obstacle_mask(70, 91, 45, 70)

        fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True
        )
        fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.1, True
        )
        change.state = "lane1"

        first = fusion.update(
            [lane1_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.2, True
        )
        second = fusion.update(
            [lane1_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.3, True
        )

        self.assertIsNone(first)
        self.assertIn("lane1 -> lane2", second)
        self.assertEqual(change.return_source, "obstacle_fusion")

    def test_new_obstacle_can_toggle_lane_after_stable_clear_rearm(self):
        fusion = planner(rearm_clear_frames=3)
        change = controller()
        lane2_mask = obstacle_mask(130, 151, 45, 70)
        lane1_mask = obstacle_mask(70, 91, 45, 70)

        fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.0, True
        )
        fusion.update(
            [lane2_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.1, True
        )
        change.state = "lane1"
        no_echo = ultrasound(fc=0, fr=0, fl=0)
        for now in (1.2, 1.3, 1.4):
            fusion.update([], SHAPE, CENTERLINE, lane(), change, no_echo, now, True)

        fusion.update(
            [lane1_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.5, True
        )
        event = fusion.update(
            [lane1_mask], SHAPE, CENTERLINE, lane(), change, ultrasound(), 1.6, True
        )

        self.assertIn("lane1 -> lane2", event)
        self.assertEqual(change.return_source, "obstacle_fusion")

    def test_yolo_mode_supports_offline_video_replay(self):
        fusion = planner(fusion_mode="yolo")
        change = controller()
        mask = obstacle_mask(130, 151, 45, 70)
        stale = UltrasonicSnapshot()

        fusion.update([mask], SHAPE, CENTERLINE, lane(), change, stale, 1.0, True)
        event = fusion.update(
            [mask], SHAPE, CENTERLINE, lane(), change, stale, 1.1, True
        )

        self.assertIn("lane2 -> lane1", event)

    def test_curved_bev_centerline_is_used_for_path_occupancy(self):
        fusion = planner(path_half_width_px=10.0)
        change = controller()
        curved_centerline = [
            (80.0 + (0.7 * y), float(y)) for y in range(0, 100, 5)
        ]
        mask = obstacle_mask(121, 132, 55, 72)
        misleading_lane = replace(lane(), center_x=175.0)

        fusion.update(
            [mask], SHAPE, curved_centerline, misleading_lane, change, ultrasound(), 1.0, True
        )
        event = fusion.update(
            [mask], SHAPE, curved_centerline, misleading_lane, change, ultrasound(), 1.1, True
        )

        self.assertIn("lane2 -> lane1", event)

    def test_real_center_with_virtual_outer_boundary_can_confirm_arrival(self):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        tier2 = YoloLaneMask(
            mask=mask,
            confidence=0.9,
            class_id=-1,
            class_name="center+virtual-right-side lane-corridor_tier2",
            device="cpu",
            inference_ms=1.0,
        )
        fallback = replace(
            tier2,
            class_name="left-side+virtual-right-side lane-corridor_tier3",
        )

        self.assertTrue(lane_change_geometry_reliable(tier2, lane()))
        self.assertFalse(lane_change_geometry_reliable(fallback, lane()))

    def test_coasted_geometry_cannot_confirm_lane_change(self):
        coast = YoloLaneMask(
            mask=np.zeros(SHAPE, dtype=np.uint8),
            confidence=0.9,
            class_id=-1,
            class_name="coast lane=coast:no_corridor(1):lane_change",
            device="cpu",
            inference_ms=1.0,
        )

        self.assertFalse(lane_change_geometry_reliable(coast, lane()))

    def test_cli_builds_fusion_and_ultrasonic_thresholds(self):
        args = parse_args(
            [
                "--obstacle-visual-trigger-y",
                "0.60",
                "--obstacle-frame-visual-trigger-y",
                "0.16",
                "--obstacle-action-confidence",
                "0.78",
                "--obstacle-trigger-mm",
                "900",
                "--obstacle-min-front-sensors",
                "2",
                "--obstacle-range-confirm-frames",
                "3",
                "--obstacle-rearm-clear-frames",
                "4",
                "--obstacle-ttc-seconds",
                "1.6",
                "--obstacle-solid-crossing-margin-px",
                "7",
                "--lane-change-target-capture-error",
                "0.22",
                "--lane-change-target-capture-frames",
                "3",
                "--lane-change-stable-near-error",
                "0.19",
                "--lane-change-stabilizing-steering-min",
                "75",
                "--obstacle-side-clearance-mm",
                "350",
            ]
        )
        config = build_obstacle_fusion_config(args)
        lane_config = build_lane_change_config(args)

        self.assertAlmostEqual(config.visual_trigger_y_ratio, 0.60)
        self.assertAlmostEqual(config.frame_visual_trigger_y_ratio, 0.16)
        self.assertAlmostEqual(config.visual_action_confidence, 0.78)
        self.assertEqual(config.ultrasonic_trigger_mm, 900.0)
        self.assertEqual(config.min_front_sensors, 2)
        self.assertEqual(config.range_confirm_frames, 3)
        self.assertEqual(config.rearm_clear_frames, 4)
        self.assertAlmostEqual(config.ttc_trigger_seconds, 1.6)
        self.assertEqual(config.solid_crossing_margin_px, 7.0)
        self.assertEqual(config.side_clearance_mm, 350.0)
        self.assertAlmostEqual(lane_config.target_capture_error, 0.22)
        self.assertEqual(lane_config.target_capture_frames, 3)
        self.assertAlmostEqual(lane_config.stable_near_lateral_error, 0.19)
        self.assertEqual(lane_config.stabilizing_steering_min, 75)

    def test_competition_defaults_use_early_range_and_strong_stabilization(self):
        args = parse_args([])

        fusion = build_obstacle_fusion_config(args)
        lane_config = build_lane_change_config(args)

        self.assertEqual(fusion.ultrasonic_trigger_mm, 2600.0)
        self.assertEqual(fusion.ultrasonic_clear_mm, 2900.0)
        self.assertEqual(lane_config.stabilizing_steering_min, 70)

    def test_old_clearance_cli_name_maps_to_capture_error(self):
        args = parse_args(
            ["--lane-change-target-clearance-margin", "0.17"]
        )

        self.assertAlmostEqual(
            build_lane_change_config(args).target_capture_error,
            0.17,
        )

    def test_obstacle_mode_is_disabled_and_side_effect_free_by_default(self):
        args = parse_args([])
        mode = ObstacleDriveMode(args, object(), object())
        vehicle = RecordingVehicle()
        command = ControlCommand(speed=120, steering=-30, brake=False, reason="lane")

        mode.start_serial(vehicle)
        mode.stop_serial(vehicle)

        self.assertFalse(mode.enabled)
        self.assertEqual(mode.status_text, "off")
        self.assertEqual(vehicle.lines, [])
        self.assertEqual(mode.apply_steering(command), command)
        self.assertEqual(mode.apply_speed_cap(command), command)
        self.assertEqual(mode.apply_safety(command, True), command)

    def test_obstacle_mode_enables_ultrasonic_stream_only_when_requested(self):
        args = parse_args(["--obstacle-avoidance", "on"])
        mode = ObstacleDriveMode(args, object(), object())
        vehicle = RecordingVehicle()

        mode.start_serial(vehicle)
        mode.stop_serial(vehicle)

        self.assertTrue(mode.enabled)
        self.assertEqual(vehicle.lines, ["USON", "USOFF"])


class RecordingVehicle:
    def __init__(self):
        self.lines = []

    def write_line(self, line):
        self.lines.append(line)


if __name__ == "__main__":
    unittest.main()
