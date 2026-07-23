import tempfile
import unittest
import zipfile
from pathlib import Path

from skku_autocar.parking_config import load_parking_config
from skku_autocar.estimation.parking_geometry import ParkingGeometry, ParkingLine
from skku_autocar.runtime.parking_app import (
    apply_cli_overrides,
    extract_recording_zip,
    parking_mask_color,
    parse_args,
)


ROOT = Path(__file__).resolve().parents[1]


class ParkingConfigTest(unittest.TestCase):
    def test_repository_parking_config_loads_nested_rois(self):
        config = load_parking_config(str(ROOT / "configs" / "parking.json"))

        self.assertEqual(config.yolo.model_path, "trained_model/parking_best.pt")
        self.assertEqual(config.geometry.min_confirm_frames, 3)
        self.assertEqual(config.lidar.parking_space_width_mm, 950.0)
        self.assertEqual(config.lidar.parking_space_depth_mm, 1500.0)
        self.assertFalse(config.lidar.clockwise_angles)
        self.assertEqual(config.lidar.angle_offset_deg, -90.0)
        self.assertGreater(
            config.lidar.car_detection_roi.x_max_mm,
            config.lidar.car_detection_roi.x_min_mm,
        )
        self.assertEqual(config.lidar.quality_min, 1)
        self.assertEqual(config.lidar.car_detection_roi.x_min_mm, 250.0)
        self.assertEqual(config.lidar.car_detection_roi.x_max_mm, 2600.0)
        self.assertIsNotNone(config.lidar.slot_tracking_roi)
        self.assertEqual(config.lidar.slot_tracking_roi.x_min_mm, -1800.0)
        self.assertEqual(config.lidar.slot_tracking_roi.x_max_mm, 2600.0)
        self.assertEqual(config.lidar.car_cluster_radius_mm, 350.0)
        self.assertEqual(config.lidar.car_cluster_min_points, 3)
        self.assertEqual(config.lidar.gap_cluster_min_points, 5)
        self.assertEqual(config.lidar.gap_pair_min_points, 10)
        self.assertEqual(config.lidar.gap_center_x_min_mm, 0.0)
        self.assertEqual(config.lidar.gap_center_y_back_min_mm, 200.0)
        self.assertEqual(config.lidar.gap_confirm_scans, 3)
        self.assertEqual(config.lidar.gap_candidate_hold_s, 1.2)
        self.assertTrue(config.lidar.gap_single_cluster_track_enabled)
        self.assertEqual(config.lidar.gap_single_cluster_max_edge_jump_mm, 700.0)
        self.assertEqual(config.lidar.gap_coast_scans, 15)
        self.assertTrue(config.lidar.gap_hold_confirmed_until_reset)
        self.assertEqual(config.lidar.gap_orientation_smooth_alpha, 0.25)
        self.assertEqual(config.lidar.gap_max_orientation_jump_deg, 35.0)
        self.assertGreater(config.lidar.expected_observed_gap_mm, config.lidar.parking_space_width_mm)
        self.assertLess(config.planner.reverse_entry_speed, 0)
        self.assertGreater(config.path.lookahead_px, 0.0)
        self.assertEqual(config.runtime.lidar_display_rotation_deg, 0.0)
        self.assertFalse(config.runtime.auto_start)
        self.assertTrue(config.runtime.camera_enabled)
        self.assertEqual(config.fusion.camera_min_confidence, 0.35)
        self.assertEqual(config.fusion.max_heading_disagreement_deg, 25.0)
        self.assertTrue(config.fusion.prefer_camera_back_line)
        self.assertEqual(config.runtime.lidar_debug_vehicle_width_mm, 600.0)
        self.assertEqual(config.runtime.lidar_debug_vehicle_length_mm, 1000.0)
        self.assertEqual(config.runtime.lidar_debug_sensor_behind_vehicle_rear_mm, 100.0)
        self.assertEqual(config.lidar.sensor_to_rear_axle_y_back_mm, -300.0)
        self.assertEqual(config.lidar.first_car_turn_target_y_back_mm, -650.0)
        self.assertEqual(config.lidar.first_car_confirm_scans, 6)
        self.assertEqual(config.lidar.first_car_min_x_right_mm, 450.0)
        self.assertEqual(config.lidar.first_car_cluster_min_points, 5)
        self.assertEqual(config.lidar.first_car_min_span_mm, 80.0)
        self.assertEqual(config.lidar.first_car_max_center_jump_mm, 450.0)
        self.assertTrue(config.planner.first_car_preemptive_turn_enabled)
        self.assertFalse(config.planner.first_car_only_prealign_enabled)
        self.assertEqual(config.planner.start_forward_s, 0.8)
        self.assertEqual(config.planner.search_speed, 100)
        self.assertEqual(config.planner.straight_steering_trim, -20)
        self.assertEqual(config.planner.gap_tracking_speed, 100)
        self.assertEqual(config.planner.position_speed, 100)
        self.assertEqual(config.planner.first_car_approach_speed, 100)
        self.assertEqual(config.planner.first_car_straight_s, 2.2)
        self.assertTrue(config.planner.prealign_enabled)
        self.assertEqual(config.planner.prealign_speed, 100)
        self.assertEqual(config.planner.prealign_steering, -150)
        self.assertEqual(config.planner.prealign_gap_acquire_timeout_s, 0.0)
        self.assertEqual(config.planner.prealign_timeout_s, 6.0)
        self.assertEqual(config.planner.prealign_entry_bearing_tolerance_deg, 12.0)
        self.assertEqual(config.planner.prealign_center_x_tolerance_mm, 180.0)
        self.assertEqual(config.planner.prealign_curve_slot_heading_tolerance_deg, 55.0)
        self.assertEqual(config.planner.prealign_curve_entry_bearing_tolerance_deg, 45.0)
        self.assertEqual(config.planner.prealign_curve_center_x_tolerance_mm, 1400.0)
        self.assertEqual(config.planner.prealign_target_distance_min_mm, 900.0)
        self.assertEqual(config.planner.prealign_target_distance_max_mm, 2600.0)
        self.assertEqual(config.planner.path_confirm_frames, 3)
        self.assertEqual(config.planner.ultrasonic_emergency_mm, 100.0)
        self.assertFalse(config.planner.emergency_stop_enabled)
        self.assertEqual(config.planner.ultrasonic_max_correction, 35)
        self.assertEqual(config.planner.ultrasonic_stale_after_s, 0.8)
        self.assertEqual(config.planner.max_steering, 150)
        self.assertEqual(config.planner.reverse_entry_min_steering, 90)
        self.assertEqual(config.planner.reverse_entry_steer_settle_s, 0.40)
        self.assertEqual(config.planner.reverse_entry_release_heading_deg, 12.0)
        self.assertEqual(config.planner.reverse_entry_release_confirm_frames, 3)
        self.assertEqual(config.planner.entry_curve_timeout_s, 16.0)
        self.assertTrue(config.planner.correction_enabled)
        self.assertEqual(config.planner.reverse_entry_speed, -100)
        self.assertEqual(config.planner.reverse_center_speed, -100)
        self.assertEqual(config.planner.correction_forward_speed, 100)
        self.assertEqual(config.planner.correction_reverse_speed, -100)
        self.assertEqual(config.planner.correction_steering, 130)
        self.assertEqual(config.planner.correction_depth_trigger_px, 760.0)
        self.assertEqual(config.planner.correction_trigger_frames, 3)
        self.assertEqual(config.planner.correction_max_attempts, 3)
        self.assertEqual(config.planner.park_hold_s, 3.0)
        self.assertEqual(config.planner.exit_speed, 100)
        self.assertEqual(config.planner.exit_turn_steering, 80)
        self.assertEqual(config.planner.exit_turn_s, 1.6)
        self.assertEqual(config.planner.exit_straight_s, 0.0)
        self.assertEqual(config.planner.exit_right_min_clearance_mm, 180.0)
        self.assertEqual(config.planner.reverse_steering_sign, 1.0)
        self.assertEqual(tuple(config.bev.src_top_left), (0.18, 0.56))
        self.assertEqual(tuple(config.bev.src_top_right), (0.82, 0.56))
        self.assertTrue(config.runtime.locked_slot_tracking_enabled)
        self.assertEqual(config.runtime.locked_slot_min_points, 8)
        self.assertEqual(config.runtime.locked_slot_max_range_mm, 3500.0)
        self.assertEqual(config.runtime.locked_slot_trim_ratio, 0.65)
        self.assertEqual(config.runtime.locked_slot_max_hold_scans, 3)

    def test_recording_zip_finds_video_and_lidar_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "recording.zip"
            with zipfile.ZipFile(str(archive_path), "w") as archive:
                archive.writestr("session/run.mp4", b"video")
                archive.writestr(
                    "session/run_lidar.csv",
                    "timestamp,quality,angle_deg,distance_mm\n",
                )
            output = root / "output"
            video, lidar = extract_recording_zip(archive_path, output)

            self.assertTrue(video.exists())
            self.assertTrue(lidar.exists())
            self.assertEqual(video.name, "run.mp4")
            self.assertEqual(lidar.name, "run_lidar.csv")

    def test_cpu_replay_options_are_parsed(self):
        args = parse_args([
            "--recording-zip",
            "recording.zip",
            "--device",
            "cpu",
            "--imgsz",
            "512",
            "--frame-stride",
            "2",
            "--auto-start",
        ])

        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.frame_stride, 2)
        self.assertTrue(args.auto_start)

    def test_bev_and_lidar_debug_cli_overrides(self):
        args = parse_args([
            "--bev-top-y", "0.42",
            "--bev-top-left-x", "-0.18",
            "--bev-top-right-x", "1.18",
            "--bev-dst-margin", "0.12",
            "--lidar-display-rotation", "-90",
            "--lidar-angle-offset", "5",
            "--lidar-behind-vehicle-rear-cm", "8",
            "--lidar-to-rear-axle-cm", "-28",
            "--first-car-turn-target-cm", "-72",
            "--first-car-preemptive-turn", "on",
            "--camera",
            "--prealign-speed", "42",
            "--prealign-steering", "-120",
            "--prealign-timeout-s", "7.5",
            "--park-hold-s", "2.5",
            "--exit-speed", "20",
            "--exit-turn-steering", "70",
            "--exit-turn-s", "1.2",
            "--exit-straight-s", "0",
            "--exit-right-min-clearance-cm", "22",
        ])
        original = load_parking_config(str(ROOT / "configs" / "parking.json"))
        config = apply_cli_overrides(original, args)

        self.assertEqual(config.bev.src_top_left, (-0.18, 0.42))
        self.assertEqual(config.bev.src_top_right, (1.18, 0.42))
        self.assertEqual(config.bev.dst_x_margin, 0.12)
        self.assertEqual(config.runtime.lidar_display_rotation_deg, -90.0)
        self.assertEqual(config.runtime.lidar_debug_sensor_behind_vehicle_rear_mm, 80.0)
        self.assertEqual(config.lidar.angle_offset_deg, 5.0)
        self.assertEqual(config.lidar.sensor_to_rear_axle_y_back_mm, -280.0)
        self.assertEqual(config.lidar.first_car_turn_target_y_back_mm, -720.0)
        self.assertTrue(config.planner.first_car_preemptive_turn_enabled)
        self.assertTrue(config.runtime.camera_enabled)
        self.assertEqual(config.planner.prealign_speed, 42)
        self.assertEqual(config.planner.prealign_steering, -120)
        self.assertEqual(config.planner.max_steering, original.planner.max_steering)
        self.assertEqual(config.planner.prealign_timeout_s, 7.5)
        self.assertEqual(config.planner.park_hold_s, 2.5)
        self.assertEqual(config.planner.exit_speed, 20)
        self.assertEqual(config.planner.exit_turn_steering, 70)
        self.assertEqual(config.planner.exit_turn_s, 1.2)
        self.assertEqual(config.planner.exit_straight_s, 0.0)
        self.assertEqual(config.planner.exit_right_min_clearance_mm, 220.0)

    def test_parking_mask_colors_follow_semantic_line_role(self):
        def line(index):
            return ParkingLine(0, 0, 0, -1, 100, 1, 1, 100, mask_index=index)

        geometry = ParkingGeometry(
            left=line(2),
            right=line(0),
            back=line(1),
        )

        self.assertEqual(parking_mask_color(2, geometry), (255, 255, 0))
        self.assertEqual(parking_mask_color(0, geometry), (0, 255, 0))
        self.assertEqual(parking_mask_color(1, geometry), (0, 0, 255))
        self.assertEqual(parking_mask_color(3, geometry), (255, 0, 255))


if __name__ == "__main__":
    unittest.main()
