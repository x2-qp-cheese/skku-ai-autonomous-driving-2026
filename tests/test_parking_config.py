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
        self.assertEqual(config.lidar.car_detection_roi.x_min_mm, -1800.0)
        self.assertEqual(config.lidar.gap_center_x_min_mm, 0.0)
        self.assertEqual(config.lidar.gap_coast_scans, 15)
        self.assertTrue(config.lidar.gap_hold_confirmed_until_reset)
        self.assertEqual(config.lidar.gap_orientation_smooth_alpha, 0.25)
        self.assertEqual(config.lidar.gap_max_orientation_jump_deg, 35.0)
        self.assertGreater(config.lidar.expected_observed_gap_mm, config.lidar.parking_space_width_mm)
        self.assertLess(config.planner.reverse_entry_speed, 0)
        self.assertGreater(config.path.lookahead_px, 0.0)
        self.assertEqual(config.runtime.lidar_display_rotation_deg, 0.0)
        self.assertEqual(config.runtime.lidar_debug_vehicle_width_mm, 550.0)
        self.assertEqual(config.runtime.lidar_debug_vehicle_length_mm, 1000.0)
        self.assertEqual(config.runtime.lidar_debug_sensor_behind_vehicle_rear_mm, 100.0)
        self.assertEqual(config.lidar.sensor_to_rear_axle_y_back_mm, -300.0)
        self.assertEqual(config.lidar.first_car_turn_target_y_back_mm, -650.0)
        self.assertEqual(config.lidar.first_car_confirm_scans, 2)
        self.assertTrue(config.planner.first_car_preemptive_turn_enabled)
        self.assertEqual(config.planner.first_car_approach_speed, 10)
        self.assertTrue(config.planner.prealign_enabled)
        self.assertEqual(config.planner.prealign_speed, 35)
        self.assertEqual(config.planner.prealign_steering, -150)
        self.assertEqual(config.planner.prealign_timeout_s, 6.0)
        self.assertEqual(config.planner.ultrasonic_emergency_mm, 100.0)
        self.assertEqual(config.planner.ultrasonic_max_correction, 35)
        self.assertEqual(config.planner.ultrasonic_stale_after_s, 0.8)
        self.assertEqual(config.planner.park_hold_s, 3.0)
        self.assertEqual(config.planner.exit_speed, 24)
        self.assertEqual(config.planner.exit_turn_steering, 80)
        self.assertEqual(config.planner.exit_turn_s, 1.6)
        self.assertEqual(config.planner.exit_straight_s, 0.0)
        self.assertEqual(config.planner.exit_right_min_clearance_mm, 180.0)
        self.assertEqual(tuple(config.bev.src_top_left), (0.18, 0.56))
        self.assertEqual(tuple(config.bev.src_top_right), (0.82, 0.56))
        self.assertTrue(config.geometry.synthesize_back_from_side_pair)
        self.assertAlmostEqual(config.geometry.synthetic_back_depth_to_width_ratio, 1.58)

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
