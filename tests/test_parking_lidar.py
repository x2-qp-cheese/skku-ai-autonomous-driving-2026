import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from skku_autocar.estimation.parking_lidar import (
    CarCluster,
    LidarParkingConfig,
    LidarParkingSpaceEstimator,
    RectangleRoi,
    infer_dynamic_slot_polygon,
    choose_gap,
)
from skku_autocar.sensors.lidar import (
    LidarCsvReplay,
    LidarPoint,
    LidarScan,
    load_lidar_csv,
)


def point_at(x_right, y_back, quality=15):
    distance = math.hypot(x_right, y_back)
    angle = math.degrees(math.atan2(x_right, -y_back)) % 360.0
    return LidarPoint(quality, angle, distance)


class ParkingLidarTest(unittest.TestCase):
    def make_estimator(self):
        return LidarParkingSpaceEstimator(
            LidarParkingConfig(
                clockwise_angles=False,
                angle_offset_deg=0.0,
                car_detection_roi=RectangleRoi(500.0, 1500.0, -1800.0, 1800.0),
                safety_roi=RectangleRoi(-300.0, 300.0, 200.0, 800.0),
                min_observed_points=5,
                car_cluster_radius_mm=120.0,
                car_cluster_min_points=3,
                expected_observed_gap_mm=1300.0,
                observed_gap_min_mm=1100.0,
                observed_gap_max_mm=1500.0,
                gap_confirm_scans=2,
                gap_center_smooth_alpha=1.0,
                sensor_to_rear_axle_y_back_mm=0.0,
                entry_tolerance_mm=50.0,
            )
        )

    def test_supplied_sensor_yaw_maps_vehicle_front_to_screen_up(self):
        estimator = LidarParkingSpaceEstimator(
            LidarParkingConfig(
                angle_offset_deg=-90.0,
                clockwise_angles=False,
                quality_min=1,
            )
        )
        scan = LidarScan(
            1.0,
            (
                LidarPoint(15, 90.0, 1000.0),   # physical front
                LidarPoint(15, 180.0, 1000.0),  # physical right
                LidarPoint(15, 270.0, 1000.0),  # physical rear
                LidarPoint(15, 0.0, 1000.0),    # physical left
            ),
        )

        front, right, rear, left = estimator.vehicle_points(scan)

        self.assertAlmostEqual(front[0], 0.0, delta=1.0)
        self.assertAlmostEqual(front[1], -1000.0, delta=1.0)
        self.assertAlmostEqual(right[0], 1000.0, delta=1.0)
        self.assertAlmostEqual(right[1], 0.0, delta=1.0)
        self.assertAlmostEqual(rear[0], 0.0, delta=1.0)
        self.assertAlmostEqual(rear[1], 1000.0, delta=1.0)
        self.assertAlmostEqual(left[0], -1000.0, delta=1.0)
        self.assertAlmostEqual(left[1], 0.0, delta=1.0)

    @staticmethod
    def two_car_points(offset=0.0):
        return tuple(
            point_at(x, y + offset)
            for x, y in (
                (950.0, -720.0), (1000.0, -700.0), (1050.0, -680.0),
                (950.0, 620.0), (1000.0, 640.0), (1050.0, 660.0),
            )
        )

    def test_requires_distinct_scans_to_confirm_two_car_gap(self):
        estimator = self.make_estimator()
        points = self.two_car_points()

        first = estimator.estimate(LidarScan(1.0, points), now=1.0)
        repeated = estimator.estimate(LidarScan(1.0, points), now=1.0)
        second = estimator.estimate(LidarScan(2.0, points), now=2.0)

        self.assertTrue(first.valid)
        self.assertEqual(first.car_count, 2)
        self.assertTrue(first.gap_found)
        self.assertFalse(first.gap_confirmed)
        self.assertFalse(repeated.gap_confirmed)
        self.assertTrue(second.gap_confirmed)
        self.assertAlmostEqual(second.gap_width_mm, 1300.0, delta=50.0)
        self.assertTrue(second.entry_reached)

    def test_first_car_slot_edge_triggers_before_second_car_is_visible(self):
        config = replace(
            self.make_estimator().config,
            min_observed_points=3,
            first_car_confirm_scans=2,
            first_car_turn_target_y_back_mm=-650.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        points = tuple(
            point_at(x, y)
            for x, y in (
                (950.0, -640.0),
                (1000.0, -620.0),
                (1050.0, -600.0),
            )
        )

        first = estimator.estimate(LidarScan(1.0, points), now=1.0)
        second = estimator.estimate(LidarScan(2.0, points), now=2.0)

        self.assertTrue(first.first_car_seen)
        self.assertFalse(first.first_car_confirmed)
        self.assertTrue(second.first_car_confirmed)
        self.assertTrue(second.first_car_turn_reached)
        self.assertEqual(second.car_count, 1)
        self.assertFalse(second.gap_found)
        self.assertAlmostEqual(second.first_car_slot_edge_y_back_mm, -640.0)

    def test_gap_center_produces_rear_axle_position_error(self):
        estimator = self.make_estimator()
        points = self.two_car_points(offset=250.0)

        estimator.estimate(LidarScan(1.0, points), now=1.0)
        observation = estimator.estimate(LidarScan(2.0, points), now=2.0)

        self.assertTrue(observation.gap_confirmed)
        self.assertAlmostEqual(observation.entry_error_mm, 230.0, delta=30.0)
        self.assertFalse(observation.entry_reached)

    def test_dynamic_slot_box_starts_at_sensor_facing_obstacle_edges(self):
        estimator = self.make_estimator()
        observation = estimator.estimate(
            LidarScan(1.0, self.two_car_points()),
            now=1.0,
        )

        polygon = infer_dynamic_slot_polygon(observation, depth_mm=1500.0)

        self.assertIsNotNone(polygon)
        # LiDAR is at x=0 and the car returns span x=950..1050. The slot
        # entrance must therefore begin at the detected near surfaces x=950,
        # then extend its full 1500 mm depth away from the sensor.
        self.assertAlmostEqual(polygon[0][0], 950.0, delta=1.0)
        self.assertAlmostEqual(polygon[0][1], -680.0, delta=1.0)
        self.assertAlmostEqual(polygon[1][0], 950.0, delta=1.0)
        self.assertAlmostEqual(polygon[1][1], 620.0, delta=1.0)
        self.assertAlmostEqual(polygon[3][0] - polygon[0][0], 1500.0, delta=1.0)

    def test_dynamic_slot_box_moves_with_detected_cars(self):
        estimator = self.make_estimator()
        first = estimator.estimate(
            LidarScan(1.0, self.two_car_points(offset=0.0)),
            now=1.0,
        )
        second = estimator.estimate(
            LidarScan(2.0, self.two_car_points(offset=250.0)),
            now=2.0,
        )

        first_polygon = infer_dynamic_slot_polygon(first, depth_mm=1500.0)
        second_polygon = infer_dynamic_slot_polygon(second, depth_mm=1500.0)

        self.assertIsNotNone(first_polygon)
        self.assertIsNotNone(second_polygon)
        for before, after in zip(first_polygon, second_polygon):
            self.assertAlmostEqual(after[0], before[0], delta=1.0)
            self.assertAlmostEqual(after[1], before[1] + 250.0, delta=1.0)

    def test_oriented_gap_keeps_car_that_rotates_into_left_half_plane(self):
        clusters = (
            CarCluster(5, -750.0, -650.0, -50.0, 50.0),
            CarCluster(5, 650.0, 750.0, -50.0, 50.0),
        )
        config = LidarParkingConfig(
            expected_observed_gap_mm=1300.0,
            observed_gap_min_mm=1100.0,
            observed_gap_max_mm=1500.0,
            gap_center_x_min_mm=0.0,
        )

        gap = choose_gap(clusters, config)

        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap[0], 1300.0, delta=1.0)
        self.assertAlmostEqual(gap[3], -650.0, delta=1.0)
        self.assertAlmostEqual(gap[4], 650.0, delta=1.0)
        self.assertAlmostEqual(gap[1], -50.0, delta=1.0)
        self.assertAlmostEqual(gap[2], -50.0, delta=1.0)

    def test_tracked_slot_moves_and_rotates_without_freezing(self):
        config = replace(
            self.make_estimator().config,
            car_detection_roi=RectangleRoi(-1800.0, 1800.0, -2500.0, 2500.0),
            gap_center_x_min_mm=-1800.0,
            gap_confirm_scans=1,
            gap_center_smooth_alpha=1.0,
            gap_orientation_smooth_alpha=1.0,
            gap_max_center_jump_mm=1000.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        base = (
            (950.0, -720.0), (1000.0, -700.0), (1050.0, -680.0),
            (950.0, 620.0), (1000.0, 640.0), (1050.0, 660.0),
        )

        first = estimator.estimate(
            LidarScan(1.0, tuple(point_at(x, y) for x, y in base)),
            now=1.0,
        )
        angle = math.radians(15.0)
        rotated = tuple(
            point_at(
                math.cos(angle) * x - math.sin(angle) * y,
                math.sin(angle) * x + math.cos(angle) * y,
            )
            for x, y in base
        )
        second = estimator.estimate(LidarScan(2.0, rotated), now=2.0)

        self.assertTrue(second.gap_confirmed)
        self.assertFalse(second.coasted)
        self.assertNotAlmostEqual(
            second.gap_center_y_back_mm,
            first.gap_center_y_back_mm,
            delta=20.0,
        )
        self.assertGreater(abs(second.slot_depth_y_back), 0.1)
        self.assertGreater(
            first.slot_depth_x_right * second.slot_depth_x_right
            + first.slot_depth_y_back * second.slot_depth_y_back,
            0.9,
        )

    def test_tracked_slot_depth_does_not_flip_after_sensor_crosses_entrance(self):
        config = replace(
            self.make_estimator().config,
            car_detection_roi=RectangleRoi(-1800.0, 1800.0, -2500.0, 2500.0),
            gap_center_x_min_mm=-1800.0,
            gap_confirm_scans=1,
            gap_center_smooth_alpha=1.0,
            gap_orientation_smooth_alpha=1.0,
            gap_max_center_jump_mm=3000.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        initial = estimator.estimate(
            LidarScan(1.0, self.two_car_points()),
            now=1.0,
        )
        crossed_points = tuple(
            point_at(x - 2000.0, y)
            for x, y in (
                (950.0, -720.0), (1000.0, -700.0), (1050.0, -680.0),
                (950.0, 620.0), (1000.0, 640.0), (1050.0, 660.0),
            )
        )
        crossed = estimator.estimate(LidarScan(2.0, crossed_points), now=2.0)
        polygon = infer_dynamic_slot_polygon(crossed, depth_mm=1500.0)

        self.assertGreater(initial.slot_depth_x_right, 0.9)
        self.assertGreater(crossed.slot_depth_x_right, 0.9)
        self.assertFalse(crossed.coasted)
        self.assertIsNotNone(polygon)
        self.assertGreater(polygon[3][0], polygon[0][0])

    def test_confirmed_slot_remains_visible_when_both_cars_are_lost(self):
        config = replace(
            self.make_estimator().config,
            gap_confirm_scans=1,
            gap_coast_scans=2,
            gap_hold_confirmed_until_reset=True,
        )
        estimator = LidarParkingSpaceEstimator(config)
        initial = estimator.estimate(
            LidarScan(1.0, self.two_car_points()),
            now=1.0,
        )
        held = initial
        for index in range(1, 8):
            held = estimator.estimate(
                LidarScan(1.0 + index, ()),
                now=1.0 + index,
            )

        self.assertTrue(initial.gap_confirmed)
        self.assertTrue(held.gap_found)
        self.assertTrue(held.gap_confirmed)
        self.assertTrue(held.coasted)
        self.assertEqual(
            infer_dynamic_slot_polygon(held, 1500.0),
            infer_dynamic_slot_polygon(initial, 1500.0),
        )

    def test_safety_envelope_is_independent_of_gap_detection(self):
        estimator = self.make_estimator()
        points = self.two_car_points() + (point_at(0.0, 500.0),)

        observation = estimator.estimate(LidarScan(1.0, points), now=1.0)

        self.assertTrue(observation.gap_found)
        self.assertTrue(observation.unsafe)
        self.assertEqual(observation.reason, "safety_obstacle")

    def test_recording_csv_parser_groups_timestamp_scans(self):
        content = (
            "timestamp,quality,angle_deg,distance_mm\n"
            "10.0,15,0.0,1000.0\n"
            "10.0,8,1.0,1100.0\n"
            "10.1,15,2.0,1200.0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lidar.csv"
            path.write_text(content, encoding="utf-8")
            scans = load_lidar_csv(str(path))
            replay = LidarCsvReplay(str(path))

        self.assertEqual(len(scans), 2)
        self.assertEqual(len(scans[0].points), 2)
        self.assertEqual(replay.scan_at_elapsed(0.09).timestamp, 10.1)


if __name__ == "__main__":
    unittest.main()
