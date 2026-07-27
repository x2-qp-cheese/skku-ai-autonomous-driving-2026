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
    LidarCsvRecorder,
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
                gap_cluster_min_points=3,
                gap_pair_min_points=6,
                expected_observed_gap_mm=1300.0,
                observed_gap_min_mm=1100.0,
                observed_gap_max_mm=1500.0,
                gap_confirm_scans=2,
                gap_center_y_back_min_mm=-100.0,
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
        self.assertTrue(second.gap_pair_observed)
        self.assertAlmostEqual(second.gap_width_mm, 1300.0, delta=50.0)
        self.assertTrue(second.entry_reached)

    def test_first_car_slot_edge_triggers_before_second_car_is_visible(self):
        config = replace(
            self.make_estimator().config,
            min_observed_points=3,
            first_car_cluster_min_points=3,
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

    def test_first_car_confirmation_resets_when_candidate_jumps(self):
        config = replace(
            self.make_estimator().config,
            min_observed_points=3,
            first_car_cluster_min_points=3,
            first_car_confirm_scans=2,
            first_car_max_center_jump_mm=120.0,
            first_car_turn_target_y_back_mm=-650.0,
        )
        estimator = LidarParkingSpaceEstimator(config)

        def first_car_points(y_offset):
            return tuple(
                point_at(x, y + y_offset)
                for x, y in (
                    (950.0, -640.0),
                    (1000.0, -620.0),
                    (1050.0, -600.0),
                )
            )

        first = estimator.estimate(LidarScan(1.0, first_car_points(0.0)), now=1.0)
        jumped = estimator.estimate(LidarScan(2.0, first_car_points(500.0)), now=2.0)
        stable = estimator.estimate(LidarScan(3.0, first_car_points(500.0)), now=3.0)

        self.assertTrue(first.first_car_seen)
        self.assertFalse(first.first_car_confirmed)
        self.assertTrue(jumped.first_car_seen)
        self.assertFalse(jumped.first_car_confirmed)
        self.assertTrue(stable.first_car_confirmed)

    def test_non_right_side_clusters_do_not_trigger_first_car(self):
        config = replace(
            self.make_estimator().config,
            min_observed_points=3,
            first_car_confirm_scans=1,
        )
        estimator = LidarParkingSpaceEstimator(config)
        points = tuple(
            point_at(x, y)
            for x, y in (
                (-900.0, -640.0),
                (-950.0, -620.0),
                (-1000.0, -600.0),
                (0.0, -1000.0),
                (30.0, -980.0),
            )
        )

        observation = estimator.estimate(LidarScan(1.0, points), now=1.0)

        self.assertTrue(observation.valid)
        self.assertEqual(observation.car_roi_points, 0)
        self.assertEqual(observation.left_roi_points, 3)
        self.assertEqual(observation.car_count, 0)
        self.assertFalse(observation.first_car_seen)
        self.assertFalse(observation.first_car_confirmed)
        self.assertFalse(observation.first_car_turn_reached)
        self.assertFalse(observation.gap_found)

    def test_sparse_right_car_returns_accumulate_across_distinct_scans(self):
        config = replace(
            self.make_estimator().config,
            detection_accumulation_scans=3,
            detection_accumulation_max_age_s=1.0,
            gap_cluster_min_points=4,
            gap_cluster_min_scans=2,
            gap_pair_min_points=8,
            gap_cluster_min_span_mm=20.0,
            gap_pair_max_lateral_offset_mm=300.0,
            gap_pair_min_longitudinal_alignment=0.9,
            gap_confirm_scans=1,
        )
        estimator = LidarParkingSpaceEstimator(config)
        sparse = tuple(
            point_at(x, y)
            for x, y in (
                (980.0, -680.0),
                (1020.0, -660.0),
                (980.0, 660.0),
                (1020.0, 680.0),
            )
        )

        first = estimator.estimate(LidarScan(1.0, sparse), now=1.0)
        second = estimator.estimate(LidarScan(1.1, sparse), now=1.1)

        self.assertEqual(first.car_count, 0)
        self.assertFalse(first.gap_found)
        self.assertEqual(second.car_count, 2)
        self.assertTrue(second.gap_confirmed)
        self.assertEqual(second.car_clusters[0].scan_count, 2)
        self.assertEqual(second.car_roi_points, 4)
        self.assertEqual(second.accumulated_car_roi_points, 8)

    def test_signal_gantry_crossbar_pair_is_not_a_parking_gap(self):
        config = replace(
            self.make_estimator().config,
            gap_cluster_min_span_mm=60.0,
            gap_pair_max_lateral_offset_mm=450.0,
            gap_pair_min_longitudinal_alignment=0.82,
        )
        left_support = CarCluster(8, 700.0, 800.0, 500.0, 620.0, 2)
        right_support = CarCluster(8, 2050.0, 2150.0, 500.0, 620.0, 2)

        self.assertIsNone(choose_gap((left_support, right_support), config))

    def test_slender_signal_pole_cannot_pair_with_a_car(self):
        config = replace(
            self.make_estimator().config,
            gap_cluster_min_span_mm=60.0,
            gap_pair_max_lateral_offset_mm=450.0,
            gap_pair_min_longitudinal_alignment=0.82,
        )
        pole = CarCluster(8, 990.0, 1010.0, -665.0, -635.0, 2)
        car = CarCluster(8, 950.0, 1050.0, 660.0, 760.0, 2)

        self.assertIsNone(choose_gap((pole, car), config))

    def test_close_right_side_sparse_returns_do_not_trigger_first_car(self):
        config = LidarParkingConfig(
            angle_offset_deg=0.0,
            clockwise_angles=False,
            quality_min=1,
            car_detection_roi=RectangleRoi(250.0, 2600.0, -2500.0, 2500.0),
            min_observed_points=5,
            car_cluster_radius_mm=350.0,
            car_cluster_min_points=2,
            first_car_min_x_right_mm=250.0,
            first_car_confirm_scans=1,
            first_car_turn_target_y_back_mm=-650.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        scan = LidarScan(
            1.0,
            (
                point_at(320.0, -640.0, quality=1),
                point_at(360.0, -620.0, quality=1),
            ),
        )

        observation = estimator.estimate(scan, now=1.0)

        self.assertTrue(observation.valid)
        self.assertEqual(observation.car_roi_points, 2)
        self.assertEqual(observation.car_count, 1)
        self.assertFalse(observation.first_car_seen)
        self.assertFalse(observation.first_car_confirmed)
        self.assertFalse(observation.first_car_turn_reached)
        self.assertEqual(observation.reason, "one_cluster_filtered")

    def test_weak_two_cluster_returns_do_not_confirm_slot_gap(self):
        config = LidarParkingConfig(
            angle_offset_deg=0.0,
            clockwise_angles=False,
            quality_min=1,
            car_detection_roi=RectangleRoi(500.0, 1500.0, -1800.0, 1800.0),
            min_observed_points=5,
            car_cluster_radius_mm=120.0,
            car_cluster_min_points=2,
            gap_cluster_min_points=5,
            gap_pair_min_points=12,
            expected_observed_gap_mm=1300.0,
            observed_gap_min_mm=1100.0,
            observed_gap_max_mm=1500.0,
            gap_confirm_scans=1,
            gap_center_y_back_min_mm=-100.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        weak_points = tuple(
            point_at(x, y)
            for x, y in (
                (950.0, -720.0), (980.0, -705.0),
                (1020.0, -695.0), (1050.0, -680.0),
                (950.0, 620.0), (980.0, 635.0),
                (1020.0, 645.0), (1050.0, 660.0),
            )
        )

        observation = estimator.estimate(LidarScan(1.0, weak_points), now=1.0)

        self.assertTrue(observation.valid)
        self.assertEqual(observation.car_count, 2)
        self.assertTrue(observation.second_car_seen)
        self.assertFalse(observation.gap_found)
        self.assertFalse(observation.gap_confirmed)
        self.assertEqual(observation.reason, "two_cars_no_valid_gap")

    def test_strong_two_cluster_returns_can_confirm_slot_gap(self):
        config = LidarParkingConfig(
            angle_offset_deg=0.0,
            clockwise_angles=False,
            quality_min=1,
            car_detection_roi=RectangleRoi(500.0, 1500.0, -1800.0, 1800.0),
            min_observed_points=5,
            car_cluster_radius_mm=120.0,
            car_cluster_min_points=2,
            gap_cluster_min_points=5,
            gap_pair_min_points=12,
            expected_observed_gap_mm=1300.0,
            observed_gap_min_mm=1100.0,
            observed_gap_max_mm=1500.0,
            gap_confirm_scans=1,
            gap_center_y_back_min_mm=-100.0,
            gap_center_smooth_alpha=1.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        strong_points = tuple(
            point_at(x, y)
            for x, y in (
                (940.0, -730.0), (965.0, -718.0), (990.0, -706.0),
                (1015.0, -694.0), (1040.0, -682.0), (1065.0, -670.0),
                (940.0, 610.0), (965.0, 622.0), (990.0, 634.0),
                (1015.0, 646.0), (1040.0, 658.0), (1065.0, 670.0),
            )
        )

        observation = estimator.estimate(LidarScan(1.0, strong_points), now=1.0)

        self.assertTrue(observation.gap_found)
        self.assertTrue(observation.gap_confirmed)
        self.assertAlmostEqual(observation.gap_width_mm, 1300.0, delta=80.0)

    def test_initial_slot_candidate_must_be_rearward_enough(self):
        config = LidarParkingConfig(
            angle_offset_deg=0.0,
            clockwise_angles=False,
            quality_min=1,
            car_detection_roi=RectangleRoi(500.0, 1500.0, -1800.0, 1800.0),
            min_observed_points=5,
            car_cluster_radius_mm=120.0,
            car_cluster_min_points=2,
            gap_cluster_min_points=5,
            gap_pair_min_points=10,
            expected_observed_gap_mm=1300.0,
            observed_gap_min_mm=1100.0,
            observed_gap_max_mm=1500.0,
            gap_center_y_back_min_mm=200.0,
            gap_confirm_scans=1,
            gap_center_smooth_alpha=1.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        strong_points = tuple(
            point_at(x, y)
            for x, y in (
                (940.0, -730.0), (965.0, -718.0), (990.0, -706.0),
                (1015.0, -694.0), (1040.0, -682.0), (1065.0, -670.0),
                (940.0, 610.0), (965.0, 622.0), (990.0, 634.0),
                (1015.0, 646.0), (1040.0, 658.0), (1065.0, 670.0),
            )
        )

        observation = estimator.estimate(LidarScan(1.0, strong_points), now=1.0)

        self.assertEqual(observation.car_count, 2)
        self.assertFalse(observation.gap_found)
        self.assertFalse(observation.gap_confirmed)
        self.assertEqual(observation.reason, "two_cars_no_valid_gap")

    def test_interrupted_candidate_confirms_without_waiting_for_consecutive_scans(self):
        config = LidarParkingConfig(
            angle_offset_deg=0.0,
            clockwise_angles=False,
            quality_min=1,
            car_detection_roi=RectangleRoi(500.0, 1500.0, -1800.0, 1800.0),
            min_observed_points=5,
            car_cluster_radius_mm=120.0,
            car_cluster_min_points=2,
            gap_cluster_min_points=5,
            gap_pair_min_points=10,
            expected_observed_gap_mm=1300.0,
            observed_gap_min_mm=1100.0,
            observed_gap_max_mm=1500.0,
            gap_center_y_back_min_mm=200.0,
            gap_confirm_scans=3,
            gap_candidate_hold_s=1.2,
            gap_center_smooth_alpha=1.0,
        )
        estimator = LidarParkingSpaceEstimator(config)

        def shifted_two_car_points(offset):
            return tuple(
                point_at(x, y + offset)
                for x, y in (
                    (940.0, -730.0), (965.0, -718.0), (990.0, -706.0),
                    (1015.0, -694.0), (1040.0, -682.0), (1065.0, -670.0),
                    (940.0, 610.0), (965.0, 622.0), (990.0, 634.0),
                    (1015.0, 646.0), (1040.0, 658.0), (1065.0, 670.0),
                )
            )

        first = estimator.estimate(LidarScan(1.0, shifted_two_car_points(350.0)), now=1.0)
        missed = estimator.estimate(
            LidarScan(1.5, shifted_two_car_points(350.0)[:6]),
            now=1.5,
        )
        second = estimator.estimate(LidarScan(2.0, shifted_two_car_points(400.0)), now=2.0)
        third = estimator.estimate(LidarScan(2.8, shifted_two_car_points(450.0)), now=2.8)

        self.assertTrue(first.gap_found)
        self.assertFalse(first.gap_confirmed)
        self.assertFalse(missed.gap_found)
        self.assertTrue(second.gap_found)
        self.assertFalse(second.gap_confirmed)
        self.assertTrue(third.gap_confirmed)

    def test_confirmed_slot_tracks_one_visible_border_cluster(self):
        config = replace(
            self.make_estimator().config,
            gap_confirm_scans=1,
            gap_single_cluster_track_enabled=True,
            gap_single_cluster_max_edge_jump_mm=700.0,
            slot_tracking_roi=RectangleRoi(-1800.0, 1800.0, -2500.0, 2500.0),
            gap_center_smooth_alpha=1.0,
            gap_orientation_smooth_alpha=1.0,
        )
        estimator = LidarParkingSpaceEstimator(config)
        initial = estimator.estimate(
            LidarScan(1.0, self.two_car_points(offset=350.0)),
            now=1.0,
        )
        first_car_only = tuple(
            point_at(x, y + 500.0)
            for x, y in (
                (390.0, -720.0),
                (400.0, -700.0),
                (410.0, -680.0),
            )
        )

        tracked = estimator.estimate(LidarScan(2.0, first_car_only), now=2.0)

        self.assertTrue(initial.gap_confirmed)
        self.assertEqual(tracked.car_count, 1)
        self.assertTrue(tracked.gap_confirmed)
        self.assertFalse(tracked.gap_pair_observed)
        self.assertFalse(tracked.coasted)
        self.assertLess(
            tracked.gap_center_x_right_mm,
            initial.gap_center_x_right_mm - 500.0,
        )
        self.assertAlmostEqual(
            tracked.gap_center_y_back_mm - initial.gap_center_y_back_mm,
            150.0,
            delta=1.0,
        )

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
            gap_center_y_back_min_mm=-100.0,
            gap_pair_min_points=10,
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

    def test_official_width_override_keeps_detected_center_and_exact_size(self):
        estimator = self.make_estimator()
        observation = estimator.estimate(
            LidarScan(1.0, self.two_car_points()),
            now=1.0,
        )

        polygon = infer_dynamic_slot_polygon(
            observation,
            depth_mm=1500.0,
            width_mm=950.0,
        )

        self.assertIsNotNone(polygon)
        width = math.hypot(
            polygon[1][0] - polygon[0][0],
            polygon[1][1] - polygon[0][1],
        )
        depth = math.hypot(
            polygon[3][0] - polygon[0][0],
            polygon[3][1] - polygon[0][1],
        )
        self.assertAlmostEqual(width, 950.0, delta=0.01)
        self.assertAlmostEqual(depth, 1500.0, delta=0.01)

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

    def test_lidar_csv_recorder_writes_each_scan_once(self):
        first = LidarScan(
            10.0,
            (
                LidarPoint(15, 1.0, 1000.0),
                LidarPoint(8, 2.0, 1100.0),
            ),
        )
        second = LidarScan(10.1, (LidarPoint(12, 3.0, 1200.0),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_lidar.csv"
            recorder = LidarCsvRecorder(path)
            self.assertTrue(recorder.write(first))
            self.assertFalse(recorder.write(first))
            self.assertTrue(recorder.write(second))
            recorder.close()
            scans = load_lidar_csv(str(path))

        self.assertEqual(recorder.scans_written, 2)
        self.assertEqual(recorder.points_written, 3)
        self.assertEqual(scans, [first, second])


if __name__ == "__main__":
    unittest.main()
