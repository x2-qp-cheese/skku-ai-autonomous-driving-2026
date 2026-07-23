import math
import unittest

from skku_autocar.estimation.locked_slot import (
    LockedSlotGeometryEstimator,
    LockedSlotTracker,
    LockedSlotTrackerConfig,
)
from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.estimation.parking_lidar import LidarParkingObservation


def static_scene():
    points = []
    for index in range(45):
        x = -1200.0 + index * 55.0
        points.append((x, 420.0 + 0.00025 * x * x))
    for index in range(35):
        y = -1100.0 + index * 67.0
        points.append((1350.0, y))
    return points


class LockedSlotTrackerTest(unittest.TestCase):
    def make_tracker(self, hold_scans=2):
        return LockedSlotTracker(
            LockedSlotTrackerConfig(
                min_points=10,
                max_points=120,
                max_correspondence_mm=180.0,
                trim_ratio=0.8,
                iterations=8,
                max_translation_per_scan_mm=100.0,
                max_rotation_per_scan_deg=5.0,
                max_hold_scans=hold_scans,
            )
        )

    def test_consecutive_scan_motion_moves_same_locked_rectangle(self):
        previous = static_scene()
        # The LiDAR moved +20 mm right and +12 mm rearward in the world, so
        # stationary world features and the physical bay both appear offset by
        # the inverse amount in the new sensor frame.
        current = [(x - 20.0, y - 12.0) for x, y in previous]
        polygon = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))
        tracker = self.make_tracker()

        locked = tracker.lock(polygon, previous)
        tracked = tracker.update(current)

        self.assertTrue(locked.locked)
        self.assertTrue(tracked.tracked, tracked.reason)
        self.assertAlmostEqual(tracked.polygon[0][0], -495.0, delta=3.0)
        self.assertAlmostEqual(tracked.polygon[0][1], -12.0, delta=3.0)
        width = math.hypot(
            tracked.polygon[1][0] - tracked.polygon[0][0],
            tracked.polygon[1][1] - tracked.polygon[0][1],
        )
        depth = math.hypot(
            tracked.polygon[3][0] - tracked.polygon[0][0],
            tracked.polygon[3][1] - tracked.polygon[0][1],
        )
        self.assertAlmostEqual(width, 950.0, delta=0.01)
        self.assertAlmostEqual(depth, 1500.0, delta=0.01)

    def test_missing_scans_hold_then_report_lost_without_resizing_box(self):
        tracker = self.make_tracker(hold_scans=2)
        polygon = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))
        tracker.lock(polygon, static_scene())

        first = tracker.update([])
        second = tracker.update([])
        lost = tracker.update([])

        self.assertTrue(first.held)
        self.assertTrue(second.held)
        self.assertEqual(first.polygon, polygon)
        self.assertTrue(lost.lost)
        self.assertEqual(lost.reason, "locked_slot_lost")

    def test_eight_nearby_points_are_enough_for_stationary_slot_tracking(self):
        tracker = LockedSlotTracker(
            LockedSlotTrackerConfig(
                min_points=8,
                max_correspondence_mm=180.0,
                trim_ratio=1.0,
                iterations=4,
            )
        )
        points = [
            (-700.0, 450.0),
            (-500.0, 420.0),
            (-300.0, 410.0),
            (-100.0, 405.0),
            (100.0, 405.0),
            (300.0, 410.0),
            (500.0, 420.0),
            (700.0, 450.0),
        ]
        polygon = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))

        locked = tracker.lock(polygon, points)
        tracked = tracker.update(points)

        self.assertTrue(locked.locked)
        self.assertTrue(tracked.tracked, tracked.reason)
        self.assertFalse(tracked.held)

    def test_reanchor_replaces_accumulated_pose_without_resizing_box(self):
        tracker = self.make_tracker()
        points = static_scene()
        initial = ((-475.0, 0.0), (475.0, 0.0), (475.0, 1500.0), (-475.0, 1500.0))
        corrected = ((-275.0, 100.0), (675.0, 100.0), (675.0, 1600.0), (-275.0, 1600.0))
        tracker.lock(initial, points)

        pose = tracker.reanchor(corrected, points)

        self.assertTrue(pose.tracked)
        self.assertEqual(pose.reason, "locked_slot_reanchored")
        self.assertEqual(pose.polygon, corrected)
        self.assertAlmostEqual(
            math.hypot(
                pose.polygon[1][0] - pose.polygon[0][0],
                pose.polygon[1][1] - pose.polygon[0][1],
            ),
            950.0,
            delta=0.01,
        )


class RecordingProjector:
    def __init__(self):
        self.polygons = []

    def project(self, observation):
        return ParkingGeometry(reason="dynamic_slot")

    def project_polygon(self, polygon, **kwargs):
        self.polygons.append(tuple(polygon))
        return ParkingGeometry(reason=kwargs["reason"])


class LockedSlotGeometryEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.points = static_scene()
        self.projector = RecordingProjector()
        self.estimator = LockedSlotGeometryEstimator(
            self.projector,
            LockedSlotTracker(
                LockedSlotTrackerConfig(
                    min_points=10,
                    max_points=120,
                    max_correspondence_mm=180.0,
                    trim_ratio=0.8,
                    iterations=8,
                )
            ),
            width_mm=950.0,
            depth_mm=1500.0,
        )

    @staticmethod
    def observation(center_x, *, pair_observed):
        return LidarParkingObservation(
            valid=True,
            car_count=2 if pair_observed else 1,
            gap_found=True,
            gap_confirmed=True,
            gap_pair_observed=pair_observed,
            gap_near_edge_x_right_mm=center_x,
            gap_near_edge_y_back_mm=-650.0,
            gap_far_edge_x_right_mm=center_x,
            gap_far_edge_y_back_mm=650.0,
            slot_depth_x_right=1.0,
            slot_depth_y_back=0.0,
        )

    def test_fresh_two_car_gap_reanchors_locked_slot(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        geometry = self.estimator.update(
            self.observation(700.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "locked_slot_reanchored")
        self.assertAlmostEqual(
            sum(point[0] for point in self.estimator.pose.polygon) / 4.0,
            1450.0,
            delta=0.01,
        )

    def test_single_border_tracking_does_not_reanchor_slot(self):
        self.estimator.update(
            self.observation(1000.0, pair_observed=True),
            self.points,
            lock_requested=True,
        )
        initial_polygon = self.estimator.pose.polygon

        geometry = self.estimator.update(
            self.observation(700.0, pair_observed=False),
            self.points,
            lock_requested=True,
        )

        self.assertEqual(geometry.reason, "locked_slot_tracked")
        for before, after in zip(initial_polygon, self.estimator.pose.polygon):
            self.assertAlmostEqual(after[0], before[0], delta=0.01)
            self.assertAlmostEqual(after[1], before[1], delta=0.01)


if __name__ == "__main__":
    unittest.main()
