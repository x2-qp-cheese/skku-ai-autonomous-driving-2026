import math
import unittest

from skku_autocar.estimation.lidar_slot_geometry import LidarSlotGeometryProjector
from skku_autocar.estimation.parking_geometry import ParkingGeometryConfig
from skku_autocar.estimation.parking_lidar import (
    LidarParkingConfig,
    LidarParkingObservation,
)
from skku_autocar.planning.reverse_parking_path import (
    ReverseParkingPathGenerator,
    ReversePathConfig,
)


def slot_observation(
    *,
    center_x=0.0,
    center_y=500.0,
    heading_deg=0.0,
    confirmed=True,
    coasted=False,
):
    angle = math.radians(heading_deg)
    depth = (math.sin(angle), math.cos(angle))
    width_axis = (math.cos(angle), -math.sin(angle))
    half_width = 475.0
    first = (
        center_x - width_axis[0] * half_width,
        center_y - width_axis[1] * half_width,
    )
    second = (
        center_x + width_axis[0] * half_width,
        center_y + width_axis[1] * half_width,
    )
    return LidarParkingObservation(
        timestamp=1.0,
        valid=not coasted,
        observed_points=20 if not coasted else 0,
        car_count=2 if not coasted else 0,
        gap_found=True,
        gap_confirmed=confirmed,
        gap_width_mm=950.0,
        gap_near_edge_x_right_mm=first[0],
        gap_near_edge_y_back_mm=first[1],
        gap_far_edge_x_right_mm=second[0],
        gap_far_edge_y_back_mm=second[1],
        gap_center_x_right_mm=center_x,
        gap_center_y_back_mm=center_y,
        slot_depth_x_right=depth[0],
        slot_depth_y_back=depth[1],
        coasted=coasted,
        reason="gap_confirmed" if not coasted else "insufficient_lidar_points",
    )


class LidarSlotGeometryProjectorTest(unittest.TestCase):
    def setUp(self):
        self.lidar_config = LidarParkingConfig(
            parking_space_width_mm=950.0,
            parking_space_depth_mm=1500.0,
            sensor_to_rear_axle_y_back_mm=-300.0,
        )
        self.geometry_config = ParkingGeometryConfig(
            expected_slot_width_px=220.0,
            desired_back_clearance_px=60.0,
            min_geometry_confidence=0.20,
        )
        self.projector = LidarSlotGeometryProjector(
            self.lidar_config,
            self.geometry_config,
            canvas_width=600,
            canvas_height=600,
        )

    def test_confirmed_box_creates_all_virtual_lines_and_reverse_path(self):
        geometry = self.projector.project(slot_observation())

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertTrue(geometry.has_back_line)
        self.assertEqual(geometry.reason, "lidar_slot_box")
        self.assertEqual(geometry.observed_line_count, 0)
        self.assertAlmostEqual(geometry.slot_width_px, 220.0, delta=0.1)
        self.assertAlmostEqual(geometry.heading_error_deg, 0.0, delta=0.1)
        self.assertAlmostEqual(geometry.lateral_error_norm, 0.0, delta=0.1)
        self.assertGreater(geometry.depth_remaining_px, 45.0)

        path = ReverseParkingPathGenerator(
            ReversePathConfig(maximum_curvature_per_px=0.05)
        ).generate(geometry)
        self.assertTrue(path.found, path.reason)

    def test_unconfirmed_box_cannot_arm_path(self):
        geometry = self.projector.project(slot_observation(confirmed=False))

        self.assertFalse(geometry.found)
        self.assertTrue(geometry.has_side_pair)
        self.assertEqual(geometry.reason, "lidar_slot_box_confirming")

    def test_rotated_box_keeps_its_heading_in_bev(self):
        geometry = self.projector.project(slot_observation(heading_deg=30.0))

        self.assertTrue(geometry.found)
        self.assertAlmostEqual(geometry.heading_error_deg, 30.0, delta=0.1)
        self.assertGreater(geometry.slot_direction_x, 0.0)
        self.assertLess(geometry.slot_direction_y, 0.0)

    def test_box_motion_toward_rear_axle_reduces_remaining_depth(self):
        first = self.projector.project(slot_observation(center_y=500.0))
        closer = self.projector.project(slot_observation(center_y=200.0))

        self.assertLess(closer.depth_remaining_px, first.depth_remaining_px)

    def test_coasted_box_stays_visible_but_reports_hold(self):
        geometry = self.projector.project(slot_observation(coasted=True))

        self.assertTrue(geometry.found)
        self.assertTrue(geometry.coasted)
        self.assertEqual(geometry.reason, "lidar_slot_box_hold")
        self.assertLess(geometry.confidence, 0.95)

    def test_missing_box_returns_no_geometry(self):
        geometry = self.projector.project(LidarParkingObservation(reason="no_scan"))

        self.assertFalse(geometry.found)
        self.assertEqual(geometry.reason, "lidar_slot_box_unavailable")


if __name__ == "__main__":
    unittest.main()
