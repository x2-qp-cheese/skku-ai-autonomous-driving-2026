import unittest

from skku_autocar.estimation.parking_fusion import (
    ParkingFusionConfig,
    fuse_parking_geometry,
)
from skku_autocar.estimation.parking_geometry import ParkingGeometry


def lidar_geometry():
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=True,
        lateral_error_norm=0.10,
        heading_error_deg=5.0,
        depth_remaining_px=180.0,
        stop_target_x_px=310.0,
        stop_target_y_px=180.0,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        slot_direction_x=0.10,
        slot_direction_y=-0.99,
        vehicle_inside_ratio=0.85,
        vehicle_fully_inside=True,
        confidence=0.95,
        reason="locked_slot_tracked",
    )


def camera_geometry(*, lateral=0.18, heading=8.0, back=True):
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=back,
        lateral_error_norm=lateral,
        heading_error_deg=heading,
        depth_remaining_px=160.0 if back else None,
        stop_target_x_px=320.0 if back else None,
        stop_target_y_px=175.0 if back else None,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        slot_direction_x=0.12,
        slot_direction_y=-0.99,
        confidence=0.70,
        observed_line_count=3 if back else 2,
        reason="parking_bay" if back else "side_pair",
    )


class ParkingFusionTest(unittest.TestCase):
    def test_compatible_camera_geometry_takes_over_but_keeps_inside_state(self):
        fused = fuse_parking_geometry(lidar_geometry(), camera_geometry())

        self.assertEqual(fused.reason, "camera_lidar_fused")
        self.assertAlmostEqual(fused.heading_error_deg, 8.0)
        self.assertTrue(fused.vehicle_fully_inside)
        self.assertAlmostEqual(fused.vehicle_inside_ratio, 0.85)

    def test_neighboring_camera_slot_is_rejected_by_disagreement_gate(self):
        lidar = lidar_geometry()
        fused = fuse_parking_geometry(
            lidar,
            camera_geometry(lateral=1.4, heading=8.0),
            ParkingFusionConfig(max_lateral_disagreement_norm=0.4),
        )

        self.assertIs(fused, lidar)

    def test_side_only_camera_updates_alignment_without_replacing_lidar_depth(self):
        fused = fuse_parking_geometry(
            lidar_geometry(),
            camera_geometry(lateral=-0.20, heading=-6.0, back=False),
        )

        self.assertEqual(fused.reason, "camera_side_lidar_depth")
        self.assertAlmostEqual(fused.lateral_error_norm, -0.20)
        self.assertAlmostEqual(fused.heading_error_deg, -6.0)
        self.assertAlmostEqual(fused.depth_remaining_px, 180.0)
        self.assertTrue(fused.vehicle_fully_inside)


if __name__ == "__main__":
    unittest.main()
