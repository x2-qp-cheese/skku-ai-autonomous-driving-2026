import unittest

from skku_autocar.estimation.parking_geometry import ParkingGeometry
from skku_autocar.planning.reverse_parking_path import (
    ReverseParkingPathGenerator,
    ReversePathConfig,
)


def geometry(target_x=300.0, target_y=100.0):
    return ParkingGeometry(
        found=True,
        has_side_pair=True,
        has_back_line=True,
        vehicle_x_px=300.0,
        vehicle_y_px=570.0,
        slot_direction_x=0.0,
        slot_direction_y=-1.0,
        stop_target_x_px=target_x,
        stop_target_y_px=target_y,
    )


class ReverseParkingPathTest(unittest.TestCase):
    def make_generator(self):
        return ReverseParkingPathGenerator(
            ReversePathConfig(
                samples=31,
                start_tangent_px=100.0,
                end_tangent_px=100.0,
                lookahead_px=80.0,
                maximum_curvature_per_px=0.05,
            )
        )

    def test_path_starts_at_vehicle_and_ends_at_local_lookahead(self):
        path = self.make_generator().generate(geometry())

        self.assertTrue(path.found)
        self.assertEqual(len(path.points), 31)
        self.assertAlmostEqual(path.points[0][0], 300.0)
        self.assertAlmostEqual(path.points[0][1], 570.0)
        self.assertAlmostEqual(path.points[-1][0], 300.0)
        self.assertAlmostEqual(path.points[-1][1], 490.0)
        self.assertAlmostEqual(path.lookahead_point[1], 490.0)
        self.assertEqual(path.reason, "local_target_ready")
        self.assertAlmostEqual(path.curvature_per_px, 0.0, places=5)

    def test_target_to_right_generates_signed_curvature(self):
        path = self.make_generator().generate(geometry(target_x=380.0))

        self.assertTrue(path.found)
        self.assertGreater(path.curvature_per_px, 0.0)
        self.assertIsNotNone(path.lookahead_point)

    def test_back_line_is_required(self):
        incomplete = ParkingGeometry(
            found=True,
            has_side_pair=True,
            vehicle_x_px=300.0,
            vehicle_y_px=570.0,
        )

        path = self.make_generator().generate(incomplete)

        self.assertFalse(path.found)
        self.assertEqual(path.reason, "parking_back_line_missing")


if __name__ == "__main__":
    unittest.main()
