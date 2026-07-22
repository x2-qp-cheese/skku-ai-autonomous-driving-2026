import unittest

import numpy as np

from skku_autocar.estimation.lane_geometry import LaneGeometry
from skku_autocar.planning.lane_change import LaneChangeConfig, LaneChangeController
from skku_autocar.planning.local_occupancy import (
    LocalOccupancyConfig,
    LocalOccupancyGrid,
)
from skku_autocar.planning.obstacle_fusion import (
    ObstacleFusionConfig,
    ObstacleFusionPlanner,
)
from skku_autocar.sensors.ultrasonic import UltrasonicSnapshot


SHAPE = (100, 200)
CENTERLINE = [(140.0, float(y)) for y in range(0, 100, 5)]


def obstacle(x0=120, x1=160, y0=60, y1=95):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


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


class LocalOccupancyGridTest(unittest.TestCase):
    def test_grid_inflates_and_keeps_instance_boundaries(self):
        grid = LocalOccupancyGrid(
            LocalOccupancyConfig(inflation_radius_px=4)
        )

        snapshot = grid.update([obstacle()], SHAPE, 0.9, 0.0, True)

        self.assertTrue(snapshot.found)
        self.assertEqual(len(snapshot.instances), 1)
        self.assertGreater(np.count_nonzero(snapshot.mask), 40 * 35)
        self.assertGreaterEqual(snapshot.confidence, 0.75)

    def test_grid_bridges_short_detection_dropout_then_decays(self):
        grid = LocalOccupancyGrid(
            LocalOccupancyConfig(
                decay_seconds=0.20,
                inflation_radius_px=0,
            )
        )

        detected = grid.update([obstacle()], SHAPE, 1.0, 0.0, True)
        short_dropout = grid.update([], SHAPE, 0.0, 0.05, True)
        expired = grid.update([], SHAPE, 0.0, 1.0, True)

        self.assertTrue(detected.found)
        self.assertTrue(short_dropout.found)
        self.assertFalse(expired.found)

    def test_pausing_clears_stale_map(self):
        grid = LocalOccupancyGrid(LocalOccupancyConfig(inflation_radius_px=0))
        grid.update([obstacle()], SHAPE, 1.0, 0.0, True)

        paused = grid.update([], SHAPE, 0.0, 0.1, False)

        self.assertFalse(paused.found)
        self.assertEqual(np.count_nonzero(paused.mask), 0)

    def test_fusion_planner_uses_map_to_select_clear_adjacent_lane(self):
        grid = LocalOccupancyGrid(LocalOccupancyConfig(inflation_radius_px=0))
        mapped = grid.update([obstacle()], SHAPE, 1.0, 0.0, True)
        controller = LaneChangeController(
            LaneChangeConfig(
                mode="external",
                target_lane_width_px=60.0,
                stable_required_frames=0,
            )
        )
        planner = ObstacleFusionPlanner(
            ObstacleFusionConfig(
                fusion_mode="yolo",
                lane_width_px=60.0,
                visual_confirm_frames=1,
                visual_trigger_y_ratio=0.20,
                target_block_y_ratio=0.20,
                path_half_width_px=22.0,
                min_path_overlap_ratio=0.10,
            )
        )

        event = planner.update(
            mapped.instances,
            SHAPE,
            CENTERLINE,
            lane(),
            controller,
            UltrasonicSnapshot(),
            0.0,
            True,
            obstacle_confidence=mapped.confidence,
        )

        self.assertIsNotNone(event)
        self.assertIn("lane2 -> lane1", event)
        self.assertEqual(controller.state, "armed")

    def test_fusion_planner_rejects_occupied_adjacent_lane(self):
        grid = LocalOccupancyGrid(LocalOccupancyConfig(inflation_radius_px=0))
        mapped = grid.update(
            [obstacle(), obstacle(60, 100, 60, 95)],
            SHAPE,
            1.0,
            0.0,
            True,
        )
        controller = LaneChangeController(
            LaneChangeConfig(mode="external", target_lane_width_px=60.0)
        )
        planner = ObstacleFusionPlanner(
            ObstacleFusionConfig(
                fusion_mode="yolo",
                lane_width_px=60.0,
                visual_confirm_frames=1,
                visual_trigger_y_ratio=0.20,
                target_block_y_ratio=0.20,
                path_half_width_px=22.0,
                min_path_overlap_ratio=0.10,
            )
        )

        event = planner.update(
            mapped.instances,
            SHAPE,
            CENTERLINE,
            lane(),
            controller,
            UltrasonicSnapshot(),
            0.0,
            True,
            obstacle_confidence=mapped.confidence,
        )

        self.assertIsNone(event)
        self.assertTrue(planner.observation.target_blocked)
        self.assertEqual(controller.state, "lane2")


if __name__ == "__main__":
    unittest.main()
