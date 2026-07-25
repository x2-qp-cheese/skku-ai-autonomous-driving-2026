import unittest
from types import SimpleNamespace

import numpy as np

from skku_autocar.estimation.bev_corridor import (
    BevClassMasks,
    BevCorridorConfig,
    BevCorridorLaneEstimator,
    warp_class_masks,
)


def line_mask(x: int, shape=(100, 200)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, x : x + 4] = 255
    return mask


def slanted_line_mask(x_at_target: float, slope: float, shape=(100, 200)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    target_y = shape[0] * BevCorridorConfig.lookahead_y_ratio
    for y in range(shape[0]):
        x = int(round(x_at_target + slope * (y - target_y)))
        if 0 <= x < shape[1] - 3:
            mask[y, x : x + 4] = 255
    return mask


def crosswalk_mask(shape=(100, 200)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[40:60, :] = 255
    return mask


def bev_at(center_x: int, *, crosswalk: bool) -> BevClassMasks:
    return BevClassMasks(
        center=[line_mask(center_x)],
        crosswalk=[crosswalk_mask()] if crosswalk else [],
        center_conf=1.0,
        shape=(100, 200),
    )


class BevCorridorCrosswalkTest(unittest.TestCase):
    def test_crosswalk_tracks_lane_with_stronger_smoothing(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                crosswalk_lane_width_px=60.0,
                center_smooth_alpha=1.0,
                crosswalk_center_smooth_alpha=0.1,
                crosswalk_max_center_jump_px=30.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        before = estimator.estimate(bev_at(60, crosswalk=False))
        during = estimator.estimate(bev_at(70, crosswalk=True))

        self.assertAlmostEqual(before.center_x, 91.5, delta=0.2)
        self.assertAlmostEqual(during.center_x, 92.5, delta=0.2)
        self.assertEqual(during.reason, "corridor_tier2")
        self.assertEqual(estimator.last_class_name, "crosswalk-virtual-center")

    def test_crosswalk_specific_jump_gate_coasts_on_outlier(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                crosswalk_lane_width_px=60.0,
                center_smooth_alpha=1.0,
                crosswalk_center_smooth_alpha=0.1,
                max_center_jump_px=80.0,
                crosswalk_max_center_jump_px=15.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        before = estimator.estimate(bev_at(60, crosswalk=False))
        outlier = estimator.estimate(bev_at(80, crosswalk=True))

        self.assertAlmostEqual(outlier.center_x, before.center_x)
        self.assertTrue(outlier.reason.startswith("coast:center_jump"))
        self.assertEqual(estimator.last_class_name, "coast")

    def test_heading_jump_gate_coasts_on_slanted_outlier(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                max_center_jump_px=80.0,
                max_heading_jump=0.08,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        before = estimator.estimate(bev_at(60, crosswalk=False))
        outlier = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(60.0, 0.25)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertAlmostEqual(outlier.center_x, before.center_x)
        self.assertTrue(outlier.reason.startswith("coast:heading_jump"))
        self.assertEqual(estimator.last_class_name, "coast")

    def test_crosswalk_option_b_follows_right_boundary_offset(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                crosswalk_option="b",
                crosswalk_right_offset_px=30.0,
                center_smooth_alpha=1.0,
                crosswalk_center_smooth_alpha=1.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        lane = estimator.estimate(BevClassMasks(
            side=[line_mask(160)],
            crosswalk=[crosswalk_mask()],
            side_conf=1.0,
            shape=(100, 200),
        ))

        self.assertTrue(lane.found)
        self.assertAlmostEqual(lane.center_x, 131.5, delta=0.2)
        self.assertEqual(lane.reason, "corridor_tier3")
        self.assertEqual(estimator.last_class_name, "crosswalk-right-side-b")

    def test_crosswalk_option_b_holds_previous_right_lane_geometry(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                crosswalk_option="b",
                crosswalk_right_offset_px=30.0,
                center_smooth_alpha=1.0,
                crosswalk_center_smooth_alpha=1.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        before = estimator.estimate(
            BevClassMasks(
                center=[line_mask(60)],
                side=[line_mask(120)],
                center_conf=1.0,
                side_conf=1.0,
                shape=(100, 200),
            )
        )
        during = estimator.estimate(
            BevClassMasks(
                side=[line_mask(90)],
                crosswalk=[crosswalk_mask()],
                side_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(during.found)
        self.assertAlmostEqual(during.center_x, before.center_x, delta=0.2)
        self.assertAlmostEqual(during.lateral_error_norm, before.lateral_error_norm)
        self.assertEqual(estimator.last_class_name, "crosswalk-hold-right-lane")

    def test_center_anchor_does_not_push_target_past_detected_right_boundary(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=120.0,
                min_lane_width_px=60.0,
                center_anchor=True,
                centerline_bias=0.5,
                center_smooth_alpha=1.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        lane = estimator.estimate(
            BevClassMasks(
                center=[line_mask(60)],
                side=[line_mask(110)],
                center_conf=1.0,
                side_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(lane.found)
        self.assertEqual(lane.reason, "corridor_tier1")
        self.assertEqual(estimator.last_class_name, "center+right-side")
        self.assertAlmostEqual(lane.center_x, 86.5, delta=0.2)
        self.assertLess(lane.center_x, 111.5)

    def test_virtual_hold_preserves_last_curve_direction(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                max_coast_frames=0,
                virtual_hold=True,
                virtual_hold_recenter_alpha=0.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        before = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(60.0, 0.20)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        virtual = estimator.estimate(BevClassMasks(shape=(100, 200)))

        self.assertTrue(virtual.found)
        self.assertEqual(estimator.last_class_name, "virtual-hold")
        self.assertTrue(virtual.reason.startswith("virtual_hold:no_corridor"))
        self.assertAlmostEqual(virtual.center_x, before.center_x, delta=0.2)
        self.assertAlmostEqual(virtual.heading_error, before.heading_error, delta=0.01)
        self.assertNotAlmostEqual(
            estimator.last_centerline_bev[0][0],
            estimator.last_centerline_bev[-1][0],
            delta=1.0,
        )

    def test_virtual_hold_recenter_shifts_curve_without_flattening_it(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                max_coast_frames=0,
                virtual_hold=True,
                virtual_hold_recenter_alpha=0.25,
                vehicle_center_x_offset_ratio=0.0,
            )
        )

        before = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(60.0, 0.20)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        virtual = estimator.estimate(BevClassMasks(shape=(100, 200)))

        self.assertLess(abs(virtual.lateral_error_norm), abs(before.lateral_error_norm))
        self.assertAlmostEqual(virtual.heading_error, before.heading_error, delta=0.01)

    def test_disabled_obstacle_mode_skips_obstacle_bev_warp(self):
        transformer = CountingTransformer()
        masks = SimpleNamespace(
            center=[],
            side=[],
            lane=[],
            crosswalk=[],
            obstacle=[np.ones((2, 2), dtype=np.uint8)],
            center_conf=0.0,
            side_conf=0.0,
            lane_conf=0.0,
            crosswalk_conf=0.0,
            obstacle_conf=0.9,
        )

        bev = warp_class_masks(transformer, masks, include_obstacle=False)

        self.assertEqual(transformer.warp_calls, 0)
        self.assertEqual(bev.obstacle, [])
        self.assertEqual(bev.obstacle_conf, 0.0)


class CountingTransformer:
    out_size = (20, 10)

    def __init__(self):
        self.warp_calls = 0

    def warp_mask(self, mask):
        self.warp_calls += 1
        return mask


if __name__ == "__main__":
    unittest.main()
