import unittest

import numpy as np

from skku_autocar.estimation.bev_corridor import (
    BevClassMasks,
    BevCorridorConfig,
    BevCorridorLaneEstimator,
)


def line_mask(x: int, shape=(100, 200)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, x : x + 4] = 255
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


if __name__ == "__main__":
    unittest.main()
