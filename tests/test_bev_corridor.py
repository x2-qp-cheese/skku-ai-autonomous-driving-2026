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


def crosswalk_mask_at(top: int, shape=(100, 200)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top : top + 20, :] = 255
    return mask


def bev_at(center_x: int, *, crosswalk: bool) -> BevClassMasks:
    return BevClassMasks(
        center=[line_mask(center_x)],
        crosswalk=[crosswalk_mask()] if crosswalk else [],
        center_conf=1.0,
        shape=(100, 200),
    )


class BevCorridorCrosswalkTest(unittest.TestCase):
    def test_competition_transit_prefers_fresh_lane_over_cache(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                path_smooth_alpha=1.0,
                crosswalk_transit_enabled=True,
                crosswalk_transit_recenter_alpha=0.0,
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

        during = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(120.0, -0.20)],
                crosswalk=[crosswalk_mask()],
                center_conf=1.0,
                crosswalk_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertEqual(during.reason, "corridor_tier2")
        self.assertNotEqual(during.center_x, before.center_x)
        self.assertNotEqual(during.path_points, before.path_points)

    def test_competition_transit_advances_cache_only_when_lane_is_hidden(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                path_smooth_alpha=1.0,
                crosswalk_transit_enabled=True,
                crosswalk_transit_advance_smooth_alpha=1.0,
                crosswalk_transit_max_advance_px=18.0,
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
        first_hold = estimator.estimate(
            BevClassMasks(
                crosswalk=[crosswalk_mask_at(20)],
                crosswalk_conf=1.0,
                shape=(100, 200),
            )
        )
        advanced_hold = estimator.estimate(
            BevClassMasks(
                crosswalk=[crosswalk_mask_at(30)],
                crosswalk_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(first_hold.reason.startswith("crosswalk_transit_hold:"))
        self.assertTrue(advanced_hold.reason.startswith("crosswalk_transit_hold:"))
        self.assertAlmostEqual(first_hold.center_x, before.center_x, delta=0.1)
        self.assertLess(advanced_hold.center_x, first_hold.center_x)

    def test_competition_transit_reacquires_without_heading_jump_deadlock(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                path_smooth_alpha=1.0,
                crosswalk_transit_enabled=True,
                crosswalk_recovery_max_center_jump_px=5.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )
        estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(60.0, 0.20)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        estimator.estimate(
            BevClassMasks(
                crosswalk=[crosswalk_mask_at(30)],
                crosswalk_conf=1.0,
                shape=(100, 200),
            )
        )
        recovered = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(65.0, -0.50)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertEqual(recovered.reason, "corridor_tier2")
        self.assertNotIn("heading_jump", recovered.reason)
        self.assertGreater(estimator._crosswalk_transit_remaining, 0)

        curved_exit = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(100.0, -0.55)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertEqual(curved_exit.reason, "corridor_tier2")
        self.assertNotIn("center_jump", curved_exit.reason)

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

    def test_crosswalk_specific_jump_gate_holds_stable_cache_on_outlier(self):
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
        self.assertTrue(outlier.reason.startswith("crosswalk_hold:"))
        self.assertEqual(estimator.last_class_name, "crosswalk-hold-right-lane")

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

    def test_repeated_consistent_heading_jump_is_promoted_without_teleport(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                path_smooth_alpha=0.65,
                path_max_step_px=10.0,
                max_center_jump_px=5.0,
                max_heading_jump=0.08,
                jump_confirm_frames=2,
                jump_confirm_path_delta_px=12.0,
                jump_confirm_heading_delta=0.10,
                vehicle_center_x_offset_ratio=0.0,
            )
        )
        before = estimator.estimate(bev_at(50, crosswalk=False))
        pending = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(90.0, 0.25)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        promoted = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(92.0, 0.27)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(pending.reason.startswith("coast:"))
        self.assertEqual(pending.path_points, before.path_points)
        self.assertEqual(promoted.reason, "corridor_tier2")
        self.assertTrue(
            all(
                abs(current[0] - previous[0]) <= 10.0 + 1e-9
                for previous, current in zip(
                    before.path_points,
                    promoted.path_points,
                )
            )
        )

    def test_inconsistent_second_jump_cannot_replace_tracked_path(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                path_smooth_alpha=1.0,
                max_center_jump_px=5.0,
                max_heading_jump=0.08,
                jump_confirm_frames=2,
                jump_confirm_path_delta_px=8.0,
                jump_confirm_heading_delta=0.05,
                vehicle_center_x_offset_ratio=0.0,
            )
        )
        before = estimator.estimate(bev_at(50, crosswalk=False))
        first = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(90.0, 0.25)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        second = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(130.0, -0.25)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(first.reason.startswith("coast:"))
        self.assertTrue(second.reason.startswith("coast:"))
        self.assertEqual(first.path_points, before.path_points)
        self.assertEqual(second.path_points, before.path_points)

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

    def test_crosswalk_option_b_uses_right_boundary_before_cache(self):
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

        cached = estimator.estimate(
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
                side=[line_mask(160)],
                crosswalk=[crosswalk_mask()],
                side_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(during.found)
        self.assertGreater(during.center_x, cached.center_x)
        self.assertAlmostEqual(during.center_x, 131.5, delta=0.2)
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

    def test_crosswalk_cache_ignores_bad_pre_crosswalk_geometry(self):
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

        stable = estimator.estimate(
            BevClassMasks(
                center=[line_mask(60)],
                side=[line_mask(120)],
                center_conf=1.0,
                side_conf=1.0,
                shape=(100, 200),
            )
        )
        estimator.estimate(
            BevClassMasks(
                center=[line_mask(20)],
                side=[line_mask(80)],
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
        self.assertAlmostEqual(during.center_x, stable.center_x, delta=0.2)
        self.assertEqual(during.reason, "crosswalk_hold:no_corridor")
        self.assertEqual(estimator.last_class_name, "crosswalk-hold-right-lane")

    def test_crosswalk_right_boundary_heading_jump_holds_cache(self):
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

        stable = estimator.estimate(
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
                side=[slanted_line_mask(160.0, 0.30)],
                crosswalk=[crosswalk_mask()],
                side_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertTrue(during.found)
        self.assertAlmostEqual(during.center_x, stable.center_x, delta=0.2)
        self.assertEqual(during.reason, "crosswalk_hold:cache_heading_guard")
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

    def test_trusted_two_boundary_curve_bypasses_scalar_center_jump(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                min_lane_width_px=40.0,
                max_lane_width_px=100.0,
                max_center_jump_px=5.0,
                trusted_tier1_min_confidence=0.80,
                center_smooth_alpha=1.0,
                heading_smooth_alpha=1.0,
                path_smooth_alpha=1.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )
        estimator.estimate(
            BevClassMasks(
                center=[line_mask(40)],
                side=[line_mask(100)],
                center_conf=1.0,
                side_conf=1.0,
                shape=(100, 200),
            )
        )

        curved = estimator.estimate(
            BevClassMasks(
                center=[line_mask(90)],
                side=[line_mask(150)],
                center_conf=1.0,
                side_conf=1.0,
                shape=(100, 200),
            )
        )

        self.assertEqual(curved.reason, "corridor_tier1")
        self.assertNotIn("center_jump", curved.reason)

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

    def test_path_anchors_stay_fixed_when_visible_line_span_changes(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                path_smooth_alpha=0.36,
                path_max_step_px=28.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )
        full = slanted_line_mask(70.0, 0.15)
        partial = full.copy()
        partial[:20, :] = 0
        partial[85:, :] = 0

        first = estimator.estimate(
            BevClassMasks(center=[full], center_conf=1.0, shape=(100, 200))
        )
        second = estimator.estimate(
            BevClassMasks(center=[partial], center_conf=1.0, shape=(100, 200))
        )

        self.assertEqual(len(first.path_points), 24)
        self.assertEqual(len(second.path_points), 24)
        self.assertEqual(
            [round(point[1], 6) for point in first.path_points],
            [round(point[1], 6) for point in second.path_points],
        )

    def test_lane_targets_are_derived_from_the_stabilized_path(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lane_width_px=60.0,
                path_smooth_alpha=0.36,
                path_max_step_px=28.0,
                vehicle_center_x_offset_ratio=0.0,
            )
        )
        estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(55.0, 0.10)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        lane = estimator.estimate(
            BevClassMasks(
                center=[slanted_line_mask(90.0, 0.35)],
                center_conf=1.0,
                shape=(100, 200),
            )
        )
        ys = np.asarray([point[1] for point in lane.path_points])
        xs = np.asarray([point[0] for point in lane.path_points])

        self.assertAlmostEqual(lane.center_x, np.interp(lane.target_y, ys, xs), delta=0.01)
        self.assertAlmostEqual(
            lane.near_center_x,
            np.interp(lane.near_target_y, ys, xs),
            delta=0.01,
        )

    def test_spatial_path_guard_removes_v_shaped_splice(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                path_max_abs_slope=1.0,
                path_max_slope_delta=0.25,
            )
        )

        guarded = estimator._limit_path_geometry(
            [
                (10.0, 0.0),
                (190.0, 20.0),
                (20.0, 40.0),
                (180.0, 60.0),
                (100.0, 80.0),
            ]
        )
        slopes = [
            (right[0] - left[0]) / (right[1] - left[1])
            for left, right in zip(guarded, guarded[1:])
        ]

        self.assertTrue(all(abs(slope) <= 1.0 + 1e-9 for slope in slopes))
        self.assertTrue(
            all(
                abs(current - previous) <= 0.25 + 1e-9
                for previous, current in zip(slopes, slopes[1:])
            )
        )

    def test_far_preview_is_tangent_extension_of_control_path(self):
        estimator = BevCorridorLaneEstimator(
            BevCorridorConfig(
                lookahead_y_ratio=0.58,
                sample_bottom_y_ratio=0.96,
            )
        )
        stabilized = estimator._stabilize_far_preview(
            [
                (10.0, 0.0),
                (190.0, 20.0),
                (20.0, 40.0),
                (100.0, 58.0),
                (100.0, 70.0),
                (100.0, 82.0),
                (100.0, 96.0),
            ]
        )

        far_x = [x for x, y in stabilized if y < 58.0]
        self.assertTrue(far_x)
        self.assertTrue(all(abs(x - 100.0) < 1e-6 for x in far_x))

    def test_heading_ignores_unused_far_preview_hook(self):
        estimator = BevCorridorLaneEstimator(BevCorridorConfig())
        heading = estimator._heading_from_path(
            [
                (10.0, 0.0),
                (190.0, 20.0),
                (20.0, 40.0),
                (100.0, 60.0),
                (100.0, 70.0),
                (100.0, 80.0),
                (100.0, 90.0),
                (100.0, 96.0),
            ],
            height=100,
            fallback=1.0,
        )

        self.assertAlmostEqual(heading, 0.0, delta=1e-6)

    def test_heading_uses_visible_control_segment_without_bottom_extrapolation(self):
        estimator = BevCorridorLaneEstimator(BevCorridorConfig())
        points = [
            (
                100.0 + 0.02 * (float(y) - 66.0) ** 2,
                float(y),
            )
            for y in range(0, 100, 5)
        ]

        heading = estimator._heading_from_path(
            points,
            height=100,
            fallback=-1.0,
        )

        self.assertGreater(heading, 0.0)
        self.assertLess(abs(heading), 0.15)

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
