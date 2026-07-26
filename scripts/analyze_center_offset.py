"""Measure tracked-path position against detected lane boundaries.

The same YOLO/BEV result is fed to several temporal-filter candidates so center
bias and curve-response changes can be compared without inference differences.
Only tier-1 frames, where both the center line and outer line are detected, are
used as geometric evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skku_autocar.estimation.bev_corridor import (  # noqa: E402
    BevCorridorConfig,
    BevCorridorLaneEstimator,
    warp_class_masks,
)
from skku_autocar.perception.bev import BevConfig, BevTransformer  # noqa: E402
from skku_autocar.perception.yolo_lane import (  # noqa: E402
    YoloLaneConfig,
    YoloLaneSegmenter,
)
from skku_autocar.planning.yolo_lane_follower import (  # noqa: E402
    YoloLaneFollower,
    YoloLaneFollowerConfig,
)
from skku_autocar.runtime.yolo_drive_app import resolve_model_path  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    name: str
    centerline_bias: float
    path_smooth_alpha: float
    path_max_step_px: float


DEFAULT_CANDIDATES = (
    Candidate("current", 0.50, 0.36, 28.0),
    Candidate("center_responsive", 0.50, 0.52, 44.0),
    Candidate("center_agile", 0.50, 0.65, 80.0),
    Candidate("center_fast_bounded", 0.50, 0.90, 80.0),
    Candidate("center_fast_preview", 0.50, 1.00, 80.0),
    Candidate("inner_current", 0.46, 0.36, 28.0),
    Candidate("inner_responsive", 0.46, 0.52, 44.0),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare center-path offset on recorded competition videos"
    )
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument(
        "--model",
        default="trained_model/skku_merged_yolov8n_seg_aug_best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="analyze every Nth source frame; 3 approximates a 10fps live loop",
    )
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser.parse_args(argv)


def build_estimator(candidate: Candidate) -> BevCorridorLaneEstimator:
    return BevCorridorLaneEstimator(
        BevCorridorConfig(
            lane_width_px=150.0,
            lookahead_y_ratio=0.32,
            lane_change_near_y_ratio=0.88,
            vehicle_center_x_offset_ratio=0.035,
            centerline_bias=candidate.centerline_bias,
            center_smooth_alpha=0.32,
            heading_smooth_alpha=1.0,
            path_smooth_alpha=candidate.path_smooth_alpha,
            path_max_step_px=candidate.path_max_step_px,
            max_center_jump_px=65.0,
            max_heading_jump=0.45,
            max_coast_frames=7,
            center_anchor=True,
            crosswalk_option="a",
            crosswalk_transit_enabled=True,
        )
    )


def build_follower() -> YoloLaneFollower:
    return YoloLaneFollower(
        YoloLaneFollowerConfig(
            base_speed=255,
            max_speed=255,
            min_curve_speed=255,
            max_steering=150,
            steering_rate_limit=80,
            min_steering_rate_limit=35,
            steering_release_rate_limit=55,
            speed_curve_slowdown=0,
            center_lock_enabled=False,
            path_tracking=True,
            path_lateral_gain=225.0,
            path_heading_gain=70.0,
            path_derivative_gain=18.0,
            path_near_weight=1.25,
            path_far_weight=0.70,
            path_steering_rise_alpha=0.55,
            path_steering_release_alpha=0.28,
        )
    )


def interpolate_x(
    points: Iterable[Tuple[float, float]],
    target_y: float,
) -> Optional[float]:
    ordered = sorted((float(y), float(x)) for x, y in points)
    if len(ordered) < 2:
        return None
    if target_y < ordered[0][0] or target_y > ordered[-1][0]:
        return None
    import numpy as np

    ys = np.asarray([point[0] for point in ordered], dtype=float)
    xs = np.asarray([point[1] for point in ordered], dtype=float)
    return float(np.interp(float(target_y), ys, xs))


def lane_position(
    estimator: BevCorridorLaneEstimator,
    path_x: float,
    target_y: float,
) -> Optional[Tuple[float, float]]:
    left_x = interpolate_x(estimator.last_center_line_bev, target_y)
    right_x = interpolate_x(estimator.last_right_line_bev, target_y)
    if left_x is None or right_x is None:
        return None
    width = right_x - left_x
    if width <= 20.0:
        return None
    ratio = (float(path_x) - left_x) / width
    offset_px = float(path_x) - 0.5 * (left_x + right_x)
    return ratio, offset_px


def path_positions(
    estimator: BevCorridorLaneEstimator,
    path_points: Iterable[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    positions = []
    for path_x, target_y in path_points:
        position = lane_position(estimator, path_x, target_y)
        if position is not None:
            positions.append(position)
    return positions


def curve_group(heading: float) -> str:
    if heading < -0.12:
        return "negative_curve"
    if heading > 0.12:
        return "positive_curve"
    return "straight"


def summarize(values: List[dict]) -> dict:
    if not values:
        return {"frames": 0}
    import numpy as np

    ratios = np.asarray([row["ratio"] for row in values], dtype=float)
    offsets = np.asarray([row["offset_px"] for row in values], dtype=float)
    return {
        "frames": len(values),
        "ratio_median": round(float(np.median(ratios)), 4),
        "ratio_p10": round(float(np.percentile(ratios, 10)), 4),
        "ratio_p90": round(float(np.percentile(ratios, 90)), 4),
        "center_abs_error_median_px": round(float(np.median(np.abs(offsets))), 2),
        "center_abs_error_p90_px": round(float(np.percentile(np.abs(offsets), 90)), 2),
        "signed_offset_median_px": round(float(np.median(offsets)), 2),
    }


def summarize_control(steering_values: List[int]) -> dict:
    if not steering_values:
        return {"frames": 0}
    import numpy as np

    values = np.asarray(steering_values, dtype=float)
    deltas = np.abs(np.diff(values))
    reversals = sum(
        1
        for previous, current in zip(steering_values, steering_values[1:])
        if previous * current < 0 and abs(previous) >= 20 and abs(current) >= 20
    )
    return {
        "frames": len(steering_values),
        "steering_abs_median": round(float(np.median(np.abs(values))), 2),
        "steering_abs_p90": round(float(np.percentile(np.abs(values), 90)), 2),
        "delta_abs_median": (
            round(float(np.median(deltas)), 2) if len(deltas) else 0.0
        ),
        "delta_abs_p90": (
            round(float(np.percentile(deltas, 90)), 2) if len(deltas) else 0.0
        ),
        "large_delta_frames": int((deltas > 35.0).sum()),
        "strong_sign_reversals": reversals,
    }


def analyze_video(
    cv2,
    path: Path,
    segmenter: YoloLaneSegmenter,
    transformer: BevTransformer,
    candidates: Sequence[Candidate],
    frame_step: int,
) -> Dict[str, dict]:
    estimators = {item.name: build_estimator(item) for item in candidates}
    followers = {item.name: build_follower() for item in candidates}
    rows: Dict[str, List[dict]] = {item.name: [] for item in candidates}
    path_rows: Dict[str, List[dict]] = {
        item.name: [] for item in candidates
    }
    steering: Dict[str, List[int]] = {item.name: [] for item in candidates}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("could not open video: %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    index = 0
    analyzed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_step != 0:
                index += 1
                continue
            class_masks = segmenter.segment_class_masks(frame)
            bev = warp_class_masks(transformer, class_masks)
            for item in candidates:
                estimator = estimators[item.name]
                lane = estimator.estimate(bev)
                command = followers[item.name].plan(lane)
                steering[item.name].append(int(command.steering))
                if not lane.found or estimator.last_tier != 1:
                    continue
                position = lane_position(
                    estimator,
                    lane.center_x,
                    lane.target_y,
                )
                if position is None:
                    continue
                ratio, offset_px = position
                rows[item.name].append(
                    {
                        "frame": index,
                        "group": curve_group(lane.heading_error),
                        "ratio": ratio,
                        "offset_px": offset_px,
                    }
                )
                for path_ratio, path_offset_px in path_positions(
                    estimator,
                    lane.path_points,
                ):
                    path_rows[item.name].append(
                        {
                            "frame": index,
                            "group": curve_group(lane.heading_error),
                            "ratio": path_ratio,
                            "offset_px": path_offset_px,
                        }
                    )
            analyzed += 1
            index += 1
            if analyzed % 100 == 0:
                print(
                    "%s: analyzed %d source frames (%d/%d)"
                    % (path.stem, analyzed, index, total),
                    flush=True,
                )
    finally:
        cap.release()

    output = {}
    for item in candidates:
        candidate_rows = rows[item.name]
        by_group = {
            group: summarize(
                [row for row in candidate_rows if row["group"] == group]
            )
            for group in ("straight", "negative_curve", "positive_curve")
        }
        output[item.name] = {
            "all": summarize(candidate_rows),
            "whole_path": summarize(path_rows[item.name]),
            "groups": by_group,
            "control": summarize_control(steering[item.name]),
            "outer_examples": sorted(
                candidate_rows,
                key=lambda row: row["offset_px"],
                reverse=True,
            )[:5],
            "inner_examples": sorted(
                candidate_rows,
                key=lambda row: row["offset_px"],
            )[:5],
        }
    return output


def aggregate(per_video: Dict[str, dict], candidates: Sequence[Candidate]) -> dict:
    output = {}
    for item in candidates:
        video_summaries = [
            result[item.name]["all"]
            for result in per_video.values()
            if result[item.name]["all"]["frames"] > 0
        ]
        path_summaries = [
            result[item.name]["whole_path"]
            for result in per_video.values()
            if result[item.name]["whole_path"]["frames"] > 0
        ]
        control_summaries = [
            result[item.name]["control"]
            for result in per_video.values()
            if result[item.name]["control"]["frames"] > 0
        ]
        output[item.name] = {
            "videos": len(video_summaries),
            "median_of_video_ratio_medians": round(
                statistics.median(
                    value["ratio_median"] for value in video_summaries
                ),
                4,
            ),
            "median_of_video_abs_error_p90_px": round(
                statistics.median(
                    value["center_abs_error_p90_px"]
                    for value in video_summaries
                ),
                2,
            ),
            "median_of_video_whole_path_ratio_medians": round(
                statistics.median(
                    value["ratio_median"] for value in path_summaries
                ),
                4,
            ),
            "median_of_video_whole_path_abs_error_p90_px": round(
                statistics.median(
                    value["center_abs_error_p90_px"]
                    for value in path_summaries
                ),
                2,
            ),
            "median_steering_delta_p90": round(
                statistics.median(
                    value["delta_abs_p90"] for value in control_summaries
                ),
                2,
            ),
            "total_large_delta_frames": sum(
                value["large_delta_frames"] for value in control_summaries
            ),
            "total_strong_sign_reversals": sum(
                value["strong_sign_reversals"] for value in control_summaries
            ),
            "per_video": {
                name: {
                    "center": result[item.name]["all"],
                    "whole_path": result[item.name]["whole_path"],
                    "control": result[item.name]["control"],
                }
                for name, result in per_video.items()
            },
        }
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import cv2

    model_path = resolve_model_path(args.model)
    segmenter = YoloLaneSegmenter(
        YoloLaneConfig(
            model_path=model_path,
            confidence=args.conf,
            image_size=args.imgsz,
            device=args.device,
        )
    )
    transformer = BevTransformer(BevConfig())
    candidates = DEFAULT_CANDIDATES
    per_video = {}
    for value in args.video:
        path = Path(value).expanduser().resolve()
        print("analyzing %s" % path, flush=True)
        per_video[path.stem] = analyze_video(
            cv2,
            path,
            segmenter,
            transformer,
            candidates,
            max(1, int(args.frame_step)),
        )

    report = {
        "model": str(model_path),
        "device": segmenter.device,
        "frame_step": max(1, int(args.frame_step)),
        "candidates": [item.__dict__ for item in candidates],
        "aggregate": aggregate(per_video, candidates),
        "videos": per_video,
    }
    text = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print("saved %s" % output)
    print(json.dumps(report["aggregate"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
