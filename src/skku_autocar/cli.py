import argparse
import json
from pathlib import Path

from .config import load_config
from .control.protocol import encode_command
from .planning.lane_follower import LaneFollower
from .types import ControlCommand, LaneEstimate


def _dry_run(config_path: str) -> int:
    config = load_config(config_path)
    follower = LaneFollower(config.control)
    sample_lane = LaneEstimate(center_offset_norm=0.12, heading_error=0.0, confidence=0.9)
    command = follower.plan(sample_lane)
    payload = {
        "config": str(Path(config_path)),
        "mission_mode": config.mission.mode,
        "sample_command": ControlCommand(
            speed=command.speed,
            steering=command.steering,
            brake=command.brake,
            reason=command.reason,
        ).__dict__,
        "serial_line": encode_command(
            command,
            max_speed=config.control.max_speed,
            max_steering=config.control.max_steering,
        ).strip(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skku_autocar")
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run", help="Validate config and command formatting without hardware")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        return _dry_run(args.config)
    parser.error("unknown command")
    return 2
