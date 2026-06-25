from typing import Dict

from ..types import ControlCommand


def encode_command(
    command: ControlCommand,
    max_speed: int = 255,
    max_steering: int = 255,
) -> str:
    clipped = command.clipped(max_speed=max_speed, max_steering=max_steering)
    if clipped.brake:
        return "STOP\n"
    return "DRIVE %d %d\n" % (clipped.speed, clipped.steering)


def decode_status(line: str) -> Dict[str, str]:
    parts = line.strip().split(maxsplit=1)
    if not parts:
        return {"type": "empty", "message": ""}
    if len(parts) == 1:
        return {"type": parts[0], "message": ""}
    return {"type": parts[0], "message": parts[1]}
