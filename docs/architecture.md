# Architecture

## Competition Constraints Reflected In Code

- Track driving is counter-clockwise on the outer lane.
- Time trial, obstacle/traffic-light mission, and vertical parking should be selectable modes, not separate rewrites.
- After inspection, the hardware configuration must stay unchanged across matches.
- The start signal rule means motor output should stay stopped until the operator explicitly starts the program mode.
- A judge may stand outside the lane, so LiDAR obstacle logic should avoid treating every side measurement as a mission obstacle.

## Data Flow

```text
camera frame
     |
     v
YOLO per-class segmentation (center / side / lane / crosswalk / light)
     |
     v
BEV warp + two-lane corridor geometry
     |
     v
YOLO lane follower
     |
     v
command safety + traffic-light latch
     |
     v
ControlCommand
     |
     v
Arduino serial
     |
     v
drive and steering motors
```

## Module Boundaries

- `sensors`: hardware IO only. No mission decisions.
- `perception`: runs YOLO, preserves per-class masks, and classifies light color.
- `estimation`: builds the BEV driving corridor and target geometry.
- `planning`: chooses speed and steering from the YOLO-derived geometry.
- `control`: serial protocol and actuator command formatting.
- `firmware`: receives simple serial commands and applies them to motor pins.

Keep calibration values in `configs/default.json`. Avoid burying threshold numbers inside mission code once real track tuning starts.

The competition entrypoint is `scripts/drive.py`. Its lane pipeline is always the
BEV corridor path; legacy frame-plane and single-mask runtime switches were removed
to prevent launching the wrong controller on race day. `scripts/bev_tune.py` and
`scripts/bev_replay.py` remain offline calibration/replay tools.
