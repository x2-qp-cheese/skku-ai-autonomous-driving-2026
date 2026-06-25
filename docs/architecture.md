# Architecture

## Competition Constraints Reflected In Code

- Track driving is counter-clockwise on the outer lane.
- Time trial, obstacle/traffic-light mission, and vertical parking should be selectable modes, not separate rewrites.
- After inspection, the hardware configuration must stay unchanged across matches.
- The start signal rule means motor output should stay stopped until the operator explicitly starts the program mode.
- A judge may stand outside the lane, so LiDAR obstacle logic should avoid treating every side measurement as a mission obstacle.

## Data Flow

```text
camera frame        lidar scan
     |                  |
     v                  v
lane detector     obstacle filter
     |                  |
     +--------+---------+
              v
        mission planner
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
- `perception`: converts raw sensor data into lane, obstacle, or traffic-light estimates.
- `planning`: chooses speed and steering from perception outputs and mission mode.
- `control`: serial protocol and actuator command formatting.
- `firmware`: receives simple serial commands and applies them to motor pins.

Keep calibration values in `configs/default.json`. Avoid burying threshold numbers inside mission code once real track tuning starts.
