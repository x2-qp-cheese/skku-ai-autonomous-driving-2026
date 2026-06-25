# Hardware Bring-Up Checklist

## Before Power

- Confirm SMPS is set to 12.0 V.
- Confirm motor driver polarity and shared GND with Arduino.
- Confirm camera and LiDAR are fixed firmly enough to survive all match modes without remounting.
- Label USB ports and cables for Arduino, LiDAR, and camera.

## Arduino

- Upload `firmware/arduino/vehicle_controller/vehicle_controller.ino`.
- Close Arduino Serial Monitor before running Python control code.
- Run `scripts/list_serial_ports.py` and copy the Arduino port into `configs/default.json`.
- Test `PING`, `STOP`, and low-speed `DRIVE` commands with wheels raised.

## Camera

- Run `scripts/camera_check.py --config configs/default.json`.
- Verify Logitech webcam index and resolution.
- Tune ROI values in `configs/default.json` against the real track.
- Save representative images for lane extraction assignment and debugging.

## LiDAR

- Connect LiDAR USB directly where possible.
- Run vendor utility once to confirm scan stability.
- Copy the LiDAR port into `configs/default.json`.
- Confirm front angle window matches the installed LiDAR orientation.

## Track Practice

- First stabilize time-trial mode at slow speed.
- Increase base speed only after lane-lost braking and manual stop are reliable.
- Add obstacle and traffic-light behavior after the base lane follower can complete laps repeatedly.
- Treat vertical parking as a separate mission planner because it is geometry-heavy and starts from randomized positions.
