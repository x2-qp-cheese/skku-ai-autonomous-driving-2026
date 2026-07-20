# Arduino Mega T Parking Controller

`t_parking_controller.ino` is a standalone mission state machine that expects
the vehicle project to provide these functions:

```cpp
float get_lidar_distance(int angleDeg);          // -1 = invalid
float get_line_error();                          // px, left -, right +; NAN = lost
float get_ultrasonic_distance(const char* side); // cm, -1 = invalid
void steer(int angleDeg);                        // -45 .. +45
void drive(int speed);                           // -100 .. +100
```

The state flow is `SEARCHING -> POSITIONING -> REVERSING -> FINISHED`.
Because no rear ultrasonic sensor is available, LiDAR at 180 degrees supplies
the 20 cm completion distance and 10 cm emergency distance.

The first parked car is guaranteed to be followed by the mission bay. Its first
observation immediately reduces speed to `FIRST_CAR_APPROACH_SPEED`. After two
scans, when the slot-adjacent edge reaches `-65 cm yBack`, `PREALIGN_LEFT`
commands maximum left steering and moves forward slowly without waiting for the
second car. The second car then confirms gap width and orientation during the
arc. Every LiDAR scan checks the slot-depth heading and bearing/distance from
the rear axle to the entrance center. Three aligned samples start reversing. A
6 second post-confirmation timeout or heading overshoot hands control to the
camera-guided reversing correction. Failure to see the second car for 12
seconds stops the mission.

## Before driving

1. Confirm the LiDAR convention is front=0, left=90, rear=180, right=270.
2. Set `LIDAR_TO_CM=0.1` if the LiDAR function returns millimeters.
3. Tune the full-frame Cartesian ROI, 25 cm clustering radius, and observed
   vehicle-to-vehicle gap range (default 110-165 cm) using stationary logs.
   A candidate gap center must still remain on the vehicle-right. Full-frame
   tracking keeps a bordering car visible after the ego vehicle rotates. Once
   acquired, the slot depth normal is sign-locked to the previous scan; center
   and angle still update dynamically, but a 180-degree reversal or a one-scan
   orientation jump above 35 degrees is rejected.
4. Measure LiDAR origin to rear-axle center and replace the provisional signed
   `POSITION_CENTER_Y_TARGET_CM=-30.0` value. Negative means the axle is in
   front of the rear-mounted LiDAR.
5. Verify `POSITION_DIRECTION_SIGN` with the wheels raised.
6. Verify the reverse camera steering sign with the wheels raised. The default
   assumes positive line error and positive steering both move the rear right.
7. Keep `AUTO_START=false`; send `S` over Serial only when the test area is clear.
8. Treat `PREALIGN_STEER_DEG=-45` as a configured maximum, not proof of the
   mechanical end stop. Measure the real steering linkage and reduce this
   value if the servo binds. Also tune `PREALIGN_SPEED` to the lowest value
   that moves the loaded vehicle reliably.

Serial controls:

- `S`: reset and start the mission
- `R`: reset and remain stopped
- `X`: latch emergency stop

An emergency stop is latched. Use `R`, inspect the cause, and then send `S` to
restart. Distance sensor failure is `-1`; a zero distance is therefore treated
as a real emergency. Camera error cannot use `-1` as a failure sentinel because
`-1 px` is a valid left error, so return `NAN` when the parking line is lost.

## Recording-only replay

The earlier MP4 + LiDAR CSV ZIP can exercise the LiDAR detector, camera line
error, state transitions, and virtual motor commands without connecting the
Arduino:

```bash
python scripts/arduino_parking_replay.py \
  --recording-zip "path/to/recording.zip" \
  --device cpu
```

Use `--no-camera` for a quick LiDAR/state-only check. The generated CSV and
summary JSON are written under `data/processed/`. Because that ZIP does not
contain LEFT/RIGHT ultrasonic readings, it cannot validate side P-control or
the side-sensor 10 cm emergency stop. Virtual commands also cannot change the
motion already captured in the recording.

To see the Arduino state, virtual drive/steer commands, YOLO masks, BEV, and
vehicle-frame LiDAR together in one offline dashboard:

```bash
python scripts/arduino_parking_replay.py \
  --recording-zip "path/to/recording.zip" \
  --device cpu --imgsz 512 --frame-stride 3 --show
```

Add `--save-video` to also write
`data/processed/arduino_parking_replay_overlay.mp4`. The visual replay never
opens a serial port and never sends a motor command.
