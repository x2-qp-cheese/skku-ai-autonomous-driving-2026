# Competition day runbook

## Do not tune

Do not change gains, BEV calibration, lane width, model weights, camera
resolution, or Arduino firmware at the venue. The mission launcher is
`final_obstacle.sh`; the driving-only launcher is `final.sh`.

## Before placing the car

Run the software-only checks once before connecting the car:

```bash
PYTHONPATH=src venv/bin/python -m unittest discover -s tests
PYTHONPATH=src venv/bin/python scripts/validate_lane_change_trajectory.py --fps 9
```

Both commands must finish successfully. The 9 Hz trajectory check is the
lowest supported replay rate; the recorded competition runs normally processed
about 10--15 control frames/s.

1. Connect the camera and Arduino.
2. Confirm that `scripts/list_serial_ports.py` shows an Arduino `usbmodem`
   device. With the tested hardware it was `/dev/cu.usbmodem21101` with
   `VID:PID=2341:0042`. `/dev/cu.usbserial-2130` is not the vehicle controller.
3. Point the car along the lane center before starting. Do not start while a
   person, chair, or another car fills the lower camera image.
4. For the two-lap time trial, disable mission triggers:

   ```bash
   ./final.sh --serial-port /dev/cu.usbmodem21101
   ```

   For the obstacle plus traffic-light mission, use:

   ```bash
   ./final_obstacle.sh --serial-port /dev/cu.usbmodem21101
   ```

5. Confirm the terminal prints `serial connected`, the correct port,
   `fixed_speed=on`, and `brake_policy=red-light-only`.
6. In the debug image, confirm `path_points=24`, `lane_change=LANE2`, and a
   magenta path centered in the current lane. `control` should normally read
   `straight`, `curve_entry`, or `curve_hold`.
7. Press Space once to start.

If the Arduino device number changes, replace only the `--serial-port` value
with the new `usbmodem` path. Do not choose `usbserial-2130`.

## Expected behavior

- Normal lane: whole magenta path remains between the lane boundaries; speed is
  `255`.
- Curve entry: `control=curve_entry` may appear before the target point has
  moved far from center. Steering follows the path direction without a large
  opposite-direction pulse. Every steering sign reversal is limited to at most
  80 command units per processed frame. `control=curve_transition` is expected
  briefly when the near and far parts of an S-curve point in opposite
  directions; the far preview is blended in instead of changing sign at once.
- Crosswalk: state changes to `CROSSWALK_HOLD`; the magenta path follows the
  visible current-lane boundaries. If both boundaries are briefly hidden, it
  continues the motion-adjusted entry path until a valid boundary returns.
  Zebra stripes never become lane boundaries.
- Current-lane obstacle: one `lane2 -> lane1 requested` event, a near-anchored
  S-shaped target path, lane-1 stabilization, then one
  `lane1 -> lane2 requested` event. The target never jumps a full lane in one
  frame.
- Obstacle geometry dropout: cached steering unwinds for 0.25 s and then
  becomes neutral. It must not remain at `-80` or `+80`, and it must not brake.
- Red light at the stop line: `stop=Y` and speed `0`. Green releases the stop.
- Every moving command is speed `255`. The red-contact traffic-light command is
  the only competition brake; lane loss, virtual bootstrap and obstacle
  perception cannot issue a persistent brake.

## Abort before launch

Do not press Space when any of these is true:

- the terminal did not print `serial connected`;
- the selected serial path is `usbserial-2130`;
- the camera is not `1280x720`;
- `path_points` is not `24`;
- the initial magenta path is outside the current lane for several consecutive
  frames;
- the overlay starts in `virtual_hold` or `coast` while the lane is plainly
  visible.

After Space is pressed, avoid keyboard input unless the run must be stopped.

## Preparation-track acceptance order

Do not start by placing three obstacles on the track. Record raw and debug video
for every step and continue only after the previous step passes:

1. One obstacle-free lap at speed 255. The car must remain centered through
   both S-curves with no steering delta above 80.
2. One lane-2 to lane-1 obstacle change on a straight. Confirm
   `CHANGING_TO_LANE1 -> STABILIZING_LANE1 -> LANE1`.
3. One lane-1 to lane-2 return. Confirm
   `CHANGING_TO_LANE2 -> STABILIZING_LANE2 -> COMPLETED`.
4. Repeat both directions once. A single success is not enough to accept a
   high-speed transition.
5. Run the full obstacle placement.
6. Test the red stop and green release last. Moving commands must be 255, the
   red-contact command must be speed 0, and green must release within five
   seconds.

Reject a run if a lane-change target moves more than 55 BEV px in one frame,
the steering command changes by more than 80 outside the lane-change-specific
35-unit slew, a transition holds one steering direction after the path has
crossed the vehicle, or a non-red state applies a brake.
