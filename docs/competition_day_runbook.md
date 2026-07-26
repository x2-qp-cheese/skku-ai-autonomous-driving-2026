# Competition day runbook

## Do not tune

Do not change gains, BEV calibration, lane width, model weights, camera
resolution, or Arduino firmware at the venue. The validated launcher is
`run_competition_255_v3.sh`.

## Before placing the car

1. Connect the camera and Arduino.
2. Confirm that `scripts/list_serial_ports.py` shows an Arduino `usbmodem`
   device. With the tested hardware it was `/dev/cu.usbmodem21101` with
   `VID:PID=2341:0042`. `/dev/cu.usbserial-2130` is not the vehicle controller.
3. Point the car along the lane center before starting. Do not start while a
   person, chair, or another car fills the lower camera image.
4. Run:

   ```bash
   ./run_competition_255_v3.sh --serial-port /dev/cu.usbmodem21101
   ```

5. Confirm the terminal prints `serial connected` and the correct port.
6. In the debug image, confirm `path_points=24`, `lane_change=LANE2`, and a
   magenta path centered in the current lane.
7. Press Space once to start.

If the Arduino device number changes, replace only the `--serial-port` value
with the new `usbmodem` path. Do not choose `usbserial-2130`.

## Expected behavior

- Normal lane: whole magenta path remains between the lane boundaries; speed is
  `255`.
- Crosswalk: state changes to `CROSSWALK_HOLD`; the magenta path remains the
  pre-crosswalk path and zebra stripes do not recenter it.
- Current-lane obstacle: one `lane2 -> lane1 requested` event, smooth parallel
  path translation, lane-1 hold, then one `lane1 -> lane2 requested` event.
- Red light at the stop line: `stop=Y` and speed `0`. Green releases the stop.
- Obstacle perception cannot issue an emergency brake.

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
