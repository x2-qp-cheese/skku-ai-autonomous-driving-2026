#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/venv/bin/python" ]] &&
  "$ROOT_DIR/venv/bin/python" -c "import ultralytics" >/dev/null 2>&1; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]] &&
  "$ROOT_DIR/.venv/bin/python" -c "import ultralytics" >/dev/null 2>&1; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/drive.py" \
  --speed 255 \
  --fixed-speed on \
  --max-speed 255 \
  --min-curve-speed 255 \
  --speed-curve-slowdown 0 \
  --max-steering 150 \
  --kp-lateral 205 \
  --kd-lateral 75 \
  --kp-heading 1.5 \
  --kd-heading 0.3 \
  --min-steering-rate-limit 35 \
  --steering-rate-limit 150 \
  --steering-release-rate-limit 10 \
  --lateral-priority-threshold 0.16 \
  --curve-strength-alpha 0.45 \
  --straight-steering-scale 0.50 \
  --curve-steering-scale 1.68 \
  --center-recovery-error-threshold 0.12 \
  --center-recovery-steering-boost 1.20 \
  --center-recovery-min-steering 70 \
  --center-recovery-rate-limit 150 \
  --center-recovery-max-speed 255 \
  --center-lock off \
  --lane-lost-hold-frames 3 \
  --lane-lost-speed-cap 255 \
  --vehicle-center-offset 0.085 \
  --bev-lookahead 0.55 \
  --bev-center-smooth 0.60 \
  --bev-heading-smooth 0.30 \
  --bev-heading-gain 1.6 \
  --corridor-lane-width-px 150 \
  --corridor-center-anchor on \
  --corridor-centerline-bias 0.40 \
  --corridor-max-center-jump 150 \
  --corridor-max-coast-frames 3 \
  --corridor-max-width-jump 40 \
  --corridor-crosswalk-option b \
  --corridor-crosswalk-right-offset-px 90 \
  --corridor-crosswalk-center-smooth 0.10 \
  --corridor-crosswalk-max-center-jump 150 \
  --corridor-virtual-hold off \
  --virtual-lane-speed-cap 255 \
  --virtual-lane-bootstrap-speed-cap 255 \
  --virtual-lane-max-steering 150 \
  --virtual-lane-warmup-frames 0 \
  --virtual-lane-steering-blend 1.00 \
  --virtual-lane-max-steering-step 100 \
  --virtual-lane-min-reliable-frames 1 \
  --obstacle-local-map on \
  --obstacle-trigger-mm 2600 \
  --obstacle-clear-mm 2900 \
  --obstacle-min-front-sensors 2 \
  --obstacle-range-confirm-frames 1 \
  --obstacle-range-clear-frames 2 \
  --obstacle-visual-confirm-frames 2 \
  --obstacle-rearm-clear-frames 3 \
  --obstacle-source-clear-frames 2 \
  --obstacle-current-path-min-overlap 0.30 \
  --obstacle-physical-lane-min-overlap 0.65 \
  --obstacle-current-path-max-distance-ratio 0.58 \
  --obstacle-frame-boundary-margin-px 3 \
  --obstacle-visual-commit off \
  --obstacle-range-visual-fallback on \
  --lane-change-target-width-px 120 \
  --lane-change-smooth-avoidance off \
  --lane-change-steering-override on \
  --lane-change-steering-min 150 \
  --lane-change-steering-boost 35 \
  --lane-change-steering-cap 150 \
  --lane-change-steering-slew-limit 0 \
  --lane-change-unreliable-steering-cap 150 \
  --lane-change-stabilizing-steering-min 90 \
  --lane-change-transition-seconds 1.00 \
  --lane-change-stable-lateral-error 0.12 \
  --lane-change-stable-near-error 0.18 \
  --lane-change-stable-frames 5 \
  --light-stop-during-lane-change off \
  --light-confirm-frames 2 \
  --light-red-confirm-frames 2 \
  --light-min-color-pixels 300 \
  --light-max-mask-area-ratio 0.06 \
  --obstacle-emergency-stop off \
  "$@"
