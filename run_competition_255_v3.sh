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
  --path-tracking \
  --path-lateral-gain 225 \
  --path-heading-gain 70 \
  --path-derivative-gain 18 \
  --path-near-weight 1.25 \
  --path-far-weight 0.70 \
  --path-steering-rise-alpha 0.72 \
  --path-steering-release-alpha 0.28 \
  --path-heading-lead-gain 180 \
  --path-heading-lead-span 0.15 \
  --path-heading-lead-max-steering 36 \
  --bev-lookahead 0.32 \
  --bev-center-smooth 0.32 \
  --bev-heading-smooth 1.0 \
  --bev-path-smooth 0.90 \
  --bev-path-max-step 80 \
  --vehicle-center-offset 0.035 \
  --corridor-centerline-bias 0.50 \
  --corridor-crosswalk-option a \
  --center-lock off \
  --center-recovery-error-threshold 0.20 \
  --center-recovery-min-steering 0 \
  --center-recovery-steering-boost 1.0 \
  --steering-rate-limit 80 \
  --min-steering-rate-limit 35 \
  --steering-release-rate-limit 55 \
  --corridor-max-center-jump 65 \
  --corridor-max-heading-jump 0.45 \
  --corridor-max-coast-frames 7 \
  --obstacle-local-map off \
  --obstacle-trigger-mm 2900 \
  --obstacle-clear-mm 3150 \
  --obstacle-min-front-sensors 2 \
  --obstacle-range-confirm-frames 1 \
  --obstacle-visual-confirm-frames 2 \
  --lane-change-target-width-px 150 \
  --lane-change-steering-override off \
  --lane-change-steering-min 0 \
  --lane-change-steering-boost 0 \
  --lane-change-steering-cap 130 \
  --lane-change-unreliable-steering-cap 90 \
  --lane-change-stabilizing-steering-min 0 \
  --lane-change-transition-seconds 1.2 \
  --lane-change-stable-lateral-error 0.25 \
  --lane-change-stable-near-error 0.35 \
  --lane-change-stable-frames 3 \
  --light-stop-during-lane-change on \
  --obstacle-emergency-stop off \
  "$@"
