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
  --path-heading-gain 65 \
  --path-derivative-gain 18 \
  --path-near-weight 1.45 \
  --path-far-weight 0.55 \
  --path-steering-rise-alpha 0.72 \
  --path-steering-release-alpha 0.28 \
  --path-center-recovery-error-threshold 0.07 \
  --path-center-recovery-heading-limit 0.12 \
  --path-center-recovery-min-steering 60 \
  --path-center-recovery-alpha 0.90 \
  --path-center-recovery-rate-limit 120 \
  --path-reversal-alpha 0.90 \
  --path-reversal-min-steering 25 \
  --path-reversal-min-geometry 0.08 \
  --path-reversal-output-min-steering 60 \
  --path-reversal-rate-limit 220 \
  --path-curve-guard-heading-threshold 0.25 \
  --path-curve-guard-near-error 0.18 \
  --path-curve-guard-steering-limit 105 \
  --path-heading-lead-gain 170 \
  --path-heading-lead-span 0.16 \
  --path-heading-lead-max-steering 32 \
  --path-integral-gain 45 \
  --path-integral-limit 0.25 \
  --path-integral-decay 0.65 \
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
  --obstacle-current-path-min-overlap 0.40 \
  --obstacle-physical-lane-min-overlap 0.65 \
  --obstacle-current-path-max-distance-ratio 0.48 \
  --obstacle-frame-boundary-margin-px 3 \
  --obstacle-visual-commit on \
  --obstacle-visual-commit-confidence 0.90 \
  --obstacle-visual-commit-frame-y 0.40 \
  --obstacle-range-visual-fallback off \
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
  --light-confirm-frames 2 \
  --light-red-confirm-frames 2 \
  --light-min-color-pixels 300 \
  --light-max-mask-area-ratio 0.06 \
  --obstacle-emergency-stop off \
  "$@"
