#!/usr/bin/env bash
set -euo pipefail

python3 scripts/drive.py \
  --speed 255 \
  --fixed-speed on \
  --max-speed 255 \
  --min-curve-speed 255 \
  --speed-curve-slowdown 0 \
  --pure-pursuit \
  --pp-gain 330 \
  --bev-lookahead 0.34 \
  --bev-center-smooth 0.28 \
  --bev-heading-smooth 0.22 \
  --center-lock off \
  --center-recovery-error-threshold 0.20 \
  --center-recovery-min-steering 0 \
  --center-recovery-steering-boost 1.0 \
  --steering-rate-limit 80 \
  --min-steering-rate-limit 35 \
  --steering-release-rate-limit 55 \
  --corridor-max-center-jump 65 \
  --corridor-max-heading-jump 0.25 \
  --corridor-max-coast-frames 5 \
  --corridor-crosswalk-option b \
  --corridor-crosswalk-right-offset-px 90 \
  --obstacle-local-map off \
  --obstacle-trigger-mm 2900 \
  --obstacle-clear-mm 3150 \
  --obstacle-min-front-sensors 2 \
  --obstacle-range-confirm-frames 1 \
  --obstacle-visual-confirm-frames 2 \
  --lane-change-target-width-px 0 \
  --lane-change-steering-override off \
  --lane-change-steering-min 70 \
  --lane-change-steering-boost 25 \
  --lane-change-steering-cap 140 \
  --lane-change-unreliable-steering-cap 110 \
  --lane-change-stabilizing-steering-min 55 \
  --lane-change-transition-seconds 0.9 \
  --lane-change-stable-lateral-error 0.25 \
  --lane-change-stable-near-error 0.35 \
  --lane-change-stable-frames 1 \
  --light-stop-during-lane-change on \
  --obstacle-emergency-stop on \
  "$@"
