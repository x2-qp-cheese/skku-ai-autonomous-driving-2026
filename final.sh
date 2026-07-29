#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the proven final driving parameters and force every mission controller off.
# Keep these overrides after "$@" so they cannot be accidentally re-enabled.
exec "$ROOT_DIR/final_obstacle.sh" \
  "$@" \
  --corridor-centerline-bias 0.55 \
  --path-center-recovery-error-threshold 0.07 \
  --path-integral-gain 55 \
  --path-curve-guard-steering-limit 70 \
  --normal-path-right-steering-scale 1 \
  --traffic-light off \
  --obstacle-avoidance off
