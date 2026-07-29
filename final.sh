#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the proven final driving parameters and force every mission controller off.
# Keep these overrides after "$@" so they cannot be accidentally re-enabled.
exec "$ROOT_DIR/final_obstacle.sh" \
  "$@" \
  --path-center-recovery-error-threshold 0.20 \
  --path-integral-gain 50 \
  --path-curve-guard-steering-limit 75 \
  --normal-path-right-steering-scale 1 \
  --traffic-light off \
  --obstacle-avoidance off
