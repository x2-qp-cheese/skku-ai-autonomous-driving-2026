#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the proven final driving parameters and force every mission controller off.
# Keep these overrides after "$@" so they cannot be accidentally re-enabled.
exec "$ROOT_DIR/final_obstacle.sh" \
  "$@" \
  --traffic-light off \
  --obstacle-avoidance off
