#!/usr/bin/env bash
set -euo pipefail

# Fully decode one CarryBox tactile bundle and check its source-clock length.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ROOT successful_grasp|failed_grasp|failed_closure|all" >&2
  exit 2
fi

RUN_ROOT="$1"
SCENARIO="$2"
if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$ROOT/$RUN_ROOT"
fi

exec "$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/validate_complete_carrybox_bundle.py" \
  "$RUN_ROOT" \
  "$SCENARIO"
