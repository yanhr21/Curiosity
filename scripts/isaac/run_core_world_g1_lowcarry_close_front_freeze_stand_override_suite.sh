#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 close-front freeze-stand override suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
export SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_lowcarry_close_front_freeze_stand_override}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_close_front_freeze_stand_override/${SUITE_STAMP_PREFIX}}"
export SUITE_NAME="${SUITE_NAME:-close_front_freeze_stand_override}"
export AGILE_COMMAND_HOLD_STAND_OVERRIDES_FINAL_FREEZE=1
export AGILE_COMMAND_HOLD_RESCUE_ENABLE=0

exec bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_lowcarry_close_front_freeze_stand_transition_suite.sh" "$@"
