#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 low-carry 0.60 kg tiny-mass late chest-pad suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
export SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_lowcarry_060_late_chestpad_tinymass}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_060_late_chestpad_tinymass/${SUITE_STAMP_PREFIX}}"
export CRADLE_CHEST_PAD_MASS_SCALE="${CRADLE_CHEST_PAD_MASS_SCALE:-0.001}"

exec bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_lowcarry_060_late_chestpad_suite.sh"
