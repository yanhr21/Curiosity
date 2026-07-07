#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 selected-branch horizon repair suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_selected_branch_horizon_repair}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_selected_branch_horizon_repair/${SUITE_STAMP_PREFIX}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/selected_branch_horizon_repair_status.tsv"
summary_out="${OUTPUT_ROOT}/selected_branch_horizon_repair_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

case_enabled() {
  local case_name="$1"
  local requested="${SELECTED_BRANCH_REPAIR_CASES:-all}"
  if [[ "${requested}" == "all" ]]; then
    return 0
  fi
  case ",${requested}," in
    *",${case_name},"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_case() {
  local case_name="$1"
  shift
  if ! case_enabled "${case_name}"; then
    echo "[G1-SELECTED-BRANCH-REPAIR] skip case=${case_name} SELECTED_BRANCH_REPAIR_CASES=${SELECTED_BRANCH_REPAIR_CASES:-all}"
    return 0
  fi
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-SELECTED-BRANCH-REPAIR] case=${case_name} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    RUN_NOBOX=0 \
    RUN_FIXED=0 \
    RUN_FREE=1 \
    TARGET_WINDOW_CENTER="${TARGET_WINDOW_CENTER:-2.0}" \
    TARGET_WINDOW_HALFWIDTH="${TARGET_WINDOW_HALFWIDTH:-0.35}" \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-2.35}" \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-2.35}" \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-80}" \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-50}" \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-40}" \
    MAX_FINAL_HOLD_FALL_EVENTS="${MAX_FINAL_HOLD_FALL_EVENTS:-0}" \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS="${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-0}" \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

run_case light_chestpad_1600_nearstop \
  LARGERBOX_STRICT_MODE=chestpad \
  FREE_BOX_MASS=0.25 \
  FREE_STEPS=1600 \
  AGILE_COMMAND_HOLD_YAW_CORRECTION=1 \
  AGILE_COMMAND_HOLD_YAW_GAIN=0.04 \
  AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 \
  AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 \
  AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.05 \
  AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 \
  AGILE_COMMAND_HOLD_TERMINAL_LATCH=1

run_case light_chestpad_1600_finalstop \
  LARGERBOX_STRICT_MODE=chestpad \
  FREE_BOX_MASS=0.25 \
  FREE_STEPS=1600 \
  AGILE_COMMAND_HOLD_YAW_CORRECTION=1 \
  AGILE_COMMAND_HOLD_YAW_GAIN=0.04 \
  AGILE_COMMAND_HOLD_YAW_LIMIT=0.08 \
  AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 \
  AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.05 \
  AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 \
  AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 \
  AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.65 \
  AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
  AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
  AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1

run_case heavy_boxtilt_1200_default \
  LARGERBOX_STRICT_MODE=boxtilt \
  FREE_BOX_MASS=0.75 \
  FREE_STEPS=1200

run_case heavy_boxtilt_1200_stop_finalzero \
  LARGERBOX_STRICT_MODE=boxtilt \
  FREE_BOX_MASS=0.75 \
  FREE_STEPS=1200 \
  AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0 \
  AGILE_COMMAND_HOLD_LATERAL_GAIN=0.04 \
  AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.018 \
  AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.65 \
  AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.018 \
  AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 \
  AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.80 \
  AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
  AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
  AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  set +e
  python3 scripts/isaac/summarize_core_world_g1_largerbox_strict.py \
    "${summary_args[@]}" \
    --output "${summary_out}"
  summary_status=$?
  set -e
  if [[ "${summary_status}" != "0" ]]; then
    overall_status=1
  fi
fi

exit "${overall_status}"
