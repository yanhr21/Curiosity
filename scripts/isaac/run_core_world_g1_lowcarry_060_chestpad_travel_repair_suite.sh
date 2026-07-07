#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 low-carry 0.60 kg chest-pad travel repair suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_lowcarry_060_chestpad_travel_repair}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_060_chestpad_travel_repair/${SUITE_STAMP_PREFIX}}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/lowcarry_060_chestpad_travel_repair_status.tsv"
summary_out="${OUTPUT_ROOT}/lowcarry_060_chestpad_travel_repair_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_repair_case() {
  local case_name="$1"
  local terminal_travel="$2"
  local final_travel="$3"
  shift 3
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-060-CHEST-TRAVEL] case=${case_name} terminal=${terminal_travel} final=${final_travel} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=lowcarry \
    FREE_BOX_MASS=0.60 \
    FREE_STEPS=920 \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 \
    TARGET_WINDOW_CENTER=2.0 \
    TARGET_WINDOW_HALFWIDTH=0.35 \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 \
    MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80 \
    MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50 \
    MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40 \
    MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=80 \
    MAX_FINAL_HOLD_COMMAND_X=0.001 \
    MAX_FINAL_HOLD_COMMAND_Y=0.003 \
    MAX_FINAL_HOLD_COMMAND_YAW=0.001 \
    MIN_FINAL_HOLD_ROBOT_Z=0.45 \
    MIN_FINAL_HOLD_BOX_Z=0.45 \
    MAX_FINAL_HOLD_TILT=0.35 \
    MAX_FINAL_HOLD_BOX_TILT=0.45 \
    MAX_FINAL_HOLD_FALL_EVENTS=0 \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 \
    AGILE_COMMAND_HOLD_YAW_CORRECTION=1 \
    AGILE_COMMAND_HOLD_YAW_GAIN=0.0 \
    AGILE_COMMAND_HOLD_YAW_LIMIT=0.0 \
    AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 \
    AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1 \
    AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1 \
    AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45 \
    AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1 \
    AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006 \
    AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015 \
    AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0 \
    AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30 \
    AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35 \
    AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL="${terminal_travel}" \
    AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 \
    AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 \
    AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL="${final_travel}" \
    AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
    AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
    CRADLE_TOP_LID_LOCAL_Z=0.145 \
    CRADLE_TOP_LID_THICKNESS=0.018 \
    CRADLE_CHEST_PAD_ENABLED=1 \
    CRADLE_CHEST_PAD_ENABLE_ON_FINAL_HOLD=1 \
    CRADLE_CHEST_PAD_LOCAL_X=-0.02 \
    CRADLE_CHEST_PAD_LOCAL_Z=0.10 \
    CRADLE_CHEST_PAD_SIZE_X=0.04 \
    CRADLE_CHEST_PAD_SIZE_Y=0.38 \
    CRADLE_CHEST_PAD_SIZE_Z=0.22 \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  return "${status}"
}

overall_status=0

run_repair_case final090 0.95 0.90 || overall_status=1
run_repair_case final110 1.15 1.10 || overall_status=1
run_repair_case final130 1.35 1.30 || overall_status=1
run_repair_case final150 1.55 1.50 || overall_status=1

summary_args=()
for case_root in "${summary_case_roots[@]}"; do
  summary_args+=(--case-root "${case_root}")
done

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  set +e
  python3 "${ROOT_DIR}/scripts/isaac/summarize_core_world_g1_largerbox_strict.py" \
    "${summary_args[@]}" \
    --output "${summary_out}"
  summary_status=$?
  set -e
  if [[ "${summary_status}" != "0" ]]; then
    overall_status=1
  fi
fi

exit "${overall_status}"
