#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 close-front support-timing suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_lowcarry_close_front_support_timing}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_close_front_support_timing/${SUITE_STAMP_PREFIX}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/close_front_support_timing_status.tsv"
summary_out="${OUTPUT_ROOT}/close_front_support_timing_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

run_case() {
  local case_name="$1"
  shift
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-CLOSE-FRONT-SUPPORT-TIMING] case=${case_name} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=lowcarry \
    FREE_BOX_MASS=0.60 \
    FREE_BOX_POS_X=-0.14 \
    FREE_CRADLE_LOCAL_X=-0.14 \
    FREE_STEPS=1300 \
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
    MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=399 \
    MAX_FINAL_HOLD_COMMAND_X=0.001 \
    MAX_FINAL_HOLD_COMMAND_Y=0.003 \
    MAX_FINAL_HOLD_COMMAND_YAW=0.001 \
    MIN_FINAL_HOLD_ROBOT_Z=0.45 \
    MIN_FINAL_HOLD_BOX_Z=0.45 \
    MAX_FINAL_HOLD_TILT=0.35 \
    MAX_FINAL_HOLD_BOX_TILT=0.45 \
    MAX_FINAL_HOLD_FALL_EVENTS=0 \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 \
    AGILE_COMMAND_X=0.10 \
    AGILE_COMMAND_Y=-0.04 \
    AGILE_COMMAND_YAW=0.0 \
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
    AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.25 \
    AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015 \
    AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 \
    AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.20 \
    AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
    AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
    AGILE_COMMAND_BOX_PROGRESS_CONTROLLER=1 \
    AGILE_COMMAND_BOX_PROGRESS_START_STEP=40 \
    AGILE_COMMAND_BOX_PROGRESS_TARGET=2.0 \
    AGILE_COMMAND_BOX_PROGRESS_DEADBAND=0.12 \
    AGILE_COMMAND_BOX_PROGRESS_GAIN=0.040 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_FORWARD=0.055 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_REVERSE=0.015 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_TILT=0.30 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_BOX_TILT=0.42 \
    AGILE_COMMAND_BOX_PROGRESS_SCALE_ON_HOLD=1 \
    AGILE_COMMAND_BOX_LATERAL_CONTROLLER=1 \
    AGILE_COMMAND_BOX_LATERAL_DEADBAND=0.12 \
    AGILE_COMMAND_BOX_LATERAL_GAIN=0.010 \
    AGILE_COMMAND_BOX_LATERAL_LIMIT=0.0015 \
    AGILE_COMMAND_BOX_LATERAL_SIGN=1.0 \
    AGILE_COMMAND_BOX_LATERAL_SCALE_ON_HOLD=1 \
    CRADLE_TOP_LID_LOCAL_Z=0.13 \
    CRADLE_TOP_LID_THICKNESS=0.014 \
    CRADLE_CHEST_PAD_ENABLED=1 \
    CRADLE_CHEST_PAD_SPAWN_ON_TRIGGER=1 \
    CRADLE_CHEST_PAD_ENABLE_ON_TARGET_WINDOW=1 \
    CRADLE_CHEST_PAD_TARGET_WINDOW_MIN_STEP=700 \
    CRADLE_CHEST_PAD_LOCAL_X=-0.02 \
    CRADLE_CHEST_PAD_LOCAL_Z=0.10 \
    CRADLE_CHEST_PAD_SIZE_X=0.04 \
    CRADLE_CHEST_PAD_SIZE_Y=0.38 \
    CRADLE_CHEST_PAD_SIZE_Z=0.22 \
    COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

run_case no_runtime_pad \
  CRADLE_CHEST_PAD_ENABLED=0 \
  CRADLE_CHEST_PAD_ENABLE_ON_TARGET_WINDOW=0

run_case pad760 \
  CRADLE_CHEST_PAD_TARGET_WINDOW_MIN_STEP=760

run_case pad700_small \
  CRADLE_CHEST_PAD_SIZE_X=0.025 \
  CRADLE_CHEST_PAD_SIZE_Y=0.34 \
  CRADLE_CHEST_PAD_SIZE_Z=0.16 \
  CRADLE_CHEST_PAD_LOCAL_X=-0.03 \
  CRADLE_CHEST_PAD_LOCAL_Z=0.10

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
