#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 boxtilt window-hold suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_boxtilt_window_hold}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_boxtilt_window_hold/${SUITE_STAMP_PREFIX}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/boxtilt_window_hold_status.tsv"
summary_out="${OUTPUT_ROOT}/boxtilt_window_hold_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

case_enabled() {
  local case_name="$1"
  local requested="${WINDOW_HOLD_CASES:-all}"
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
    echo "[G1-BOXTILT-WINDOW-HOLD] skip case=${case_name} WINDOW_HOLD_CASES=${WINDOW_HOLD_CASES:-all}"
    return 0
  fi
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-BOXTILT-WINDOW-HOLD] case=${case_name} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    DEVICE="${DEVICE:-cpu}" \
    STRICT=0 \
    RUN_NOBOX=0 \
    RUN_FIXED=0 \
    RUN_FREE=1 \
    TARGET_X="${TARGET_X:--1.2}" \
    TARGET_Y="${TARGET_Y:-0.0}" \
    TARGET_WINDOW_CENTER="${TARGET_WINDOW_CENTER:-2.0}" \
    TARGET_WINDOW_HALFWIDTH="${TARGET_WINDOW_HALFWIDTH:-0.35}" \
    FREE_BOX_MASS=0.75 \
    FREE_BOX_SIZE_X=0.14 \
    FREE_BOX_SIZE_Y=0.10 \
    FREE_BOX_SIZE_Z=0.08 \
    FREE_BOX_POS_X=-0.18 \
    FREE_CRADLE_LOCAL_X=-0.18 \
    FREE_CRADLE_LOCAL_Z=0.05 \
    FREE_STEPS="${FREE_STEPS:-1200}" \
    FREE_MIN_ROBOT_TRAVEL=0.02 \
    FREE_MIN_BOX_TRAVEL=0.02 \
    FREE_MAX_TILT=0.35 \
    FREE_MAX_BOX_TILT=0.45 \
    FREE_MAX_FINAL_REL=0.25 \
    FREE_MAX_ROBOT_LATERAL_ERROR=0.80 \
    FREE_MAX_BOX_LATERAL_ERROR=0.80 \
    FREE_MAX_FINAL_ROBOT_LATERAL_ERROR=0.60 \
    FREE_MAX_FINAL_BOX_LATERAL_ERROR=0.60 \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80 \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50 \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40 \
    MIN_AGILE_COMMAND_HOLD_ACTIVE_STEPS=40 \
    REQUIRE_AGILE_COMMAND_STOP_TARGET_WINDOW_LATCHED=1 \
    MAX_FINAL_HOLD_FALL_EVENTS=0 \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 \
    CRADLE_TOP_LID_ENABLED=1 \
    CRADLE_TOP_LID_ENABLE_ON_HOLD=0 \
    CRADLE_TOP_LID_LOCAL_Z=0.16 \
    CRADLE_TOP_LID_THICKNESS=0.014 \
    CRADLE_TOP_LID_X_SCALE=1.15 \
    CRADLE_TOP_LID_Y_SCALE=1.10 \
    CRADLE_SIDE_RAIL_HEIGHT=0.12 \
    CRADLE_END_STOP_HEIGHT=0.13 \
    CRADLE_CHEST_PAD_ENABLED=0 \
    CRADLE_CHEST_PAD_ENABLE_ON_HOLD=0 \
    BALANCE_FEEDBACK_CONTROLLER=0 \
    BALANCE_START_ON_AGILE_HOLD=0 \
    AGILE_COMMAND_STOP_TARGET_WINDOW=1 \
    AGILE_COMMAND_STOP_TARGET_WINDOW_MIN_STEP=500 \
    AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=-1.0 \
    AGILE_COMMAND_STOP_ROBOT_TARGET_TRAVEL=-1.0 \
    AGILE_COMMAND_HOLD_SCALE=0.0 \
    AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=0 \
    AGILE_COMMAND_HOLD_LATERAL_CORRECTION=0 \
    AGILE_COMMAND_HOLD_YAW_CORRECTION=0 \
    AGILE_COMMAND_BOX_PROGRESS_CONTROLLER=1 \
    AGILE_COMMAND_BOX_PROGRESS_TARGET="${AGILE_COMMAND_BOX_PROGRESS_TARGET:-2.0}" \
    AGILE_COMMAND_BOX_PROGRESS_DEADBAND="${AGILE_COMMAND_BOX_PROGRESS_DEADBAND:-0.12}" \
    AGILE_COMMAND_BOX_PROGRESS_GAIN=0.040 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_FORWARD=0.055 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_REVERSE=0.015 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_TILT=0.45 \
    AGILE_COMMAND_BOX_PROGRESS_MAX_BOX_TILT=0.60 \
    AGILE_COMMAND_BOX_PROGRESS_SCALE_ON_HOLD=1 \
    AGILE_COMMAND_BOX_LATERAL_CONTROLLER=1 \
    AGILE_COMMAND_BOX_LATERAL_SIGN=1.0 \
    AGILE_COMMAND_BOX_LATERAL_DEADBAND=0.22 \
    AGILE_COMMAND_BOX_LATERAL_GAIN=0.012 \
    AGILE_COMMAND_BOX_LATERAL_LIMIT=0.006 \
    AGILE_COMMAND_BOX_LATERAL_SCALE_ON_HOLD=1 \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_agile_policy_low_cradle_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

run_case window_zero

run_case window_freeze \
  AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.70 \
  AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
  AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
  AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 \
  AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1 \
  AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT=0.50 \
  AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT=0.65

run_case window_brake \
  AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.70 \
  AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
  AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
  AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 \
  AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X=-0.012 \
  AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS=80

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
