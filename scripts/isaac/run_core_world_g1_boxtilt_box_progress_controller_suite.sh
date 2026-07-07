#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 boxtilt box-progress controller suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_boxtilt_box_progress_controller}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_boxtilt_box_progress_controller/${SUITE_STAMP_PREFIX}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/boxtilt_box_progress_controller_status.tsv"
summary_out="${OUTPUT_ROOT}/boxtilt_box_progress_controller_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

case_enabled() {
  local case_name="$1"
  local requested="${BOX_PROGRESS_CONTROLLER_CASES:-all}"
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
    echo "[G1-BOXTILT-BOX-PROGRESS] skip case=${case_name} BOX_PROGRESS_CONTROLLER_CASES=${BOX_PROGRESS_CONTROLLER_CASES:-all}"
    return 0
  fi
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-BOXTILT-BOX-PROGRESS] case=${case_name} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=boxtilt \
    FREE_BOX_MASS=0.75 \
    FREE_STEPS="${FREE_STEPS:-1200}" \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-2.35}" \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-2.35}" \
    TARGET_WINDOW_CENTER="${TARGET_WINDOW_CENTER:-2.0}" \
    TARGET_WINDOW_HALFWIDTH="${TARGET_WINDOW_HALFWIDTH:-0.35}" \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-80}" \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-50}" \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-40}" \
    MAX_FINAL_HOLD_FALL_EVENTS="${MAX_FINAL_HOLD_FALL_EVENTS:-0}" \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS="${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-0}" \
    AGILE_COMMAND_BOX_PROGRESS_CONTROLLER=1 \
    AGILE_COMMAND_BOX_PROGRESS_START_STEP="${AGILE_COMMAND_BOX_PROGRESS_START_STEP:-0}" \
    AGILE_COMMAND_BOX_PROGRESS_TARGET="${AGILE_COMMAND_BOX_PROGRESS_TARGET:-2.0}" \
    AGILE_COMMAND_BOX_PROGRESS_DEADBAND="${AGILE_COMMAND_BOX_PROGRESS_DEADBAND:-0.10}" \
    AGILE_COMMAND_BOX_PROGRESS_GAIN="${AGILE_COMMAND_BOX_PROGRESS_GAIN:-0.07}" \
    AGILE_COMMAND_BOX_PROGRESS_MAX_FORWARD="${AGILE_COMMAND_BOX_PROGRESS_MAX_FORWARD:-0.095}" \
    AGILE_COMMAND_BOX_PROGRESS_MAX_REVERSE="${AGILE_COMMAND_BOX_PROGRESS_MAX_REVERSE:-0.020}" \
    AGILE_COMMAND_BOX_PROGRESS_MAX_TILT="${AGILE_COMMAND_BOX_PROGRESS_MAX_TILT:-0.55}" \
    AGILE_COMMAND_BOX_PROGRESS_MAX_BOX_TILT="${AGILE_COMMAND_BOX_PROGRESS_MAX_BOX_TILT:-0.70}" \
    AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.68 \
    AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.030 \
    AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 \
    AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.95 \
    AGILE_COMMAND_HOLD_FINAL_SCALE=0.020 \
    AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

run_case progress_only

run_case progress_lateral_neg \
  AGILE_COMMAND_BOX_LATERAL_CONTROLLER=1 \
  AGILE_COMMAND_BOX_LATERAL_SIGN=-1.0 \
  AGILE_COMMAND_BOX_LATERAL_DEADBAND=0.18 \
  AGILE_COMMAND_BOX_LATERAL_GAIN=0.018 \
  AGILE_COMMAND_BOX_LATERAL_LIMIT=0.010

run_case progress_lateral_pos \
  AGILE_COMMAND_BOX_LATERAL_CONTROLLER=1 \
  AGILE_COMMAND_BOX_LATERAL_SIGN=1.0 \
  AGILE_COMMAND_BOX_LATERAL_DEADBAND=0.18 \
  AGILE_COMMAND_BOX_LATERAL_GAIN=0.018 \
  AGILE_COMMAND_BOX_LATERAL_LIMIT=0.010

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
