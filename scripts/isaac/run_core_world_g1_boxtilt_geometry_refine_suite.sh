#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 boxtilt geometry refine suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_boxtilt_geometry_refine_760}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_boxtilt_geometry_refine/${SUITE_STAMP_PREFIX}}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/boxtilt_geometry_refine_status.tsv"
summary_out="${OUTPUT_ROOT}/boxtilt_geometry_refine_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

case_enabled() {
  local case_name="$1"
  local requested="${GEOMETRY_REFINE_CASES:-all}"
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
    echo "[G1-BOXTILT-GEOMETRY-REFINE] skip case=${case_name} GEOMETRY_REFINE_CASES=${GEOMETRY_REFINE_CASES:-all}"
    return 0
  fi
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-BOXTILT-GEOMETRY-REFINE] case=${case_name} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=boxtilt \
    FREE_BOX_MASS=0.75 \
    FREE_STEPS="${FREE_STEPS:-760}" \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-2.35}" \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-2.35}" \
    TARGET_WINDOW_CENTER="${TARGET_WINDOW_CENTER:-2.0}" \
    TARGET_WINDOW_HALFWIDTH="${TARGET_WINDOW_HALFWIDTH:-0.35}" \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-80}" \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-50}" \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-40}" \
    MAX_FINAL_HOLD_FALL_EVENTS="${MAX_FINAL_HOLD_FALL_EVENTS:-0}" \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS="${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-0}" \
    AGILE_COMMAND_HOLD_LATERAL_SIGN=-1.0 \
    AGILE_COMMAND_HOLD_LATERAL_GAIN=0.04 \
    AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.018 \
    AGILE_COMMAND_HOLD_TERMINAL_LATCH=1 \
    AGILE_COMMAND_HOLD_FINAL_LATCH=1 \
    AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=1.65 \
    AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.018 \
    AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=1.80 \
    AGILE_COMMAND_HOLD_FINAL_SCALE=0.0 \
    AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 \
    BALANCE_ROLL_TARGET_FROM_LATERAL=1 \
    BALANCE_ROLL_TARGET_LATERAL_SOURCE=average \
    BALANCE_ROLL_TARGET_LATERAL_SIGN=1.0 \
    BALANCE_ROLL_TARGET_LATERAL_GAIN=0.020 \
    BALANCE_ROLL_TARGET_LATERAL_LIMIT=0.030 \
    BALANCE_ROLL_TARGET_LATERAL_DEADBAND=0.45 \
    BALANCE_ROLL_TARGET_LATERAL_START_AFTER_HOLD_STEPS=24 \
    BALANCE_ROLL_TARGET_LATERAL_RAMP_STEPS=80 \
    BALANCE_ROLL_TARGET_LATERAL_MAX_TILT=0.45 \
    BALANCE_ROLL_TARGET_LATERAL_MAX_BOX_TILT=0.60 \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

run_case box_cradle_y_neg003 \
  FREE_BOX_POS_Y=-0.03 \
  FREE_CRADLE_LOCAL_Y=-0.03

run_case box_cradle_y_pos003 \
  FREE_BOX_POS_Y=0.03 \
  FREE_CRADLE_LOCAL_Y=0.03

run_case wider_lid_rails \
  CRADLE_TOP_LID_Y_SCALE=1.35 \
  CRADLE_SIDE_RAIL_HEIGHT=0.15 \
  CRADLE_END_STOP_HEIGHT=0.16

run_case final_chest_pad \
  CRADLE_CHEST_PAD_ENABLED=1 \
  CRADLE_CHEST_PAD_ENABLE_ON_HOLD=0 \
  CRADLE_CHEST_PAD_ENABLE_ON_TERMINAL_HOLD=0 \
  CRADLE_CHEST_PAD_ENABLE_ON_FINAL_HOLD=1 \
  CRADLE_CHEST_PAD_LOCAL_X=-0.04 \
  CRADLE_CHEST_PAD_SIZE_X=0.05 \
  CRADLE_CHEST_PAD_SIZE_Y=0.42 \
  CRADLE_CHEST_PAD_SIZE_Z=0.24

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
