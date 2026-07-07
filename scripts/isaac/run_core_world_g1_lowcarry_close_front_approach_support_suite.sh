#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 close-front approach-support suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_lowcarry_close_front_approach_support}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_close_front_approach_support/${SUITE_STAMP_PREFIX}}"
CASE_SET="${APPROACH_SUPPORT_CASE_SET:-default}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/close_front_approach_support_status.tsv"
summary_out="${OUTPUT_ROOT}/close_front_approach_support_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_case() {
  local case_name="$1"
  local steps="$2"
  local terminal_travel="$3"
  local final_travel="$4"
  local travel_start="$5"
  local travel_full="$6"
  local blend="$7"
  local hip_offset="$8"
  local knee_offset="$9"
  local ankle_offset="${10}"
  local waist_offset="${11}"
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-CLOSE-APPROACH-SUPPORT] case=${case_name} steps=${steps} terminal=${terminal_travel} final=${final_travel} start=${travel_start} full=${travel_full} blend=${blend} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=lowcarry \
    FREE_BOX_MASS=0.60 \
    FREE_BOX_POS_X=-0.14 \
    FREE_CRADLE_LOCAL_X=-0.14 \
    FREE_STEPS="${steps}" \
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
    APPROACH_SUPPORT_POSTURE_CONTROLLER=1 \
    APPROACH_SUPPORT_POSTURE_DISABLE_ON_FINAL_HOLD=1 \
    APPROACH_SUPPORT_POSTURE_TRAVEL_START="${travel_start}" \
    APPROACH_SUPPORT_POSTURE_TRAVEL_FULL="${travel_full}" \
    APPROACH_SUPPORT_POSTURE_BLEND_RATE="${blend}" \
    APPROACH_SUPPORT_POSTURE_HIP_PITCH_OFFSET="${hip_offset}" \
    APPROACH_SUPPORT_POSTURE_KNEE_OFFSET="${knee_offset}" \
    APPROACH_SUPPORT_POSTURE_ANKLE_PITCH_OFFSET="${ankle_offset}" \
    APPROACH_SUPPORT_POSTURE_WAIST_PITCH_OFFSET="${waist_offset}" \
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
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  return "${status}"
}

overall_status=0

case "${CASE_SET}" in
  quick)
    run_case soft1050_active 1050 1.25 1.20 0.65 1.15 0.015 -0.03 0.06 -0.03 -0.015 || overall_status=1
    ;;
  default)
    run_case soft1050_active 1050 1.25 1.20 0.65 1.15 0.015 -0.03 0.06 -0.03 -0.015 || overall_status=1
    run_case support1200_active 1200 1.25 1.20 0.65 1.20 0.020 -0.04 0.08 -0.04 -0.020 || overall_status=1
    ;;
  *)
    echo "Unknown APPROACH_SUPPORT_CASE_SET=${CASE_SET}; expected default or quick" >&2
    exit 2
    ;;
esac

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
