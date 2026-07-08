#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 posture gauntlet on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
GAUNTLET_STAMP="${GAUNTLET_STAMP:-$(date +%Y%m%d_g1_posture_gauntlet)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_posture_gauntlet/${GAUNTLET_STAMP}}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUTPUT_ROOT}/g1_posture_gauntlet_summary.json}"
CASES_RAW="${GAUNTLET_CASES:-lowcarry_base chestpad_terminal boxtilt_diagnostic lowcarry_lightbox lowcarry_heavybox}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/g1_posture_gauntlet_status.tsv"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

run_case() {
  local case_name="$1"
  shift
  local suite_stamp="${GAUNTLET_STAMP}_${case_name}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-POSTURE-GAUNTLET] case=${case_name} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env SUITE_STAMP="${suite_stamp}" "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

common_target_hold_env=(
  FREE_STEPS=819
  TARGET_WINDOW_CENTER=2.0
  TARGET_WINDOW_HALFWIDTH=0.35
  FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35
  FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35
  MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=80
  MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=50
  MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=40
  MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS=80
  MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS=50
  MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS=40
  MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=399
  MAX_FINAL_HOLD_COMMAND_X=0.001
  MAX_FINAL_HOLD_COMMAND_Y=0.003
  MAX_FINAL_HOLD_COMMAND_YAW=0.001
  MIN_FINAL_HOLD_ROBOT_Z=0.45
  MIN_FINAL_HOLD_BOX_Z=0.45
  MAX_FINAL_HOLD_TILT=0.35
  MAX_FINAL_HOLD_BOX_TILT=0.45
  MAX_FINAL_HOLD_FALL_EVENTS=0
  MAX_FINAL_HOLD_BOX_DROP_EVENTS=0
)

lowcarry_controller_env=(
  LARGERBOX_STRICT_MODE=lowcarry
  AGILE_COMMAND_HOLD_YAW_CORRECTION=1
  AGILE_COMMAND_HOLD_YAW_GAIN=0.0
  AGILE_COMMAND_HOLD_YAW_LIMIT=0.0
  AGILE_COMMAND_HOLD_YAW_SIGN=-1.0
  AGILE_COMMAND_HOLD_LATERAL_CORRECTION=1
  AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY=1
  AGILE_COMMAND_HOLD_LATERAL_ERROR_START=0.45
  AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR=1
  AGILE_COMMAND_HOLD_LATERAL_GAIN=0.006
  AGILE_COMMAND_HOLD_LATERAL_LIMIT=0.0015
  AGILE_COMMAND_HOLD_LATERAL_SIGN=1.0
  AGILE_COMMAND_HOLD_LATERAL_MAX_TILT=0.30
  AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT=0.35
  AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL=0.65
  AGILE_COMMAND_HOLD_TERMINAL_SCALE=0.015
  AGILE_COMMAND_HOLD_TERMINAL_LATCH=1
  AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL=0.6
  AGILE_COMMAND_HOLD_FINAL_SCALE=0.0
  AGILE_COMMAND_HOLD_FINAL_LATCH=1
  ARM_POSE_MODE=both_front_reach
  ARM_POSE_START_STEP=0
  ARM_POSE_RAMP_STEPS=120
)

for case_name in ${CASES_RAW}; do
  case "${case_name}" in
    lowcarry_base)
      run_case "${case_name}" \
        "${common_target_hold_env[@]}" \
        "${lowcarry_controller_env[@]}"
      ;;
    chestpad_terminal)
      run_case "${case_name}" \
        "${common_target_hold_env[@]}" \
        LARGERBOX_STRICT_MODE=lowcarry \
        CRADLE_CHEST_PAD_ENABLED=1 \
        CRADLE_CHEST_PAD_ENABLE_ON_HOLD=0 \
        CRADLE_CHEST_PAD_ENABLE_ON_TERMINAL_HOLD=1 \
        CRADLE_CHEST_PAD_LOCAL_X=-0.02 \
        CRADLE_CHEST_PAD_LOCAL_Z=0.10 \
        CRADLE_CHEST_PAD_SIZE_X=0.04 \
        CRADLE_CHEST_PAD_SIZE_Y=0.38 \
        CRADLE_CHEST_PAD_SIZE_Z=0.22 \
        "${lowcarry_controller_env[@]}"
      ;;
    boxtilt_diagnostic)
      run_case "${case_name}" \
        LARGERBOX_STRICT_MODE=boxtilt \
        FREE_STEPS=819 \
        TARGET_WINDOW_CENTER=2.0 \
        TARGET_WINDOW_HALFWIDTH=0.35 \
        MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=50 \
        MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=30 \
        MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=20 \
        FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 \
        FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 \
        ARM_POSE_MODE=right_front_reach \
        ARM_POSE_START_STEP=0 \
        ARM_POSE_RAMP_STEPS=120
      ;;
    lowcarry_lightbox)
      run_case "${case_name}" \
        "${common_target_hold_env[@]}" \
        "${lowcarry_controller_env[@]}" \
        FREE_BOX_MASS=0.25
      ;;
    lowcarry_heavybox)
      run_case "${case_name}" \
        "${common_target_hold_env[@]}" \
        "${lowcarry_controller_env[@]}" \
        FREE_BOX_MASS=0.75
      ;;
    *)
      echo "Unknown GAUNTLET_CASES entry: ${case_name}" >&2
      overall_status=1
      ;;
  esac
done

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  set +e
  python3 "${ROOT_DIR}/scripts/isaac/summarize_core_world_g1_largerbox_strict.py" \
    "${summary_args[@]}" \
    --output "${SUMMARY_OUT}"
  summary_status=$?
  set -e
  if [[ "${summary_status}" != "0" ]]; then
    overall_status=1
  fi
fi

exit "${overall_status}"
