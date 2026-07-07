#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 held-out geometry instability-planted suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_closefront_heldout_geometry_instability_planted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_closefront_heldout_geometry_instability_planted/${SUITE_STAMP_PREFIX}}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/closefront_heldout_geometry_instability_planted_status.tsv"
summary_out="${OUTPUT_ROOT}/closefront_heldout_geometry_instability_planted_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_case() {
  local case_name="$1"
  shift
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-HELDOUT-GEOMETRY-INSTABILITY-PLANTED] case=${case_name} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 \
    TARGET_WINDOW_CENTER=2.0 \
    TARGET_WINDOW_HALFWIDTH=0.35 \
    COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
    LARGERBOX_STRICT_MODE=chestpad \
    FREE_STEPS=1000 \
    FREE_BOX_MASS=0.50 \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=100 \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=100 \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=100 \
    AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL=-1 \
    AGILE_COMMAND_STOP_ROBOT_TARGET_TRAVEL=-1 \
    AGILE_COMMAND_STOP_TILT_MIN_STEP=420 \
    AGILE_COMMAND_STOP_TILT_MIN_BOX_TARGET_TRAVEL=0.75 \
    AGILE_COMMAND_STOP_TILT_THRESHOLD=0.30 \
    AGILE_COMMAND_STOP_BOX_TILT_THRESHOLD=0.34 \
    AGILE_COMMAND_HOLD_MODE=policy_then_stand \
    AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS=8 \
    AGILE_COMMAND_HOLD_STAND_BLEND_RATE=0.16 \
    AGILE_COMMAND_HOLD_SCALE=0.010 \
    AGILE_COMMAND_HOLD_ADAPTIVE_SCALE=0 \
    AGILE_COMMAND_HOLD_YAW_CORRECTION=1 \
    AGILE_COMMAND_HOLD_YAW_GAIN=0.035 \
    AGILE_COMMAND_HOLD_YAW_LIMIT=0.06 \
    AGILE_COMMAND_HOLD_YAW_SIGN=-1.0 \
    AGILE_COMMAND_HOLD_STAND_HIP_PITCH=-0.15 \
    AGILE_COMMAND_HOLD_STAND_KNEE=0.46 \
    AGILE_COMMAND_HOLD_STAND_ANKLE_PITCH=-0.28 \
    AGILE_COMMAND_HOLD_STAND_HIP_ROLL=0.0 \
    AGILE_COMMAND_HOLD_STAND_ANKLE_ROLL=0.0 \
    AGILE_COMMAND_HOLD_STAND_WAIST_PITCH=-0.045 \
    BALANCE_FEEDBACK_CONTROLLER=1 \
    BALANCE_START_ON_AGILE_HOLD=1 \
    BALANCE_FEEDBACK_BASE=command \
    BALANCE_PITCH_SIGN=1.0 \
    BALANCE_ROLL_SIGN=-1.0 \
    BALANCE_PITCH_GAIN=0.07 \
    BALANCE_PITCH_RATE_GAIN=0.003 \
    BALANCE_ROLL_GAIN=0.07 \
    BALANCE_ROLL_RATE_GAIN=0.003 \
    BALANCE_ADJUSTMENT_LIMIT=0.05 \
    AGILE_COMMAND_TERMINAL_SUPPORT_CONTROLLER=1 \
    AGILE_COMMAND_TERMINAL_SUPPORT_START_BOX_TARGET_TRAVEL=0.95 \
    AGILE_COMMAND_TERMINAL_SUPPORT_TARGET=2.0 \
    AGILE_COMMAND_TERMINAL_SUPPORT_DEADBAND=0.08 \
    AGILE_COMMAND_TERMINAL_SUPPORT_GAIN=0.025 \
    AGILE_COMMAND_TERMINAL_SUPPORT_MAX_FORWARD=0.012 \
    AGILE_COMMAND_TERMINAL_SUPPORT_MAX_REVERSE=0.012 \
    AGILE_COMMAND_TERMINAL_SUPPORT_LATERAL_SOURCE=average \
    AGILE_COMMAND_TERMINAL_SUPPORT_LATERAL_DEADBAND=0.06 \
    AGILE_COMMAND_TERMINAL_SUPPORT_LATERAL_GAIN=0.010 \
    AGILE_COMMAND_TERMINAL_SUPPORT_LATERAL_LIMIT=0.003 \
    AGILE_COMMAND_TERMINAL_SUPPORT_LATERAL_SIGN=1.0 \
    AGILE_COMMAND_TERMINAL_SUPPORT_YAW_GAIN=0.008 \
    AGILE_COMMAND_TERMINAL_SUPPORT_YAW_LIMIT=0.010 \
    AGILE_COMMAND_TERMINAL_SUPPORT_YAW_SIGN=1.0 \
    AGILE_COMMAND_TERMINAL_SUPPORT_MAX_TILT=0.85 \
    AGILE_COMMAND_TERMINAL_SUPPORT_MAX_BOX_TILT=0.95 \
    TERMINAL_SUPPORT_POSTURE_CONTROLLER=0 \
    TERMINAL_CENTROIDAL_SUPPORT_CONTROLLER=0 \
    CRADLE_FINAL_SIDE_GUARDS=1 \
    CRADLE_FINAL_SIDE_GUARD_SPAWN_ON_TRIGGER=1 \
    CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_HOLD=1 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_X=-0.19 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_Y=0.0 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_Z=0.10 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_X=0.18 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_Y=0.018 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_Z=0.18 \
    CRADLE_FINAL_SIDE_GUARD_HALF_SPACING=0.12 \
    CRADLE_FINAL_SIDE_GUARD_MASS_SCALE=0.25 \
    "$@" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  return "${status}"
}

overall_status=0

run_case wide_y012_instability_planted \
  FREE_BOX_SIZE_Y=0.12 \
  CRADLE_FINAL_SIDE_GUARD_HALF_SPACING=0.13 \
  CRADLE_FINAL_SIDE_GUARD_MASS_SCALE=0.15 || overall_status=1

run_case tall_z009_instability_planted \
  FREE_BOX_SIZE_Z=0.09 || overall_status=1

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
