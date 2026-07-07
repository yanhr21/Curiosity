#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 held-out geometry rescue-freeze suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_closefront_heldout_geometry_rescue_freeze}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_closefront_heldout_geometry_rescue_freeze/${SUITE_STAMP_PREFIX}}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/closefront_heldout_geometry_rescue_freeze_status.tsv"
summary_out="${OUTPUT_ROOT}/closefront_heldout_geometry_rescue_freeze_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_case() {
  local case_name="$1"
  shift
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-HELDOUT-GEOMETRY-RESCUE-FREEZE] case=${case_name} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 \
    TARGET_WINDOW_CENTER=2.0 \
    TARGET_WINDOW_HALFWIDTH=0.35 \
    MAX_FINAL_HOLD_COMMAND_X=0.001 \
    MAX_FINAL_HOLD_COMMAND_Y=0.003 \
    MAX_FINAL_HOLD_COMMAND_YAW=0.001 \
    MAX_FINAL_HOLD_FALL_EVENTS=0 \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 \
    COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
    LARGERBOX_STRICT_MODE=chestpad \
    FREE_STEPS=1000 \
    FREE_BOX_MASS=0.50 \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=100 \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=100 \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=100 \
    MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=100 \
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
    AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS=1 \
    AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW=1 \
    AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT=0.35 \
    AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT=0.45 \
    AGILE_COMMAND_HOLD_RESCUE_ENABLE=1 \
    AGILE_COMMAND_HOLD_RESCUE_OVERRIDES_FINAL_FREEZE=1 \
    AGILE_COMMAND_HOLD_RESCUE_ABS_ROLL_THRESHOLD=0.42 \
    AGILE_COMMAND_HOLD_RESCUE_FORWARD_PITCH_THRESHOLD=-999.0 \
    AGILE_COMMAND_HOLD_RESCUE_BLEND_RATE=0.025 \
    AGILE_COMMAND_HOLD_RESCUE_HIP_PITCH=-0.08 \
    AGILE_COMMAND_HOLD_RESCUE_KNEE=0.28 \
    AGILE_COMMAND_HOLD_RESCUE_ANKLE_PITCH=-0.18 \
    AGILE_COMMAND_HOLD_RESCUE_WAIST_PITCH=-0.02 \
    CRADLE_FINAL_SIDE_GUARDS=1 \
    CRADLE_FINAL_SIDE_GUARD_SPAWN_ON_TRIGGER=1 \
    CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_FINAL_HOLD=1 \
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

run_case wide_y012_hold_guard_rescue_freeze \
  FREE_BOX_SIZE_Y=0.12 \
  CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_FINAL_HOLD=0 \
  CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_HOLD=1 \
  CRADLE_FINAL_SIDE_GUARD_HALF_SPACING=0.13 \
  CRADLE_FINAL_SIDE_GUARD_MASS_SCALE=0.15 || overall_status=1

run_case tall_z009_rescue_freeze \
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
