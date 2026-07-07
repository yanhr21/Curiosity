#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 chest-pad final-stop side-guard suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_chestpad_finalstop_side_guard}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_chestpad_finalstop_side_guard/${SUITE_STAMP_PREFIX}}"
CASE_SET="${SIDE_GUARD_CASE_SET:-quick}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/chestpad_finalstop_side_guard_status.tsv"
summary_out="${OUTPUT_ROOT}/chestpad_finalstop_side_guard_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_case() {
  local case_name="$1"
  local enable_mode="$2"
  local half_spacing="$3"
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-CHESTPAD-SIDE-GUARD] case=${case_name} enable_mode=${enable_mode} half_spacing=${half_spacing} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")

  local guard_hold=0
  local guard_final=0
  local guard_target=0
  case "${enable_mode}" in
    hold) guard_hold=1 ;;
    final) guard_final=1 ;;
    target_window) guard_target=1 ;;
    *)
      echo "Unknown side-guard enable mode: ${enable_mode}" >&2
      return 2
      ;;
  esac

  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=chestpad \
    FREE_STEPS=1000 \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL=2.35 \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL=2.35 \
    TARGET_WINDOW_CENTER=2.0 \
    TARGET_WINDOW_HALFWIDTH=0.35 \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS=100 \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS=100 \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS=100 \
    MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS=100 \
    MAX_FINAL_HOLD_COMMAND_X=0.001 \
    MAX_FINAL_HOLD_COMMAND_Y=0.003 \
    MAX_FINAL_HOLD_COMMAND_YAW=0.001 \
    MAX_FINAL_HOLD_FALL_EVENTS=0 \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS=0 \
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
    CRADLE_FINAL_SIDE_GUARDS=1 \
    CRADLE_FINAL_SIDE_GUARD_SPAWN_ON_TRIGGER=1 \
    CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_HOLD="${guard_hold}" \
    CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_FINAL_HOLD="${guard_final}" \
    CRADLE_FINAL_SIDE_GUARD_ENABLE_ON_TARGET_WINDOW="${guard_target}" \
    CRADLE_FINAL_SIDE_GUARD_TARGET_WINDOW_MIN_STEP=700 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_X=-0.18 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_Y=0.0 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_Z=0.10 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_X=0.18 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_Y=0.018 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_Z=0.18 \
    CRADLE_FINAL_SIDE_GUARD_HALF_SPACING="${half_spacing}" \
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
    run_case final_guard_hs090 final 0.09 || overall_status=1
    ;;
  default)
    run_case final_guard_hs090 final 0.09 || overall_status=1
    run_case target_guard_hs090 target_window 0.09 || overall_status=1
    ;;
  *)
    echo "Unknown SIDE_GUARD_CASE_SET=${CASE_SET}; expected quick or default" >&2
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
