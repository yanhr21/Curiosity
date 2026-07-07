#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 close-front mass-0.525 side-guard timing suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_closefront_mass0525_side_guard_timing}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_closefront_mass0525_side_guard_timing/${SUITE_STAMP_PREFIX}}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/closefront_mass0525_side_guard_timing_status.tsv"
summary_out="${OUTPUT_ROOT}/closefront_mass0525_side_guard_timing_summary.json"
printf "case\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_case() {
  local case_name="$1"
  local enable_mode="$2"
  local terminal_box_travel="$3"
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-CLOSEFRONT-M0525-GUARD-TIMING] case=${case_name} enable_mode=${enable_mode} terminal_box_travel=${terminal_box_travel} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")

  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    SIDE_GUARD_CASE_SET=quick \
    SIDE_GUARD_QUICK_CASE_NAME="${case_name}" \
    SIDE_GUARD_QUICK_ENABLE_MODE="${enable_mode}" \
    SIDE_GUARD_QUICK_HALF_SPACING=0.12 \
    FREE_BOX_MASS=0.525 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_X=-0.19 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_Y=0.0 \
    CRADLE_FINAL_SIDE_GUARD_LOCAL_Z=0.10 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_X=0.18 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_Y=0.018 \
    CRADLE_FINAL_SIDE_GUARD_SIZE_Z=0.18 \
    CRADLE_FINAL_SIDE_GUARD_MASS_SCALE=0.25 \
    AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL="${terminal_box_travel}" \
    COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_chestpad_finalstop_side_guard_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\n" "${case_name}" "${status}" "${suite_stamp}" >> "${status_file}"
  return "${status}"
}

overall_status=0

run_case terminal_guard_lx19 terminal 1.05 || overall_status=1
run_case hold_guard_lx19 hold 1.05 || overall_status=1

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
