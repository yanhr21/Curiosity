#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 low-carry load validation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_lowcarry_load_validation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_lowcarry_load_validation/${SUITE_STAMP_PREFIX}}"

mkdir -p "${OUTPUT_ROOT}"
status_file="${OUTPUT_ROOT}/lowcarry_load_validation_status.tsv"
summary_out="${OUTPUT_ROOT}/lowcarry_load_validation_summary.json"
printf "case\tmass_kg\tstatus\tsuite_stamp\n" > "${status_file}"
summary_case_roots=()

run_mass_case() {
  local case_name="$1"
  local mass_kg="$2"
  local suite_stamp="${SUITE_STAMP_PREFIX}_${case_name}"
  echo "[G1-LOWCARRY-LOAD] case=${case_name} mass=${mass_kg} suite_stamp=${suite_stamp}"
  summary_case_roots+=("${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}")
  set +e
  env \
    SUITE_STAMP_PREFIX="${suite_stamp}" \
    RUN_LOWCARRY_BASELINE=1 \
    RUN_CHESTPAD_LONGHOLD=0 \
    RUN_LOWCARRY_LIGHTBOX=0 \
    RUN_CHESTPAD_HEAVYBOX=0 \
    FREE_BOX_MASS="${mass_kg}" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\t%s\n" "${case_name}" "${mass_kg}" "${status}" "${suite_stamp}" >> "${status_file}"
  return "${status}"
}

overall_status=0
run_mass_case light025 0.25 || overall_status=1
run_mass_case nominal050 0.50 || overall_status=1
run_mass_case heavy075 0.75 || overall_status=1

summary_args=()
for case_root in "${summary_case_roots[@]}"; do
  summary_args+=(--case-root "${case_root}_lowcarry_targethold819")
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
