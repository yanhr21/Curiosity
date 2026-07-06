#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 boxtilt load probe suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP_PREFIX="${SUITE_STAMP_PREFIX:-20260707_g1_boxtilt_load_probe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_boxtilt_load_probe/${SUITE_STAMP_PREFIX}}"
MASSES_CSV="${MASSES_CSV:-0.25,0.50,0.75}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/boxtilt_load_probe_status.tsv"
summary_out="${OUTPUT_ROOT}/boxtilt_load_probe_summary.json"
printf "case\tmass_kg\tstatus\tsuite_stamp\n" > "${status_file}"

summary_args=()
overall_status=0

run_mass_case() {
  local label="$1"
  local mass="$2"
  local suite_stamp="${SUITE_STAMP_PREFIX}_${label}"
  local case_root="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${suite_stamp}"
  echo "[G1-BOXTILT-LOAD] case=${label} mass=${mass} suite_stamp=${suite_stamp}"
  summary_args+=(--case-root "${case_root}")
  set +e
  env \
    SUITE_STAMP="${suite_stamp}" \
    LARGERBOX_STRICT_MODE=boxtilt \
    FREE_BOX_MASS="${mass}" \
    FREE_STEPS="${FREE_STEPS:-819}" \
    FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-2.35}" \
    FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-2.35}" \
    TARGET_WINDOW_CENTER="${TARGET_WINDOW_CENTER:-2.0}" \
    TARGET_WINDOW_HALFWIDTH="${TARGET_WINDOW_HALFWIDTH:-0.35}" \
    MIN_TARGET_WINDOW_BOTH_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-80}" \
    MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-50}" \
    MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-40}" \
    MAX_FINAL_HOLD_FALL_EVENTS="${MAX_FINAL_HOLD_FALL_EVENTS:-0}" \
    MAX_FINAL_HOLD_BOX_DROP_EVENTS="${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-0}" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  local status=$?
  set -e
  printf "%s\t%s\t%s\t%s\n" "${label}" "${mass}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
}

IFS=',' read -r -a masses <<< "${MASSES_CSV}"
for raw_mass in "${masses[@]}"; do
  mass="$(echo "${raw_mass}" | tr -d '[:space:]')"
  [[ -n "${mass}" ]] || continue
  label="mass$(echo "${mass}" | tr '.' 'p')"
  run_mass_case "${label}" "${mass}"
done

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
