#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run larger-box posture matrix on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
MATRIX_STAMP="${MATRIX_STAMP:-$(date +%Y%m%d_g1_largerbox_posture_matrix)}"
MODES_RAW="${LARGERBOX_POSTURE_MODES:-boxtilt lowcarry chestpad}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_largerbox_posture_matrix/${MATRIX_STAMP}}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUTPUT_ROOT}/largerbox_posture_matrix_summary.json}"

mkdir -p "${OUTPUT_ROOT}"

status_file="${OUTPUT_ROOT}/largerbox_posture_matrix_status.tsv"
printf "mode\tstatus\tsuite_stamp\n" > "${status_file}"

overall_status=0
for mode in ${MODES_RAW}; do
  suite_stamp="${MATRIX_STAMP}_${mode}"
  echo "[LARGERBOX-MATRIX] mode=${mode} suite_stamp=${suite_stamp}"
  set +e
  env \
    LARGERBOX_STRICT_MODE="${mode}" \
    SUITE_STAMP="${suite_stamp}" \
    bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
  status=$?
  set -e
  printf "%s\t%s\t%s\n" "${mode}" "${status}" "${suite_stamp}" >> "${status_file}"
  if [[ "${status}" != "0" ]]; then
    overall_status=1
  fi
done

summary_args=()
for mode in ${MODES_RAW}; do
  summary_args+=(
    --case-root
    "${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${MATRIX_STAMP}_${mode}"
  )
done

set +e
python3 "${ROOT_DIR}/scripts/isaac/summarize_core_world_g1_largerbox_strict.py" \
  "${summary_args[@]}" \
  --output "${SUMMARY_OUT}"
summary_status=$?
set -e

if [[ "${summary_status}" != "0" ]]; then
  overall_status=1
fi

exit "${overall_status}"
