#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

MULTISEED_STAMP="${MULTISEED_STAMP:-20260705_probe_parameter_search_multiseed_7068_7070}"
MULTISEED_OUTPUT="experiments/outputs/probe_parameter_search_multiseed/${MULTISEED_STAMP}"
LOG_DIR="logs/probe_parameter_search_multiseed"
mkdir -p "${MULTISEED_OUTPUT}" "${LOG_DIR}"

SEEDS=(${SEEDS:-7068 7069 7070})
declare -a STAMPS=()

for seed in "${SEEDS[@]}"; do
  stamp="${MULTISEED_STAMP}_seed${seed}"
  STAMPS+=("${stamp}")
  echo "[SEED] seed=${seed} stamp=${stamp}"
  set +e
  STAMP="${stamp}" BOX_SEED="${seed}" \
    bash scripts/isaac/run_probe_parameter_search_carry_diag.sh \
    2>&1 | tee "${LOG_DIR}/${stamp}.log"
  status=${PIPESTATUS[0]}
  set -e
  echo "${status}" > "${MULTISEED_OUTPUT}/${stamp}_status.txt"
  if [[ "${status}" -ne 0 ]]; then
    echo "[WARN] seed ${seed} parameter-search runner exited ${status}; continuing to aggregate." >&2
  fi
done

summary_args=()
for stamp in "${STAMPS[@]}"; do
  summary_args+=(--stamp "${stamp}")
done

python3 scripts/isaac/summarize_probe_parameter_search_multiseed.py \
  "${summary_args[@]}" \
  --output "${MULTISEED_OUTPUT}/probe_parameter_search_multiseed_summary.json"

echo "[INFO] Multi-seed probe parameter-search output: ${MULTISEED_OUTPUT}"
