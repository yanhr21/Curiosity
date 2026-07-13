#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing SUGAR rollout visualization generation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_DIR}/eval/refiner_model5000_rollout_eval_novideo/raw_npz}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_DIR}/visualizations}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-model_5000}"
POLICY_STAGE="${POLICY_STAGE:-refiner}"
EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-16}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-refiner_model5000_rollout_summary}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/sugar/render_official_sugar_refiner5000_rollout_summary.py" \
  --rollout-dir "${ROLLOUT_DIR}" \
  --expected-windows "${EXPECTED_WINDOWS}" \
  --checkpoint-label "${CHECKPOINT_LABEL}" \
  --policy-stage "${POLICY_STAGE}" \
  --output "${PLOT_DIR}/${OUTPUT_BASENAME}.png" \
  --summary-json "${PLOT_DIR}/${OUTPUT_BASENAME}.json"
