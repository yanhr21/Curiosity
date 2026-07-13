#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing SUGAR visualization generation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_DIR}/visualizations}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/sugar/render_official_sugar_training_curves.py" \
  --output-dir "${OUTPUT_DIR}" \
  --plot-dir "${PLOT_DIR}"
