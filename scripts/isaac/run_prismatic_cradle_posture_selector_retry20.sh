#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run posture selector data processing on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

MANIFEST="${MANIFEST:-experiments/configs/prismatic_cradle_posture_selector_retry20_manifest.json}"
REPORT_DIR="${REPORT_DIR:-experiments/reports/prismatic_cradle_posture_selector}"
STAMP="${STAMP:-20260706_retry20_prismatic_cradle_posture_selector}"
OUTPUT="${OUTPUT:-${REPORT_DIR}/${STAMP}_report.json}"
TABLE_OUTPUT="${TABLE_OUTPUT:-${REPORT_DIR}/${STAMP}_candidates.jsonl}"

python3 scripts/isaac/summarize_prismatic_cradle_posture_selector.py \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --table-output "${TABLE_OUTPUT}"
