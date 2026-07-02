#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_dense_preflight_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/newton/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/dense/preflight/config_v1.json}"
EVIDENCE_MANIFEST="${EVIDENCE_MANIFEST:-$ROOT/experiments/configs/phase00/dense_tactile_infant/active_evidence_manifest_20260701_v1.json}"
DESIGN_CONTRACT="${DESIGN_CONTRACT:-$ROOT/experiments/configs/phase00/dense_tactile_infant/closed_loop_curiosity_design_v1.json}"
BASELINE_CONTRACT="${BASELINE_CONTRACT:-$ROOT/experiments/configs/phase00/dense_tactile_infant/baseline_eval_contract_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase01/dense/preflight/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase01/dense/preflight/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase01/dense/preflight/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in \
  "$PYTHON_BIN" \
  "$CONFIG" \
  "$EVIDENCE_MANIFEST" \
  "$DESIGN_CONTRACT" \
  "$BASELINE_CONTRACT" \
  "$ROOT/src/newton_tactile_curiosity/dense_training_preflight.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

run_log="$LOG_DIR/dense_training_preflight.log"

echo "PHASE01_DENSE_PREFLIGHT_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG=$CONFIG"
echo "EVIDENCE_MANIFEST=$EVIDENCE_MANIFEST"
echo "DESIGN_CONTRACT=$DESIGN_CONTRACT"
echo "BASELINE_CONTRACT=$BASELINE_CONTRACT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=preflight_not_training_not_curiosity_success"

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
  "$PYTHON_BIN" "$ROOT/src/newton_tactile_curiosity/dense_training_preflight.py" \
    --root "$ROOT" \
    --run-tag "$RUN_TAG" \
    --evidence-manifest "$EVIDENCE_MANIFEST" \
    --design-contract "$DESIGN_CONTRACT" \
    --baseline-contract "$BASELINE_CONTRACT" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE01_DENSE_PREFLIGHT_END"
