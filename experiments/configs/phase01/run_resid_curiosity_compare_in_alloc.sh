#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_resid_cur_compare_$(date +%Y%m%d_%H%M%S)}"
CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ -z "$CANDIDATE_SUMMARY" || ! -f "$CANDIDATE_SUMMARY" ]]; then
  echo "ERROR: CANDIDATE_SUMMARY must point to completed curiosity eval summary." >&2
  exit 3
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing trainer venv: $TRAINER_VENV/bin/python" >&2
  exit 4
fi

cd "$ROOT"
out_json="$ROOT/experiments/outputs/phase01/core/resid/curiosity_eval/${RUN_TAG}_comparison.json"
out_report="$ROOT/experiments/reports/phase01/core/resid/${RUN_TAG}_comparison.md"

"$TRAINER_VENV/bin/python" "$ROOT/experiments/configs/phase01/build_resid_curiosity_comparison.py" \
  --root "$ROOT" \
  --candidate-summary "$CANDIDATE_SUMMARY" \
  --output-json "$out_json" \
  --output-report "$out_report"
