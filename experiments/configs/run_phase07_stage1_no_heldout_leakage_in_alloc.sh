#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports

RUN_TAG="${RUN_TAG:-phase07_stage1_no_heldout_leakage_v1_20260627}"
{
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'SLURM_JOB_ID=%q\n' "$SLURM_JOB_ID"
  printf 'HOSTNAME=%q\n' "$(hostname)"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'CLASSIFICATION=%q\n' "stage1_no_heldout_leakage_audit_only"
} >"$ROOT/logs/newton/${RUN_TAG}_env.sh"

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/audit_phase07_stage1_no_heldout_leakage_v1.py" \
  --root "$ROOT" \
  --stage1-dir "$ROOT/experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627" \
  --output "$ROOT/experiments/outputs/phase07_stage1_no_heldout_leakage_v1_20260627.json" \
  --report "$ROOT/experiments/reports/2026-06-27_phase07_stage1_no_heldout_leakage_v1.md" \
  --require-pass
