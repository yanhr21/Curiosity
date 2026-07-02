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

RUN_TAG="${RUN_TAG:-phase07_official_method_readiness_v1_20260627}"
{
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'SLURM_JOB_ID=%q\n' "$SLURM_JOB_ID"
  printf 'HOSTNAME=%q\n' "$(hostname)"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'CLASSIFICATION=%q\n' "official_method_readiness_audit_only_not_training_not_inference"
} >"$ROOT/logs/newton/${RUN_TAG}_env.sh"

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/audit_phase07_official_method_readiness_v1.py" \
  --root "$ROOT" \
  --output "$ROOT/experiments/outputs/phase07_official_method_readiness_v1_20260627.json" \
  --report "$ROOT/experiments/reports/2026-06-27_phase07_official_method_readiness_v1.md"
