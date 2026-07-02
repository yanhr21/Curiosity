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
mkdir -p logs/newton experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627

RUN_TAG="${RUN_TAG:-phase07_mainstream_stage1_dataset_index_v1_20260627}"
{
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'SLURM_JOB_ID=%q\n' "$SLURM_JOB_ID"
  printf 'HOSTNAME=%q\n' "$(hostname)"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'CLASSIFICATION=%q\n' "stage1_indices_only_not_training_not_official_method_success"
} >"$ROOT/logs/newton/${RUN_TAG}_env.sh"

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_phase07_mainstream_stage1_dataset_index_v1.py" \
  --root "$ROOT" \
  --backfill-manifest "$ROOT/experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json" \
  --output-dir "$ROOT/experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627"
