#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
TREX_VENV="${TREX_VENV:-$ROOT/envs/trex/.venv}"
RUN_TAG="${RUN_TAG:-trex_checkpoint_current_sanity_20260627}"
CUDA_ID="${CUDA_ID:-0}"

cd "$ROOT"

sed -n '1,140p' AGENTS.md >/dev/null

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

if [[ ! -x "$TREX_VENV/bin/python" ]]; then
  echo "ERROR: missing local T-Rex venv at $TREX_VENV; configure envs/ locally before compute use." >&2
  exit 4
fi

source "$TREX_VENV/bin/activate"

integrity_json="$ROOT/experiments/outputs/${RUN_TAG}_integrity.json"
model_load_json="$ROOT/experiments/outputs/${RUN_TAG}_midtrain_model_load.json"

echo "compute_host=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-missing}"
nvidia-smi

PYTHONPATH="$ROOT/external/T-Rex:${PYTHONPATH:-}" \
  "$TREX_VENV/bin/python" "$ROOT/experiments/configs/trex_checkpoint_integrity_sanity.py" \
  --root "$ROOT" \
  --output "$integrity_json"

PYTHONPATH="$ROOT/external/T-Rex:${PYTHONPATH:-}" \
  "$TREX_VENV/bin/python" "$ROOT/experiments/configs/trex_midtrain_model_load_sanity.py" \
  --workspace "$ROOT" \
  --cuda "$CUDA_ID" \
  --output "$model_load_json"

echo "TREX_CHECKPOINT_INTEGRITY_JSON=$integrity_json"
echo "TREX_MIDTRAIN_MODEL_LOAD_JSON=$model_load_json"
