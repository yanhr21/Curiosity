#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/outputs/phase07_action_bridge_backfill_v1_20260627}"
MANIFEST="${MANIFEST:-experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi

echo "PHASE07_ACTION_BRIDGE_BACKFILL_START"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MANIFEST=$MANIFEST"
echo "NOTE=data_processing_inside_held_allocation_not_training_not_success_claim"

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/backfill_phase07_candidate_action_bridge_v1.py" \
  --root "$ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --manifest "$MANIFEST"

echo "PHASE07_ACTION_BRIDGE_BACKFILL_END"
