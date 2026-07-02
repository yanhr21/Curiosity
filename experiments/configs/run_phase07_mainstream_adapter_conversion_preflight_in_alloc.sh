#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
BACKFILL_MANIFEST="${BACKFILL_MANIFEST:-experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json}"
OUTPUT="${OUTPUT:-experiments/outputs/phase07_mainstream_adapter_conversion_preflight_v1_20260627/manifest.json}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi

echo "PHASE07_MAINSTREAM_ADAPTER_CONVERSION_PREFLIGHT_START"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "BACKFILL_MANIFEST=$BACKFILL_MANIFEST"
echo "OUTPUT=$OUTPUT"
echo "NOTE=preflight_only_not_dataset_conversion_not_training_not_success_claim"

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_phase07_mainstream_adapter_conversion_preflight_v1.py" \
  --root "$ROOT" \
  --backfill-manifest "$BACKFILL_MANIFEST" \
  --output "$OUTPUT"

echo "PHASE07_MAINSTREAM_ADAPTER_CONVERSION_PREFLIGHT_END"
