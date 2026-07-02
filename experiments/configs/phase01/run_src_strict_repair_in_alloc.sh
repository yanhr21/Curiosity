#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_src_strict_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/src_strict_repair.json}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/resid/src_strict}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR" \
  "$ROOT/data/processed/phase01/src_strict" \
  "$ROOT/experiments/reports/phase01/core/resid"

if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing trainer venv python: $TRAINER_VENV/bin/python" >&2
  exit 3
fi

echo "PHASE01_SRC_STRICT_REPAIR_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG=$CONFIG"
echo "TRAINER_VENV=$TRAINER_VENV"
echo "NOTE=train_only_preflight_not_training_not_curiosity_success"

"$TRAINER_VENV/bin/python" "$ROOT/experiments/configs/phase01/build_src_strict_repair.py" \
  --root "$ROOT" \
  --config "$CONFIG" \
  --run-tag "$RUN_TAG"

echo "PHASE01_SRC_STRICT_REPAIR_END"
