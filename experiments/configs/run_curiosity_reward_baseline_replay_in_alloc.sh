#!/usr/bin/env bash
set -euo pipefail

# Evaluate Phase 03 curiosity reward components inside an existing allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/curiosity_reward_baseline_replay_v1.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT/experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.csv}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: missing curiosity reward config: $CONFIG" >&2
  exit 4
fi

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG=$CONFIG"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "OUTPUT_CSV=$OUTPUT_CSV"
echo "NEWTON_VENV=$NEWTON_VENV"

echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

"$NEWTON_VENV/bin/python" experiments/configs/evaluate_curiosity_reward_baseline_replay.py \
  --config "$CONFIG" \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV" \
  --root "$ROOT"
