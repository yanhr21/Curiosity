#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
NPZ="${NPZ:-$ROOT/experiments/outputs/${RUN_TAG}.npz}"
SUMMARY="${SUMMARY:-$ROOT/experiments/outputs/${RUN_TAG}_summary.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/experiments/outputs/${RUN_TAG}_accel_peak_analysis.json}"
TOP_K="${TOP_K:-12}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
for path in "$NPZ" "$SUMMARY"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required accel-analysis input: $path" >&2
    exit 4
  fi
done

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NPZ=$NPZ"
echo "SUMMARY=$SUMMARY"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "TOP_K=$TOP_K"

echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

"$NEWTON_VENV/bin/python" experiments/configs/analyze_lift_hold_accel_peak.py \
  --npz "$NPZ" \
  --summary "$SUMMARY" \
  --output "$OUTPUT_JSON" \
  --run-tag "$RUN_TAG" \
  --top-k "$TOP_K"
