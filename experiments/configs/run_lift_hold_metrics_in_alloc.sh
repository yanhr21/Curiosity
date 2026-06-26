#!/usr/bin/env bash
set -euo pipefail

# Extract Phase 02 lift-hold metrics inside an existing Slurm allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
SCHEMA="${SCHEMA:-$ROOT/experiments/configs/lift_hold_metrics_schema_v1.json}"
NPZ="${NPZ:-$ROOT/experiments/outputs/${RUN_TAG}.npz}"
SUMMARY="${SUMMARY:-$ROOT/experiments/outputs/${RUN_TAG}_summary.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/experiments/outputs/${RUN_TAG}_metrics.json}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT/experiments/outputs/${RUN_TAG}_metrics.csv}"
BASELINE_NAME="${BASELINE_NAME:-no_adaptation_scripted_grasp_lift}"
MASS_LABEL="${MASS_LABEL:-nominal}"
FRICTION_LABEL="${FRICTION_LABEL:-nominal}"
POSE_SEED="${POSE_SEED:-nominal}"
MANUAL_VISUAL_INSPECTION="${MANUAL_VISUAL_INSPECTION:-not_checked}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
for path in "$SCHEMA" "$NPZ" "$SUMMARY"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required metrics input: $path" >&2
    exit 4
  fi
done

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "SCHEMA=$SCHEMA"
echo "NPZ=$NPZ"
echo "SUMMARY=$SUMMARY"
echo "OUTPUT_JSON=$OUTPUT_JSON"
echo "OUTPUT_CSV=$OUTPUT_CSV"
echo "BASELINE_NAME=$BASELINE_NAME"
echo "MASS_LABEL=$MASS_LABEL"
echo "FRICTION_LABEL=$FRICTION_LABEL"
echo "POSE_SEED=$POSE_SEED"
echo "MANUAL_VISUAL_INSPECTION=$MANUAL_VISUAL_INSPECTION"

echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

"$NEWTON_VENV/bin/python" experiments/configs/extract_lift_hold_metrics.py \
  --npz "$NPZ" \
  --summary "$SUMMARY" \
  --schema "$SCHEMA" \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV" \
  --run-tag "$RUN_TAG" \
  --baseline-name "$BASELINE_NAME" \
  --mass-label "$MASS_LABEL" \
  --friction-label "$FRICTION_LABEL" \
  --pose-seed "$POSE_SEED" \
  --manual-visual-inspection "$MANUAL_VISUAL_INSPECTION"
