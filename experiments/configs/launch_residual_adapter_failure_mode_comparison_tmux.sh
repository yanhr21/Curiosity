#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_residual_source_alloc_20260627_034021}"
JOB_ID="${JOB_ID:-154142}"
WINDOW_NAME="${WINDOW_NAME:-residual_adapter_failure_modes}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/residual_adapter_failure_mode_comparison_v1.json}"
LOG="${LOG:-$ROOT/logs/newton/residual_adapter_failure_mode_comparison_v1_20260627.log}"

if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local residual-adapter venv at $TRAINER_VENV" >&2
  exit 1
fi

mkdir -p "$ROOT/logs/newton"

remote_cmd=$(cat <<EOF
cd $(printf '%q' "$ROOT")
set -euo pipefail
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="
$(printf '%q' "$TRAINER_VENV/bin/python") $(printf '%q' "$ROOT/experiments/configs/compare_residual_adapter_failure_modes.py") \
  --root $(printf '%q' "$ROOT") \
  --config $(printf '%q' "$CONFIG")
EOF
)

tmux new-window -t "$TMUX_SESSION" -n "${WINDOW_NAME}_$(date +%H%M)" \
  "srun --jobid=$(printf '%q' "$JOB_ID") --overlap --ntasks=1 --cpus-per-task=2 --gres=gpu:0 bash -lc $(printf '%q' "$remote_cmd") 2>&1 | tee $(printf '%q' "$LOG")"

echo "TMUX_SESSION=$TMUX_SESSION"
echo "JOB_ID=$JOB_ID"
echo "LOG=$LOG"
