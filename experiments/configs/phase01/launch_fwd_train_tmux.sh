#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 01 forward-model training inside an existing tmux-held H200
# Slurm allocation. This script submits no new allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_h200}"
WINDOW_NAME="${WINDOW_NAME:-p01_fwd_a1}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/fwd_train.json}"
RUN_TAG="${RUN_TAG:-p01_fwd_a1_$(date +%Y%m%d_%H%M%S)}"
RUN_MODE="${RUN_MODE:-train}"
DEVICE="${DEVICE:-cuda:0}"
ALLOW_REAL_TRAINING="${ALLOW_REAL_TRAINING:-0}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase01/core/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p logs/newton/phase01/core

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held H200 allocation." >&2
  exit 2
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 3
fi
if ! squeue -h -j "$JOB_ID" >/dev/null 2>&1; then
  echo "ERROR: Slurm job $JOB_ID is not visible." >&2
  exit 4
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing Newton venv: $NEWTON_VENV/bin/python" >&2
  exit 5
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing trainer venv: $TRAINER_VENV/bin/python" >&2
  exit 6
fi
if [[ "$RUN_MODE" == "train" && "$ALLOW_REAL_TRAINING" != "1" ]]; then
  echo "ERROR: RUN_MODE=train requires ALLOW_REAL_TRAINING=1." >&2
  exit 7
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 8
fi

env_file="$ROOT/logs/newton/phase01/core/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'RUN_MODE=%q\n' "$RUN_MODE"
  printf 'CONFIG=%q\n' "$CONFIG"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
  printf 'DEVICE=%q\n' "$DEVICE"
  printf 'ALLOW_REAL_TRAINING=%q\n' "$ALLOW_REAL_TRAINING"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase01/run_fwd_train_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE01_FWD_TRAIN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
CONFIG=$CONFIG
RUN_MODE=$RUN_MODE
REQUIRES_H200=true
DOWNSTREAM_USE=forward_model_training_component_not_policy_success
EOF
