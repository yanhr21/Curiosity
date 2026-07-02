#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 01 transition manifest build inside an existing tmux-held H200
# Slurm allocation. This does not run conversion on the login node.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_h200}"
WINDOW_NAME="${WINDOW_NAME:-p01_manifest}"
RUN_TAG="${RUN_TAG:-phase01_core_manifest_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase01/core/${RUN_TAG}.srun.log}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/transition_manifest.json}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"

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
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 6
fi

env_file="$ROOT/logs/newton/phase01/core/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'CONFIG=%q\n' "$CONFIG"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'DEVICE=%q\n' "$DEVICE"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase01/run_transition_manifest_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE01_TRANSITION_MANIFEST_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
CONFIG=$CONFIG
REQUIRES_H200=true
DOWNSTREAM_USE=phase01_data_preparation_only_not_training
EOF
