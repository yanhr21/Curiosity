#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_src_h200}"
WINDOW_NAME="${WINDOW_NAME:-p01_src_strict}"
RUN_TAG="${RUN_TAG:-p01_src_strict_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/src_strict_repair.json}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/resid/src_strict}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held H200 allocation." >&2
  exit 2
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 3
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running." >&2
  exit 4
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 5
fi
bash -n "$ROOT/experiments/configs/phase01/run_src_strict_repair_in_alloc.sh"

env_file="$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'CONFIG=%q\n' "$CONFIG"
  printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
  printf 'LOG_SUBDIR=%q\n' "$LOG_SUBDIR"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase01/run_src_strict_repair_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=2 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE01_SRC_STRICT_REPAIR_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
CONFIG=$CONFIG
DOWNSTREAM_USE=train_only_final_attempt_source_preflight_not_training
EOF
