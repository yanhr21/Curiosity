#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 03 learning-progress scoring inside an existing Curiosity
# tmux-held allocation. This script submits no new allocation and does not
# update a policy.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_forward_alloc_20260627_105456}"
WINDOW_NAME="${WINDOW_NAME:-phase03_curiosity_learning_progress}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/curiosity_learning_progress_v1.json}"
RUN_TAG="${RUN_TAG:-curiosity_learning_progress_v1_20260627}"
DEVICE="${DEVICE:-cuda:0}"

cd "$ROOT"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a currently running tmux-held Slurm allocation." >&2
  exit 1
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: required tmux-held session not found: $TMUX_SESSION" >&2
  exit 2
fi
if ! squeue -h -j "$JOB_ID" >/dev/null 2>&1; then
  echo "ERROR: Slurm job $JOB_ID is not visible." >&2
  exit 3
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv; configure envs/ locally before compute use." >&2
  exit 4
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local trainer venv; configure envs/residual_adapter locally before compute use." >&2
  exit 6
fi

bash -n "$ROOT/experiments/configs/run_curiosity_learning_progress_in_alloc.sh"
sed -n '1,120p' AGENTS.md >/dev/null

log="$ROOT/logs/newton/${RUN_TAG}.log"
remote_cmd="cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") TRAINER_VENV=$(printf '%q' "$TRAINER_VENV") CONFIG=$(printf '%q' "$CONFIG") DEVICE=$(printf '%q' "$DEVICE") bash $(printf '%q' "$ROOT/experiments/configs/run_curiosity_learning_progress_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_CURIOSITY_LEARNING_PROGRESS_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
CONFIG=$CONFIG
TRAINER_VENV=$TRAINER_VENV
RUN_TAG=$RUN_TAG
EOF
