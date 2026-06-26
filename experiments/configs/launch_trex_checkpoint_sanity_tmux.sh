#!/usr/bin/env bash
set -euo pipefail

# Launch T-Rex checkpoint sanity inside an existing Curiosity tmux-held
# allocation. This script submits no new allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_next_source_alloc_20260626_232937}"
WINDOW_NAME="${WINDOW_NAME:-trex_checkpoint_sanity}"
RUN_TAG="${RUN_TAG:-trex_checkpoint_current_sanity_20260627}"
TREX_VENV="${TREX_VENV:-$ROOT/envs/trex/.venv}"
CUDA_ID="${CUDA_ID:-0}"

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
if [[ ! -x "$TREX_VENV/bin/python" ]]; then
  echo "ERROR: missing local T-Rex venv at $TREX_VENV; configure envs/ locally before compute use." >&2
  exit 4
fi

bash -n "$ROOT/experiments/configs/run_trex_checkpoint_sanity_in_alloc.sh"
sed -n '1,140p' AGENTS.md >/dev/null

log="$ROOT/logs/trex/${RUN_TAG}.log"
remote_cmd="cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") TREX_VENV=$(printf '%q' "$TREX_VENV") CUDA_ID=$(printf '%q' "$CUDA_ID") bash $(printf '%q' "$ROOT/experiments/configs/run_trex_checkpoint_sanity_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

mkdir -p "$ROOT/logs/trex"
window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_TREX_CHECKPOINT_SANITY_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
EOF
