#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 03 curiosity reward replay evaluation in an existing Curiosity allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_next_source_alloc_20260626_232937}"
WINDOW_NAME="${WINDOW_NAME:-phase03_reward_replay}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/curiosity_reward_baseline_replay_v1.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT/experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.csv}"

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

bash -n "$ROOT/experiments/configs/run_curiosity_reward_baseline_replay_in_alloc.sh"
sed -n '1,120p' AGENTS.md >/dev/null

log="$ROOT/logs/newton/curiosity_reward_baseline_replay_v1_20260627.log"
remote_cmd="cd $(printf '%q' "$ROOT") && NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") CONFIG=$(printf '%q' "$CONFIG") OUTPUT_JSON=$(printf '%q' "$OUTPUT_JSON") OUTPUT_CSV=$(printf '%q' "$OUTPUT_CSV") bash $(printf '%q' "$ROOT/experiments/configs/run_curiosity_reward_baseline_replay_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=2 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_CURIOSITY_REWARD_REPLAY_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
OUTPUT_JSON=$OUTPUT_JSON
OUTPUT_CSV=$OUTPUT_CSV
EOF
