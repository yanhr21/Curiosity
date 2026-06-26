#!/usr/bin/env bash
set -euo pipefail

# Launch lift-hold metrics extraction in an existing Curiosity tmux-held allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_next_source_alloc_20260626_232937}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
WINDOW_NAME="${WINDOW_NAME:-lift_hold_metrics}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"

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

bash -n "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
sed -n '1,120p' AGENTS.md >/dev/null

log="$ROOT/logs/newton/${RUN_TAG}_metrics.log"
remote_cmd="cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") MASS_LABEL=$(printf '%q' "${MASS_LABEL:-nominal}") FRICTION_LABEL=$(printf '%q' "${FRICTION_LABEL:-nominal}") POSE_SEED=$(printf '%q' "${POSE_SEED:-nominal}") MANUAL_VISUAL_INSPECTION=$(printf '%q' "${MANUAL_VISUAL_INSPECTION:-not_checked}") bash $(printf '%q' "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=2 bash -lc $(printf '%q' "$remote_cmd")"

window="${WINDOW_NAME}_${RUN_TAG##*_}"
window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$window")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_LIFT_HOLD_METRICS_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$window
JOB_ID=$JOB_ID
LOG=$log
EOF
