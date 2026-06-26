#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 05 Newton contact-source conversion inside an existing
# Curiosity tmux-held allocation. This script submits no new allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_next_source_alloc_20260626_232937}"
WINDOW_NAME="${WINDOW_NAME:-phase05_contact_source_manifest}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/newton_lift_hold_contact_source_manifest_v1.json}"

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

bash -n "$ROOT/experiments/configs/run_newton_lift_hold_contact_source_manifest_in_alloc.sh"
sed -n '1,120p' AGENTS.md >/dev/null

log="$ROOT/logs/newton/newton_lift_hold_contact_source_manifest_v1_20260627.log"
remote_cmd="cd $(printf '%q' "$ROOT") && NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") CONFIG=$(printf '%q' "$CONFIG") bash $(printf '%q' "$ROOT/experiments/configs/run_newton_lift_hold_contact_source_manifest_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=2 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_CONTACT_SOURCE_MANIFEST_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
CONFIG=$CONFIG
EOF
