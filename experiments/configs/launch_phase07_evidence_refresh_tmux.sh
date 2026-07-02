#!/usr/bin/env bash
set -euo pipefail

# Launch the Phase07 evidence-refresh audit inside an existing tmux-held Slurm
# allocation. This submits no new allocation and performs no training.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase07_evidence_refresh_alloc_20260628}"
WINDOW_NAME="${WINDOW_NAME:-phase07_evidence_refresh}"
RUN_TAG="${RUN_TAG:-phase07_evidence_refresh_v1_20260628}"
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
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running yet." >&2
  squeue -j "$JOB_ID" -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R' >&2 || true
  exit 3
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv; configure envs/newton locally before compute use." >&2
  exit 4
fi

bash -n "$ROOT/experiments/configs/run_phase07_evidence_refresh_in_alloc.sh"

log="$ROOT/logs/newton/${RUN_TAG}.log"
remote_cmd="cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") bash $(printf '%q' "$ROOT/experiments/configs/run_phase07_evidence_refresh_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:0 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_PHASE07_EVIDENCE_REFRESH_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
RUN_TAG=$RUN_TAG
DOWNSTREAM_USE=audit_refresh_only_not_training_not_success_claim
EOF
