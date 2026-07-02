#!/usr/bin/env bash
set -euo pipefail

# Launch the remaining Phase07 ablation queue inside an existing tmux-held
# Slurm allocation. This submits no new allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_contact_ablation_alloc_20260627_222339}"
WINDOW_NAME="${WINDOW_NAME:-phase07_remaining_ablation_queue}"
RUN_TAG="${RUN_TAG:-phase07_remaining_ablation_queue_20260627}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
ABLATED_VARIANTS="${ABLATED_VARIANTS:-contact_only shuffled_contact delayed_contact no_learning_progress}"

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
  echo "ERROR: missing local Newton venv; configure envs/ locally before compute use." >&2
  exit 4
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local trainer venv; configure envs/residual_adapter locally before compute use." >&2
  exit 5
fi

bash -n "$ROOT/experiments/configs/run_phase07_remaining_ablation_queue_in_alloc.sh"
sed -n '1,160p' AGENTS.md >/dev/null

log="$ROOT/logs/newton/${RUN_TAG}.log"
remote_cmd="cd $(printf '%q' "$ROOT") && NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") TRAINER_VENV=$(printf '%q' "$TRAINER_VENV") DEVICE=$(printf '%q' "$DEVICE") ABLATED_VARIANTS=$(printf '%q' "$ABLATED_VARIANTS") bash $(printf '%q' "$ROOT/experiments/configs/run_phase07_remaining_ablation_queue_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_PHASE07_REMAINING_ABLATION_QUEUE_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
RUN_TAG=$RUN_TAG
DEVICE=$DEVICE
ABLATED_VARIANTS=$ABLATED_VARIANTS
DOWNSTREAM_USE=blocked_until_manual_visual_inspection_and_report_update
EOF
