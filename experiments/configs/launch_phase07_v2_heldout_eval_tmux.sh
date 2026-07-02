#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase07_v2_source_alloc_20260628}"
WINDOW_NAME="${WINDOW_NAME:-phase07_v2_heldout_eval}"
RUN_TAG="${RUN_TAG:-phase07_v2_heldout_eval_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase07_v2_eval}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-28_phase07_v2_heldout_eval_v1.md}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.5}"
NO_CURIOSITY_CHECKPOINT="${NO_CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase07_v2_residual_adapter_trainer_v1_20260628/phase07_v2_residual_adapter_v1_train_20260628.pt}"
CURIOSITY_CHECKPOINT="${CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase07_v2_curiosity_weighted_residual_adapter_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_v1_train_20260628.pt}"

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
  exit 3
fi

bash -n "$ROOT/experiments/configs/run_phase07_v2_heldout_eval_in_alloc.sh"
log="$ROOT/logs/newton/${RUN_TAG}.log"
remote_cmd="cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") EVAL_TAG_PREFIX=$(printf '%q' "$EVAL_TAG_PREFIX") EVAL_TAG_SUFFIX=$(printf '%q' "$EVAL_TAG_SUFFIX") REPORT_PATH=$(printf '%q' "$REPORT_PATH") NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") TRAINER_VENV=$(printf '%q' "$TRAINER_VENV") DEVICE=$(printf '%q' "$DEVICE") ACTIVE_THRESHOLD=$(printf '%q' "$ACTIVE_THRESHOLD") NO_CURIOSITY_CHECKPOINT=$(printf '%q' "$NO_CURIOSITY_CHECKPOINT") CURIOSITY_CHECKPOINT=$(printf '%q' "$CURIOSITY_CHECKPOINT") bash $(printf '%q' "$ROOT/experiments/configs/run_phase07_v2_heldout_eval_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_PHASE07_V2_HELDOUT_EVAL_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG=$log
RUN_TAG=$RUN_TAG
EVAL_TAG_PREFIX=$EVAL_TAG_PREFIX
EVAL_TAG_SUFFIX=$EVAL_TAG_SUFFIX
REPORT_PATH=$REPORT_PATH
ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD
DOWNSTREAM_USE=heldout_eval_not_success_claim_until_manual_visual_and_mainstream_gate
EOF
