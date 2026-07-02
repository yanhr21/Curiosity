#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_src_h200}"
WINDOW_NAME="${WINDOW_NAME:-p01_cur_eval}"
RUN_TAG="${RUN_TAG:-p01_resid_cur_eval_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-phase01/core/resid/curiosity_eval}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/resid/curiosity_eval}"
VISUAL_PHASE_DIR="${VISUAL_PHASE_DIR:-phase01/core/resid/curiosity_eval}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "logs/newton/$LOG_SUBDIR"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held H200 allocation." >&2
  exit 2
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: CHECKPOINT must point to the trained curiosity residual checkpoint." >&2
  exit 3
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 4
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running yet." >&2
  exit 5
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 6
fi
bash -n "$ROOT/experiments/configs/phase01/run_resid_base_eval_in_alloc.sh"

env_file="$ROOT/logs/newton/$LOG_SUBDIR/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
  printf 'DEVICE=%q\n' "$DEVICE"
  printf 'OUTPUT_SUBDIR=%q\n' "$OUTPUT_SUBDIR"
  printf 'LOG_SUBDIR=%q\n' "$LOG_SUBDIR"
  printf 'VISUAL_PHASE_DIR=%q\n' "$VISUAL_PHASE_DIR"
  printf 'METHOD_LABEL=%q\n' "curiosity_resid"
  printf 'METHOD_NAME=%q\n' "phase01_curiosity_weighted_residual"
  printf 'METHOD_REPORT_TITLE=%q\n' "Phase 01 Curiosity-Weighted Residual Held-Out Eval"
  printf 'METHOD_REPORT_NOTE=%q\n' "This is held-out evaluation of the curiosity-weighted residual candidate. It is not a success claim until compared against the strongest baseline set without safety regression."
  printf 'METHOD_SUMMARY_CLASSIFICATION=%q\n' "phase01_curiosity_weighted_residual_heldout_eval_summary_v1"
  printf 'METHOD_NOT_CURIOSITY_SUCCESS=%q\n' "1"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase01/run_resid_base_eval_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE01_RESID_CURIOSITY_EVAL_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
CHECKPOINT=$CHECKPOINT
DOWNSTREAM_USE=heldout_curiosity_weighted_residual_evaluation_not_success_until_comparison
EOF
