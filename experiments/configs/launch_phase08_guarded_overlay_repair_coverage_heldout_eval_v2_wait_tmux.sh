#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
SESSION="${SESSION:-curiosity_phase08_contact_patch_alloc_20260628}"
WINDOW="${WINDOW:-phase08_overlay_eval_v2_wait}"
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-156696}"
SUMMARY_PATH="${SUMMARY_PATH:-$ROOT/experiments/outputs/phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_trainer_v2_20260629/phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_v2_train_20260629_summary.json}"
CURIOSITY_CHECKPOINT="${CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_trainer_v2_20260629/phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_v2_train_20260629.pt}"
RUN_TAG="${RUN_TAG:-phase08_guarded_overlay_repair_coverage_curiosity_heldout_eval_v2_20260629}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_guarded_overlay_repair_coverage_curiosity_v2_eval}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260629}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-29_phase08_guarded_overlay_repair_coverage_curiosity_heldout_eval_v2.md}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase08_guarded_overlay_repair_coverage_curiosity_heldout_eval_v2_20260629.srun.log}"
DEVICE="${DEVICE:-cuda:0}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.5}"
CURIOSITY_CONTROLLER_MODE="${CURIOSITY_CONTROLLER_MODE:-lift_hold_feedback_residual_overlay}"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: missing tmux session: $SESSION" >&2
  exit 2
fi

if tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$WINDOW"; then
  echo "ERROR: tmux window already exists: $SESSION:$WINDOW" >&2
  exit 3
fi

mkdir -p "$ROOT/logs/newton" "$ROOT/experiments/reports"

remote_cmd=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$ROOT")
echo "PHASE08_GUARDED_OVERLAY_REPAIR_COVERAGE_HELDOUT_EVAL_V2_WAITER_START"
echo "SUMMARY_PATH=$(printf '%q' "$SUMMARY_PATH")"
echo "CURIOSITY_CHECKPOINT=$(printf '%q' "$CURIOSITY_CHECKPOINT")"
echo "SLURM_JOB_ID_TARGET=$(printf '%q' "$SLURM_JOB_ID_TARGET")"
while true; do
  if [[ -f $(printf '%q' "$SUMMARY_PATH") ]] && jq -e '.status == "pass" and .real_training_result == true' $(printf '%q' "$SUMMARY_PATH") >/dev/null; then
    break
  fi
  if ! squeue -j $(printf '%q' "$SLURM_JOB_ID_TARGET") -h >/dev/null; then
    echo "ERROR: allocation disappeared before V2 policy training passed"
    exit 4
  fi
  date
  echo "waiting_for_phase08_v2_policy_training_summary"
  sleep 60
done
if [[ ! -f $(printf '%q' "$CURIOSITY_CHECKPOINT") ]]; then
  echo "ERROR: policy summary passed but checkpoint is missing: $(printf '%q' "$CURIOSITY_CHECKPOINT")"
  exit 5
fi
echo "PHASE08_GUARDED_OVERLAY_REPAIR_COVERAGE_HELDOUT_EVAL_V2_START"
srun --jobid=$(printf '%q' "$SLURM_JOB_ID_TARGET") --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc 'cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") EVAL_TAG_PREFIX=$(printf '%q' "$EVAL_TAG_PREFIX") EVAL_TAG_SUFFIX=$(printf '%q' "$EVAL_TAG_SUFFIX") REPORT_PATH=$(printf '%q' "$REPORT_PATH") CURIOSITY_CHECKPOINT=$(printf '%q' "$CURIOSITY_CHECKPOINT") ACTIVE_THRESHOLD=$(printf '%q' "$ACTIVE_THRESHOLD") CURIOSITY_CONTROLLER_MODE=$(printf '%q' "$CURIOSITY_CONTROLLER_MODE") DEVICE=$(printf '%q' "$DEVICE") bash experiments/configs/run_phase08_curiosity_weighted_heldout_eval_in_alloc.sh' 2>&1 | tee $(printf '%q' "$LOG_PATH")
echo "PHASE08_GUARDED_OVERLAY_REPAIR_COVERAGE_HELDOUT_EVAL_V2_END"
EOF
)

tmux new-window -t "$SESSION" -n "$WINDOW" "bash -lc $(printf '%q' "$remote_cmd")"
echo "Launched $SESSION:$WINDOW"
echo "Log: $LOG_PATH"
