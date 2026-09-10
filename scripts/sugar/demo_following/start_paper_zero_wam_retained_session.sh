#!/usr/bin/env bash
# Run the existing pipeline once, retaining the compute shell after child exit.
set -uo pipefail
ROOT=/public/home/yanhongru/Curiosity
test -n "${SLURM_JOB_ID:-}" || exit 2
test -n "${SLURM_STEP_ID:-}" || exit 2
cd "$ROOT" || exit 2
PZW_RUN_PREFIX="$ROOT/experiments/demo_following/paper_zero_wam_v1/logs/held_${SLURM_JOB_ID}_resume_latest"
bash scripts/sugar/native_tactile/launch_retained_child.sh \
    --foreground --record "$PZW_RUN_PREFIX.record" \
    --status "$PZW_RUN_PREFIX.status" --log "$PZW_RUN_PREFIX.log" \
    --tag paper_zero_wam_formal_resume_latest \
    -- bash scripts/sugar/demo_following/run_paper_zero_wam_single_gpu_held.sh
PZW_CHILD_EXIT=$?
printf 'Pipeline child exited with status %s; retaining compute shell.\n' "$PZW_CHILD_EXIT"
exec bash --noprofile --norc -i
