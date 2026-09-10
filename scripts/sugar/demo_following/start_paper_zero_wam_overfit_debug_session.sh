#!/usr/bin/env bash
# Keep the replacement allocation after this bounded diagnostic terminates.
set -uo pipefail
ROOT=/public/home/yanhongru/Curiosity
test -n "${SLURM_JOB_ID:-}" || exit 2
test -n "${SLURM_STEP_ID:-}" || exit 2
cd "$ROOT" || exit 2
PZW_DEBUG_PREFIX="$ROOT/experiments/demo_following/paper_zero_wam_v1/logs/held_${SLURM_JOB_ID}_overfit_debug"
bash scripts/sugar/native_tactile/launch_retained_child.sh \
    --foreground --record "$PZW_DEBUG_PREFIX.record" \
    --status "$PZW_DEBUG_PREFIX.status" --log "$PZW_DEBUG_PREFIX.log" \
    --tag paper_zero_wam_user_fixed_noise_overfit32 \
    -- bash scripts/sugar/demo_following/run_paper_zero_wam_overfit_debug_held.sh
PZW_DEBUG_EXIT=$?
printf 'Overfit diagnostic child exited with status %s; retaining compute shell.\n' "$PZW_DEBUG_EXIT"
exec bash --noprofile --norc -i
