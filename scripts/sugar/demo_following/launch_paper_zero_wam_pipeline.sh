#!/usr/bin/env bash
# Retained single-GPU entrypoint. The old submission body below is inactive.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
# Latest user direction: never enqueue the old eight-GPU dependency chain.
# The target refuses login hosts/missing compute steps and holds one serial lock.
exec bash "$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_single_gpu_held.sh" "$@"

# Historical batch-submission implementation (unreachable).
PYTHON_BIN=/usr/bin/python3.10
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
RUNNER="$ROOT/scripts/sugar/demo_following/run_paper_zero_wam_stage.sh"
MOTION_CASES="$EXPERIMENT/formal/motion_disjoint_cases/MOTION_DISJOINT_CASES.json"
mkdir -p "$EXPERIMENT/logs" "$(dirname "$MOTION_CASES")"

# Resolve the complete physical test grid from the immutable split before any
# learned outcome exists.  Runtime jobs compare their prompt IDs and seeds to
# this semantic case file; this branch intentionally uses no digest gate.
PYTHONPATH="$ROOT" "$PYTHON_BIN" \
    -m scripts.sugar.demo_following.paper_zero_wam.materialize_motion_cases \
    --output "$MOTION_CASES"

precompute_job=$(sbatch --parsable \
    --no-requeue \
    --job-name=pzw_latents \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gres=gpu:NVIDIAH200:1 \
    --time=08:00:00 \
    --output="$EXPERIMENT/logs/precompute_%j.out" \
    --error="$EXPERIMENT/logs/precompute_%j.err" \
    "$RUNNER" precompute)

overfit_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$precompute_job" \
    --job-name=pzw_overfit \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 \
    --time=16:00:00 \
    --output="$EXPERIMENT/logs/overfit_%j.out" \
    --error="$EXPERIMENT/logs/overfit_%j.err" \
    "$RUNNER" overfit)

overfit_recovery_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterany:$overfit_job" \
    --job-name=pzw_overfit_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/overfit_recover_%j.out" \
    --error="$EXPERIMENT/logs/overfit_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_overfit.sh")

formal_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$overfit_job" \
    --job-name=pzw_formal \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 \
    --time=4-00:00:00 \
    --output="$EXPERIMENT/logs/formal_%j.out" \
    --error="$EXPERIMENT/logs/formal_%j.err" \
    "$RUNNER" formal)

formal_recovery_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterany:$formal_job" \
    --job-name=pzw_formal_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/formal_recover_%j.out" \
    --error="$EXPERIMENT/logs/formal_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_formal.sh")

heldout_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$formal_job" \
    --job-name=pzw_heldout \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 \
    --time=1-00:00:00 \
    --output="$EXPERIMENT/logs/heldout_%j.out" \
    --error="$EXPERIMENT/logs/heldout_%j.err" \
    "$RUNNER" evaluate_heldout)

heldout_recovery_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterany:$heldout_job" \
    --job-name=pzw_heldout_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/heldout_recover_%j.out" \
    --error="$EXPERIMENT/logs/heldout_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_heldout.sh")

physical_admission_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$heldout_job" \
    --job-name=pzw_physical_admit \
    --partition=cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=01:00:00 \
    --output="$EXPERIMENT/logs/physical_admit_%j.out" \
    --error="$EXPERIMENT/logs/physical_admit_%j.err" \
    "$ROOT/scripts/sugar/demo_following/advance_paper_zero_wam_after_heldout.sh")

render_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterok:$heldout_job" \
    --job-name=pzw_render \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --gres=gpu:NVIDIAH200:8 \
    --time=12:00:00 \
    --output="$EXPERIMENT/logs/render_%j.out" \
    --error="$EXPERIMENT/logs/render_%j.err" \
    "$RUNNER" render_openloop)

render_recovery_job=$(sbatch --parsable \
    --no-requeue \
    --dependency="afterany:$render_job" \
    --job-name=pzw_render_recover --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output="$EXPERIMENT/logs/render_recover_%j.out" \
    --error="$EXPERIMENT/logs/render_recover_%j.err" \
    "$ROOT/scripts/sugar/demo_following/recover_paper_zero_wam_after_render.sh")

printf '{"precompute_job":"%s","overfit_job":"%s","overfit_recovery_job":"%s","formal_job":"%s","formal_recovery_job":"%s","heldout_job":"%s","heldout_recovery_job":"%s","render_job":"%s","render_recovery_job":"%s","physical_admission_job":"%s"}\n' \
    "$precompute_job" "$overfit_job" "$overfit_recovery_job" \
    "$formal_job" "$formal_recovery_job" "$heldout_job" \
    "$heldout_recovery_job" "$render_job" "$render_recovery_job" \
    "$physical_admission_job" | tee "$EXPERIMENT/JOBS.json"
