#!/usr/bin/env bash
# User-requested interim evaluation only; no training or physics continuation.
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
PYTHON_BIN=/usr/bin/python3.10
PZW_SITE_PACKAGES=/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages
IMPORT_BOOTSTRAP="$ROOT/scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py"
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_STEP_ID:-}"
case "$(hostname)" in login*|mgmtserver*) exit 2 ;; esac
cd "$ROOT"
export PYTHONPATH="$PZW_SITE_PACKAGES:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec 9>"$EXPERIMENT/single_gpu_pipeline.lock"
flock -n 9
jq -e '.training_stopped == true and .completed_optimizer_steps == 700' \
    "$EXPERIMENT/formal/STEP700_STOP_RESULT.json" > /dev/null
"$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
    --standalone --nproc_per_node=1 "$IMPORT_BOOTSTRAP" \
    scripts.sugar.demo_following.paper_zero_wam.evaluate_heldout \
    --single-gpu --interim-step 700 --checkpoint "$EXPERIMENT/formal/latest_checkpoint" \
    --output-dir "$EXPERIMENT/formal/heldout_prompt_gate_step700"
"$PYTHON_BIN" "$IMPORT_BOOTSTRAP" torch.distributed.run \
    --standalone --nproc_per_node=1 "$IMPORT_BOOTSTRAP" \
    scripts.sugar.demo_following.paper_zero_wam.render_openloop \
    --single-gpu --interim-step 700 --checkpoint "$EXPERIMENT/formal/latest_checkpoint" \
    --output-dir "$EXPERIMENT/formal/heldout_openloop_videos_step700"
