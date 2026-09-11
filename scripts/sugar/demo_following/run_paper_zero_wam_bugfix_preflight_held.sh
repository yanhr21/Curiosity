#!/usr/bin/env bash
# Current-code audit on the retained H200; zero training updates.
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_STEP_ID:-}"
case "$(hostname)" in login*|mgmtserver*) exit 2 ;; esac
cd "$ROOT"
export PYTHONPATH="$EXPERIMENT/repair_python_deps:/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages:$ROOT"
export OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec 9>"$EXPERIMENT/single_gpu_pipeline.lock"
flock -n 9
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/usr/bin/python3.10 -m unittest \
    scripts.sugar.demo_following.paper_zero_wam.test_remote_repairs \
    scripts.sugar.demo_following.paper_zero_wam.test_repaired_conditioning \
    scripts.sugar.demo_following.paper_zero_wam.test_single_gpu_optimizer \
    scripts.sugar.demo_following.paper_zero_wam.test_overfit_diagnostic \
    scripts.sugar.demo_following.paper_zero_wam.test_results_location -v
exec /usr/bin/python3.10 \
    scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py \
    scripts.sugar.demo_following.paper_zero_wam.audit_repaired_conditioning \
    --output "$EXPERIMENT/bugfix_audit_20260911/OFFICIAL_FORWARD_AUDIT.json"
