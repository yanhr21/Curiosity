#!/usr/bin/env bash
# One corrected user-requested repeat; never dispatch formal training or physics.
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
EXPERIMENT="$ROOT/experiments/demo_following/paper_zero_wam_v1"
PZW_SITE_PACKAGES=/public/home/yanhongru/envs/gr00t_n16_py310/lib/python3.10/site-packages
IMPORT_BOOTSTRAP="$ROOT/scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py"
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_STEP_ID:-}"
case "$(hostname)" in login*|mgmtserver*) exit 2 ;; esac
cd "$ROOT"
export PYTHONPATH="$EXPERIMENT/repair_python_deps:$PZW_SITE_PACKAGES:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec 9>"$EXPERIMENT/single_gpu_pipeline.lock"
flock -n 9
PZW_RESUME_ARGS=()
if [[ "${1:-}" == "--resume-preupdate" && "$#" -eq 1 ]]; then
    PZW_RESUME_ARGS=(--resume-repaired-preupdate)
elif [[ "$#" -ne 0 ]]; then
    exit 2
fi
/usr/bin/python3.10 "$IMPORT_BOOTSTRAP" scripts.sugar.demo_following.paper_zero_wam.audit_repaired_conditioning
exec /usr/bin/python3.10 "$IMPORT_BOOTSTRAP" torch.distributed.run \
    --standalone --nproc_per_node=1 "$IMPORT_BOOTSTRAP" \
    scripts.sugar.demo_following.paper_zero_wam.train_single_gpu \
    --mode overfit --fixed-noise-overfit-diagnostic --repaired-overfit \
    --output-dir "$EXPERIMENT/overfit_repaired_fixed_noise_20260910" "${PZW_RESUME_ARGS[@]}"
