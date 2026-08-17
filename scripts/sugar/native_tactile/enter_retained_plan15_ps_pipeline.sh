#!/usr/bin/env bash
# Entered through srun on the allocated GPU node; launch the serial pipeline and retain the shell.

set -euo pipefail

root=/public/home/yanhongru/Curiosity
runtime="$root/experiments/online_patch_tactile_mass_adaptation/runtime"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is absent; enter through srun inside an allocation" >&2
    exit 2
fi
if [[ "${PLAN15_COMPUTE_LOGIN_SHELL:-0}" != 1 ]]; then
    export PLAN15_COMPUTE_LOGIN_SHELL=1
    exec bash -lc "cd $root && exec bash $root/scripts/sugar/native_tactile/enter_retained_plan15_ps_pipeline.sh"
fi

stem="plan15_ps_remaining_pipeline_job${SLURM_JOB_ID}"
cd "$root"
bash scripts/sugar/native_tactile/launch_retained_child.sh \
    --record "$runtime/$stem.process.txt" \
    --status "$runtime/$stem.status.txt" \
    --log "$runtime/$stem.log" \
    --tag "$stem" \
    -- bash scripts/sugar/native_tactile/run_plan15_ps_remaining_pipeline.sh cuda:0

exec bash -l
