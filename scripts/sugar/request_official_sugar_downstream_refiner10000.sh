#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
DOWNSTREAM_SCRIPT="${DOWNSTREAM_SCRIPT:-${ROOT_DIR}/scripts/sugar/run_official_sugar_downstream_from_refiner10000.sh}"
SLURM_PARTITION="${SLURM_PARTITION:-cpu}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-sugar_down10k}"
SLURM_TIME="${SLURM_TIME:-1-00:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_GPUS="${SLURM_GPUS:-gpu:1}"
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-server36,server53}"

if [[ ! -x "${DOWNSTREAM_SCRIPT}" ]]; then
  echo "Missing executable downstream script: ${DOWNSTREAM_SCRIPT}" >&2
  exit 2
fi

export ROOT_DIR DOWNSTREAM_SCRIPT
export START_STAGE="${START_STAGE:-refiner_rollout}"
export STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
export TRACKER_MAX_ITERATIONS="${TRACKER_MAX_ITERATIONS:-10001}"
export TRACKER_FINAL_ITERATION="${TRACKER_FINAL_ITERATION:-10000}"
export GENERATOR_NUM_EPOCHS="${GENERATOR_NUM_EPOCHS:-1001}"
export STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_downstream_refiner10000_${SLURM_PARTITION}}"

srun_args=(
  -p "${SLURM_PARTITION}"
  --gres="${SLURM_GPUS}"
  --job-name="${SLURM_JOB_NAME}"
  --time="${SLURM_TIME}"
  --cpus-per-task="${SLURM_CPUS_PER_TASK}"
  --mem="${SLURM_MEM}"
)
if [[ -n "${SLURM_EXCLUDE_NODES}" ]]; then
  srun_args+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

echo "[SUGAR-DOWNSTREAM-REQUEST] partition=${SLURM_PARTITION}"
echo "[SUGAR-DOWNSTREAM-REQUEST] time=${SLURM_TIME}"
echo "[SUGAR-DOWNSTREAM-REQUEST] cpus=${SLURM_CPUS_PER_TASK}"
echo "[SUGAR-DOWNSTREAM-REQUEST] mem=${SLURM_MEM}"
echo "[SUGAR-DOWNSTREAM-REQUEST] start_stage=${START_STAGE}"
echo "[SUGAR-DOWNSTREAM-REQUEST] tracker_final_iteration=${TRACKER_FINAL_ITERATION}"
echo "[SUGAR-DOWNSTREAM-REQUEST] generator_num_epochs=${GENERATOR_NUM_EPOCHS}"

exec srun "${srun_args[@]}" --pty bash -lc \
  "cd '${ROOT_DIR}'; set +e; bash '${DOWNSTREAM_SCRIPT}'; status=\$?; echo '[SUGAR-DOWNSTREAM-PERSIST] pipeline_status='\${status}; echo '[SUGAR-DOWNSTREAM-PERSIST] keeping allocation shell'; exec bash"
