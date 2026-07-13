#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
REFINER_CKPT="${REFINER_CKPT:-${OUTPUT_DIR}/ckpts/refiner.pt}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-20260712_sugar_carrybox_remaining_pipeline_after_refiner}"
WAIT_LOG="${WAIT_LOG:-${LOG_DIR}/${STAMP}_wait.log}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${ROOT_DIR}/scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-432000}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
SLURM_GPUS="${SLURM_GPUS:-gpu:1}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-sugar_remaining}"
SLURM_TIME="${SLURM_TIME:-5-00:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
SLURM_MEM="${SLURM_MEM:-160G}"

mkdir -p "${LOG_DIR}"

{
  echo "[SUGAR-REMAINING-WAIT] host=$(hostname)"
  echo "[SUGAR-REMAINING-WAIT] root=${ROOT_DIR}"
  echo "[SUGAR-REMAINING-WAIT] sugar_dir=${SUGAR_DIR}"
  echo "[SUGAR-REMAINING-WAIT] task=${TASK_NAME}"
  echo "[SUGAR-REMAINING-WAIT] exp_name=${EXP_NAME}"
  echo "[SUGAR-REMAINING-WAIT] output_dir=${OUTPUT_DIR}"
  echo "[SUGAR-REMAINING-WAIT] refiner_ckpt=${REFINER_CKPT}"
  echo "[SUGAR-REMAINING-WAIT] python_bin=${PYTHON_BIN}"
  echo "[SUGAR-REMAINING-WAIT] pipeline_script=${PIPELINE_SCRIPT}"
  echo "[SUGAR-REMAINING-WAIT] slurm_partition=${SLURM_PARTITION}"
  echo "[SUGAR-REMAINING-WAIT] slurm_gpus=${SLURM_GPUS}"
  echo "[SUGAR-REMAINING-WAIT] slurm_job_name=${SLURM_JOB_NAME}"
  echo "[SUGAR-REMAINING-WAIT] slurm_time=${SLURM_TIME}"
  echo "[SUGAR-REMAINING-WAIT] slurm_cpus_per_task=${SLURM_CPUS_PER_TASK}"
  echo "[SUGAR-REMAINING-WAIT] slurm_mem=${SLURM_MEM}"
  echo "[SUGAR-REMAINING-WAIT] wait_interval_seconds=${WAIT_INTERVAL_SECONDS}"
  echo "[SUGAR-REMAINING-WAIT] wait_timeout_seconds=${WAIT_TIMEOUT_SECONDS}"
  echo "[SUGAR-REMAINING-WAIT] started=$(date '+%F %T')"
} | tee -a "${WAIT_LOG}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while [[ ! -s "${REFINER_CKPT}" ]]; do
  if (( SECONDS >= deadline )); then
    echo "[SUGAR-REMAINING-WAIT] timed out waiting for ${REFINER_CKPT} at $(date '+%F %T')" | tee -a "${WAIT_LOG}"
    exit 10
  fi
  echo "[SUGAR-REMAINING-WAIT] waiting for refiner checkpoint at $(date '+%F %T')" | tee -a "${WAIT_LOG}"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

echo "[SUGAR-REMAINING-WAIT] found refiner checkpoint at $(date '+%F %T')" | tee -a "${WAIT_LOG}"

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "[SUGAR-REMAINING-WAIT] missing root directory before launch: ${ROOT_DIR}" | tee -a "${WAIT_LOG}"
  exit 11
fi
if [[ ! -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  echo "[SUGAR-REMAINING-WAIT] missing official SUGAR clone before launch: ${SUGAR_DIR}" | tee -a "${WAIT_LOG}"
  exit 12
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[SUGAR-REMAINING-WAIT] missing executable python before launch: ${PYTHON_BIN}" | tee -a "${WAIT_LOG}"
  exit 13
fi
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "[SUGAR-REMAINING-WAIT] missing pipeline script before launch: ${PIPELINE_SCRIPT}" | tee -a "${WAIT_LOG}"
  exit 14
fi
if [[ ! -s "${REFINER_CKPT}" ]]; then
  echo "[SUGAR-REMAINING-WAIT] refiner checkpoint vanished before launch: ${REFINER_CKPT}" | tee -a "${WAIT_LOG}"
  exit 15
fi

echo "[SUGAR-REMAINING-WAIT] launching remaining official pipeline with srun" | tee -a "${WAIT_LOG}"

cd "${ROOT_DIR}"
exec srun \
  -p "${SLURM_PARTITION}" \
  --gres="${SLURM_GPUS}" \
  --job-name="${SLURM_JOB_NAME}" \
  --time="${SLURM_TIME}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem="${SLURM_MEM}" \
  --pty bash -lc \
  "cd '${ROOT_DIR}' && SUGAR_DIR='${SUGAR_DIR}' OUTPUT_DIR='${OUTPUT_DIR}' PYTHON_BIN='${PYTHON_BIN}' TASK_NAME='${TASK_NAME}' EXP_NAME='${EXP_NAME}' STAMP='${STAMP}' START_STAGE=refiner_rollout bash '${PIPELINE_SCRIPT}'"
