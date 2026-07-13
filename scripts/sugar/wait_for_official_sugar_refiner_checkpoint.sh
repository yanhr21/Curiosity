#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
TARGET_ITERATION="${TARGET_ITERATION:-3000}"
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-refiner}"
TARGET_CKPT="${TARGET_CKPT:-${OUTPUT_DIR}/logs/${CHECKPOINT_STAGE}/model_${TARGET_ITERATION}.pt}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-20260712_sugar_refiner_model_${TARGET_ITERATION}_watch}"
WAIT_LOG="${WAIT_LOG:-${LOG_DIR}/${STAMP}.log}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-432000}"
SLURM_JOB_ID="${SLURM_JOB_ID:-177561}"
AUTO_CHAIN_NEXT="${AUTO_CHAIN_NEXT:-0}"
CHAIN_STEP="${CHAIN_STEP:-1000}"
CHAIN_UNTIL_ITERATION="${CHAIN_UNTIL_ITERATION:-30000}"
CHAIN_TMUX_PREFIX="${CHAIN_TMUX_PREFIX:-curiosity_sugar_model}"
CHAIN_SESSION_SUFFIX="${CHAIN_SESSION_SUFFIX:-0712}"
CHAIN_STAMP_PREFIX="${CHAIN_STAMP_PREFIX:-20260712}"

mkdir -p "${LOG_DIR}"

{
  echo "[SUGAR-CKPT-WAIT] host=$(hostname)"
  echo "[SUGAR-CKPT-WAIT] root=${ROOT_DIR}"
  echo "[SUGAR-CKPT-WAIT] sugar_dir=${SUGAR_DIR}"
  echo "[SUGAR-CKPT-WAIT] task=${TASK_NAME}"
  echo "[SUGAR-CKPT-WAIT] exp_name=${EXP_NAME}"
  echo "[SUGAR-CKPT-WAIT] output_dir=${OUTPUT_DIR}"
  echo "[SUGAR-CKPT-WAIT] target_iteration=${TARGET_ITERATION}"
  echo "[SUGAR-CKPT-WAIT] checkpoint_stage=${CHECKPOINT_STAGE}"
  echo "[SUGAR-CKPT-WAIT] target_ckpt=${TARGET_CKPT}"
  echo "[SUGAR-CKPT-WAIT] slurm_job_id=${SLURM_JOB_ID}"
  echo "[SUGAR-CKPT-WAIT] wait_interval_seconds=${WAIT_INTERVAL_SECONDS}"
  echo "[SUGAR-CKPT-WAIT] wait_timeout_seconds=${WAIT_TIMEOUT_SECONDS}"
  echo "[SUGAR-CKPT-WAIT] auto_chain_next=${AUTO_CHAIN_NEXT}"
  echo "[SUGAR-CKPT-WAIT] chain_step=${CHAIN_STEP}"
  echo "[SUGAR-CKPT-WAIT] chain_until_iteration=${CHAIN_UNTIL_ITERATION}"
  echo "[SUGAR-CKPT-WAIT] chain_session_suffix=${CHAIN_SESSION_SUFFIX}"
  echo "[SUGAR-CKPT-WAIT] chain_stamp_prefix=${CHAIN_STAMP_PREFIX}"
  echo "[SUGAR-CKPT-WAIT] started=$(date '+%F %T %Z')"
} | tee -a "${WAIT_LOG}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while [[ ! -s "${TARGET_CKPT}" ]]; do
  if (( SECONDS >= deadline )); then
    echo "[SUGAR-CKPT-WAIT] timed out waiting for ${TARGET_CKPT} at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"
    exit 10
  fi
  if command -v squeue >/dev/null 2>&1 && ! squeue -h -j "${SLURM_JOB_ID}" | grep -q .; then
    echo "[SUGAR-CKPT-WAIT] tracked Slurm job ${SLURM_JOB_ID} is no longer active at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"
    exit 11
  fi
  echo "[SUGAR-CKPT-WAIT] waiting for ${TARGET_CKPT} at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

ls -lh "${TARGET_CKPT}" | sed 's/^/[SUGAR-CKPT-WAIT] found=/' | tee -a "${WAIT_LOG}"
echo "[SUGAR-CKPT-WAIT] found target checkpoint at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"

if [[ "${AUTO_CHAIN_NEXT}" == "1" ]]; then
  next_iteration=$((TARGET_ITERATION + CHAIN_STEP))
  if (( next_iteration <= CHAIN_UNTIL_ITERATION )); then
    next_session="${CHAIN_TMUX_PREFIX}${next_iteration}_watch_${CHAIN_SESSION_SUFFIX}"
    next_stamp="${CHAIN_STAMP_PREFIX}_sugar_${CHECKPOINT_STAGE}_model_${next_iteration}_watch"
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[SUGAR-CKPT-WAIT] auto-chain skipped: tmux not available" | tee -a "${WAIT_LOG}"
      exit 0
    fi
    if tmux has-session -t "${next_session}" 2>/dev/null; then
      echo "[SUGAR-CKPT-WAIT] auto-chain skipped: ${next_session} already present" | tee -a "${WAIT_LOG}"
      exit 0
    fi
    echo "[SUGAR-CKPT-WAIT] auto-chain starting ${next_session} for model_${next_iteration}.pt at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"
    tmux new-session -d -s "${next_session}" \
      "cd '${ROOT_DIR}' && CHECKPOINT_STAGE='${CHECKPOINT_STAGE}' TARGET_ITERATION='${next_iteration}' STAMP='${next_stamp}' WAIT_INTERVAL_SECONDS='${WAIT_INTERVAL_SECONDS}' WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}' SLURM_JOB_ID='${SLURM_JOB_ID}' AUTO_CHAIN_NEXT='1' CHAIN_STEP='${CHAIN_STEP}' CHAIN_UNTIL_ITERATION='${CHAIN_UNTIL_ITERATION}' CHAIN_TMUX_PREFIX='${CHAIN_TMUX_PREFIX}' CHAIN_SESSION_SUFFIX='${CHAIN_SESSION_SUFFIX}' CHAIN_STAMP_PREFIX='${CHAIN_STAMP_PREFIX}' bash scripts/sugar/wait_for_official_sugar_refiner_checkpoint.sh"
  else
    echo "[SUGAR-CKPT-WAIT] auto-chain complete: next_iteration=${next_iteration} exceeds chain_until_iteration=${CHAIN_UNTIL_ITERATION}" | tee -a "${WAIT_LOG}"
  fi
fi
