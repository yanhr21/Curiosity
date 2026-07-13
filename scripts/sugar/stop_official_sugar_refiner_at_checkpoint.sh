#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full}"
TARGET_ITERATION="${TARGET_ITERATION:-10000}"
TARGET_CKPT="${TARGET_CKPT:-${OUTPUT_DIR}/logs/refiner/model_${TARGET_ITERATION}.pt}"
SLURM_JOB_ID="${SLURM_JOB_ID:-178129}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-2}"
STABLE_CHECKS_REQUIRED="${STABLE_CHECKS_REQUIRED:-3}"
WAIT_LOG="${WAIT_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260713_sugar_refiner_model_${TARGET_ITERATION}_stop_watch.log}"
TRAIN_PATTERN="${TRAIN_PATTERN:-^/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python scripts/sugar_rl/train.py --task Sugar-G129dof-CarryBox-Refiner }"

mkdir -p "$(dirname "${WAIT_LOG}")"

{
  echo "[SUGAR-CKPT-STOP] host=$(hostname)"
  echo "[SUGAR-CKPT-STOP] target_iteration=${TARGET_ITERATION}"
  echo "[SUGAR-CKPT-STOP] target_ckpt=${TARGET_CKPT}"
  echo "[SUGAR-CKPT-STOP] slurm_job_id=${SLURM_JOB_ID}"
  echo "[SUGAR-CKPT-STOP] wait_interval_seconds=${WAIT_INTERVAL_SECONDS}"
  echo "[SUGAR-CKPT-STOP] stable_checks_required=${STABLE_CHECKS_REQUIRED}"
  echo "[SUGAR-CKPT-STOP] started=$(date '+%F %T %Z')"
} | tee -a "${WAIT_LOG}"

last_report=0
while [[ ! -s "${TARGET_CKPT}" ]]; do
  if ! squeue -h -j "${SLURM_JOB_ID}" 2>/dev/null | grep -q .; then
    echo "[SUGAR-CKPT-STOP] tracked job is no longer active before checkpoint: ${SLURM_JOB_ID}" | tee -a "${WAIT_LOG}"
    exit 10
  fi
  if (( SECONDS - last_report >= 30 )); then
    echo "[SUGAR-CKPT-STOP] waiting at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"
    last_report=${SECONDS}
  fi
  sleep "${WAIT_INTERVAL_SECONDS}"
done

stable_count=0
previous_size=-1
while (( stable_count < STABLE_CHECKS_REQUIRED )); do
  current_size="$(stat -c '%s' "${TARGET_CKPT}")"
  if [[ "${current_size}" -gt 0 && "${current_size}" -eq "${previous_size}" ]]; then
    stable_count=$((stable_count + 1))
  else
    stable_count=0
  fi
  previous_size="${current_size}"
  sleep 1
done

echo "[SUGAR-CKPT-STOP] checkpoint stable size=${previous_size} at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"

srun \
  --jobid="${SLURM_JOB_ID}" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --export="ALL,SUGAR_STOP_PATTERN=${TRAIN_PATTERN}" \
  bash -lc '
    pids="$(pgrep -u "$USER" -f "$SUGAR_STOP_PATTERN" || true)"
    if [[ -z "$pids" ]]; then
      echo "[SUGAR-CKPT-STOP] no matching refiner process found"
      exit 11
    fi
    echo "[SUGAR-CKPT-STOP] terminating refiner pids=$pids"
    kill -TERM $pids
    for _ in $(seq 1 10); do
      alive=""
      for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
          alive="$alive $pid"
        fi
      done
      if [[ -z "$alive" ]]; then
        echo "[SUGAR-CKPT-STOP] refiner exited after TERM"
        exit 0
      fi
      sleep 1
    done
    echo "[SUGAR-CKPT-STOP] forcing remaining refiner pids=$alive"
    kill -KILL $alive
  ' 2>&1 | tee -a "${WAIT_LOG}"

sleep 3
if ! squeue -h -j "${SLURM_JOB_ID}" 2>/dev/null | grep -q .; then
  echo "[SUGAR-CKPT-STOP] warning: allocation ${SLURM_JOB_ID} is no longer active" | tee -a "${WAIT_LOG}"
  exit 12
fi

echo "[SUGAR-CKPT-STOP] training child stopped; allocation ${SLURM_JOB_ID} remains active at $(date '+%F %T %Z')" | tee -a "${WAIT_LOG}"
