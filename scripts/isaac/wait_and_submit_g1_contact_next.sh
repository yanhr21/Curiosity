#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" != mgmtserver* ]]; then
  echo "This watcher is intended for the login node because it only waits and submits srun jobs: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
WAIT_JOB_ID="${WAIT_JOB_ID:-}"
WAIT_PATH="${WAIT_PATH:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PARTITION="${PARTITION:-cpu}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
JOB_NAME="${JOB_NAME:-g1_contact_next}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
GRES="${GRES:-gpu:1}"
FOLLOWUP_PREFIX="${FOLLOWUP_PREFIX:-20260706_after_quick_render_contact_next}"
CASE_SET="${CASE_SET:-contact_next}"

cd "${ROOT_DIR}"

if [[ -n "${WAIT_JOB_ID}" ]]; then
  echo "[CONTACT-NEXT-WAITER] waiting for job ${WAIT_JOB_ID} to leave queue"
  while squeue -j "${WAIT_JOB_ID}" -h >/tmp/contact_next_wait_squeue.$$ && [[ -s /tmp/contact_next_wait_squeue.$$ ]]; do
    rm -f /tmp/contact_next_wait_squeue.$$
    echo "[CONTACT-NEXT-WAITER] $(date '+%F %T') job ${WAIT_JOB_ID} still queued/running"
    sleep "${POLL_SECONDS}"
  done
  rm -f /tmp/contact_next_wait_squeue.$$
fi

if [[ -n "${WAIT_PATH}" ]]; then
  echo "[CONTACT-NEXT-WAITER] waiting for path ${WAIT_PATH}"
  while [[ ! -e "${WAIT_PATH}" ]]; do
    echo "[CONTACT-NEXT-WAITER] $(date '+%F %T') path still missing"
    sleep "${POLL_SECONDS}"
  done
fi

echo "[CONTACT-NEXT-WAITER] submitting CASE_SET=${CASE_SET} FOLLOWUP_PREFIX=${FOLLOWUP_PREFIX}"
srun \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --gres="${GRES}" \
  --time="${TIME_LIMIT}" \
  --job-name="${JOB_NAME}" \
  --export=ALL,FOLLOWUP_PREFIX="${FOLLOWUP_PREFIX}",CASE_SET="${CASE_SET}" \
  bash scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh
