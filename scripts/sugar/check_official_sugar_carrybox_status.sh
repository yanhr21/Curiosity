#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
REFINER_LOG="${REFINER_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_full_official.log}"
WAIT_LOG="${WAIT_LOG:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260712_sugar_carrybox_remaining_pipeline_after_refiner_wait.log}"
REFINER_TMUX="${REFINER_TMUX:-curiosity_sugar_refiner_full_0712:0}"
WATCHER_TMUX="${WATCHER_TMUX:-curiosity_sugar_remaining_after_refiner_0712:0}"
NEXT_CKPT_ITERATION="${NEXT_CKPT_ITERATION:-}"
NEXT_CKPT_WATCHER_TMUX="${NEXT_CKPT_WATCHER_TMUX:-}"
SLURM_JOB_ID="${SLURM_JOB_ID:-177561}"
REFINER_STOP_ITERATION="${REFINER_STOP_ITERATION:-5000}"
SUGAR_STATUS_COMPACT="${SUGAR_STATUS_COMPACT:-0}"
NEXT_CKPT_WAIT_LOG="${NEXT_CKPT_WAIT_LOG:-}"

strip_ansi() {
  sed -E 's/\x1b\[[0-9;]*[[:alpha:]]//g'
}

hms_to_seconds() {
  local hms="$1"
  local hours minutes seconds
  IFS=: read -r hours minutes seconds <<<"${hms}"
  echo $((10#${hours} * 3600 + 10#${minutes} * 60 + 10#${seconds}))
}

seconds_to_hms() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf '%02d:%02d:%02d\n' "${hours}" "${minutes}" "${seconds}"
}

echo "[SUGAR-STATUS] checked=$(date '+%F %T %Z')"
echo "[SUGAR-STATUS] root=${ROOT_DIR}"
echo "[SUGAR-STATUS] sugar_dir=${SUGAR_DIR}"
echo "[SUGAR-STATUS] task=${TASK_NAME}"
echo "[SUGAR-STATUS] exp_name=${EXP_NAME}"
echo "[SUGAR-STATUS] output_dir=${OUTPUT_DIR}"

tracked_refiner_rows=""
refiner_job_active=0
refiner_stopped_at_target=0
if command -v squeue >/dev/null 2>&1; then
  echo "[SUGAR-STATUS] slurm_tracked_refiner:"
  tracked_refiner_rows="$(
    squeue -h -j "${SLURM_JOB_ID}" -o '%.18i %.12P %.28j %.8T %.12M %.10l %.20R' 2>/dev/null \
      | sort -u || true
  )"
  if [[ -n "${tracked_refiner_rows}" ]]; then
    refiner_job_active=1
    echo "${tracked_refiner_rows}" | sed 's/^/[SUGAR-STATUS] slurm_tracked_refiner_row=/'
  else
    echo "[SUGAR-STATUS] slurm_tracked_refiner_row=inactive:${SLURM_JOB_ID}"
  fi
  echo "[SUGAR-STATUS] slurm_user_sugar_jobs:"
  squeue -h -u "${USER}" -o '%.18i %.12P %.28j %.8T %.12M %.10l %.20R' \
    | grep -E 'sugar' \
    | sort -u \
    | sed 's/^/[SUGAR-STATUS] slurm_user_sugar_job_row=/' || true
fi

echo "[SUGAR-STATUS] checkpoints:"
find "${OUTPUT_DIR}" -maxdepth 4 -type f \
  \( -name 'model_*.pt' -o -name 'refiner.pt' -o -name 'tracker.pt' -o -name 'generator.ckpt' -o -name '*summary*.json' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -80 || true

if [[ -s "${OUTPUT_DIR}/ckpts/refiner.pt" ]]; then
  echo "[SUGAR-STATUS] refiner_ckpt=present"
else
  echo "[SUGAR-STATUS] refiner_ckpt=missing"
fi

latest_refiner_model="$(
  find "${OUTPUT_DIR}/logs/refiner" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
    | sed -nE 's/^model_([0-9]+)\.pt$/\1/p' \
    | sort -n \
    | tail -1
)"
if [[ -n "${latest_refiner_model}" ]]; then
  echo "[SUGAR-STATUS] latest_refiner_periodic_checkpoint=model_${latest_refiner_model}.pt"
  if [[ "${latest_refiner_model}" -ge "${REFINER_STOP_ITERATION}" && "${refiner_job_active}" == "0" ]]; then
    refiner_stopped_at_target=1
  fi
fi

if [[ -s "${REFINER_LOG}" ]]; then
  if [[ "${SUGAR_STATUS_COMPACT}" != "1" ]]; then
    echo "[SUGAR-STATUS] refiner_log_tail:"
    tail -220 "${REFINER_LOG}" \
      | strip_ansi \
      | grep -E 'Learning iteration|Total timesteps|Time elapsed|ETA|Mean reward|Computation' \
      | tail -30 || true
  fi
  latest_iteration="$(
    tail -500 "${REFINER_LOG}" \
      | strip_ansi \
      | sed -nE 's/.*Learning iteration[[:space:]]+([0-9]+)\/([0-9]+).*/\1 \2/p' \
      | tail -1
  )"
  latest_elapsed="$(
    tail -500 "${REFINER_LOG}" \
      | strip_ansi \
      | sed -nE 's/.*Time elapsed:[[:space:]]+([0-9:]+).*/\1/p' \
      | tail -1
  )"
  latest_eta="$(
    tail -500 "${REFINER_LOG}" \
      | strip_ansi \
      | sed -nE 's/.*ETA:[[:space:]]+([0-9:]+).*/\1/p' \
      | tail -1
  )"
  if [[ -n "${latest_iteration}" ]]; then
    read -r current_iteration total_iterations <<<"${latest_iteration}"
    if [[ "${refiner_stopped_at_target}" == "1" ]]; then
      echo "[SUGAR-STATUS] refiner_progress=current_iteration=${current_iteration}/${total_iterations} stopped_at=model_${latest_refiner_model}.pt stop_iteration=${REFINER_STOP_ITERATION}"
      NEXT_CKPT_ITERATION=""
    else
      next_checkpoint=$(( ((current_iteration / 1000) + 1) * 1000 ))
      if (( next_checkpoint >= total_iterations )); then
        next_checkpoint=$((total_iterations - 1))
      fi
      if [[ -z "${NEXT_CKPT_ITERATION}" ]]; then
        NEXT_CKPT_ITERATION="${next_checkpoint}"
      fi
      remaining_to_next=$((next_checkpoint - current_iteration))
      echo "[SUGAR-STATUS] refiner_progress=current_iteration=${current_iteration}/${total_iterations} next_checkpoint=model_${next_checkpoint}.pt remaining_iterations=${remaining_to_next}"
      if [[ -n "${latest_elapsed}" && "${current_iteration}" -gt 0 ]]; then
        elapsed_seconds="$(hms_to_seconds "${latest_elapsed}")"
        avg_seconds_per_iteration=$((elapsed_seconds / current_iteration))
        if (( avg_seconds_per_iteration < 1 )); then
          avg_seconds_per_iteration=1
        fi
        estimated_next_checkpoint_seconds=$((remaining_to_next * avg_seconds_per_iteration))
        echo "[SUGAR-STATUS] estimated_time_to_next_checkpoint=$(seconds_to_hms "${estimated_next_checkpoint_seconds}") avg_seconds_per_iteration=${avg_seconds_per_iteration}"
      fi
    fi
  fi
  if [[ -n "${latest_elapsed}" ]]; then
    echo "[SUGAR-STATUS] latest_refiner_elapsed=${latest_elapsed}"
  fi
  if [[ -n "${latest_eta}" ]]; then
    echo "[SUGAR-STATUS] latest_refiner_eta=${latest_eta}"
  fi
  echo "[SUGAR-STATUS] refiner_fatal_patterns:"
  grep -nE 'Traceback \(most recent call last\)|FileNotFoundError|Boost\.Python\.ArgumentError|RuntimeError|CUDA out of memory|\[Error\]|status=' "${REFINER_LOG}" | tail -80 || true
else
  echo "[SUGAR-STATUS] refiner_log=missing:${REFINER_LOG}"
fi

if [[ -z "${NEXT_CKPT_ITERATION}" && -n "${latest_refiner_model}" && "${refiner_stopped_at_target}" != "1" ]]; then
  NEXT_CKPT_ITERATION=$((latest_refiner_model + 1000))
fi
if [[ -z "${NEXT_CKPT_ITERATION}" && "${refiner_stopped_at_target}" != "1" ]]; then
  NEXT_CKPT_ITERATION="3000"
fi
if [[ "${refiner_stopped_at_target}" != "1" && -z "${NEXT_CKPT_WATCHER_TMUX}" ]]; then
  NEXT_CKPT_WATCHER_TMUX="curiosity_sugar_model${NEXT_CKPT_ITERATION}_watch_0712:0"
fi
if [[ "${refiner_stopped_at_target}" != "1" && -z "${NEXT_CKPT_WAIT_LOG}" ]]; then
  NEXT_CKPT_WAIT_LOG="${ROOT_DIR}/experiments/sugar_reproduction/logs/20260712_sugar_refiner_model_${NEXT_CKPT_ITERATION}_watch.log"
fi

if command -v tmux >/dev/null 2>&1; then
  refiner_session="${REFINER_TMUX%%:*}"
  watcher_session="${WATCHER_TMUX%%:*}"
  next_ckpt_watcher_session="${NEXT_CKPT_WATCHER_TMUX%%:*}"
  if tmux has-session -t "${refiner_session}" 2>/dev/null; then
    echo "[SUGAR-STATUS] refiner_tmux=present:${refiner_session}"
  else
    echo "[SUGAR-STATUS] refiner_tmux=missing:${refiner_session}"
  fi
  if tmux has-session -t "${watcher_session}" 2>/dev/null; then
    echo "[SUGAR-STATUS] watcher_tmux=present:${watcher_session}"
  else
    echo "[SUGAR-STATUS] watcher_tmux=missing:${watcher_session}"
  fi
  if [[ "${refiner_stopped_at_target}" == "1" ]]; then
    echo "[SUGAR-STATUS] next_ckpt_watcher_tmux=stopped_at_model_${latest_refiner_model}.pt"
  elif tmux has-session -t "${next_ckpt_watcher_session}" 2>/dev/null; then
    echo "[SUGAR-STATUS] next_ckpt_watcher_tmux=present:${next_ckpt_watcher_session}"
  else
    echo "[SUGAR-STATUS] next_ckpt_watcher_tmux=missing:${next_ckpt_watcher_session}"
  fi
  if [[ "${SUGAR_STATUS_COMPACT}" != "1" ]]; then
    echo "[SUGAR-STATUS] refiner_tmux_tail:"
    tmux capture-pane -pt "${REFINER_TMUX}" -S -120 2>/dev/null \
      | strip_ansi \
      | grep -E 'Learning iteration|Total timesteps|Time elapsed|ETA|Mean reward|Computation' \
      | tail -24 || true
    echo "[SUGAR-STATUS] watcher_tmux_tail:"
    tmux capture-pane -pt "${WATCHER_TMUX}" -S -80 2>/dev/null | tail -40 || true
  fi
fi

if [[ "${refiner_stopped_at_target}" == "1" ]]; then
  echo "[SUGAR-STATUS] next_ckpt_wait_log=stopped_at_model_${latest_refiner_model}.pt"
elif [[ -s "${NEXT_CKPT_WAIT_LOG}" ]]; then
  echo "[SUGAR-STATUS] next_ckpt_wait_log_config:"
  grep -E '^\[SUGAR-CKPT-WAIT\] (host=|root=|sugar_dir=|task=|exp_name=|output_dir=|target_iteration=|target_ckpt=|slurm_job_id=|wait_interval_seconds=|wait_timeout_seconds=|auto_chain_next=|chain_step=|chain_until_iteration=|started=)' "${NEXT_CKPT_WAIT_LOG}" \
    | tail -32 || true
  echo "[SUGAR-STATUS] next_ckpt_wait_log_tail:"
  if [[ "${SUGAR_STATUS_COMPACT}" == "1" ]]; then
    tail -8 "${NEXT_CKPT_WAIT_LOG}" || true
  else
    tail -24 "${NEXT_CKPT_WAIT_LOG}" || true
  fi
else
  echo "[SUGAR-STATUS] next_ckpt_wait_log=missing:${NEXT_CKPT_WAIT_LOG}"
fi

if [[ "${refiner_stopped_at_target}" == "1" ]]; then
  echo "[SUGAR-STATUS] watcher_wait_log=stopped_at_model_${latest_refiner_model}.pt"
elif [[ -s "${WAIT_LOG}" ]]; then
  echo "[SUGAR-STATUS] watcher_wait_log_config:"
  grep -E '^\[SUGAR-REMAINING-WAIT\] (host=|root=|sugar_dir=|task=|exp_name=|output_dir=|refiner_ckpt=|python_bin=|pipeline_script=|slurm_|wait_interval_seconds=|wait_timeout_seconds=|started=)' "${WAIT_LOG}" \
    | tail -32 || true
  echo "[SUGAR-STATUS] watcher_wait_log_tail:"
  if [[ "${SUGAR_STATUS_COMPACT}" == "1" ]]; then
    tail -8 "${WAIT_LOG}" || true
  else
    tail -40 "${WAIT_LOG}" || true
  fi
else
  echo "[SUGAR-STATUS] watcher_wait_log=missing:${WAIT_LOG}"
fi
