#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner5000_rollout_eval.log}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_DIR}/eval/refiner_model5000_rollout_eval_novideo/raw_npz}"
PAPER_URL="${PAPER_URL:-https://arxiv.org/html/2605.20373v1#S4.T1}"

extract_last() {
  local pattern="$1"
  local file="$2"
  sed -nE "${pattern}" "${file}" | tail -1
}

echo "[SUGAR-REFINER5000-SUMMARY] checked=$(date '+%F %T %Z')"
echo "[SUGAR-REFINER5000-SUMMARY] root=${ROOT_DIR}"
echo "[SUGAR-REFINER5000-SUMMARY] sugar_dir=${SUGAR_DIR}"
echo "[SUGAR-REFINER5000-SUMMARY] task=${TASK_NAME}"
echo "[SUGAR-REFINER5000-SUMMARY] exp_name=${EXP_NAME}"
echo "[SUGAR-REFINER5000-SUMMARY] log=${LOG_PATH}"
echo "[SUGAR-REFINER5000-SUMMARY] rollout_dir=${ROLLOUT_DIR}"

if [[ ! -s "${LOG_PATH}" ]]; then
  echo "[SUGAR-REFINER5000-SUMMARY] status=missing_log"
  exit 2
fi

checkpoint="$(extract_last 's/^\[SUGAR-REFINER5000-EVAL\] checkpoint=(.*)$/\1/p' "${LOG_PATH}")"
num_envs="$(extract_last 's/^\[SUGAR-REFINER5000-EVAL\] num_envs=([0-9]+)$/\1/p' "${LOG_PATH}")"
expected_windows="$(extract_last 's/^\[Rollout\] Expected total rollouts \(assuming all timeouts\): ([0-9]+)$/\1/p' "${LOG_PATH}")"
saved_total="$(extract_last 's/^\[Rollout\] ====== All [0-9]+ envs completed, total ([0-9]+) trajectories saved to .*/\1/p' "${LOG_PATH}")"
reported_complete_count="$(extract_last 's/^\[SUGAR-REFINER5000-EVAL\] trajectory_complete_count=([0-9]+)$/\1/p' "${LOG_PATH}")"

file_complete_count=0
if [[ -d "${ROLLOUT_DIR}/trajectory_complete" ]]; then
  file_complete_count="$(find "${ROLLOUT_DIR}/trajectory_complete" -maxdepth 1 -type f -name '*.npz' | wc -l)"
fi

if [[ -z "${saved_total}" ]]; then
  saved_total="${reported_complete_count:-${file_complete_count}}"
fi
if [[ -z "${reported_complete_count}" ]]; then
  reported_complete_count="${file_complete_count}"
fi

completion_rate="n/a"
if [[ -n "${expected_windows}" && "${expected_windows}" -gt 0 && -n "${saved_total}" ]]; then
  completion_rate="$(awk -v saved="${saved_total}" -v expected="${expected_windows}" 'BEGIN { printf "%.2f", 100.0 * saved / expected }')"
fi

echo "[SUGAR-REFINER5000-SUMMARY] checkpoint=${checkpoint:-unknown}"
echo "[SUGAR-REFINER5000-SUMMARY] eval_num_envs=${num_envs:-unknown}"
echo "[SUGAR-REFINER5000-SUMMARY] expected_rollout_windows=${expected_windows:-unknown}"
echo "[SUGAR-REFINER5000-SUMMARY] saved_trajectory_complete=${saved_total:-unknown}"
echo "[SUGAR-REFINER5000-SUMMARY] reported_trajectory_complete_count=${reported_complete_count}"
echo "[SUGAR-REFINER5000-SUMMARY] file_trajectory_complete_count=${file_complete_count}"
echo "[SUGAR-REFINER5000-SUMMARY] sampled_refiner_window_completion_rate_percent=${completion_rate}"
echo "[SUGAR-REFINER5000-SUMMARY] metric_scope=sampled_5000_step_refiner_rollout_windows"
echo "[SUGAR-REFINER5000-SUMMARY] paper_metric_definition=CarryBox_SR_is_final_object_position_within_target_threshold; Err_is_final_object_target_euclidean_distance"
echo "[SUGAR-REFINER5000-SUMMARY] paper_table1_source=${PAPER_URL}"
echo "[SUGAR-REFINER5000-SUMMARY] paper_table1_sugar_carrybox_train_sr=84.5"
echo "[SUGAR-REFINER5000-SUMMARY] paper_table1_sugar_carrybox_train_err=0.280"
echo "[SUGAR-REFINER5000-SUMMARY] paper_table1_sugar_carrybox_test_sr=69.6"
echo "[SUGAR-REFINER5000-SUMMARY] paper_table1_sugar_carrybox_test_err=0.326"
echo "[SUGAR-REFINER5000-SUMMARY] comparable_to_paper=false"
echo "[SUGAR-REFINER5000-SUMMARY] reason=official_paper_metric_requires_final_object_target_success_and_error_for_full_refiner_tracker_generator_policy; this_run_is_a_16_env_sampled_refiner_rollout_window_check_at_model_5000.pt"
