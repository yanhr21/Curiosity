#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing SUGAR downstream pipeline on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
SOURCE_REFINER="${SOURCE_REFINER:-${OUTPUT_DIR}/ckpts/refiner_model10000.pt}"
PIPELINE_REFINER="${PIPELINE_REFINER:-${OUTPUT_DIR}/ckpts/refiner.pt}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${ROOT_DIR}/scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh}"

START_STAGE="${START_STAGE:-refiner_rollout}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
REFINER_ROLLOUT_NUM_ENVS="${REFINER_ROLLOUT_NUM_ENVS:-1000}"
TRACKER_NUM_ENVS="${TRACKER_NUM_ENVS:-4096}"
TRACKER_MAX_ITERATIONS="${TRACKER_MAX_ITERATIONS:-10001}"
TRACKER_FINAL_ITERATION="${TRACKER_FINAL_ITERATION:-10000}"
TRACKER_ROLLOUT_NUM_ENVS="${TRACKER_ROLLOUT_NUM_ENVS:-1000}"
GENERATOR_NUM_EPOCHS="${GENERATOR_NUM_EPOCHS:-1001}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_${TASK_NAME}_downstream_from_refiner10000}"

for path in "${SOURCE_REFINER}" "${PIPELINE_REFINER}" "${PIPELINE_SCRIPT}" "${PYTHON_BIN}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing required downstream input: ${path}" >&2
    exit 3
  fi
done

source_sha="$(sha256sum "${SOURCE_REFINER}" | awk '{print $1}')"
pipeline_sha="$(sha256sum "${PIPELINE_REFINER}" | awk '{print $1}')"
if [[ "${source_sha}" != "${pipeline_sha}" ]]; then
  echo "Refiner export does not match operator-selected model_10000: source=${source_sha} pipeline=${pipeline_sha}" >&2
  exit 4
fi

{
  echo "[SUGAR-DOWNSTREAM-10000] host=$(hostname)"
  echo "[SUGAR-DOWNSTREAM-10000] source_refiner=${SOURCE_REFINER}"
  echo "[SUGAR-DOWNSTREAM-10000] pipeline_refiner=${PIPELINE_REFINER}"
  echo "[SUGAR-DOWNSTREAM-10000] refiner_sha256=${source_sha}"
  echo "[SUGAR-DOWNSTREAM-10000] refiner_training_frozen_at=10000"
  echo "[SUGAR-DOWNSTREAM-10000] start_stage=${START_STAGE}"
  echo "[SUGAR-DOWNSTREAM-10000] stop_after_stage=${STOP_AFTER_STAGE}"
  echo "[SUGAR-DOWNSTREAM-10000] tracker_num_envs=${TRACKER_NUM_ENVS}"
  echo "[SUGAR-DOWNSTREAM-10000] tracker_final_iteration=${TRACKER_FINAL_ITERATION}"
  echo "[SUGAR-DOWNSTREAM-10000] generator_num_epochs=${GENERATOR_NUM_EPOCHS}"
  echo "[SUGAR-DOWNSTREAM-10000] fidelity=official_code_tasks_assets_with_operator_selected_refiner10000"
}

exec env \
  ROOT_DIR="${ROOT_DIR}" \
  SUGAR_DIR="${SUGAR_DIR}" \
  TASK_NAME="${TASK_NAME}" \
  EXP_NAME="${EXP_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  START_STAGE="${START_STAGE}" \
  STOP_AFTER_STAGE="${STOP_AFTER_STAGE}" \
  REFINER_MAX_ITERATIONS=10001 \
  REFINER_FINAL_ITERATION=10000 \
  REFINER_ROLLOUT_NUM_ENVS="${REFINER_ROLLOUT_NUM_ENVS}" \
  TRACKER_NUM_ENVS="${TRACKER_NUM_ENVS}" \
  TRACKER_MAX_ITERATIONS="${TRACKER_MAX_ITERATIONS}" \
  TRACKER_FINAL_ITERATION="${TRACKER_FINAL_ITERATION}" \
  TRACKER_ROLLOUT_NUM_ENVS="${TRACKER_ROLLOUT_NUM_ENVS}" \
  GENERATOR_NUM_EPOCHS="${GENERATOR_NUM_EPOCHS}" \
  STAMP="${STAMP}" \
  bash "${PIPELINE_SCRIPT}"
