#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run SUGAR training on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
NUM_ENVS="${NUM_ENVS:-128}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_${TASK_NAME}_refiner_train}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${STAMP}.log}"
EXP_NAME="${EXP_NAME:-${STAMP}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}/logs/refiner}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACLAB_GROUND_PLANE_USD="${ISAACLAB_GROUND_PLANE_USD:-${SUGAR_DIR}/descriptions/terrain/sugar_ground_plane.usda}"
export ISAACLAB_USE_LOCAL_FRAME_MARKER="${ISAACLAB_USE_LOCAL_FRAME_MARKER:-1}"

if [[ ! -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  echo "Missing official SUGAR clone at ${SUGAR_DIR}" >&2
  exit 3
fi

cd "${SUGAR_DIR}"

required_paths=(
  "data/${TASK_NAME}"
  "descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
  "descriptions/objects/small_box/obj_aligned.usd"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required official SUGAR asset: ${SUGAR_DIR}/${path}" >&2
    exit 4
  fi
done

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  echo "[SUGAR-TRAIN-REFINER] running environment preflight" | tee "${LOG_PATH}"
  ROOT_DIR="${ROOT_DIR}" SUGAR_DIR="${SUGAR_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/scripts/sugar/preflight_official_sugar_env.sh" \
    2>&1 | tee -a "${LOG_PATH}"
fi

{
  echo "[SUGAR-TRAIN-REFINER] host=$(hostname)"
  echo "[SUGAR-TRAIN-REFINER] root=${ROOT_DIR}"
  echo "[SUGAR-TRAIN-REFINER] sugar_dir=${SUGAR_DIR}"
  echo "[SUGAR-TRAIN-REFINER] task=${TASK_NAME}"
  echo "[SUGAR-TRAIN-REFINER] python_bin=${PYTHON_BIN}"
  echo "[SUGAR-TRAIN-REFINER] num_envs=${NUM_ENVS}"
  echo "[SUGAR-TRAIN-REFINER] max_iterations=${MAX_ITERATIONS}"
  echo "[SUGAR-TRAIN-REFINER] train_log_dir=${TRAIN_LOG_DIR}"
  echo "[SUGAR-TRAIN-REFINER] sugar_commit=$(git rev-parse HEAD)"
  if [[ -f "${ROOT_DIR}/IsaacLab/VERSION" ]]; then
    echo "[SUGAR-TRAIN-REFINER] isaaclab_version=v$(tr -d '[:space:]' < "${ROOT_DIR}/IsaacLab/VERSION")-curiosity-glue"
  fi
  echo "[SUGAR-TRAIN-REFINER] command starts at $(date '+%F %T')"
  echo "[SUGAR-TRAIN-REFINER] log=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

set +e
timeout "${TIMEOUT_SECONDS}" \
  "${PYTHON_BIN}" scripts/sugar_rl/train.py \
    --task "Sugar-G129dof-${TASK_NAME}-Refiner" \
    --num_envs "${NUM_ENVS}" \
    --log_dir "${TRAIN_LOG_DIR}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --motion_folder "data/${TASK_NAME}" \
    --headless \
  2>&1 | tee -a "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e
if grep -Eq "Traceback \\(most recent call last\\)|FileNotFoundError|Boost\\.Python\\.ArgumentError" "${LOG_PATH}"; then
  echo "[SUGAR-TRAIN-REFINER] fatal traceback detected in log" | tee -a "${LOG_PATH}"
  status=20
fi
echo "[SUGAR-TRAIN-REFINER] command finished at $(date '+%F %T') with status=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
