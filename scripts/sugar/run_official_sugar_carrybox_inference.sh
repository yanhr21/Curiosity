#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run SUGAR inference on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
NUM_ENVS="${NUM_ENVS:-16}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_${TASK_NAME}_inference}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${STAMP}.log}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRACKER_CHECKPOINT="${TRACKER_CHECKPOINT:-demo_ckpts/${TASK_NAME}/tracker.pt}"
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-demo_ckpts/${TASK_NAME}/generator.ckpt}"

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
  "${TRACKER_CHECKPOINT}"
  "${GENERATOR_CHECKPOINT}"
  "descriptions/robots/g1"
  "descriptions/objects/small_box"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required official SUGAR asset: ${SUGAR_DIR}/${path}" >&2
    exit 4
  fi
done

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  echo "[SUGAR-INFERENCE] running environment preflight" | tee "${LOG_PATH}"
  ROOT_DIR="${ROOT_DIR}" SUGAR_DIR="${SUGAR_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/scripts/sugar/preflight_official_sugar_env.sh" \
    2>&1 | tee -a "${LOG_PATH}"
fi

{
  echo "[SUGAR-INFERENCE] host=$(hostname)"
  echo "[SUGAR-INFERENCE] root=${ROOT_DIR}"
  echo "[SUGAR-INFERENCE] sugar_dir=${SUGAR_DIR}"
  echo "[SUGAR-INFERENCE] task=${TASK_NAME}"
  echo "[SUGAR-INFERENCE] tracker_checkpoint=${TRACKER_CHECKPOINT}"
  echo "[SUGAR-INFERENCE] generator_checkpoint=${GENERATOR_CHECKPOINT}"
  echo "[SUGAR-INFERENCE] python_bin=${PYTHON_BIN}"
  echo "[SUGAR-INFERENCE] omni_kit_accept_eula=${OMNI_KIT_ACCEPT_EULA}"
  echo "[SUGAR-INFERENCE] sugar_commit=$(git rev-parse HEAD)"
  if [[ -f "${ROOT_DIR}/IsaacLab/VERSION" ]]; then
    echo "[SUGAR-INFERENCE] isaaclab_version=v$(tr -d '[:space:]' < "${ROOT_DIR}/IsaacLab/VERSION")-curiosity-glue"
  fi
  echo "[SUGAR-INFERENCE] command starts at $(date '+%F %T')"
  echo "[SUGAR-INFERENCE] log=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

set +e
timeout "${TIMEOUT_SECONDS}" \
  "${PYTHON_BIN}" scripts/sugar_rl/play.py \
    --task "Sugar-G129dof-${TASK_NAME}-Inference" \
    --checkpoint "${TRACKER_CHECKPOINT}" \
    --generator_checkpoint "${GENERATOR_CHECKPOINT}" \
    --motion_folder "data/${TASK_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --eval_random_motion \
    --headless \
    --video \
    --video_length "${VIDEO_LENGTH}" \
  2>&1 | tee -a "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e
if grep -Eq "Traceback \\(most recent call last\\)|FileNotFoundError|Boost\\.Python\\.ArgumentError|RuntimeError|CUDA out of memory|\\[Error\\]" "${LOG_PATH}"; then
  echo "[SUGAR-INFERENCE] fatal traceback detected in log" | tee -a "${LOG_PATH}"
  status=20
fi
echo "[SUGAR-INFERENCE] command finished at $(date '+%F %T') with status=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
