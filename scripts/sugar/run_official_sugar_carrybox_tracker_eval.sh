#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing SUGAR Tracker eval on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/ckpts/tracker.pt}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${OUTPUT_DIR}/ckpts/refiner.pt}"
MOTION_FOLDER="${MOTION_FOLDER:-${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset}"
TEACHER_MOTION_FOLDER="${TEACHER_MOTION_FOLDER:-data/${TASK_NAME}}"
EVAL_NAME="${EVAL_NAME:-tracker_eval}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_DIR}/eval/${EVAL_NAME}/raw_npz}"
NUM_ENVS="${NUM_ENVS:-16}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
ENABLE_VIDEO="${ENABLE_VIDEO:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_${TASK_NAME}_tracker_eval}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${STAMP}.log}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SUGAR_DISABLE_RENDERER_MULTIGPU="${SUGAR_DISABLE_RENDERER_MULTIGPU:-0}"

mkdir -p "${LOG_DIR}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACLAB_GROUND_PLANE_USD="${ISAACLAB_GROUND_PLANE_USD:-${SUGAR_DIR}/descriptions/terrain/sugar_ground_plane.usda}"
export ISAACLAB_USE_LOCAL_FRAME_MARKER="${ISAACLAB_USE_LOCAL_FRAME_MARKER:-1}"

cd "${SUGAR_DIR}"
for path in "${CHECKPOINT}" "${TEACHER_CHECKPOINT}" "${MOTION_FOLDER}" "${TEACHER_MOTION_FOLDER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required official Tracker eval input: ${SUGAR_DIR}/${path}" >&2
    exit 3
  fi
done

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  ROOT_DIR="${ROOT_DIR}" SUGAR_DIR="${SUGAR_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/scripts/sugar/preflight_official_sugar_env.sh" 2>&1 | tee "${LOG_PATH}"
fi

{
  echo "[SUGAR-TRACKER-EVAL] host=$(hostname)"
  echo "[SUGAR-TRACKER-EVAL] checkpoint=${CHECKPOINT}"
  echo "[SUGAR-TRACKER-EVAL] teacher_checkpoint=${TEACHER_CHECKPOINT}"
  echo "[SUGAR-TRACKER-EVAL] motion_folder=${MOTION_FOLDER}"
  echo "[SUGAR-TRACKER-EVAL] rollout_dir=${ROLLOUT_DIR}"
  echo "[SUGAR-TRACKER-EVAL] num_envs=${NUM_ENVS}"
  echo "[SUGAR-TRACKER-EVAL] enable_video=${ENABLE_VIDEO}"
  echo "[SUGAR-TRACKER-EVAL] video_length=${VIDEO_LENGTH}"
  echo "[SUGAR-TRACKER-EVAL] command starts at $(date '+%F %T')"
} | tee -a "${LOG_PATH}"

cmd=(
  "${PYTHON_BIN}" scripts/sugar_rl/play.py
  --task "Sugar-G129dof-${TASK_NAME}-Tracker-Rollout"
  --checkpoint "${CHECKPOINT}"
  --teacher_ckpt "${TEACHER_CHECKPOINT}"
  --motion_folder "${MOTION_FOLDER}"
  --teacher_motion_folder "${TEACHER_MOTION_FOLDER}"
  --rollout_dir "${ROLLOUT_DIR}"
  --num_envs "${NUM_ENVS}"
  --eval_mode
  --eval_random_motion
  --headless
)
if [[ "${SUGAR_DISABLE_RENDERER_MULTIGPU}" == "1" ]]; then
  cmd+=(--kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1")
fi
if [[ "${ENABLE_VIDEO}" == "1" ]]; then
  cmd+=(--video --video_length "${VIDEO_LENGTH}")
fi

set +e
timeout "${TIMEOUT_SECONDS}" "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" != "0" ]] \
    && grep -Eq '\[Rollout\] ====== All [0-9]+ envs completed, total [0-9]+ trajectories saved to ' "${LOG_PATH}" \
    && ! grep -Eq 'Traceback \(most recent call last\)|FileNotFoundError|Boost\.Python\.ArgumentError|RuntimeError|CUDA out of memory|\[Error\]' "${LOG_PATH}"; then
  status=0
fi
if grep -Eq 'Traceback \(most recent call last\)|FileNotFoundError|Boost\.Python\.ArgumentError|RuntimeError|CUDA out of memory|\[Error\]' "${LOG_PATH}"; then
  status=20
fi

complete_count=0
if [[ -d "${ROLLOUT_DIR}/trajectory_complete" ]]; then
  complete_count="$(find "${ROLLOUT_DIR}/trajectory_complete" -maxdepth 1 -type f -name '*.npz' | wc -l)"
fi
echo "[SUGAR-TRACKER-EVAL] trajectory_complete_count=${complete_count}" | tee -a "${LOG_PATH}"
echo "[SUGAR-TRACKER-EVAL] command finished at $(date '+%F %T') with status=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
