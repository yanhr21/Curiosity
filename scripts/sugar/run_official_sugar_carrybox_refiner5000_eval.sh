#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run SUGAR eval on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-20260712_official_carrybox_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/logs/refiner/model_5000.pt}"
EVAL_NAME="${EVAL_NAME:-refiner_model5000_eval}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_DIR}/eval/${EVAL_NAME}/raw_npz}"
NUM_ENVS="${NUM_ENVS:-16}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_${TASK_NAME}_refiner5000_eval}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${STAMP}.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ENABLE_VIDEO="${ENABLE_VIDEO:-1}"

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
  "${CHECKPOINT}"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required SUGAR eval input: ${SUGAR_DIR}/${path}" >&2
    exit 4
  fi
done

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  echo "[SUGAR-REFINER5000-EVAL] running environment preflight" | tee "${LOG_PATH}"
  ROOT_DIR="${ROOT_DIR}" SUGAR_DIR="${SUGAR_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/scripts/sugar/preflight_official_sugar_env.sh" \
    2>&1 | tee -a "${LOG_PATH}"
fi

{
  echo "[SUGAR-REFINER5000-EVAL] host=$(hostname)"
  echo "[SUGAR-REFINER5000-EVAL] root=${ROOT_DIR}"
  echo "[SUGAR-REFINER5000-EVAL] sugar_dir=${SUGAR_DIR}"
  echo "[SUGAR-REFINER5000-EVAL] task=${TASK_NAME}"
  echo "[SUGAR-REFINER5000-EVAL] output_dir=${OUTPUT_DIR}"
  echo "[SUGAR-REFINER5000-EVAL] checkpoint=${CHECKPOINT}"
  echo "[SUGAR-REFINER5000-EVAL] rollout_dir=${ROLLOUT_DIR}"
  echo "[SUGAR-REFINER5000-EVAL] num_envs=${NUM_ENVS}"
  echo "[SUGAR-REFINER5000-EVAL] enable_video=${ENABLE_VIDEO}"
  echo "[SUGAR-REFINER5000-EVAL] video_length=${VIDEO_LENGTH}"
  echo "[SUGAR-REFINER5000-EVAL] python_bin=${PYTHON_BIN}"
  echo "[SUGAR-REFINER5000-EVAL] sugar_commit=$(git rev-parse HEAD)"
  if [[ -f "${ROOT_DIR}/IsaacLab/VERSION" ]]; then
    echo "[SUGAR-REFINER5000-EVAL] isaaclab_version=v$(tr -d '[:space:]' < "${ROOT_DIR}/IsaacLab/VERSION")-curiosity-glue"
  fi
  echo "[SUGAR-REFINER5000-EVAL] command starts at $(date '+%F %T')"
  echo "[SUGAR-REFINER5000-EVAL] log=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

cmd=(
  "${PYTHON_BIN}" scripts/sugar_rl/play.py
  --task "Sugar-G129dof-${TASK_NAME}-Refiner-Rollout"
  --checkpoint "${CHECKPOINT}"
  --rollout_dir "${ROLLOUT_DIR}"
  --motion_folder "data/${TASK_NAME}"
  --num_envs "${NUM_ENVS}"
  --eval_mode
  --eval_random_motion
  --headless
)

if [[ "${ENABLE_VIDEO}" == "1" ]]; then
  cmd+=(--video --video_length "${VIDEO_LENGTH}")
fi

set +e
timeout "${TIMEOUT_SECONDS}" "${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" != "0" ]] \
    && grep -Eq "\\[Rollout\\] ====== All [0-9]+ envs completed, total [0-9]+ trajectories saved to " "${LOG_PATH}" \
    && ! grep -Eq "Traceback \\(most recent call last\\)|FileNotFoundError|Boost\\.Python\\.ArgumentError|RuntimeError|CUDA out of memory|\\[Error\\]" "${LOG_PATH}"; then
  echo "[SUGAR-REFINER5000-EVAL] rollout completion SystemExit detected; treating as success" | tee -a "${LOG_PATH}"
  status=0
fi

if grep -Eq "Traceback \\(most recent call last\\)|FileNotFoundError|Boost\\.Python\\.ArgumentError|RuntimeError|CUDA out of memory|\\[Error\\]" "${LOG_PATH}"; then
  echo "[SUGAR-REFINER5000-EVAL] fatal pattern detected in log" | tee -a "${LOG_PATH}"
  status=20
fi

complete_count=0
if [[ -d "${ROLLOUT_DIR}/trajectory_complete" ]]; then
  complete_count="$(find "${ROLLOUT_DIR}/trajectory_complete" -maxdepth 1 -type f -name '*.npz' | wc -l)"
fi
echo "[SUGAR-REFINER5000-EVAL] trajectory_complete_count=${complete_count}" | tee -a "${LOG_PATH}"
echo "[SUGAR-REFINER5000-EVAL] command finished at $(date '+%F %T') with status=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
