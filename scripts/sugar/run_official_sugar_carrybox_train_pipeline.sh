#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run SUGAR training pipeline on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
TASK_NAME="${TASK_NAME:-CarryBox}"
EXP_NAME="${EXP_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/${TASK_NAME}_${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/experiments/sugar_reproduction/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_sugar_${TASK_NAME}_train_pipeline}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${STAMP}.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
START_STAGE="${START_STAGE:-refiner_train}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
PIPELINE_LOCK_ENABLED="${PIPELINE_LOCK_ENABLED:-1}"

REFINER_NUM_ENVS="${REFINER_NUM_ENVS:-4096}"
REFINER_MAX_ITERATIONS="${REFINER_MAX_ITERATIONS:-30001}"
REFINER_FINAL_ITERATION="${REFINER_FINAL_ITERATION:-$((REFINER_MAX_ITERATIONS - 1))}"
REFINER_RESUME_CHECKPOINT="${REFINER_RESUME_CHECKPOINT:-}"
REFINER_CANONICAL_RESUME_CHECKPOINT="${REFINER_CANONICAL_RESUME_CHECKPOINT:-${OUTPUT_DIR}/resume_sources/server23_original/model_5000.pt}"
REFINER_ROLLOUT_NUM_ENVS="${REFINER_ROLLOUT_NUM_ENVS:-1000}"
TRACKER_NUM_ENVS="${TRACKER_NUM_ENVS:-4096}"
TRACKER_MAX_ITERATIONS="${TRACKER_MAX_ITERATIONS:-30001}"
TRACKER_FINAL_ITERATION="${TRACKER_FINAL_ITERATION:-$((TRACKER_MAX_ITERATIONS - 1))}"
TRACKER_RESUME_CHECKPOINT="${TRACKER_RESUME_CHECKPOINT:-}"
TRACKER_ROLLOUT_NUM_ENVS="${TRACKER_ROLLOUT_NUM_ENVS:-1000}"
GENERATOR_NUM_EPOCHS="${GENERATOR_NUM_EPOCHS:-1001}"

mkdir -p "${LOG_DIR}"
: > "${LOG_PATH}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ISAACLAB_GROUND_PLANE_USD="${ISAACLAB_GROUND_PLANE_USD:-${SUGAR_DIR}/descriptions/terrain/sugar_ground_plane.usda}"
export ISAACLAB_USE_LOCAL_FRAME_MARKER="${ISAACLAB_USE_LOCAL_FRAME_MARKER:-1}"
export SUGAR_DISABLE_TRAIN_DEBUG_VIS="${SUGAR_DISABLE_TRAIN_DEBUG_VIS:-0}"
export SUGAR_DISABLE_FABRIC="${SUGAR_DISABLE_FABRIC:-0}"
export SUGAR_DISABLE_RENDERER_MULTIGPU="${SUGAR_DISABLE_RENDERER_MULTIGPU:-0}"
export SUGAR_DISABLE_RENDERER="${SUGAR_DISABLE_RENDERER:-0}"
export SUGAR_EXPERIENCE="${SUGAR_EXPERIENCE:-}"

if [[ ! -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  echo "Missing official SUGAR clone at ${SUGAR_DIR}" >&2
  exit 3
fi

cd "${SUGAR_DIR}"

if [[ "${PIPELINE_LOCK_ENABLED}" == "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  PIPELINE_LOCK_PATH="${OUTPUT_DIR}/.official_sugar_pipeline.lock"
  exec 9>"${PIPELINE_LOCK_PATH}"
  echo "[SUGAR-TRAIN-PIPELINE] waiting for exclusive output lock: ${PIPELINE_LOCK_PATH}" | tee -a "${LOG_PATH}"
  flock 9
  echo "[SUGAR-TRAIN-PIPELINE] acquired exclusive output lock: ${PIPELINE_LOCK_PATH}" | tee -a "${LOG_PATH}"
fi

# A resumed runner saves model_<current_iteration>.pt again after its first
# update. Redirect the known live model_5000 path to an immutable copy so a
# failed primary cannot change the checkpoint from which a waiting backup
# allocation resumes.
if [[ -n "${REFINER_RESUME_CHECKPOINT}" && -s "${REFINER_CANONICAL_RESUME_CHECKPOINT}" ]]; then
  requested_resume_path="$(realpath -m "${REFINER_RESUME_CHECKPOINT}")"
  live_resume_path="$(realpath -m "${OUTPUT_DIR}/logs/refiner/model_5000.pt")"
  if [[ "${requested_resume_path}" == "${live_resume_path}" ]]; then
    REFINER_RESUME_CHECKPOINT="$(realpath "${REFINER_CANONICAL_RESUME_CHECKPOINT}")"
    echo "[SUGAR-TRAIN-PIPELINE] redirected mutable resume checkpoint to canonical source: ${REFINER_RESUME_CHECKPOINT}" | tee -a "${LOG_PATH}"
  fi
fi

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

run_logged() {
  local stage_name="$1"
  shift
  echo "[SUGAR-TRAIN-PIPELINE] stage ${stage_name} starts at $(date '+%F %T')" | tee -a "${LOG_PATH}"
  local stage_log_start
  stage_log_start="$(wc -l < "${LOG_PATH}" 2>/dev/null || echo 0)"
  set +e
  if [[ "${TIMEOUT_SECONDS}" == "0" ]]; then
    "$@" 2>&1 | tee -a "${LOG_PATH}"
  else
    timeout "${TIMEOUT_SECONDS}" "$@" 2>&1 | tee -a "${LOG_PATH}"
  fi
  local status="${PIPESTATUS[0]}"
  set -e
  local stage_log
  stage_log="$(mktemp)"
  tail -n +"$((stage_log_start + 1))" "${LOG_PATH}" > "${stage_log}"
  local fatal_detected=0
  if grep -Eq "Traceback \\(most recent call last\\)|FileNotFoundError|Boost\\.Python\\.ArgumentError|RuntimeError|CUDA out of memory|\\[Error\\]" "${stage_log}"; then
    echo "[SUGAR-TRAIN-PIPELINE] fatal traceback detected in log" | tee -a "${LOG_PATH}"
    fatal_detected=1
    status=20
  fi
  if [[ "${fatal_detected}" == "0" && "${status}" != "0" && ( "${stage_name}" == "refiner_rollout" || "${stage_name}" == "tracker_rollout" ) ]] \
      && grep -Eq "\\[Rollout\\] ====== All [0-9]+ envs completed, total [0-9]+ trajectories saved to " "${stage_log}"; then
    echo "[SUGAR-TRAIN-PIPELINE] stage ${stage_name} reported rollout completion via SystemExit; treating as success" | tee -a "${LOG_PATH}"
    status=0
  fi
  rm -f "${stage_log}"
  echo "[SUGAR-TRAIN-PIPELINE] stage ${stage_name} finished at $(date '+%F %T') with status=${status}" | tee -a "${LOG_PATH}"
  return "${status}"
}

stage_index() {
  case "$1" in
    refiner_train) echo 0 ;;
    refiner_rollout) echo 1 ;;
    process_refiner_rollout) echo 2 ;;
    tracker_train) echo 3 ;;
    tracker_rollout) echo 4 ;;
    process_tracker_rollout) echo 5 ;;
    generator_train) echo 6 ;;
    *) echo "Unknown stage: $1" >&2; exit 5 ;;
  esac
}

should_run_stage() {
  local stage="$1"
  [[ "$(stage_index "${stage}")" -ge "$(stage_index "${START_STAGE}")" ]]
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "[SUGAR-TRAIN-PIPELINE] missing required ${label}: ${path}" | tee -a "${LOG_PATH}"
    exit 30
  fi
  echo "[SUGAR-TRAIN-PIPELINE] verified ${label}: ${path}" | tee -a "${LOG_PATH}"
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[SUGAR-TRAIN-PIPELINE] missing required ${label}: ${path}" | tee -a "${LOG_PATH}"
    exit 31
  fi
  echo "[SUGAR-TRAIN-PIPELINE] verified ${label}: ${path}" | tee -a "${LOG_PATH}"
}

require_glob_nonempty() {
  local label="$1"
  local pattern="$2"
  local old_nullglob
  old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  local matches=( ${pattern} )
  eval "${old_nullglob}" 2>/dev/null || shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    echo "[SUGAR-TRAIN-PIPELINE] missing required ${label}: pattern=${pattern}" | tee -a "${LOG_PATH}"
    exit 32
  fi
  echo "[SUGAR-TRAIN-PIPELINE] verified ${label}: count=${#matches[@]} pattern=${pattern}" | tee -a "${LOG_PATH}"
}

validate_resume_request() {
  local stage_name="$1"
  local checkpoint="$2"
  local train_iterations="$3"
  local final_iteration="$4"
  if [[ -z "${checkpoint}" ]]; then
    return 0
  fi
  require_file "${stage_name} resume checkpoint" "${checkpoint}"
  local checkpoint_name
  checkpoint_name="$(basename "${checkpoint}")"
  if [[ ! "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
    echo "[SUGAR-TRAIN-PIPELINE] ${stage_name} resume checkpoint must use official model_<iteration>.pt naming: ${checkpoint}" | tee -a "${LOG_PATH}"
    exit 33
  fi
  local resume_iteration="${BASH_REMATCH[1]}"
  local computed_final_iteration=$((resume_iteration + train_iterations - 1))
  if [[ "${computed_final_iteration}" -ne "${final_iteration}" ]]; then
    echo "[SUGAR-TRAIN-PIPELINE] ${stage_name} resume iteration mismatch: resume=${resume_iteration} train_iterations=${train_iterations} computed_final=${computed_final_iteration} requested_final=${final_iteration}" | tee -a "${LOG_PATH}"
    exit 34
  fi
  echo "[SUGAR-TRAIN-PIPELINE] validated ${stage_name} resume: checkpoint=${checkpoint} resume_iteration=${resume_iteration} train_iterations=${train_iterations} final_iteration=${final_iteration}" | tee -a "${LOG_PATH}"
}

maybe_stop_after() {
  local stage="$1"
  if [[ -n "${STOP_AFTER_STAGE}" && "${STOP_AFTER_STAGE}" == "${stage}" ]]; then
    echo "[SUGAR-TRAIN-PIPELINE] stopping after requested stage ${stage}" | tee -a "${LOG_PATH}"
    exit 0
  fi
}

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  echo "[SUGAR-TRAIN-PIPELINE] running environment preflight" | tee -a "${LOG_PATH}"
  ROOT_DIR="${ROOT_DIR}" SUGAR_DIR="${SUGAR_DIR}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/scripts/sugar/preflight_official_sugar_env.sh" \
    2>&1 | tee -a "${LOG_PATH}"
fi

{
  echo "[SUGAR-TRAIN-PIPELINE] host=$(hostname)"
  echo "[SUGAR-TRAIN-PIPELINE] root=${ROOT_DIR}"
  echo "[SUGAR-TRAIN-PIPELINE] sugar_dir=${SUGAR_DIR}"
  echo "[SUGAR-TRAIN-PIPELINE] task=${TASK_NAME}"
  echo "[SUGAR-TRAIN-PIPELINE] output_dir=${OUTPUT_DIR}"
  echo "[SUGAR-TRAIN-PIPELINE] start_stage=${START_STAGE}"
  echo "[SUGAR-TRAIN-PIPELINE] stop_after_stage=${STOP_AFTER_STAGE}"
  echo "[SUGAR-TRAIN-PIPELINE] refiner_num_envs=${REFINER_NUM_ENVS}"
  echo "[SUGAR-TRAIN-PIPELINE] refiner_max_iterations=${REFINER_MAX_ITERATIONS}"
  echo "[SUGAR-TRAIN-PIPELINE] refiner_final_iteration=${REFINER_FINAL_ITERATION}"
  echo "[SUGAR-TRAIN-PIPELINE] refiner_resume_checkpoint=${REFINER_RESUME_CHECKPOINT}"
  echo "[SUGAR-TRAIN-PIPELINE] tracker_num_envs=${TRACKER_NUM_ENVS}"
  echo "[SUGAR-TRAIN-PIPELINE] tracker_max_iterations=${TRACKER_MAX_ITERATIONS}"
  echo "[SUGAR-TRAIN-PIPELINE] tracker_final_iteration=${TRACKER_FINAL_ITERATION}"
  echo "[SUGAR-TRAIN-PIPELINE] tracker_resume_checkpoint=${TRACKER_RESUME_CHECKPOINT}"
  echo "[SUGAR-TRAIN-PIPELINE] generator_num_epochs=${GENERATOR_NUM_EPOCHS}"
  echo "[SUGAR-TRAIN-PIPELINE] disable_train_debug_vis=${SUGAR_DISABLE_TRAIN_DEBUG_VIS}"
  echo "[SUGAR-TRAIN-PIPELINE] disable_fabric=${SUGAR_DISABLE_FABRIC}"
  echo "[SUGAR-TRAIN-PIPELINE] disable_renderer_multigpu=${SUGAR_DISABLE_RENDERER_MULTIGPU}"
  echo "[SUGAR-TRAIN-PIPELINE] disable_renderer=${SUGAR_DISABLE_RENDERER}"
  echo "[SUGAR-TRAIN-PIPELINE] experience=${SUGAR_EXPERIENCE}"
  echo "[SUGAR-TRAIN-PIPELINE] sugar_commit=$(git rev-parse HEAD)"
  if [[ -f "${ROOT_DIR}/IsaacLab/VERSION" ]]; then
    echo "[SUGAR-TRAIN-PIPELINE] isaaclab_version=v$(tr -d '[:space:]' < "${ROOT_DIR}/IsaacLab/VERSION")-curiosity-glue"
  fi
  echo "[SUGAR-TRAIN-PIPELINE] command starts at $(date '+%F %T')"
  echo "[SUGAR-TRAIN-PIPELINE] log=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

mkdir -p "${OUTPUT_DIR}/ckpts"

fabric_args=()
if [[ "${SUGAR_DISABLE_FABRIC}" == "1" ]]; then
  fabric_args=(--disable_fabric)
fi
renderer_args=()
if [[ "${SUGAR_DISABLE_RENDERER}" == "1" ]]; then
  renderer_args=(--kit_args "--/renderer/enabled= --/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1")
elif [[ "${SUGAR_DISABLE_RENDERER_MULTIGPU}" == "1" ]]; then
  renderer_args=(--kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1")
fi
experience_args=()
if [[ -n "${SUGAR_EXPERIENCE}" ]]; then
  require_file "IsaacLab experience" "${SUGAR_EXPERIENCE}"
  experience_args=(--experience "${SUGAR_EXPERIENCE}")
fi

if should_run_stage refiner_train; then
  validate_resume_request refiner "${REFINER_RESUME_CHECKPOINT}" "${REFINER_MAX_ITERATIONS}" "${REFINER_FINAL_ITERATION}"
  refiner_resume_args=()
  if [[ -n "${REFINER_RESUME_CHECKPOINT}" ]]; then
    refiner_resume_args=(--resume_checkpoint_path "${REFINER_RESUME_CHECKPOINT}")
  fi
  run_logged refiner_train \
    "${PYTHON_BIN}" scripts/sugar_rl/train.py \
      --task "Sugar-G129dof-${TASK_NAME}-Refiner" \
      --num_envs "${REFINER_NUM_ENVS}" \
      --log_dir "${OUTPUT_DIR}/logs/refiner" \
      --max_iterations "${REFINER_MAX_ITERATIONS}" \
      --motion_folder "data/${TASK_NAME}" \
      "${refiner_resume_args[@]}" \
      "${fabric_args[@]}" \
      "${renderer_args[@]}" \
      "${experience_args[@]}" \
      --headless
  require_file "full refiner final checkpoint" "${OUTPUT_DIR}/logs/refiner/model_${REFINER_FINAL_ITERATION}.pt"
  cp "${OUTPUT_DIR}/logs/refiner/model_${REFINER_FINAL_ITERATION}.pt" "${OUTPUT_DIR}/ckpts/refiner.pt"
  require_file "exported refiner checkpoint" "${OUTPUT_DIR}/ckpts/refiner.pt"
  maybe_stop_after refiner_train
fi

if should_run_stage refiner_rollout; then
  require_file "exported refiner checkpoint" "${OUTPUT_DIR}/ckpts/refiner.pt"
  run_logged refiner_rollout \
    "${PYTHON_BIN}" scripts/sugar_rl/play.py \
      --task "Sugar-G129dof-${TASK_NAME}-Refiner-Rollout" \
      --num_envs "${REFINER_ROLLOUT_NUM_ENVS}" \
      --checkpoint "${OUTPUT_DIR}/ckpts/refiner.pt" \
      --rollout_dir "${OUTPUT_DIR}/rollout_datasets/refiner/raw_npz" \
      --motion_folder "data/${TASK_NAME}" \
      "${fabric_args[@]}" \
      "${renderer_args[@]}" \
      "${experience_args[@]}" \
      --headless
  require_glob_nonempty "refiner rollout complete trajectories" "${OUTPUT_DIR}/rollout_datasets/refiner/raw_npz/trajectory_complete/*.npz"
  maybe_stop_after refiner_rollout
fi

if should_run_stage process_refiner_rollout; then
  require_glob_nonempty "refiner rollout complete trajectories" "${OUTPUT_DIR}/rollout_datasets/refiner/raw_npz/trajectory_complete/*.npz"
  run_logged process_refiner_rollout \
    "${PYTHON_BIN}" scripts/sugar_rl/process_refiner_rollout.py \
      --data_dir "${OUTPUT_DIR}/rollout_datasets/refiner" \
      --task_name "${TASK_NAME}"
  require_dir "refiner processed RL dataset" "${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset"
  maybe_stop_after process_refiner_rollout
fi

if should_run_stage tracker_train; then
  require_file "exported refiner checkpoint" "${OUTPUT_DIR}/ckpts/refiner.pt"
  require_dir "refiner processed RL dataset" "${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset"
  validate_resume_request tracker "${TRACKER_RESUME_CHECKPOINT}" "${TRACKER_MAX_ITERATIONS}" "${TRACKER_FINAL_ITERATION}"
  tracker_resume_args=()
  if [[ -n "${TRACKER_RESUME_CHECKPOINT}" ]]; then
    tracker_resume_args=(--resume_checkpoint_path "${TRACKER_RESUME_CHECKPOINT}")
  fi
  run_logged tracker_train \
    "${PYTHON_BIN}" scripts/sugar_rl/train.py \
      --task "Sugar-G129dof-${TASK_NAME}-Tracker" \
      --num_envs "${TRACKER_NUM_ENVS}" \
      --teacher_ckpt "${OUTPUT_DIR}/ckpts/refiner.pt" \
      --motion_folder "${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset" \
      --teacher_motion_folder "data/${TASK_NAME}" \
      --log_dir "${OUTPUT_DIR}/logs/tracker" \
      --max_iterations "${TRACKER_MAX_ITERATIONS}" \
      "${tracker_resume_args[@]}" \
      "${fabric_args[@]}" \
      "${renderer_args[@]}" \
      "${experience_args[@]}" \
      --headless
  require_file "tracker final checkpoint" "${OUTPUT_DIR}/logs/tracker/model_${TRACKER_FINAL_ITERATION}.pt"
  cp "${OUTPUT_DIR}/logs/tracker/model_${TRACKER_FINAL_ITERATION}.pt" "${OUTPUT_DIR}/ckpts/tracker.pt"
  require_file "exported tracker checkpoint" "${OUTPUT_DIR}/ckpts/tracker.pt"
  maybe_stop_after tracker_train
fi

if should_run_stage tracker_rollout; then
  require_file "exported tracker checkpoint" "${OUTPUT_DIR}/ckpts/tracker.pt"
  require_file "exported refiner checkpoint" "${OUTPUT_DIR}/ckpts/refiner.pt"
  require_dir "refiner processed RL dataset" "${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset"
  run_logged tracker_rollout \
    "${PYTHON_BIN}" scripts/sugar_rl/play.py \
      --task "Sugar-G129dof-${TASK_NAME}-Tracker-Rollout" \
      --checkpoint "${OUTPUT_DIR}/ckpts/tracker.pt" \
      --num_envs "${TRACKER_ROLLOUT_NUM_ENVS}" \
      --rollout_dir "${OUTPUT_DIR}/rollout_datasets/tracker/raw_npz" \
      --motion_folder "${OUTPUT_DIR}/rollout_datasets/refiner/rl_dataset" \
      --teacher_motion_folder "data/${TASK_NAME}" \
      --teacher_ckpt "${OUTPUT_DIR}/ckpts/refiner.pt" \
      "${fabric_args[@]}" \
      "${renderer_args[@]}" \
      "${experience_args[@]}" \
      --headless
  require_glob_nonempty "tracker rollout complete trajectories" "${OUTPUT_DIR}/rollout_datasets/tracker/raw_npz/trajectory_complete/*.npz"
  maybe_stop_after tracker_rollout
fi

if should_run_stage process_tracker_rollout; then
  require_glob_nonempty "tracker rollout complete trajectories" "${OUTPUT_DIR}/rollout_datasets/tracker/raw_npz/trajectory_complete/*.npz"
  run_logged process_tracker_rollout \
    "${PYTHON_BIN}" scripts/sugar_rl/process_tracker_rollout.py \
      --data_dir "${OUTPUT_DIR}/rollout_datasets/tracker"
  require_dir "tracker processed IL dataset" "${OUTPUT_DIR}/rollout_datasets/tracker/il_dataset"
  maybe_stop_after process_tracker_rollout
fi

if should_run_stage generator_train; then
  require_dir "tracker processed IL dataset" "${OUTPUT_DIR}/rollout_datasets/tracker/il_dataset"
  case "${TASK_NAME}" in
    CarryBox | PickBox | PushBox)
      USE_TARGET="True"
      ;;
    PickBottle | StandBottle | SitChair)
      USE_TARGET="False"
      ;;
    *)
      echo "Unsupported SUGAR task for generator target flag: ${TASK_NAME}" >&2
      exit 6
      ;;
  esac
  run_logged generator_train \
    "${PYTHON_BIN}" scripts/sugar_il/train.py \
      --config-name train_generator_workspace.yaml \
      "task=${TASK_NAME}" \
      "use_target=${USE_TARGET}" \
      "num_epochs=${GENERATOR_NUM_EPOCHS}" \
      "log_path=${OUTPUT_DIR}/logs/generator" \
      "dataset_path=${OUTPUT_DIR}/rollout_datasets/tracker/il_dataset"
  require_file "generator final checkpoint" "${OUTPUT_DIR}/logs/generator/epoch_checkpoints/epoch=$((GENERATOR_NUM_EPOCHS - 1)).ckpt"
  cp "${OUTPUT_DIR}/logs/generator/epoch_checkpoints/epoch=$((GENERATOR_NUM_EPOCHS - 1)).ckpt" "${OUTPUT_DIR}/ckpts/generator.ckpt"
  require_file "exported generator checkpoint" "${OUTPUT_DIR}/ckpts/generator.ckpt"
  maybe_stop_after generator_train
fi

echo "[SUGAR-TRAIN-PIPELINE] pipeline finished at $(date '+%F %T')" | tee -a "${LOG_PATH}"
