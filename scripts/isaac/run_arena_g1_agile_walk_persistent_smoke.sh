#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/arena_g1_agile_walk_persistent_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/arena_g1_agile_walk_persistent_smoke/${STAMP}}"

export ARENA_LOCAL_ASSET_ROOT="${ARENA_LOCAL_ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
export ARENA_G1_29DOF_WITH_HAND_USD="${ARENA_G1_29DOF_WITH_HAND_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
export OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
export PYTHONPATH="${ROOT_DIR}/external/IsaacLab-Arena:${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab:${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab_tasks:${PYTHONPATH:-}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_DIR}/arena_g1_agile_walk_persistent_smoke_${STAMP}.log"
echo "[CONFIG] STAMP=${STAMP} persistent Arena G1 AGILE walk STEPS=${STEPS:-260} COMMAND=(${COMMAND_X:-0.25},${COMMAND_Y:-0.0},${COMMAND_YAW:-0.0}) COMMAND_START_STEP=${COMMAND_START_STEP:-80}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/run_arena_g1_agile_walk_persistent_smoke.py" \
  --steps "${STEPS:-260}" \
  --warmup-steps "${WARMUP_STEPS:-40}" \
  --command-start-step "${COMMAND_START_STEP:-80}" \
  --command "${COMMAND_X:-0.25}" "${COMMAND_Y:-0.0}" "${COMMAND_YAW:-0.0}" \
  --base-height-command "${BASE_HEIGHT_COMMAND:-0.75}" \
  --min-root-height "${MIN_ROOT_HEIGHT:-0.40}" \
  --max-tilt "${MAX_TILT:-0.85}" \
  --min-commanded-travel-x "${MIN_COMMANDED_TRAVEL_X:-0.05}" \
  --ground-usd "${GROUND_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd}" \
  --device "${DEVICE:-cuda:0}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
