#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_simapp_quasistatic_walker_carry}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_simapp_quasistatic_walker_carry/${STAMP}}"
STEPS="${STEPS:-420}"
TARGET_X="${TARGET_X:-0.38}"
PAYLOAD_MASS="${PAYLOAD_MASS:-8.0}"
PAYLOAD_COM_X="${PAYLOAD_COM_X:-0.04}"
ROBOT_MASS="${ROBOT_MASS:-48.0}"
ROBOT_HEIGHT="${ROBOT_HEIGHT:-1.20}"
ARM_LENGTH="${ARM_LENGTH:-0.52}"
MAX_PAYLOAD="${MAX_PAYLOAD:-16.0}"
BASE_SPEED="${BASE_SPEED:-0.30}"
GAIT_FREQUENCY="${GAIT_FREQUENCY:-1.15}"
DEVICE="${DEVICE:-cpu}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
export ISAAC_SIMAPP_EXPERIENCE="${ISAAC_SIMAPP_EXPERIENCE:-/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_simapp_quasistatic_walker_carry.py" \
  --steps "${STEPS}" \
  --target-x "${TARGET_X}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --payload-com-x "${PAYLOAD_COM_X}" \
  --robot-mass "${ROBOT_MASS}" \
  --robot-height "${ROBOT_HEIGHT}" \
  --arm-length "${ARM_LENGTH}" \
  --max-payload "${MAX_PAYLOAD}" \
  --base-speed "${BASE_SPEED}" \
  --gait-frequency "${GAIT_FREQUENCY}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/core_world_simapp_quasistatic_walker_carry_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/core_world_simapp_quasistatic_walker_carry_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
