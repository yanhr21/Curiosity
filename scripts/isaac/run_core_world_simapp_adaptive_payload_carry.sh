#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_simapp_adaptive_payload_carry}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_simapp_adaptive_payload_carry/${STAMP}}"
STEPS="${STEPS:-360}"
TARGET_X="${TARGET_X:-1.2}"
PAYLOAD_MASS="${PAYLOAD_MASS:-6.0}"
PAYLOAD_SIZE_X="${PAYLOAD_SIZE_X:-0.34}"
PAYLOAD_SIZE_Y="${PAYLOAD_SIZE_Y:-0.24}"
PAYLOAD_SIZE_Z="${PAYLOAD_SIZE_Z:-0.24}"
PAYLOAD_COM_X="${PAYLOAD_COM_X:-0.0}"
ROBOT_HEIGHT="${ROBOT_HEIGHT:-1.35}"
ROBOT_MASS="${ROBOT_MASS:-48.0}"
ARM_LENGTH="${ARM_LENGTH:-0.55}"
MAX_PAYLOAD="${MAX_PAYLOAD:-16.0}"
BASE_SPEED="${BASE_SPEED:-0.34}"
GAIT_FREQUENCY="${GAIT_FREQUENCY:-1.25}"
FOOT_CLEARANCE="${FOOT_CLEARANCE:-0.035}"
DEVICE="${DEVICE:-cpu}"
PRESET_SWEEP="${PRESET_SWEEP:-none}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
export ISAAC_SIMAPP_EXPERIENCE="${ISAAC_SIMAPP_EXPERIENCE:-/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_simapp_adaptive_payload_carry.py" \
  --steps "${STEPS}" \
  --target-x "${TARGET_X}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --payload-size-x "${PAYLOAD_SIZE_X}" \
  --payload-size-y "${PAYLOAD_SIZE_Y}" \
  --payload-size-z "${PAYLOAD_SIZE_Z}" \
  --payload-com-x "${PAYLOAD_COM_X}" \
  --robot-height "${ROBOT_HEIGHT}" \
  --robot-mass "${ROBOT_MASS}" \
  --arm-length "${ARM_LENGTH}" \
  --max-payload "${MAX_PAYLOAD}" \
  --base-speed "${BASE_SPEED}" \
  --gait-frequency "${GAIT_FREQUENCY}" \
  --foot-clearance "${FOOT_CLEARANCE}" \
  --device "${DEVICE}" \
  --preset-sweep "${PRESET_SWEEP}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/core_world_simapp_adaptive_payload_carry_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/core_world_simapp_adaptive_payload_carry_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
