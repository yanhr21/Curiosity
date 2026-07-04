#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_simapp_fixed_payload_carry}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_simapp_fixed_payload_carry/${STAMP}}"
STEPS="${STEPS:-240}"
TARGET_SPEED="${TARGET_SPEED:-0.30}"
PAYLOAD_MASS="${PAYLOAD_MASS:-4.0}"
JOINT_MODE="${JOINT_MODE:-center_weld}"
DEVICE="${DEVICE:-cpu}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
export ISAAC_SIMAPP_EXPERIENCE="${ISAAC_SIMAPP_EXPERIENCE:-/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_simapp_fixed_payload_carry.py" \
  --steps "${STEPS}" \
  --target-speed "${TARGET_SPEED}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --joint-mode "${JOINT_MODE}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/core_world_simapp_fixed_payload_carry_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/core_world_simapp_fixed_payload_carry_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
