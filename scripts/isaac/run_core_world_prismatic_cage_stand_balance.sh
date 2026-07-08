#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_prismatic_carrier_stand}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_prismatic_carrier_stand/${STAMP}}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_DIR}/core_world_prismatic_carrier_stand_${STAMP}.log"
echo "[CONFIG] STAMP=${STAMP} cage stand balance STEPS=${STEPS:-350} ROLL_GAIN=${BALANCE_ROLL_GAIN:--0.08} PITCH_GAIN=${BALANCE_PITCH_GAIN:--0.08}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_prismatic_carrier_stand.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE:-cpu}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS:-350}" \
  --payload-mode tray_contact_free_box \
  --payload-mass "${PAYLOAD_MASS:-1.0}" \
  --torso-mass "${TORSO_MASS:-52.0}" \
  --torso-z "${TORSO_Z:-0.58}" \
  --payload-local-x "${PAYLOAD_LOCAL_X:-0.03}" \
  --payload-local-z "${PAYLOAD_LOCAL_Z:-0.04}" \
  --tray-local-x "${TRAY_LOCAL_X:-0.03}" \
  --tray-local-z "${TRAY_LOCAL_Z:-0.07}" \
  --tray-size "${TRAY_SIZE_X:-0.72}" "${TRAY_SIZE_Y:-0.56}" "${TRAY_SIZE_Z:-0.04}" \
  --tray-rail-height "${TRAY_RAIL_HEIGHT:-0.34}" \
  --tray-rail-thickness "${TRAY_RAIL_THICKNESS:-0.055}" \
  --tray-mass "${TRAY_MASS:-8.0}" \
  --enable-tray-lid \
  --tray-lid-clearance "${TRAY_LID_CLEARANCE:-0.015}" \
  --tray-lid-thickness "${TRAY_LID_THICKNESS:-0.04}" \
  --tray-lid-mass "${TRAY_LID_MASS:-3.0}" \
  --motion-mode stand \
  --target-x 0.0 \
  --settle-steps 0 \
  --ramp-steps 1 \
  --stance-half-length "${STANCE_HALF_LENGTH:-0.50}" \
  --stance-half-width "${STANCE_HALF_WIDTH:-0.42}" \
  --foot-length "${FOOT_LENGTH:-0.62}" \
  --foot-width "${FOOT_WIDTH:-0.32}" \
  --foot-height "${FOOT_HEIGHT:-0.055}" \
  --foot-mass "${FOOT_MASS:-7.0}" \
  --leg-target "${LEG_TARGET:--0.50}" \
  --leg-lower "${LEG_LOWER:--0.75}" \
  --leg-upper "${LEG_UPPER:--0.25}" \
  --leg-stiffness "${LEG_STIFFNESS:-30000.0}" \
  --leg-damping "${LEG_DAMPING:-3500.0}" \
  --leg-max-force "${LEG_MAX_FORCE:-45000.0}" \
  --enable-balance-leg-servo \
  --balance-roll-gain "${BALANCE_ROLL_GAIN:--0.08}" \
  --balance-pitch-gain "${BALANCE_PITCH_GAIN:--0.08}" \
  --balance-max-correction "${BALANCE_MAX_CORRECTION:-0.04}" \
  --static-friction "${STATIC_FRICTION:-5.0}" \
  --dynamic-friction "${DYNAMIC_FRICTION:-4.5}" \
  --fall-z "${FALL_Z:-0.42}" \
  --drop-z "${DROP_Z:-0.24}" \
  --max-stand-drift "${MAX_STAND_DRIFT:-0.08}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
