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
echo "[CONFIG] STAMP=${STAMP} dedicated stance_translate tray/free-box launcher STEPS=${STEPS:-500} TARGET_X=${TARGET_X:-0.035} SETTLE_STEPS=${SETTLE_STEPS:-250} RAMP_STEPS=${RAMP_STEPS:-200} PAYLOAD_MASS=${PAYLOAD_MASS:-2.0} TORSO_MASS=${TORSO_MASS:-44.0} TRAY_SIZE=${TRAY_SIZE_X:-0.70},${TRAY_SIZE_Y:-0.50},${TRAY_SIZE_Z:-0.04} ENABLE_TRAY_LID=${ENABLE_TRAY_LID:-1} ENABLE_BALANCE_LEG_SERVO=${ENABLE_BALANCE_LEG_SERVO:-0}"
EXTRA_ARGS=()
if [[ "${ENABLE_TRAY_LID:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-tray-lid)
fi
if [[ "${ENABLE_BALANCE_LEG_SERVO:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-balance-leg-servo)
fi
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_prismatic_carrier_stand.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE:-cpu}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS:-500}" \
  --payload-mode tray_contact_free_box \
  --payload-mass "${PAYLOAD_MASS:-2.0}" \
  --torso-mass "${TORSO_MASS:-44.0}" \
  --torso-z "${TORSO_Z:-0.60}" \
  --payload-local-x "${PAYLOAD_LOCAL_X:-0.08}" \
  --payload-local-z "${PAYLOAD_LOCAL_Z:-0.04}" \
  --tray-local-x "${TRAY_LOCAL_X:-0.08}" \
  --tray-local-z "${TRAY_LOCAL_Z:-0.11}" \
  --tray-size "${TRAY_SIZE_X:-0.70}" "${TRAY_SIZE_Y:-0.50}" "${TRAY_SIZE_Z:-0.04}" \
  --tray-rail-height "${TRAY_RAIL_HEIGHT:-0.16}" \
  --tray-rail-thickness "${TRAY_RAIL_THICKNESS:-0.05}" \
  --tray-mass "${TRAY_MASS:-6.0}" \
  --tray-lid-clearance "${TRAY_LID_CLEARANCE:-0.025}" \
  --tray-lid-thickness "${TRAY_LID_THICKNESS:-0.035}" \
  --tray-lid-mass "${TRAY_LID_MASS:-2.0}" \
  --motion-mode stance_translate \
  --enable-horizontal-legs \
  --target-x "${TARGET_X:-0.035}" \
  --step-length "${STEP_LENGTH:-0.08}" \
  --step-height "${STEP_HEIGHT:-0.09}" \
  --gait-period-steps "${GAIT_PERIOD_STEPS:-160}" \
  --swing-fraction "${SWING_FRACTION:-0.22}" \
  --settle-steps "${SETTLE_STEPS:-250}" \
  --ramp-steps "${RAMP_STEPS:-200}" \
  --stance-half-length "${STANCE_HALF_LENGTH:-0.44}" \
  --stance-half-width "${STANCE_HALF_WIDTH:-0.34}" \
  --foot-length "${FOOT_LENGTH:-0.52}" \
  --foot-width "${FOOT_WIDTH:-0.26}" \
  --foot-height "${FOOT_HEIGHT:-0.055}" \
  --foot-mass "${FOOT_MASS:-5.0}" \
  --leg-target "${LEG_TARGET:--0.50}" \
  --leg-lower "${LEG_LOWER:--0.75}" \
  --leg-upper "${LEG_UPPER:--0.25}" \
  --leg-stiffness "${LEG_STIFFNESS:-24000.0}" \
  --leg-damping "${LEG_DAMPING:-2400.0}" \
  --leg-max-force "${LEG_MAX_FORCE:-35000.0}" \
  --balance-roll-gain "${BALANCE_ROLL_GAIN:-0.0}" \
  --balance-pitch-gain "${BALANCE_PITCH_GAIN:-0.0}" \
  --balance-max-correction "${BALANCE_MAX_CORRECTION:-0.05}" \
  --x-slide-limit "${X_SLIDE_LIMIT:-0.08}" \
  --x-slide-stiffness "${X_SLIDE_STIFFNESS:-16000.0}" \
  --x-slide-damping "${X_SLIDE_DAMPING:-1800.0}" \
  --x-slide-max-force "${X_SLIDE_MAX_FORCE:-24000.0}" \
  --static-friction "${STATIC_FRICTION:-4.5}" \
  --dynamic-friction "${DYNAMIC_FRICTION:-4.0}" \
  --fall-z "${FALL_Z:-0.42}" \
  --drop-z "${DROP_Z:-0.24}" \
  --max-stand-drift "${MAX_STAND_DRIFT:-0.08}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
