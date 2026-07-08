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
echo "[CONFIG] STAMP=${STAMP} prismatic cradle sync_inchworm TARGET_X=${TARGET_X:--0.23} STEPS=${STEPS:-2350}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_prismatic_carrier_stand.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE:-cpu}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS:-2350}" \
  --payload-mode cradle_free_box \
  --payload-mass "${PAYLOAD_MASS:-8.0}" \
  --torso-mass "${TORSO_MASS:-36.0}" \
  --torso-z "${TORSO_Z:-0.62}" \
  --payload-size "${PAYLOAD_SIZE_X:-0.34}" "${PAYLOAD_SIZE_Y:-0.24}" "${PAYLOAD_SIZE_Z:-0.24}" \
  --payload-local-x "${PAYLOAD_LOCAL_X:-0.50}" \
  --payload-local-z "${PAYLOAD_LOCAL_Z:-0.16}" \
  --cradle-clearance-x "${CRADLE_CLEARANCE_X:-0.025}" \
  --cradle-clearance-y "${CRADLE_CLEARANCE_Y:-0.040}" \
  --cradle-wall-height "${CRADLE_WALL_HEIGHT:-0.26}" \
  --cradle-wall-thickness "${CRADLE_WALL_THICKNESS:-0.030}" \
  --cradle-part-mass "${CRADLE_PART_MASS:-1.0}" \
  --motion-mode sync_inchworm \
  --enable-horizontal-legs \
  --target-x "${TARGET_X:--0.23}" \
  --step-length "${STEP_LENGTH:-0.06}" \
  --step-height "${STEP_HEIGHT:-0.05}" \
  --gait-period-steps "${GAIT_PERIOD_STEPS:-360}" \
  --swing-fraction "${SWING_FRACTION:-0.22}" \
  --sync-inchworm-min-cycles "${SYNC_INCHWORM_MIN_CYCLES:-5}" \
  --sync-inchworm-stride-override "${SYNC_INCHWORM_STRIDE_OVERRIDE:-0.07}" \
  --settle-steps "${SETTLE_STEPS:-260}" \
  --ramp-steps "${RAMP_STEPS:-260}" \
  --stance-half-length "${STANCE_HALF_LENGTH:-0.65}" \
  --stance-half-width "${STANCE_HALF_WIDTH:-0.24}" \
  --foot-length "${FOOT_LENGTH:-0.65}" \
  --foot-width "${FOOT_WIDTH:-0.18}" \
  --foot-height "${FOOT_HEIGHT:-0.055}" \
  --foot-mass "${FOOT_MASS:-2.8}" \
  --leg-target "${LEG_TARGET:--0.57}" \
  --leg-lower "${LEG_LOWER:--0.82}" \
  --leg-upper "${LEG_UPPER:--0.25}" \
  --leg-stiffness "${LEG_STIFFNESS:-32000.0}" \
  --leg-damping "${LEG_DAMPING:-3200.0}" \
  --leg-max-force "${LEG_MAX_FORCE:-56000.0}" \
  --x-slide-limit "${X_SLIDE_LIMIT:-0.20}" \
  --x-slide-stiffness "${X_SLIDE_STIFFNESS:-30000.0}" \
  --x-slide-damping "${X_SLIDE_DAMPING:-3000.0}" \
  --x-slide-max-force "${X_SLIDE_MAX_FORCE:-62000.0}" \
  --static-friction "${STATIC_FRICTION:-3.0}" \
  --dynamic-friction "${DYNAMIC_FRICTION:-2.5}" \
  --fall-z "${FALL_Z:-0.42}" \
  --drop-z "${DROP_Z:-0.24}" \
  --max-stand-drift "${MAX_STAND_DRIFT:-0.08}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
