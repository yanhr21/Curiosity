#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_cradle_cart_free_box_carry}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_cradle_cart_free_box_carry/${STAMP}}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_DIR}/core_world_cradle_cart_free_box_carry_${STAMP}.log"
echo "[CONFIG] STAMP=${STAMP} STEPS=${STEPS:-420} TARGET_X=${TARGET_X:-0.08} BOX_MASS=${BOX_MASS:-0.5}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_cradle_cart_free_box_carry.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE:-cpu}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS:-420}" \
  --settle-steps "${SETTLE_STEPS:-80}" \
  --carry-steps "${CARRY_STEPS:-260}" \
  --target-x "${TARGET_X:-0.08}" \
  --box-mass "${BOX_MASS:-0.5}" \
  --cradle-gap-x "${CRADLE_GAP_X:-0.025}" \
  --cradle-gap-y "${CRADLE_GAP_Y:-0.040}" \
  --wall-thickness "${WALL_THICKNESS:-0.030}" \
  --wall-height "${WALL_HEIGHT:-0.20}" \
  --deck-thickness "${DECK_THICKNESS:-0.035}" \
  --cart-z "${CART_Z:-0.13}" \
  --drive-stiffness "${DRIVE_STIFFNESS:-12000.0}" \
  --drive-damping "${DRIVE_DAMPING:-2500.0}" \
  --drive-max-force "${DRIVE_MAX_FORCE:-80000.0}" \
  --static-friction "${STATIC_FRICTION:-0.2}" \
  --dynamic-friction "${DYNAMIC_FRICTION:-0.1}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
