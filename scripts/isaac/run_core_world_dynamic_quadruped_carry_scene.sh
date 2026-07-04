#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_dynamic_quadruped_carry_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_dynamic_quadruped_carry_scene/${STAMP}}"
STEPS="${STEPS:-480}"
PAYLOAD_MASS="${PAYLOAD_MASS:-4.0}"
TARGET_X="${TARGET_X:-0.8}"
GAIT_FREQUENCY="${GAIT_FREQUENCY:-1.1}"
HIP_AMPLITUDE_DEG="${HIP_AMPLITUDE_DEG:-18.0}"
KNEE_AMPLITUDE_DEG="${KNEE_AMPLITUDE_DEG:-16.0}"
DEVICE="${DEVICE:-cpu}"
RENDER="${RENDER:-0}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

LOG_PATH="${LOG_DIR}/core_world_dynamic_quadruped_carry_scene_${STAMP}.log"
EXTRA_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --target-x "${TARGET_X}" \
  --gait-frequency "${GAIT_FREQUENCY}" \
  --hip-amplitude-deg "${HIP_AMPLITUDE_DEG}" \
  --knee-amplitude-deg "${KNEE_AMPLITUDE_DEG}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
