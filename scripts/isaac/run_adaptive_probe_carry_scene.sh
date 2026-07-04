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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/adaptive_probe_carry_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/adaptive_probe_carry_scene/${STAMP}}"
STEPS="${STEPS:-360}"
BOX_MASS="${BOX_MASS:-8.0}"
BOX_SIZE_X="${BOX_SIZE_X:-0.58}"
BOX_SIZE_Y="${BOX_SIZE_Y:-0.38}"
BOX_SIZE_Z="${BOX_SIZE_Z:-0.36}"
BOX_COM_X="${BOX_COM_X:-0.04}"
BOX_COM_Y="${BOX_COM_Y:-0.0}"
BOX_COM_Z="${BOX_COM_Z:-0.0}"
ROBOT_HEIGHT="${ROBOT_HEIGHT:-1.45}"
ROBOT_MASS="${ROBOT_MASS:-52.0}"
ARM_LENGTH="${ARM_LENGTH:-0.58}"
MAX_PAYLOAD="${MAX_PAYLOAD:-16.0}"
TARGET_X="${TARGET_X:-2.15}"
DEVICE="${DEVICE:-cuda:0}"
RENDER="${RENDER:-0}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

LOG_PATH="${LOG_DIR}/adaptive_probe_carry_scene_${STAMP}.log"
EXTRA_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_adaptive_probe_carry_scene.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --box-mass "${BOX_MASS}" \
  --box-size "${BOX_SIZE_X}" "${BOX_SIZE_Y}" "${BOX_SIZE_Z}" \
  --box-com-offset "${BOX_COM_X}" "${BOX_COM_Y}" "${BOX_COM_Z}" \
  --robot-height "${ROBOT_HEIGHT}" \
  --robot-mass "${ROBOT_MASS}" \
  --arm-length "${ARM_LENGTH}" \
  --max-payload "${MAX_PAYLOAD}" \
  --target-x "${TARGET_X}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
