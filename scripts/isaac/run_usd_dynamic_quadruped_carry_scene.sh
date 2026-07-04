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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/usd_dynamic_quadruped_carry_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/usd_dynamic_quadruped_carry_scene/${STAMP}}"
STEPS="${STEPS:-900}"
PAYLOAD_MASS="${PAYLOAD_MASS:-4.0}"
TARGET_X="${TARGET_X:-1.0}"
GAIT_FREQUENCY="${GAIT_FREQUENCY:-1.2}"
HIP_AMPLITUDE_DEG="${HIP_AMPLITUDE_DEG:-18.0}"
KNEE_AMPLITUDE_DEG="${KNEE_AMPLITUDE_DEG:-16.0}"
ARTICULATION_ROOT="${ARTICULATION_ROOT:-0}"
CONTROL_MODE="${CONTROL_MODE:-usd_drive_attr}"
DEVICE="${DEVICE:-cuda:0}"
RENDER="${RENDER:-0}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

LOG_PATH="${LOG_DIR}/usd_dynamic_quadruped_carry_scene_${STAMP}.log"
EXTRA_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi
if [[ "${ARTICULATION_ROOT}" == "1" ]]; then
  EXTRA_ARGS+=(--articulation-root)
fi

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_usd_dynamic_quadruped_carry_scene.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --target-x "${TARGET_X}" \
  --gait-frequency "${GAIT_FREQUENCY}" \
  --hip-amplitude-deg "${HIP_AMPLITUDE_DEG}" \
  --knee-amplitude-deg "${KNEE_AMPLITUDE_DEG}" \
  --control-mode "${CONTROL_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
