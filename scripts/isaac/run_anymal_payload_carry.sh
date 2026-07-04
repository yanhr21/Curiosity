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

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/anymal_payload_carry}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/anymal_payload_carry/${STAMP}}"
TASK="${TASK:-Isaac-Velocity-Flat-Anymal-C-Play-v0}"
CHECKPOINT="${CHECKPOINT:-}"
USE_PRETRAINED_CHECKPOINT="${USE_PRETRAINED_CHECKPOINT:-1}"
STEPS="${STEPS:-600}"
NUM_ENVS="${NUM_ENVS:-1}"
PAYLOAD_MASS="${PAYLOAD_MASS:-5.0}"
PAYLOAD_COM_X="${PAYLOAD_COM_X:-0.18}"
PAYLOAD_COM_Y="${PAYLOAD_COM_Y:-0.0}"
PAYLOAD_COM_Z="${PAYLOAD_COM_Z:-0.06}"
BOX_SIZE_X="${BOX_SIZE_X:-0.55}"
BOX_SIZE_Y="${BOX_SIZE_Y:-0.35}"
BOX_SIZE_Z="${BOX_SIZE_Z:-0.25}"
COMMAND_X="${COMMAND_X:-0.35}"
COMMAND_Y="${COMMAND_Y:-0.0}"
COMMAND_YAW="${COMMAND_YAW:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
RENDER="${RENDER:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
PHYSICS_BACKEND="${PHYSICS_BACKEND:-physx}"
USE_REMOTE_ASSETS="${USE_REMOTE_ASSETS:-0}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
elif [[ "${USE_PRETRAINED_CHECKPOINT}" == "1" ]]; then
  EXTRA_ARGS+=(--use-pretrained-checkpoint)
fi
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_cameras)
fi
if [[ "${DISABLE_FABRIC}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-fabric)
fi
if [[ "${USE_REMOTE_ASSETS}" == "1" ]]; then
  EXTRA_ARGS+=(--use-remote-assets)
fi

LOG_PATH="${LOG_DIR}/anymal_payload_carry_${STAMP}.log"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}/external/IsaacLab-Arena:${PYTHONPATH:-}" "${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/run_anymal_payload_carry.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --task "${TASK}" \
  --steps "${STEPS}" \
  --num-envs "${NUM_ENVS}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --payload-com "${PAYLOAD_COM_X}" "${PAYLOAD_COM_Y}" "${PAYLOAD_COM_Z}" \
  --box-size "${BOX_SIZE_X}" "${BOX_SIZE_Y}" "${BOX_SIZE_Z}" \
  --command "${COMMAND_X}" "${COMMAND_Y}" "${COMMAND_YAW}" \
  --physics-backend "${PHYSICS_BACKEND}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
