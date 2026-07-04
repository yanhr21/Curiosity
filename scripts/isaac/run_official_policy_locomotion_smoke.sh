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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/official_policy_locomotion_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/official_policy_locomotion_smoke/${STAMP}}"
ROBOT="${ROBOT:-h1}"
STEPS="${STEPS:-240}"
COMMAND_X="${COMMAND_X:-1.0}"
COMMAND_Y="${COMMAND_Y:-0.0}"
COMMAND_YAW="${COMMAND_YAW:-0.0}"
DEVICE="${DEVICE:-cuda}"
RENDER="${RENDER:-0}"
ASSET_ROOT="${ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
PAYLOAD_MODE="${PAYLOAD_MODE:-none}"
PAYLOAD_MASS="${PAYLOAD_MASS:-2.0}"
PAYLOAD_SIZE_X="${PAYLOAD_SIZE_X:-0.28}"
PAYLOAD_SIZE_Y="${PAYLOAD_SIZE_Y:-0.18}"
PAYLOAD_SIZE_Z="${PAYLOAD_SIZE_Z:-0.14}"
PAYLOAD_OFFSET_X="${PAYLOAD_OFFSET_X:-0.22}"
PAYLOAD_OFFSET_Y="${PAYLOAD_OFFSET_Y:-0.0}"
PAYLOAD_OFFSET_Z="${PAYLOAD_OFFSET_Z:-0.08}"
export ISAACSIM_ASSET_ROOT="${ASSET_ROOT}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/persistent/isaac/asset_root/default=${ASSET_ROOT} --/persistent/isaac/asset_root/timeout=1.0 --/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/run_official_policy_locomotion_smoke.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --robot "${ROBOT}" \
  --steps "${STEPS}" \
  --command "${COMMAND_X}" "${COMMAND_Y}" "${COMMAND_YAW}" \
  --asset-root "${ASSET_ROOT}" \
  --payload-mode "${PAYLOAD_MODE}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --payload-size "${PAYLOAD_SIZE_X}" "${PAYLOAD_SIZE_Y}" "${PAYLOAD_SIZE_Z}" \
  --payload-offset "${PAYLOAD_OFFSET_X}" "${PAYLOAD_OFFSET_Y}" "${PAYLOAD_OFFSET_Z}" \
  --output-dir "${OUTPUT_DIR}" \
  $(if [[ "${RENDER}" == "1" ]]; then echo "--render"; fi) \
  "$@" \
  2>&1 | tee "${LOG_DIR}/official_policy_locomotion_smoke_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/official_policy_locomotion_smoke_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
