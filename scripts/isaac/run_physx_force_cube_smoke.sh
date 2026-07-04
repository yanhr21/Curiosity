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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/physx_force_cube_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/physx_force_cube_smoke/${STAMP}}"
STEPS="${STEPS:-180}"
FORCE_X="${FORCE_X:-180.0}"
FORCE_Z="${FORCE_Z:-0.0}"
STEP_MODE="${STEP_MODE:-sim_step}"
REUSE_STAGE="${REUSE_STAGE:-0}"
DEVICE="${DEVICE:-cpu}"
RENDER="${RENDER:-0}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_physx_force_cube_smoke.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --force-x "${FORCE_X}" \
  --force-z "${FORCE_Z}" \
  --step-mode "${STEP_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  $(if [[ "${REUSE_STAGE}" == "1" ]]; then echo "--reuse-stage"; fi) \
  2>&1 | tee "${LOG_DIR}/physx_force_cube_smoke_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/physx_force_cube_smoke_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
