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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/anymal_experimental_articulation_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/anymal_experimental_articulation_smoke/${STAMP}}"
STEPS="${STEPS:-180}"
WARMUP_STEPS="${WARMUP_STEPS:-8}"
TARGET_AMPLITUDE="${TARGET_AMPLITUDE:-0.22}"
TARGET_FREQUENCY="${TARGET_FREQUENCY:-0.8}"
DEVICE="${DEVICE:-cpu}"
RENDER="${RENDER:-0}"
ASSET_USD="${ASSET_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd}"
GROUND_USD="${GROUND_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

LOG_PATH="${LOG_DIR}/anymal_experimental_articulation_smoke_${STAMP}.log"
EXTRA_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/run_anymal_experimental_articulation_smoke.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --target-amplitude "${TARGET_AMPLITUDE}" \
  --target-frequency "${TARGET_FREQUENCY}" \
  --asset-usd "${ASSET_USD}" \
  --ground-usd "${GROUND_USD}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
