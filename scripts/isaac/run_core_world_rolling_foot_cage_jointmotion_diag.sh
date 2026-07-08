#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run rolling-foot joint-motion diagnostic on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-45}"
python3 -m py_compile scripts/isaac/build_core_world_rolling_foot_cage_carrier.py

STAMP="${STAMP:-20260705_rolling_foot_cage_jointmotion_diag10}"
OUTPUT_DIR="experiments/outputs/core_world_rolling_foot_cage_carrier/${STAMP}"
echo "[BATCH] rolling_foot_jointmotion stamp=${STAMP}"

"${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_rolling_foot_cage_carrier.py \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE:-cpu}" \
  --kit_args "${KIT_ARGS}" \
  --steps 520 \
  --settle-steps 180 \
  --drive-steps 220 \
  --target-x 0.30 \
  --wheel-control-mode "${WHEEL_CONTROL_MODE:-velocity}" \
  --wheel-velocity 1.2 \
  --wheel-effort "${WHEEL_EFFORT:-200.0}" \
  --payload-mass 1.0 \
  --wheel-damping 350.0 \
  --wheel-max-force 1200.0 \
  --wheel-static-friction 3.0 \
  --wheel-dynamic-friction 2.5 \
  --static-friction 2.0 \
  --dynamic-friction 1.8 \
  --output-dir "${OUTPUT_DIR}"
