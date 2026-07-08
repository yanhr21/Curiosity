#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run rolling-foot cage carrier batch on login/management node: $(hostname)" >&2
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

run_diag() {
  local diag="$1"
  local target_x="$2"
  local wheel_velocity="$3"
  local payload_mass="$4"
  local stamp_prefix="${STAMP_PREFIX:-20260705_rolling_foot_cage}"
  local stamp="${stamp_prefix}_${diag}_target${target_x}_vel${wheel_velocity}_mass${payload_mass}"
  local output_dir="experiments/outputs/core_world_rolling_foot_cage_carrier/${stamp}"
  echo "[BATCH] ${diag} rolling_foot target=${target_x} wheel_velocity=${wheel_velocity} payload_mass=${payload_mass}"
  "${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_rolling_foot_cage_carrier.py \
    --viz none \
    --experience "${EXPERIENCE}" \
    --device "${DEVICE:-cpu}" \
    --kit_args "${KIT_ARGS}" \
    --steps 820 \
    --settle-steps 180 \
    --drive-steps 420 \
    --target-x "${target_x}" \
    --wheel-velocity "${wheel_velocity}" \
    --payload-mass "${payload_mass}" \
    --wheel-damping 350.0 \
    --wheel-max-force 1200.0 \
    --wheel-static-friction 3.0 \
    --wheel-dynamic-friction 2.5 \
    --static-friction 2.0 \
    --dynamic-friction 1.8 \
    --output-dir "${output_dir}"
}

run_diag "${DIAG_A:-diag1}" 0.30 1.2 1.0
run_diag "${DIAG_B:-diag2}" -0.30 1.2 1.0
run_diag "${DIAG_C:-diag3}" 0.30 1.2 2.0
