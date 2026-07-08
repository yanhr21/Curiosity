#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run cradle-cart contact baseline on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-30}"
python3 -m py_compile scripts/isaac/build_core_world_cradle_cart_free_box_carry.py

run_diag() {
  local diag="$1"
  local target_x="$2"
  local box_mass="$3"
  local static_friction="$4"
  local dynamic_friction="$5"
  local carry_steps="$6"
  local steps="$7"
  local stamp="20260705_cradle_cart_contact_${diag}_target${target_x}_mass${box_mass}_mu${static_friction}"
  local output_dir="experiments/outputs/core_world_cradle_cart_free_box_carry/${stamp}"
  echo "[BATCH] ${diag} target=${target_x} mass=${box_mass} mu=${static_friction}/${dynamic_friction} carry_steps=${carry_steps}"
  "${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_cradle_cart_free_box_carry.py \
    --viz none \
    --experience "${EXPERIENCE}" \
    --device "${DEVICE:-cpu}" \
    --kit_args "${KIT_ARGS}" \
    --steps "${steps}" \
    --settle-steps 80 \
    --carry-steps "${carry_steps}" \
    --target-x "${target_x}" \
    --box-mass "${box_mass}" \
    --box-size 0.20 0.16 0.16 \
    --cradle-gap-x 0.025 \
    --cradle-gap-y 0.040 \
    --wall-thickness 0.030 \
    --wall-height 0.20 \
    --deck-thickness 0.035 \
    --cart-z 0.13 \
    --drive-stiffness 12000.0 \
    --drive-damping 2500.0 \
    --drive-max-force 80000.0 \
    --static-friction "${static_friction}" \
    --dynamic-friction "${dynamic_friction}" \
    --output-dir "${output_dir}"
}

run_diag diag1 0.30 0.50 0.20 0.10 300 520
run_diag diag2 0.60 0.50 0.20 0.10 420 700
run_diag diag3 0.60 2.00 0.20 0.10 420 700
run_diag diag4 0.30 0.50 0.05 0.03 300 520
