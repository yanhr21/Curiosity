#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run low-CG rear-anchor push batch on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-30}"
python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py

run_diag() {
  local diag="$1"
  local target_x="$2"
  local front_mu="$3"
  local rear_mu="$4"
  local stamp="20260705_lowcg_rear_anchor_${diag}_target${target_x}_fmu${front_mu}_rmu${rear_mu}"
  local output_dir="experiments/outputs/core_world_prismatic_carrier_stand/${stamp}"
  echo "[BATCH] ${diag} rear_anchor_push target=${target_x} front_mu=${front_mu} rear_mu=${rear_mu}"
  "${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_prismatic_carrier_stand.py \
    --viz none \
    --experience "${EXPERIENCE}" \
    --device "${DEVICE:-cpu}" \
    --kit_args "${KIT_ARGS}" \
    --steps 760 \
    --payload-mode tray_contact_free_box \
    --payload-mass 1.0 \
    --torso-mass 100.0 \
    --torso-z 0.58 \
    --payload-local-x 0.03 \
    --payload-local-z 0.04 \
    --tray-local-x 0.03 \
    --tray-local-z 0.07 \
    --tray-size 0.72 0.56 0.04 \
    --tray-rail-height 0.30 \
    --tray-rail-thickness 0.055 \
    --tray-mass 1.0 \
    --enable-tray-lid \
    --tray-lid-clearance 0.015 \
    --tray-lid-thickness 0.04 \
    --tray-lid-mass 0.3 \
    --motion-mode rear_anchor_push \
    --enable-horizontal-legs \
    --target-x "${target_x}" \
    --step-height 0.035 \
    --settle-steps 350 \
    --ramp-steps 300 \
    --stance-half-length 0.58 \
    --stance-half-width 0.65 \
    --foot-length 0.78 \
    --foot-width 0.44 \
    --foot-height 0.055 \
    --foot-mass 14.0 \
    --leg-target -0.50 \
    --leg-lower -0.75 \
    --leg-upper -0.25 \
    --leg-stiffness 30000.0 \
    --leg-damping 3500.0 \
    --leg-max-force 45000.0 \
    --x-slide-limit 0.08 \
    --x-slide-stiffness 9000.0 \
    --x-slide-damping 3500.0 \
    --x-slide-max-force 12000.0 \
    --static-friction "${rear_mu}" \
    --dynamic-friction "${rear_mu}" \
    --front-foot-static-friction "${front_mu}" \
    --front-foot-dynamic-friction "${front_mu}" \
    --rear-foot-static-friction "${rear_mu}" \
    --rear-foot-dynamic-friction "${rear_mu}" \
    --fall-z 0.42 \
    --drop-z 0.24 \
    --max-stand-drift 0.08 \
    --output-dir "${output_dir}"
}

run_diag diag6 0.030 0.30 6.00
run_diag diag7 -0.030 0.30 6.00
run_diag diag8 0.030 0.05 8.00
