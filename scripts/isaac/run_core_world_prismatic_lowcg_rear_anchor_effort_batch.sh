#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run low-CG rear-anchor effort batch on login/management node: $(hostname)" >&2
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
  local x_effort="$3"
  local stamp="20260705_lowcg_rear_anchor_effort_${diag}_target${target_x}_effort${x_effort}"
  local output_dir="experiments/outputs/core_world_prismatic_carrier_stand/${stamp}"
  echo "[BATCH] ${diag} rear_anchor_effort target=${target_x} x_effort=${x_effort}"
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
    --motion-mode rear_anchor_effort_push \
    --enable-horizontal-legs \
    --target-x "${target_x}" \
    --x-slide-effort "${x_effort}" \
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
    --x-slide-stiffness 0.0 \
    --x-slide-damping 0.0 \
    --x-slide-max-force 600000.0 \
    --static-friction 8.0 \
    --dynamic-friction 8.0 \
    --front-foot-static-friction 0.05 \
    --front-foot-dynamic-friction 0.05 \
    --rear-foot-static-friction 8.0 \
    --rear-foot-dynamic-friction 8.0 \
    --fall-z 0.42 \
    --drop-z 0.24 \
    --max-stand-drift 0.08 \
    --output-dir "${output_dir}"
}

run_diag diag21 0.030 5000.0
run_diag diag22 -0.030 5000.0
run_diag diag23 0.030 20000.0
