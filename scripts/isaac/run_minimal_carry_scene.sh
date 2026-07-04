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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/minimal_carry_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/minimal_carry_scene/$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-300}"
BOX_MASS="${BOX_MASS:-5.0}"
BOX_SIZE_X="${BOX_SIZE_X:-0.55}"
BOX_SIZE_Y="${BOX_SIZE_Y:-0.35}"
BOX_SIZE_Z="${BOX_SIZE_Z:-0.35}"
BOX_POS_X="${BOX_POS_X:-0.85}"
BOX_POS_Y="${BOX_POS_Y:-0.0}"
BOX_POS_Z="${BOX_POS_Z:-0.45}"
G1_USD="${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
SKIP_ROBOT="${SKIP_ROBOT:-0}"
RENDER="${RENDER:-0}"
DISABLE_FABRIC="${DISABLE_FABRIC:-0}"
DISABLE_USD_PHYSICS_UPDATES="${DISABLE_USD_PHYSICS_UPDATES:-0}"
SKIP_EXPLICIT_STATE_RESET="${SKIP_EXPLICIT_STATE_RESET:-0}"
DEVICE="${DEVICE:-cuda:0}"
WBC_MODE="${WBC_MODE:-none}"
WALK_COMMAND_X="${WALK_COMMAND_X:-0.25}"
WALK_COMMAND_Y="${WALK_COMMAND_Y:-0.0}"
WALK_COMMAND_YAW="${WALK_COMMAND_YAW:-0.0}"
BASE_HEIGHT_COMMAND="${BASE_HEIGHT_COMMAND:-0.75}"
ATTACH_BOX="${ATTACH_BOX:-none}"
ATTACH_BODY_PATH="${ATTACH_BODY_PATH:-/World/G1/torso_link}"
ATTACH_LOCAL_POS0_X="${ATTACH_LOCAL_POS0_X:-0.28}"
ATTACH_LOCAL_POS0_Y="${ATTACH_LOCAL_POS0_Y:-0.0}"
ATTACH_LOCAL_POS0_Z="${ATTACH_LOCAL_POS0_Z:-0.0}"
WBC_ASSET_ROOT="${WBC_ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

if [[ "${SKIP_ROBOT}" != "1" && ! -f "${G1_USD}" ]]; then
  echo "G1 USD not found: ${G1_USD}" >&2
  exit 4
fi

LOG_PATH="${LOG_DIR}/minimal_carry_scene_$(date +%Y%m%d_%H%M%S).log"
EXTRA_ARGS=()
if [[ "${SKIP_ROBOT}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-robot)
fi
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi
if [[ "${DISABLE_FABRIC}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-fabric)
fi
if [[ "${DISABLE_USD_PHYSICS_UPDATES}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-usd-physics-updates)
fi
if [[ "${SKIP_EXPLICIT_STATE_RESET}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-explicit-state-reset)
fi

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}/external/IsaacLab-Arena:${PYTHONPATH:-}" "${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_minimal_carry_scene.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --box-mass "${BOX_MASS}" \
  --box-size "${BOX_SIZE_X}" "${BOX_SIZE_Y}" "${BOX_SIZE_Z}" \
  --box-position "${BOX_POS_X}" "${BOX_POS_Y}" "${BOX_POS_Z}" \
  --g1-usd "${G1_USD}" \
  --wbc-mode "${WBC_MODE}" \
  --walk-command "${WALK_COMMAND_X}" "${WALK_COMMAND_Y}" "${WALK_COMMAND_YAW}" \
  --base-height-command "${BASE_HEIGHT_COMMAND}" \
  --wbc-asset-root "${WBC_ASSET_ROOT}" \
  --attach-box "${ATTACH_BOX}" \
  --attach-body-path "${ATTACH_BODY_PATH}" \
  --attach-local-pos0 "${ATTACH_LOCAL_POS0_X}" "${ATTACH_LOCAL_POS0_Y}" "${ATTACH_LOCAL_POS0_Z}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
