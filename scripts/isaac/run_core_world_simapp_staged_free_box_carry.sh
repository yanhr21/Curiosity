#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_simapp_staged_free_box_carry}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_simapp_staged_free_box_carry/${STAMP}}"
STEPS="${STEPS:-560}"
TARGET_X="${TARGET_X:-0.48}"
BOX_X="${BOX_X:-0.28}"
BOX_MASS="${BOX_MASS:-8.0}"
BOX_SIZE_X="${BOX_SIZE_X:-0.36}"
BOX_SIZE_Y="${BOX_SIZE_Y:-0.24}"
BOX_SIZE_Z="${BOX_SIZE_Z:-0.24}"
BOX_COM_X="${BOX_COM_X:-0.04}"
ROBOT_MASS="${ROBOT_MASS:-48.0}"
ROBOT_HEIGHT="${ROBOT_HEIGHT:-1.20}"
ARM_LENGTH="${ARM_LENGTH:-0.52}"
MAX_PAYLOAD="${MAX_PAYLOAD:-16.0}"
BASE_SPEED="${BASE_SPEED:-0.30}"
GAIT_FREQUENCY="${GAIT_FREQUENCY:-1.15}"
PROBE_SPEED="${PROBE_SPEED:-0.045}"
ATTACH_AFTER_STEP="${ATTACH_AFTER_STEP:-260}"
CARRY_GEOMETRY_MODE="${CARRY_GEOMETRY_MODE:-legacy}"
CARRY_CLEARANCE="${CARRY_CLEARANCE:-0.03}"
CARRY_Z_OFFSET="${CARRY_Z_OFFSET:-0.0}"
CONTACT_PROXY_GAIN="${CONTACT_PROXY_GAIN:-10.0}"
CONTACT_PROXY_MAX_SPEED="${CONTACT_PROXY_MAX_SPEED:-0.95}"
PALM_PROXY_MASS="${PALM_PROXY_MASS:-60.0}"
CHEST_PROXY_MASS="${CHEST_PROXY_MASS:-80.0}"
SHELF_PROXY_MASS="${SHELF_PROXY_MASS:-90.0}"
FRONT_STOP_PROXY_MASS="${FRONT_STOP_PROXY_MASS:-75.0}"
PALM_PROXY_THICKNESS="${PALM_PROXY_THICKNESS:-0.055}"
CHEST_PROXY_THICKNESS="${CHEST_PROXY_THICKNESS:-0.040}"
FRONT_STOP_PROXY_THICKNESS="${FRONT_STOP_PROXY_THICKNESS:-0.035}"
TARGET_HOLD_RADIUS="${TARGET_HOLD_RADIUS:-0.015}"
TARGET_SLOW_RADIUS="${TARGET_SLOW_RADIUS:-0.080}"
TARGET_BODY_MARGIN="${TARGET_BODY_MARGIN:-0.020}"
BODY_VERTICAL_MODE="${BODY_VERTICAL_MODE:-zero}"
BODY_HEIGHT_GAIN="${BODY_HEIGHT_GAIN:-18.0}"
BODY_HEIGHT_MAX_Z_SPEED="${BODY_HEIGHT_MAX_Z_SPEED:-0.80}"
PHYSICAL_SUPPORT_MODE="${PHYSICAL_SUPPORT_MODE:-none}"
SUPPORT_DECK_GAP="${SUPPORT_DECK_GAP:-0.0}"
ATTACHMENT_MODE="${ATTACHMENT_MODE:-fixed-joint}"
CARRIER_MODE="${CARRIER_MODE:-dynamic-velocity}"
CARRIER_EVIDENCE_MODE="${CARRIER_EVIDENCE_MODE:-support-proxy}"
DEVICE="${DEVICE:-cpu}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
export ISAAC_SIMAPP_EXPERIENCE="${ISAAC_SIMAPP_EXPERIENCE:-/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_simapp_staged_free_box_carry.py" \
  --steps "${STEPS}" \
  --target-x "${TARGET_X}" \
  --box-x "${BOX_X}" \
  --box-mass "${BOX_MASS}" \
  --box-size "${BOX_SIZE_X}" "${BOX_SIZE_Y}" "${BOX_SIZE_Z}" \
  --box-com-x "${BOX_COM_X}" \
  --robot-mass "${ROBOT_MASS}" \
  --robot-height "${ROBOT_HEIGHT}" \
  --arm-length "${ARM_LENGTH}" \
  --max-payload "${MAX_PAYLOAD}" \
  --base-speed "${BASE_SPEED}" \
  --gait-frequency "${GAIT_FREQUENCY}" \
  --probe-speed "${PROBE_SPEED}" \
  --attach-after-step "${ATTACH_AFTER_STEP}" \
  --carry-geometry-mode "${CARRY_GEOMETRY_MODE}" \
  --carry-clearance "${CARRY_CLEARANCE}" \
  --carry-z-offset "${CARRY_Z_OFFSET}" \
  --contact-proxy-gain "${CONTACT_PROXY_GAIN}" \
  --contact-proxy-max-speed "${CONTACT_PROXY_MAX_SPEED}" \
  --palm-proxy-mass "${PALM_PROXY_MASS}" \
  --chest-proxy-mass "${CHEST_PROXY_MASS}" \
  --shelf-proxy-mass "${SHELF_PROXY_MASS}" \
  --front-stop-proxy-mass "${FRONT_STOP_PROXY_MASS}" \
  --palm-proxy-thickness "${PALM_PROXY_THICKNESS}" \
  --chest-proxy-thickness "${CHEST_PROXY_THICKNESS}" \
  --front-stop-proxy-thickness "${FRONT_STOP_PROXY_THICKNESS}" \
  --target-hold-radius "${TARGET_HOLD_RADIUS}" \
  --target-slow-radius "${TARGET_SLOW_RADIUS}" \
  --target-body-margin "${TARGET_BODY_MARGIN}" \
  --body-vertical-mode "${BODY_VERTICAL_MODE}" \
  --body-height-gain "${BODY_HEIGHT_GAIN}" \
  --body-height-max-z-speed "${BODY_HEIGHT_MAX_Z_SPEED}" \
  --physical-support-mode "${PHYSICAL_SUPPORT_MODE}" \
  --support-deck-gap "${SUPPORT_DECK_GAP}" \
  --attachment-mode "${ATTACHMENT_MODE}" \
  --carrier-mode "${CARRIER_MODE}" \
  --carrier-evidence-mode "${CARRIER_EVIDENCE_MODE}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/core_world_simapp_staged_free_box_carry_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/core_world_simapp_staged_free_box_carry_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
