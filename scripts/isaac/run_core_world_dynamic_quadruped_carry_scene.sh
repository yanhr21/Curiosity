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
export BASE_MAT_DIR="${BASE_MAT_DIR:-}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_dynamic_quadruped_carry_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_dynamic_quadruped_carry_scene/${STAMP}}"
STEPS="${STEPS:-480}"
PAYLOAD_MASS="${PAYLOAD_MASS:-4.0}"
PAYLOAD_MODE="${PAYLOAD_MODE:-fixed_joint_to_torso}"
STAGED_ATTACH_MODE="${STAGED_ATTACH_MODE:-pose-lock}"
BOX_X="${BOX_X:-0.30}"
ATTACH_AFTER_STEP="${ATTACH_AFTER_STEP:-90}"
PROBE_SPEED="${PROBE_SPEED:-0.035}"
TARGET_X="${TARGET_X:-0.8}"
TARGET_SPEED="${TARGET_SPEED:-0.24}"
TARGET_HOLD_RADIUS="${TARGET_HOLD_RADIUS:-0.025}"
TARGET_BODY_MARGIN="${TARGET_BODY_MARGIN:-0.03}"
MIN_HOLD_TORSO_TRAVEL="${MIN_HOLD_TORSO_TRAVEL:-0.0}"
CARRY_LOCAL_X="${CARRY_LOCAL_X:-0.26}"
CARRY_LOCAL_Z="${CARRY_LOCAL_Z:-0.03}"
CONTACT_PROXY_GAIN="${CONTACT_PROXY_GAIN:-14.0}"
CONTACT_PROXY_MAX_SPEED="${CONTACT_PROXY_MAX_SPEED:-0.95}"
BASE_VELOCITY_ASSIST="${BASE_VELOCITY_ASSIST:-0}"
BASE_ASSIST_MODE="${BASE_ASSIST_MODE:-velocity}"
BASE_X_GAIN="${BASE_X_GAIN:-3.0}"
BASE_MAX_X_SPEED="${BASE_MAX_X_SPEED:-0.8}"
BASE_X_COMMAND_SCALE="${BASE_X_COMMAND_SCALE:-1.0}"
BASE_LATERAL_GAIN="${BASE_LATERAL_GAIN:-2.0}"
BASE_HEIGHT_GAIN="${BASE_HEIGHT_GAIN:-8.0}"
BASE_MAX_Z_SPEED="${BASE_MAX_Z_SPEED:-0.8}"
BASE_UPRIGHT_GAIN="${BASE_UPRIGHT_GAIN:-8.0}"
BASE_MAX_ANGULAR_SPEED="${BASE_MAX_ANGULAR_SPEED:-4.0}"
BASE_POST_STEP_VELOCITY_ASSIST="${BASE_POST_STEP_VELOCITY_ASSIST:-0}"
SUPPORT_DRIVE="${SUPPORT_DRIVE:-0}"
SUPPORT_DRIVE_GAIN="${SUPPORT_DRIVE_GAIN:-3.0}"
SUPPORT_DRIVE_MAX_SPEED="${SUPPORT_DRIVE_MAX_SPEED:-0.45}"
SUPPORT_PAD_Z="${SUPPORT_PAD_Z:-0.018}"
TORSO_Z="${TORSO_Z:-0.62}"
STANCE_HALF_LENGTH="${STANCE_HALF_LENGTH:-0.18}"
STANCE_HALF_WIDTH="${STANCE_HALF_WIDTH:-0.16}"
FOOT_LENGTH="${FOOT_LENGTH:-0.18}"
FOOT_WIDTH="${FOOT_WIDTH:-0.075}"
FOOT_HEIGHT="${FOOT_HEIGHT:-0.045}"
STATIC_FRICTION="${STATIC_FRICTION:-1.0}"
DYNAMIC_FRICTION="${DYNAMIC_FRICTION:-0.8}"
HIP_STIFFNESS="${HIP_STIFFNESS:-1800.0}"
HIP_DAMPING="${HIP_DAMPING:-120.0}"
HIP_MAX_FORCE="${HIP_MAX_FORCE:-1100.0}"
KNEE_STIFFNESS="${KNEE_STIFFNESS:-1500.0}"
KNEE_DAMPING="${KNEE_DAMPING:-100.0}"
KNEE_MAX_FORCE="${KNEE_MAX_FORCE:-900.0}"
GAIT_FREQUENCY="${GAIT_FREQUENCY:-1.1}"
HIP_NEUTRAL_DEG="${HIP_NEUTRAL_DEG:--5.0}"
KNEE_NEUTRAL_DEG="${KNEE_NEUTRAL_DEG:--18.0}"
HIP_AMPLITUDE_DEG="${HIP_AMPLITUDE_DEG:-18.0}"
KNEE_AMPLITUDE_DEG="${KNEE_AMPLITUDE_DEG:-16.0}"
DEVICE="${DEVICE:-cpu}"
RENDER="${RENDER:-0}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

LOG_PATH="${LOG_DIR}/core_world_dynamic_quadruped_carry_scene_${STAMP}.log"
echo "[INFO] BASE_X_COMMAND_SCALE=${BASE_X_COMMAND_SCALE}"
EXTRA_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi
if [[ "${BASE_VELOCITY_ASSIST}" == "1" ]]; then
  EXTRA_ARGS+=(--base-velocity-assist)
fi
if [[ "${BASE_POST_STEP_VELOCITY_ASSIST}" == "1" ]]; then
  EXTRA_ARGS+=(--base-post-step-velocity-assist)
fi
if [[ "${SUPPORT_DRIVE}" == "1" ]]; then
  EXTRA_ARGS+=(--support-drive)
fi

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_core_world_dynamic_quadruped_carry_scene.py" \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --payload-mode "${PAYLOAD_MODE}" \
  --staged-attach-mode "${STAGED_ATTACH_MODE}" \
  --box-x "${BOX_X}" \
  --attach-after-step "${ATTACH_AFTER_STEP}" \
  --probe-speed "${PROBE_SPEED}" \
  --target-x "${TARGET_X}" \
  --target-speed "${TARGET_SPEED}" \
  --target-hold-radius "${TARGET_HOLD_RADIUS}" \
  --target-body-margin "${TARGET_BODY_MARGIN}" \
  --min-hold-torso-travel "${MIN_HOLD_TORSO_TRAVEL}" \
  --carry-local-x "${CARRY_LOCAL_X}" \
  --carry-local-z "${CARRY_LOCAL_Z}" \
  --contact-proxy-gain "${CONTACT_PROXY_GAIN}" \
  --contact-proxy-max-speed "${CONTACT_PROXY_MAX_SPEED}" \
  --base-assist-mode "${BASE_ASSIST_MODE}" \
  --base-x-gain "${BASE_X_GAIN}" \
  --base-max-x-speed "${BASE_MAX_X_SPEED}" \
  --base-x-command-scale "${BASE_X_COMMAND_SCALE}" \
  --base-lateral-gain "${BASE_LATERAL_GAIN}" \
  --base-height-gain "${BASE_HEIGHT_GAIN}" \
  --base-max-z-speed "${BASE_MAX_Z_SPEED}" \
  --base-upright-gain "${BASE_UPRIGHT_GAIN}" \
  --base-max-angular-speed "${BASE_MAX_ANGULAR_SPEED}" \
  --support-drive-gain "${SUPPORT_DRIVE_GAIN}" \
  --support-drive-max-speed "${SUPPORT_DRIVE_MAX_SPEED}" \
  --support-pad-z "${SUPPORT_PAD_Z}" \
  --torso-z "${TORSO_Z}" \
  --stance-half-length "${STANCE_HALF_LENGTH}" \
  --stance-half-width "${STANCE_HALF_WIDTH}" \
  --foot-length "${FOOT_LENGTH}" \
  --foot-width "${FOOT_WIDTH}" \
  --foot-height "${FOOT_HEIGHT}" \
  --static-friction "${STATIC_FRICTION}" \
  --dynamic-friction "${DYNAMIC_FRICTION}" \
  --hip-stiffness "${HIP_STIFFNESS}" \
  --hip-damping "${HIP_DAMPING}" \
  --hip-max-force "${HIP_MAX_FORCE}" \
  --knee-stiffness "${KNEE_STIFFNESS}" \
  --knee-damping "${KNEE_DAMPING}" \
  --knee-max-force "${KNEE_MAX_FORCE}" \
  --gait-frequency "${GAIT_FREQUENCY}" \
  --hip-neutral-deg "${HIP_NEUTRAL_DEG}" \
  --knee-neutral-deg "${KNEE_NEUTRAL_DEG}" \
  --hip-amplitude-deg "${HIP_AMPLITUDE_DEG}" \
  --knee-amplitude-deg "${KNEE_AMPLITUDE_DEG}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
