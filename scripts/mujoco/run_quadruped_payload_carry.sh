#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/mujoco_quadruped_payload}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/mujoco_quadruped_payload/${STAMP}}"
STEPS="${STEPS:-3000}"
PAYLOAD_MASS="${PAYLOAD_MASS:-4.0}"
TARGET_SPEED="${TARGET_SPEED:-0.45}"
TARGET_HEIGHT="${TARGET_HEIGHT:-0.56}"
ASSIST_MODE="${ASSIST_MODE:-body_force}"
MAX_ASSIST_FORCE_X="${MAX_ASSIST_FORCE_X:-120.0}"
MAX_ASSIST_FORCE_Z="${MAX_ASSIST_FORCE_Z:-250.0}"
MAX_ASSIST_TORQUE="${MAX_ASSIST_TORQUE:-80.0}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_PATH="${LOG_DIR}/mujoco_quadruped_payload_${STAMP}.log"

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/mujoco/run_quadruped_payload_carry.py" \
  --steps "${STEPS}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --target-speed "${TARGET_SPEED}" \
  --target-height "${TARGET_HEIGHT}" \
  --assist-mode "${ASSIST_MODE}" \
  --max-assist-force-x "${MAX_ASSIST_FORCE_X}" \
  --max-assist-force-z "${MAX_ASSIST_FORCE_Z}" \
  --max-assist-torque "${MAX_ASSIST_TORQUE}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
