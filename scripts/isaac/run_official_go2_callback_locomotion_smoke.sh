#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/official_go2_callback_locomotion_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/official_go2_callback_locomotion_smoke/${STAMP}}"
STEPS="${STEPS:-220}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
COMMAND_X="${COMMAND_X:-1.0}"
COMMAND_Y="${COMMAND_Y:-0.0}"
COMMAND_YAW="${COMMAND_YAW:-0.0}"
DEVICE="${DEVICE:-cuda}"
SIMULATION_MANAGER_MODE="${SIMULATION_MANAGER_MODE:-official_device_dt}"
OFFICIAL_TEST_KIT_ARGS="${OFFICIAL_TEST_KIT_ARGS:-0}"
PAYLOAD_MODE="${PAYLOAD_MODE:-none}"
PAYLOAD_MASS="${PAYLOAD_MASS:-2.0}"
ASSET_ROOT="${ASSET_ROOT:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0}"
export ISAACSIM_ASSET_ROOT="${ASSET_ROOT}"
export ISAAC_SIMAPP_EXPERIENCE="${ISAAC_SIMAPP_EXPERIENCE:-/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/run_official_go2_callback_locomotion_smoke.py" \
  --device "${DEVICE}" \
  --simulation-manager-mode "${SIMULATION_MANAGER_MODE}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --command "${COMMAND_X}" "${COMMAND_Y}" "${COMMAND_YAW}" \
  --payload-mode "${PAYLOAD_MODE}" \
  --payload-mass "${PAYLOAD_MASS}" \
  --asset-root "${ASSET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  $(if [[ "${OFFICIAL_TEST_KIT_ARGS}" == "1" ]]; then printf '%s' "--official-test-kit-args"; fi) \
  "$@" \
  2>&1 | tee "${LOG_DIR}/official_go2_callback_locomotion_smoke_${STAMP}.log"

echo "[INFO] Log: ${LOG_DIR}/official_go2_callback_locomotion_smoke_${STAMP}.log"
echo "[INFO] Output: ${OUTPUT_DIR}"
