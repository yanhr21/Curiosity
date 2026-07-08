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

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/direct_carry_task_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/direct_carry_task_scene/${STAMP}}"
STEPS="${STEPS:-480}"
CONTROLLER_MODE="${CONTROLLER_MODE:-kinematic_proxy}"
BOX_MASS="${BOX_MASS:-6.0}"
BOX_MASS_MIN="${BOX_MASS_MIN:-}"
BOX_MASS_MAX="${BOX_MASS_MAX:-}"
BOX_SIZE_X="${BOX_SIZE_X:-0.55}"
BOX_SIZE_Y="${BOX_SIZE_Y:-0.35}"
BOX_SIZE_Z="${BOX_SIZE_Z:-0.35}"
BOX_SIZE_JITTER="${BOX_SIZE_JITTER:-0.0}"
BOX_SEED="${BOX_SEED:-}"
WALK_SPEED="${WALK_SPEED:-0.32}"
CARRY_HEIGHT="${CARRY_HEIGHT:-0.84}"
TARGET_X="${TARGET_X:-2.2}"
DEVICE="${DEVICE:-cuda:0}"
RENDER="${RENDER:-0}"

OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 3
fi

LOG_PATH="${LOG_DIR}/direct_carry_task_scene_${STAMP}.log"
EXTRA_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  EXTRA_ARGS+=(--render --enable_cameras)
fi
if [[ -n "${BOX_SEED}" ]]; then
  EXTRA_ARGS+=(--box-seed "${BOX_SEED}")
fi
if [[ -n "${BOX_MASS_MIN}" || -n "${BOX_MASS_MAX}" ]]; then
  if [[ -z "${BOX_MASS_MIN}" || -z "${BOX_MASS_MAX}" ]]; then
    echo "BOX_MASS_MIN and BOX_MASS_MAX must be set together" >&2
    exit 4
  fi
  EXTRA_ARGS+=(--box-mass-min "${BOX_MASS_MIN}" --box-mass-max "${BOX_MASS_MAX}")
fi

cd "${ROOT_DIR}"
echo "[CONFIG] STAMP=${STAMP} CONTROLLER_MODE=${CONTROLLER_MODE} STEPS=${STEPS} BOX_MASS=${BOX_MASS} BOX_MASS_RANGE=${BOX_MASS_MIN},${BOX_MASS_MAX} BOX_SIZE=${BOX_SIZE_X},${BOX_SIZE_Y},${BOX_SIZE_Z} BOX_SIZE_JITTER=${BOX_SIZE_JITTER} BOX_SEED=${BOX_SEED} TARGET_X=${TARGET_X}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/build_direct_carry_task_scene.py" \
  --viz none \
  --experience isaaclab.python.headless.kit \
  --device "${DEVICE}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS}" \
  --controller-mode "${CONTROLLER_MODE}" \
  --box-mass "${BOX_MASS}" \
  --box-size "${BOX_SIZE_X}" "${BOX_SIZE_Y}" "${BOX_SIZE_Z}" \
  --box-size-jitter "${BOX_SIZE_JITTER}" \
  --walk-speed "${WALK_SPEED}" \
  --carry-height "${CARRY_HEIGHT}" \
  --target-x "${TARGET_X}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
