#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_NAME="${SWEEP_NAME:-adaptive_probe_sweep_${STAMP}}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/adaptive_probe_carry_scene_sweeps/${SWEEP_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/adaptive_probe_carry_scene_sweeps}"
DEVICE="${DEVICE:-cuda:0}"
RENDER="${RENDER:-0}"
STEPS="${STEPS:-220}"

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

mkdir -p "${SWEEP_OUTPUT_ROOT}" "${LOG_DIR}"

run_case() {
  local case_name="$1"
  shift
  echo "[INFO] Running adaptive carry sweep case: ${case_name}"
  (
    export STAMP="${SWEEP_NAME}_${case_name}"
    export OUTPUT_DIR="${SWEEP_OUTPUT_ROOT}/${case_name}"
    export DEVICE RENDER STEPS
    for kv in "$@"; do
      export "${kv}"
    done
    bash "${ROOT_DIR}/scripts/isaac/run_adaptive_probe_carry_scene.sh"
  ) 2>&1 | tee "${LOG_DIR}/${SWEEP_NAME}_${case_name}.log"
}

cd "${ROOT_DIR}"

run_case "front_light_tall" \
  BOX_MASS=4.5 BOX_SIZE_X=0.46 BOX_SIZE_Y=0.30 BOX_SIZE_Z=0.30 \
  BOX_COM_X=0.00 ROBOT_HEIGHT=1.55 ROBOT_MASS=58.0 ARM_LENGTH=0.70 \
  MAX_PAYLOAD=18.0 TARGET_X=2.10

run_case "low_com_biased" \
  BOX_MASS=8.5 BOX_SIZE_X=0.60 BOX_SIZE_Y=0.38 BOX_SIZE_Z=0.36 \
  BOX_COM_X=0.06 ROBOT_HEIGHT=1.45 ROBOT_MASS=52.0 ARM_LENGTH=0.58 \
  MAX_PAYLOAD=16.0 TARGET_X=2.15

run_case "chest_short_arm_heavy" \
  BOX_MASS=11.0 BOX_SIZE_X=0.68 BOX_SIZE_Y=0.42 BOX_SIZE_Z=0.40 \
  BOX_COM_X=0.02 ROBOT_HEIGHT=1.25 ROBOT_MASS=44.0 ARM_LENGTH=0.48 \
  MAX_PAYLOAD=15.0 TARGET_X=1.85

run_case "near_limit_wide" \
  BOX_MASS=13.5 BOX_SIZE_X=0.72 BOX_SIZE_Y=0.50 BOX_SIZE_Z=0.42 \
  BOX_COM_X=-0.03 ROBOT_HEIGHT=1.38 ROBOT_MASS=50.0 ARM_LENGTH=0.55 \
  MAX_PAYLOAD=16.0 TARGET_X=1.95

run_case "compact_mid_payload" \
  BOX_MASS=6.5 BOX_SIZE_X=0.50 BOX_SIZE_Y=0.34 BOX_SIZE_Z=0.34 \
  BOX_COM_X=-0.02 ROBOT_HEIGHT=1.35 ROBOT_MASS=47.0 ARM_LENGTH=0.54 \
  MAX_PAYLOAD=16.0 TARGET_X=2.00

"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/isaac/aggregate_adaptive_probe_sweep.py" \
  --sweep-dir "${SWEEP_OUTPUT_ROOT}" \
  --output "${SWEEP_OUTPUT_ROOT}/adaptive_probe_sweep_summary.json"

echo "[INFO] Sweep output: ${SWEEP_OUTPUT_ROOT}"
