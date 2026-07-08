#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-multiload}"

cd "${ROOT_DIR}"

run_case() {
  local mass="$1"
  local tag="$2"
  local stamp="20260707_mujoco_quad_freebox_${tag}_v024_stop015_hold012_retention_spring_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} mass=${mass}kg host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS="${mass}" \
  TARGET_SPEED=0.24 \
  TARGET_HEIGHT=0.56 \
  ASSIST_MODE=body_force \
  MAX_ASSIST_FORCE_X=180.0 \
  MAX_ASSIST_FORCE_Z=360.0 \
  MAX_ASSIST_TORQUE=260.0 \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
  STOP_AFTER_BOX_TRAVEL=0.15 \
  HOLD_TARGET_SPEED=0.12 \
  RETENTION_FORCE_MODE=relative_spring \
  RETENTION_KP_X=260.0 \
  RETENTION_KD_X=28.0 \
  RETENTION_KP_Z=180.0 \
  RETENTION_KD_Z=20.0 \
  RETENTION_MAX_FORCE_X=90.0 \
  RETENTION_MAX_FORCE_Z=70.0 \
  bash scripts/mujoco/run_quadruped_freebox_carry.sh

  "${PYTHON_BIN}" scripts/mujoco/check_quadruped_freebox_summary.py "${summary}" \
    --expect-assist-mode body_force \
    --expect-retention-force-mode relative_spring \
    --min-retention-force-writes 1 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --min-box-travel-x 0.15 \
    --min-final-box-travel-x 0.12 \
    --max-tilt 0.70 \
    --min-box-z 0.58 \
    --max-relative-offset-error 0.22 \
    --max-final-relative-offset-error 0.20 \
    --max-root-pose-writes 0 \
    --max-root-velocity-writes 0 \
    --max-box-pose-writes 0 \
    --max-box-velocity-writes 0 \
    --min-external-force-writes 1 \
    --require-target-stop-latched \
    --min-target-stop-hold-steps 600 | tee "${check}"
}

run_case 1.0 1kg
run_case 2.0 2kg
run_case 3.0 3kg

echo "[DONE] MuJoCo free-box retention multiload suite suffix=${SUITE_SUFFIX}"
