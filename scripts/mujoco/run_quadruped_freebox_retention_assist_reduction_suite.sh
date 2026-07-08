#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-assist_reduce}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local force_x="$2"
  local force_z="$3"
  local torque="$4"
  local stamp="20260707_mujoco_quad_freebox_2kg_v024_stop015_hold012_retention_spring_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} assist_fx=${force_x} assist_fz=${force_z} assist_tau=${torque} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS=2.0 \
  TARGET_SPEED=0.24 \
  TARGET_HEIGHT=0.56 \
  ASSIST_MODE=body_force \
  MAX_ASSIST_FORCE_X="${force_x}" \
  MAX_ASSIST_FORCE_Z="${force_z}" \
  MAX_ASSIST_TORQUE="${torque}" \
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

run_case assist075 135.0 270.0 195.0
run_case assist050 90.0 180.0 130.0
run_case assist033 60.0 120.0 90.0

echo "[DONE] MuJoCo free-box retention assist-reduction suite suffix=${SUITE_SUFFIX}"
