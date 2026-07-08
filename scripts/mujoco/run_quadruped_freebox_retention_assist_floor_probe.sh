#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-assist_floor}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local assist_mode="$2"
  local force_x="$3"
  local force_z="$4"
  local torque="$5"
  local require_external="$6"
  local stamp="20260707_mujoco_quad_freebox_2kg_v024_stop015_hold012_retention_spring_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"
  local checker_args=()

  if [[ "${require_external}" == "1" ]]; then
    checker_args+=(--min-external-force-writes 1)
  fi

  echo "[CASE] ${stamp} assist=${assist_mode} fx=${force_x} fz=${force_z} tau=${torque} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS=2.0 \
  TARGET_SPEED=0.24 \
  TARGET_HEIGHT=0.56 \
  ASSIST_MODE="${assist_mode}" \
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

  set +e
  "${PYTHON_BIN}" scripts/mujoco/check_quadruped_freebox_summary.py "${summary}" \
    --expect-assist-mode "${assist_mode}" \
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
    --require-target-stop-latched \
    --min-target-stop-hold-steps 600 \
    "${checker_args[@]}" | tee "${check}"
  local check_status=${PIPESTATUS[0]}
  set -e
  echo "[CHECK_STATUS] ${stamp} ${check_status}"
}

run_case assist010 body_force 18.0 36.0 26.0 1
run_case assist000 body_force 0.0 0.0 0.0 1
run_case noassist none 0.0 0.0 0.0 0

echo "[DONE] MuJoCo free-box retention assist-floor probe suffix=${SUITE_SUFFIX}"
