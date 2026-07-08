#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-foot_ik_closed_loop}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local stride="$2"
  local hip_base="$3"
  local foot_roll="$4"
  local v_gain="$5"
  local p_gain="$6"
  local stride_clip="$7"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} stride=${stride} hip_base=${hip_base} foot_roll=${foot_roll} v_gain=${v_gain} p_gain=${p_gain} clip=${stride_clip} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS=2.0 \
  TARGET_SPEED=0.24 \
  TARGET_HEIGHT=0.56 \
  ASSIST_MODE=none \
  LEG_DRIVE_MODE=foot_ik \
  GAIT_FREQUENCY_HZ=1.1 \
  STRIDE_LENGTH="${stride}" \
  STANCE_DUTY=0.72 \
  STANCE_FOOT_Z_DOWN=0.42 \
  SWING_FOOT_Z_DOWN=0.28 \
  FOOT_ROLL_Z_GAIN="${foot_roll}" \
  HIP_ROLL_BASE="${hip_base}" \
  HIP_ROLL_FEEDBACK_GAIN=0.0 \
  CLOSED_LOOP_FOOT_PLACEMENT=1 \
  STRIDE_VELOCITY_GAIN="${v_gain}" \
  STRIDE_POSITION_GAIN="${p_gain}" \
  STRIDE_CLIP="${stride_clip}" \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
  STOP_AFTER_BOX_TRAVEL=0.08 \
  HOLD_TARGET_SPEED=0.0 \
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
    --expect-assist-mode none \
    --expect-leg-drive-mode foot_ik \
    --require-closed-loop-foot-placement \
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
    --min-target-stop-hold-steps 600 | tee "${check}"
  local check_status=${PIPESTATUS[0]}
  set -e
  echo "[CHECK_STATUS] ${stamp} ${check_status}"
}

run_case cl_v020_p030 -0.10 0.00 -0.04 0.20 0.30 0.18
run_case cl_v035_p040 -0.10 0.00 -0.04 0.35 0.40 0.18
run_case cl_v050_p060 -0.08 0.00 -0.04 0.50 0.60 0.18
run_case cl_hip005_v035 -0.10 0.05 -0.04 0.35 0.40 0.18
run_case cl_hip010_v050 -0.12 0.10 -0.04 0.50 0.60 0.20

echo "[DONE] MuJoCo free-box foot-IK closed-loop foot-placement suite suffix=${SUITE_SUFFIX}"
