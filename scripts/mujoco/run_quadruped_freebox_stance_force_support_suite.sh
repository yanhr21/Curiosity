#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-stance_force_support}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local force_scale="$2"
  local kp_z="$3"
  local kp_vx="$4"
  local max_fx="$5"
  local max_fz="$6"
  local max_tau="$7"
  local hip_base="$8"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} scale=${force_scale} kp_z=${kp_z} kp_vx=${kp_vx} max_fx=${max_fx} max_fz=${max_fz} max_tau=${max_tau} hip_base=${hip_base} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS=2.0 \
  TARGET_SPEED=0.24 \
  TARGET_HEIGHT=0.56 \
  ACTUATOR_KP=80.0 \
  ACTUATOR_KV=8.0 \
  ASSIST_MODE=none \
  LEG_DRIVE_MODE=foot_ik \
  GAIT_FREQUENCY_HZ=1.1 \
  STRIDE_LENGTH=-0.10 \
  STANCE_DUTY=0.72 \
  STANCE_FOOT_Z_DOWN=0.42 \
  SWING_FOOT_Z_DOWN=0.28 \
  FOOT_ROLL_Z_GAIN=-0.04 \
  HIP_ROLL_BASE="${hip_base}" \
  HIP_ROLL_FEEDBACK_GAIN=0.0 \
  HOLD_STANCE_FOOT_Z_DOWN=0.46 \
  HOLD_HIP_ROLL_BASE=0.08 \
  HOLD_FRONT_FOOT_X=0.14 \
  HOLD_REAR_FOOT_X=-0.10 \
  HOLD_PITCH_FOOT_X_GAIN=0.0 \
  CLOSED_LOOP_FOOT_PLACEMENT=1 \
  STRIDE_VELOCITY_GAIN=0.20 \
  STRIDE_POSITION_GAIN=0.30 \
  STRIDE_CLIP=0.18 \
  SUPPORT_CONTROLLER_MODE=stance_force \
  SUPPORT_FORCE_SCALE="${force_scale}" \
  SUPPORT_KP_Z="${kp_z}" \
  SUPPORT_KD_Z=180.0 \
  SUPPORT_KP_ROLL=260.0 \
  SUPPORT_KD_ROLL=38.0 \
  SUPPORT_KP_PITCH=220.0 \
  SUPPORT_KD_PITCH=32.0 \
  SUPPORT_KP_VX="${kp_vx}" \
  SUPPORT_MAX_TOTAL_FX="${max_fx}" \
  SUPPORT_MIN_FOOT_FZ=10.0 \
  SUPPORT_MAX_FOOT_FZ="${max_fz}" \
  SUPPORT_MAX_JOINT_TORQUE="${max_tau}" \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
  STOP_AFTER_BOX_TRAVEL=0.08 \
  HOLD_TARGET_SPEED=0.0 \
  RETENTION_FORCE_MODE=relative_spring \
  RETENTION_KP_X=260.0 \
  RETENTION_KD_X=28.0 \
  RETENTION_KP_Y=180.0 \
  RETENTION_KD_Y=28.0 \
  RETENTION_KP_Z=180.0 \
  RETENTION_KD_Z=20.0 \
  RETENTION_MAX_FORCE_X=90.0 \
  RETENTION_MAX_FORCE_Y=90.0 \
  RETENTION_MAX_FORCE_Z=70.0 \
  bash scripts/mujoco/run_quadruped_freebox_carry.sh

  set +e
  "${PYTHON_BIN}" scripts/mujoco/check_quadruped_freebox_summary.py "${summary}" \
    --expect-assist-mode none \
    --expect-leg-drive-mode foot_ik \
    --expect-support-controller-mode stance_force \
    --min-support-joint-torque-writes 1 \
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

run_case sf_pos_nominal 1.0 2600.0 520.0 260.0 260.0 220.0 0.00
run_case sf_neg_nominal -1.0 2600.0 520.0 260.0 260.0 220.0 0.00
run_case sf_pos_hip003 1.0 2600.0 520.0 260.0 260.0 220.0 0.03
run_case sf_neg_hip003 -1.0 2600.0 520.0 260.0 260.0 220.0 0.03
run_case sf_pos_strong 1.0 4200.0 760.0 360.0 360.0 320.0 0.00
run_case sf_neg_strong -1.0 4200.0 760.0 360.0 360.0 320.0 0.00

echo "[DONE] MuJoCo free-box stance-force support suite suffix=${SUITE_SUFFIX}"
