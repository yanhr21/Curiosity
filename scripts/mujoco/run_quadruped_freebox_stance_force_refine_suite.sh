#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-stance_force_refine}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local force_scale="$2"
  local hold_speed="$3"
  local kp_vx="$4"
  local max_fx="$5"
  local hip_base="$6"
  local kp_z="$7"
  local kp_roll="$8"
  local kp_pitch="$9"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} scale=${force_scale} hold=${hold_speed} kp_vx=${kp_vx} max_fx=${max_fx} hip=${hip_base} kp_z=${kp_z} kp_roll=${kp_roll} kp_pitch=${kp_pitch} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS=2.0 \
  TARGET_SPEED=0.16 \
  TARGET_HEIGHT=0.56 \
  ACTUATOR_KP=80.0 \
  ACTUATOR_KV=8.0 \
  ASSIST_MODE=none \
  LEG_DRIVE_MODE=foot_ik \
  GAIT_FREQUENCY_HZ=1.1 \
  STRIDE_LENGTH=-0.07 \
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
  STRIDE_VELOCITY_GAIN=0.16 \
  STRIDE_POSITION_GAIN=0.40 \
  STRIDE_CLIP=0.12 \
  SUPPORT_CONTROLLER_MODE=stance_force \
  SUPPORT_FORCE_SCALE="${force_scale}" \
  SUPPORT_KP_Z="${kp_z}" \
  SUPPORT_KD_Z=240.0 \
  SUPPORT_KP_ROLL="${kp_roll}" \
  SUPPORT_KD_ROLL=58.0 \
  SUPPORT_KP_PITCH="${kp_pitch}" \
  SUPPORT_KD_PITCH=48.0 \
  SUPPORT_KP_VX="${kp_vx}" \
  SUPPORT_MAX_TOTAL_FX="${max_fx}" \
  SUPPORT_MIN_FOOT_FZ=10.0 \
  SUPPORT_MAX_FOOT_FZ=360.0 \
  SUPPORT_MAX_JOINT_TORQUE=320.0 \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
  STOP_AFTER_BOX_TRAVEL=0.06 \
  HOLD_TARGET_SPEED="${hold_speed}" \
  RETENTION_FORCE_MODE=relative_spring \
  RETENTION_KP_X=360.0 \
  RETENTION_KD_X=36.0 \
  RETENTION_KP_Y=180.0 \
  RETENTION_KD_Y=28.0 \
  RETENTION_KP_Z=260.0 \
  RETENTION_KD_Z=28.0 \
  RETENTION_MAX_FORCE_X=130.0 \
  RETENTION_MAX_FORCE_Y=90.0 \
  RETENTION_MAX_FORCE_Z=100.0 \
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

run_case refine_h004_s050_h0 -0.50 0.00 220.0 90.0 0.04 4200.0 420.0 360.0
run_case refine_h004_s050_hn02 -0.50 -0.02 220.0 90.0 0.04 4200.0 420.0 360.0
run_case refine_h005_s050_h0 -0.50 0.00 220.0 90.0 0.05 4200.0 420.0 360.0
run_case refine_h005_s035_h0 -0.35 0.00 220.0 90.0 0.05 4200.0 420.0 360.0
run_case refine_h004_s035_h0 -0.35 0.00 220.0 90.0 0.04 4200.0 420.0 360.0
run_case refine_h003_s035_h0 -0.35 0.00 220.0 90.0 0.03 4200.0 420.0 360.0

echo "[DONE] MuJoCo free-box stance-force refine suite suffix=${SUITE_SUFFIX}"
