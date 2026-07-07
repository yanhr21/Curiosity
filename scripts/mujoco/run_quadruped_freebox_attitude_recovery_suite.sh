#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-attitude_recovery}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local start_tilt="$2"
  local full_tilt="$3"
  local roll_kp_delta="$4"
  local roll_kd_delta="$5"
  local hip_delta="$6"
  local foot_delta="$7"
  local height_offset="$8"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} recovery tilt=${start_tilt}/${full_tilt} roll=${roll_kp_delta}/${roll_kd_delta} hip=${hip_delta} foot=${foot_delta} height=${height_offset} host=$(hostname)"
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
  HIP_ROLL_BASE=0.05 \
  HIP_ROLL_FEEDBACK_GAIN=0.0 \
  HOLD_STANCE_FOOT_Z_DOWN=0.46 \
  HOLD_HIP_ROLL_BASE=0.08 \
  HOLD_HIP_ROLL_FEEDBACK_GAIN=-0.30 \
  HOLD_FOOT_ROLL_Z_GAIN=-0.04 \
  HOLD_FRONT_FOOT_X=0.22 \
  HOLD_REAR_FOOT_X=-0.22 \
  HOLD_PITCH_FOOT_X_GAIN=0.0 \
  CLOSED_LOOP_FOOT_PLACEMENT=1 \
  STRIDE_VELOCITY_GAIN=0.16 \
  STRIDE_POSITION_GAIN=0.40 \
  STRIDE_CLIP=0.12 \
  SUPPORT_CONTROLLER_MODE=lqr_additive_stance_force \
  SUPPORT_FORCE_SCALE=-0.36 \
  SUPPORT_KP_Z=4200.0 \
  SUPPORT_KD_Z=240.0 \
  SUPPORT_KP_ROLL=420.0 \
  SUPPORT_KD_ROLL=58.0 \
  SUPPORT_KP_PITCH=360.0 \
  SUPPORT_KD_PITCH=48.0 \
  SUPPORT_KP_VX=220.0 \
  SUPPORT_MAX_TOTAL_FX=90.0 \
  SUPPORT_MIN_FOOT_FZ=10.0 \
  SUPPORT_MAX_FOOT_FZ=380.0 \
  SUPPORT_MAX_JOINT_TORQUE=340.0 \
  HOLD_SUPPORT_FX_SCALE=0.30 \
  HOLD_SUPPORT_MAX_FX_SCALE=0.60 \
  HOLD_SUPPORT_KP_VX_SCALE=1.5 \
  HOLD_SUPPORT_KD_Z_SCALE=1.5 \
  HOLD_SUPPORT_KD_ROLL_SCALE=1.5 \
  HOLD_SUPPORT_KD_PITCH_SCALE=1.5 \
  SUPPORT_COM_PRE_LATCH_SCALE=0.0 \
  HOLD_SUPPORT_COM_SCALE=0.0 \
  SUPPORT_FY_PRE_LATCH_SCALE=0.0 \
  HOLD_SUPPORT_FY_SCALE=0.0 \
  SUPPORT_LQR_POST_LATCH_ONLY=1 \
  SUPPORT_LQR_HORIZON_STEPS=90 \
  SUPPORT_LQR_Q_POS=30.0 \
  SUPPORT_LQR_Q_VEL=12.0 \
  SUPPORT_LQR_R=2.0 \
  SUPPORT_LQR_MAX_FX=60.0 \
  SUPPORT_LQR_MAX_FY=180.0 \
  SUPPORT_ATTITUDE_RECOVERY=1 \
  SUPPORT_ATTITUDE_RECOVERY_START_TILT="${start_tilt}" \
  SUPPORT_ATTITUDE_RECOVERY_FULL_TILT="${full_tilt}" \
  SUPPORT_ATTITUDE_RECOVERY_ROLL_KP_DELTA="${roll_kp_delta}" \
  SUPPORT_ATTITUDE_RECOVERY_ROLL_KD_DELTA="${roll_kd_delta}" \
  SUPPORT_ATTITUDE_RECOVERY_PITCH_KP_DELTA=120.0 \
  SUPPORT_ATTITUDE_RECOVERY_PITCH_KD_DELTA=20.0 \
  SUPPORT_ATTITUDE_RECOVERY_HIP_ROLL_DELTA="${hip_delta}" \
  SUPPORT_ATTITUDE_RECOVERY_FOOT_ROLL_Z_DELTA="${foot_delta}" \
  SUPPORT_ATTITUDE_RECOVERY_HEIGHT_OFFSET="${height_offset}" \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
  STOP_AFTER_BOX_TRAVEL=0.05 \
  HOLD_TARGET_SPEED=0.00 \
  RETENTION_FORCE_MODE=relative_spring \
  RETENTION_KP_X=380.0 \
  RETENTION_KD_X=42.0 \
  RETENTION_KP_Y=220.0 \
  RETENTION_KD_Y=34.0 \
  RETENTION_KP_Z=300.0 \
  RETENTION_KD_Z=34.0 \
  RETENTION_MAX_FORCE_X=140.0 \
  RETENTION_MAX_FORCE_Y=100.0 \
  RETENTION_MAX_FORCE_Z=115.0 \
  bash scripts/mujoco/run_quadruped_freebox_carry.sh

  set +e
  "${PYTHON_BIN}" scripts/mujoco/check_quadruped_freebox_summary.py "${summary}" \
    --expect-assist-mode none \
    --expect-leg-drive-mode foot_ik \
    --expect-support-controller-mode lqr_additive_stance_force \
    --min-support-joint-torque-writes 1 \
    --min-support-lqr-active-steps 600 \
    --min-support-attitude-recovery-active-steps 1 \
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

run_case recover_roll_gain 0.34 0.62 420.0 80.0 0.0 0.0 0.00
run_case recover_posture 0.34 0.62 0.0 0.0 -0.70 -0.10 0.00
run_case recover_combo 0.34 0.62 360.0 70.0 -0.60 -0.08 0.02
run_case recover_combo_early 0.28 0.56 300.0 60.0 -0.55 -0.07 0.02

echo "[DONE] MuJoCo free-box attitude recovery suite suffix=${SUITE_SUFFIX}"
