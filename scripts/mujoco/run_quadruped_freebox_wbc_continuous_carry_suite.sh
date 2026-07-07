#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-wbc_continuous_carry}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local target_speed="$2"
  local support_scale="$3"
  local box_com_weight="$4"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} wbc_continuous target_speed=${target_speed} scale=${support_scale} box_com_weight=${box_com_weight} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=2400 \
  BOX_MASS=2.0 \
  TARGET_SPEED="${target_speed}" \
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
  CLOSED_LOOP_FOOT_PLACEMENT=1 \
  STRIDE_VELOCITY_GAIN=0.16 \
  STRIDE_POSITION_GAIN=0.0 \
  STRIDE_CLIP=0.12 \
  SUPPORT_CONTROLLER_MODE=wbc_carried_mass_qp \
  SUPPORT_FORCE_SCALE="${support_scale}" \
  SUPPORT_KP_Z=4200.0 \
  SUPPORT_KD_Z=260.0 \
  SUPPORT_KP_ROLL=420.0 \
  SUPPORT_KD_ROLL=64.0 \
  SUPPORT_KP_PITCH=360.0 \
  SUPPORT_KD_PITCH=54.0 \
  SUPPORT_KP_VX=220.0 \
  SUPPORT_MAX_TOTAL_FX=90.0 \
  SUPPORT_MIN_FOOT_FZ=10.0 \
  SUPPORT_MAX_FOOT_FZ=440.0 \
  SUPPORT_MAX_JOINT_TORQUE=380.0 \
  SUPPORT_COM_PRE_LATCH_SCALE=0.0 \
  HOLD_SUPPORT_COM_SCALE=0.0 \
  SUPPORT_FY_PRE_LATCH_SCALE=0.0 \
  HOLD_SUPPORT_FY_SCALE=0.0 \
  SUPPORT_LQR_POST_LATCH_ONLY=0 \
  SUPPORT_LQR_HORIZON_STEPS=90 \
  SUPPORT_LQR_Q_POS=30.0 \
  SUPPORT_LQR_Q_VEL=12.0 \
  SUPPORT_LQR_R=2.0 \
  SUPPORT_LQR_MAX_FX=60.0 \
  SUPPORT_LQR_MAX_FY=180.0 \
  SUPPORT_QP_ITERATIONS=36 \
  SUPPORT_QP_REGULARIZATION=0.001 \
  SUPPORT_QP_FRICTION_MU=0.90 \
  SUPPORT_QP_ANGULAR_WEIGHT=1.0 \
  SUPPORT_QP_MOMENT_CLIP_SCALE=0.50 \
  SUPPORT_WBC_POST_LATCH_ONLY=0 \
  SUPPORT_WBC_INCLUDE_BOX_MASS=1 \
  SUPPORT_WBC_BOX_COM_WEIGHT="${box_com_weight}" \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
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
    --expect-support-controller-mode wbc_carried_mass_qp \
    --min-support-joint-torque-writes 1 \
    --min-support-lqr-active-steps 600 \
    --min-support-qp-active-steps 600 \
    --min-support-wbc-active-steps 600 \
    --require-closed-loop-foot-placement \
    --expect-retention-force-mode relative_spring \
    --min-retention-force-writes 1 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --min-box-travel-x 0.20 \
    --min-final-box-travel-x 0.16 \
    --max-tilt 0.70 \
    --min-box-z 0.58 \
    --max-relative-offset-error 0.22 \
    --max-final-relative-offset-error 0.20 \
    --max-root-pose-writes 0 \
    --max-root-velocity-writes 0 \
    --max-box-pose-writes 0 \
    --max-box-velocity-writes 0 \
    | tee "${check}"
  local check_status=${PIPESTATUS[0]}
  set -e
  echo "[CHECK_STATUS] ${stamp} ${check_status}"
}

run_case wbc_cont_slow     0.08 -0.55 1.0
run_case wbc_cont_medium   0.12 -0.55 1.0
run_case wbc_cont_halfcom  0.12 -0.55 0.5
run_case wbc_cont_stronger 0.12 -0.70 1.0

echo "[DONE] MuJoCo free-box WBC continuous-carry suite suffix=${SUITE_SUFFIX}"
