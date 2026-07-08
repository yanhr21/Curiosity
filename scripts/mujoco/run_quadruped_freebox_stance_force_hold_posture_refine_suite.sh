#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-stance_force_holdposture_refine}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local hold_hip_feedback="$2"
  local hold_foot_roll_z="$3"
  local hold_hip_base="$4"
  local hold_stance_z="$5"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} hip_fb=${hold_hip_feedback} foot_roll_z=${hold_foot_roll_z} hip_base=${hold_hip_base} stance_z=${hold_stance_z} host=$(hostname)"
  STAMP="${stamp}" \
  STEPS=3000 \
  BOX_MASS=2.0 \
  TARGET_SPEED=0.16 \
  TARGET_HEIGHT=0.56 \
  ACTUATOR_KP=90.0 \
  ACTUATOR_KV=10.0 \
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
  HOLD_STANCE_FOOT_Z_DOWN="${hold_stance_z}" \
  HOLD_HIP_ROLL_BASE="${hold_hip_base}" \
  HOLD_HIP_ROLL_FEEDBACK_GAIN="${hold_hip_feedback}" \
  HOLD_FOOT_ROLL_Z_GAIN="${hold_foot_roll_z}" \
  HOLD_FRONT_FOOT_X=0.22 \
  HOLD_REAR_FOOT_X=-0.22 \
  HOLD_PITCH_FOOT_X_GAIN=0.0 \
  CLOSED_LOOP_FOOT_PLACEMENT=1 \
  STRIDE_VELOCITY_GAIN=0.16 \
  STRIDE_POSITION_GAIN=0.40 \
  STRIDE_CLIP=0.12 \
  SUPPORT_CONTROLLER_MODE=stance_force \
  SUPPORT_FORCE_SCALE=-0.36 \
  SUPPORT_KP_Z=4400.0 \
  SUPPORT_KD_Z=280.0 \
  SUPPORT_KP_ROLL=440.0 \
  SUPPORT_KD_ROLL=70.0 \
  SUPPORT_KP_PITCH=380.0 \
  SUPPORT_KD_PITCH=54.0 \
  SUPPORT_KP_VX=220.0 \
  SUPPORT_MAX_TOTAL_FX=90.0 \
  SUPPORT_MIN_FOOT_FZ=10.0 \
  SUPPORT_MAX_FOOT_FZ=400.0 \
  SUPPORT_MAX_JOINT_TORQUE=360.0 \
  HOLD_SUPPORT_FX_SCALE=0.30 \
  HOLD_SUPPORT_MAX_FX_SCALE=0.60 \
  HOLD_SUPPORT_KP_VX_SCALE=1.5 \
  HOLD_SUPPORT_KD_Z_SCALE=1.5 \
  HOLD_SUPPORT_KD_ROLL_SCALE=1.8 \
  HOLD_SUPPORT_KD_PITCH_SCALE=1.5 \
  SUPPORT_COM_PRE_LATCH_SCALE=0.0 \
  HOLD_SUPPORT_COM_SCALE=0.0 \
  SUPPORT_FY_PRE_LATCH_SCALE=0.0 \
  HOLD_SUPPORT_FY_SCALE=0.0 \
  TRAY_HALF_LENGTH=0.22 \
  TRAY_HALF_WIDTH=0.25 \
  WALL_HEIGHT=0.30 \
  STOP_AFTER_BOX_TRAVEL=0.05 \
  HOLD_TARGET_SPEED=0.00 \
  RETENTION_FORCE_MODE=relative_spring \
  RETENTION_KP_X=400.0 \
  RETENTION_KD_X=46.0 \
  RETENTION_KP_Y=240.0 \
  RETENTION_KD_Y=38.0 \
  RETENTION_KP_Z=320.0 \
  RETENTION_KD_Z=38.0 \
  RETENTION_MAX_FORCE_X=150.0 \
  RETENTION_MAX_FORCE_Y=110.0 \
  RETENTION_MAX_FORCE_Z=125.0 \
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

run_case refine_hip_neg050 -0.50 -0.04 0.08 0.46
run_case refine_hip_neg060 -0.60 -0.04 0.08 0.46
run_case refine_hip_neg080 -0.80 -0.04 0.08 0.46
run_case refine_hip_neg060_base06 -0.60 -0.04 0.06 0.46
run_case refine_hip_neg060_base10 -0.60 -0.04 0.10 0.46
run_case refine_hip_neg060_footneg10 -0.60 -0.10 0.08 0.46
run_case refine_hip_neg060_footzero -0.60 0.00 0.08 0.46
run_case refine_hip_neg060_high -0.60 -0.04 0.08 0.44

echo "[DONE] MuJoCo free-box stance-force hold posture refine suite suffix=${SUITE_SUFFIX}"
