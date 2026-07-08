#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
SUITE_SUFFIX="${SUITE_SUFFIX:-foot_ik_static_hold_support}"

cd "${ROOT_DIR}"

run_case() {
  local tag="$1"
  local stride="$2"
  local hip_base="$3"
  local foot_roll="$4"
  local v_gain="$5"
  local p_gain="$6"
  local hold_z="$7"
  local hold_hip="$8"
  local hold_front_x="$9"
  local hold_rear_x="${10}"
  local hold_pitch_gain="${11}"
  local stamp="20260707_mujoco_quad_freebox_2kg_${tag}_${SUITE_SUFFIX}"
  local summary="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_summary.json"
  local check="experiments/outputs/mujoco_quadruped_freebox/${stamp}/mujoco_quadruped_freebox_check.json"

  echo "[CASE] ${stamp} stride=${stride} hip_base=${hip_base} hold_z=${hold_z} hold_hip=${hold_hip} hold_x=${hold_front_x}/${hold_rear_x} pitch_gain=${hold_pitch_gain} host=$(hostname)"
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
  HOLD_STANCE_FOOT_Z_DOWN="${hold_z}" \
  HOLD_HIP_ROLL_BASE="${hold_hip}" \
  HOLD_FRONT_FOOT_X="${hold_front_x}" \
  HOLD_REAR_FOOT_X="${hold_rear_x}" \
  HOLD_PITCH_FOOT_X_GAIN="${hold_pitch_gain}" \
  CLOSED_LOOP_FOOT_PLACEMENT=1 \
  STRIDE_VELOCITY_GAIN="${v_gain}" \
  STRIDE_POSITION_GAIN="${p_gain}" \
  STRIDE_CLIP=0.18 \
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

run_case hold_x10_08 -0.10 0.00 -0.04 0.20 0.30 0.46 0.08 0.10 -0.08 0.00
run_case hold_x14_10 -0.10 0.00 -0.04 0.20 0.30 0.46 0.08 0.14 -0.10 0.00
run_case hold_x18_12 -0.10 0.00 -0.04 0.20 0.30 0.46 0.10 0.18 -0.12 0.00
run_case hold_x14_pitch -0.10 0.00 -0.04 0.20 0.30 0.46 0.08 0.14 -0.10 0.10
run_case hold_hip003 -0.10 0.03 -0.04 0.20 0.30 0.46 0.10 0.14 -0.10 0.00
run_case hold_hip005 -0.10 0.05 -0.04 0.20 0.30 0.46 0.10 0.14 -0.10 0.00

echo "[DONE] MuJoCo free-box foot-IK static hold support suite suffix=${SUITE_SUFFIX}"
