#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

STAMP=20260705_direct_physical_backend_alternating_anchor_feet_8cm_8kg_frontmid \
SUPPORT_MODE=alternating_anchor_feet \
CARRY_POSTURE=front_mid \
TARGET_X=0.08 \
PAYLOAD_MASS=8.0 \
STEPS=620 \
STEP_LENGTH=0.04 \
STANCE_STEPS=120 \
SETTLE_STEPS=80 \
RAIL_JOINT_COUNT=2 \
RAIL_LOWER=-0.04 \
RAIL_UPPER=0.10 \
SUPPORT_FOOT_MASS=30.0 \
SUPPORT_FOOT_X_LOWER=-0.12 \
SUPPORT_FOOT_X_UPPER=0.12 \
SUPPORT_FOOT_Z_LOWER=-0.005 \
SUPPORT_FOOT_Z_UPPER=0.12 \
SUPPORT_FOOT_STEP_HEIGHT=0.055 \
SUPPORT_FOOT_STANCE_X=-0.060 \
SUPPORT_FOOT_SWING_X=0.060 \
SUPPORT_FOOT_DRIVE_STIFFNESS=18000.0 \
SUPPORT_FOOT_DRIVE_DAMPING=3000.0 \
SUPPORT_FOOT_DRIVE_MAX_FORCE=90000.0 \
SUPPORT_FOOT_Z_DRIVE_STIFFNESS=15000.0 \
SUPPORT_FOOT_Z_DRIVE_DAMPING=2200.0 \
SUPPORT_FOOT_Z_DRIVE_MAX_FORCE=80000.0 \
DRIVE_STIFFNESS=22000.0 \
DRIVE_DAMPING=3500.0 \
DRIVE_MAX_FORCE=80000.0 \
STATIC_FRICTION=4.0 \
DYNAMIC_FRICTION=3.5 \
bash scripts/isaac/run_direct_carry_task_physical_backend.sh

python3 scripts/isaac/check_direct_carry_task_summary.py \
  experiments/outputs/direct_carry_task_physical_backend/20260705_direct_physical_backend_alternating_anchor_feet_8cm_8kg_frontmid/direct_carry_task_physical_backend_summary.json \
  --min-steps 600 \
  --expect-controller-mode physical_alternating_anchor_feet_cradle \
  --expect-backend-support-mode dynamic_anchor \
  --expect-support-foot-mode xz_prismatic_to_anchor \
  --min-support-foot-joint-count 8 \
  --min-support-foot-z-joint-count 4 \
  --min-support-foot-x-joint-motion 0.08 \
  --min-support-foot-z-joint-motion 0.02 \
  --min-box-travel 0.04 \
  --max-final-box-target-distance-x 0.05 \
  --max-fall-events 0 \
  --max-box-drop-events 0 \
  --require-root-shortcut-free \
  --max-support-root-pose-write-count 0 \
  --max-anchor-world-joint-retarget-count 0 \
  --max-foot-pose-write-count 0 \
  --max-stance-anchor-pose-write-count 0 \
  --forbid-fixed-world-support \
  --require-non-success-claim
