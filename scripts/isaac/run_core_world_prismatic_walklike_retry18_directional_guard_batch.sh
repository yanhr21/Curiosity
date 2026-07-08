#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

COMMON_ENV=(
  PAYLOAD_MODE=cradle_free_box
  PAYLOAD_MASS=8.0
  TORSO_MASS=36.0
  TORSO_Z=0.62
  ENABLE_HORIZONTAL_LEGS=1
  TARGET_X=-0.17
  GAIT_DRIVE_TARGET_X=-0.42
  GUARDED_STOP_TARGET_X=-0.17
  STEP_LENGTH=0.10
  STEP_HEIGHT=0.05
  GAIT_PERIOD_STEPS=300
  SETTLE_STEPS=260
  RAMP_STEPS=260
  STANCE_HALF_LENGTH=0.65
  STANCE_HALF_WIDTH=0.24
  FOOT_LENGTH=0.65
  FOOT_WIDTH=0.18
  LEG_TARGET=-0.57
  LEG_LOWER=-0.82
  LEG_UPPER=-0.25
  LEG_STIFFNESS=36000
  LEG_DAMPING=3600
  LEG_MAX_FORCE=68000
  X_SLIDE_LIMIT=0.28
  X_SLIDE_STIFFNESS=36000
  X_SLIDE_DAMPING=3600
  X_SLIDE_MAX_FORCE=76000
  MOTION_MODE=guarded_prelift_quasistatic_step_cycle
  PRELIFT_RESET_LIFT_FRACTION=0.35
  PRELIFT_RESET_LOWER_FRACTION=0.35
  GATED_STEP_MAX_TRAVEL_LOSS=0.04
  GATED_STEP_RECOVERY_PHASE=0.35
  STEPS=2800
  PAYLOAD_LOCAL_Z=0.18
)

env "${COMMON_ENV[@]}" \
  STAMP=20260705_prismatic_cradle_walklike_directional_guard_high_mid_x050_z018_retry18a \
  PAYLOAD_LOCAL_X=0.50 \
  GUARDED_STEP_TARGET_TOLERANCE=0.030 \
  bash scripts/isaac/run_core_world_prismatic_carrier_stand.sh

env "${COMMON_ENV[@]}" \
  STAMP=20260705_prismatic_cradle_walklike_directional_guard_high_close_x045_z018_overdrive16_retry18b \
  PAYLOAD_LOCAL_X=0.45 \
  PRELIFT_STANCE_OVERDRIVE=1.6 \
  GUARDED_STEP_TARGET_TOLERANCE=0.030 \
  bash scripts/isaac/run_core_world_prismatic_carrier_stand.sh
