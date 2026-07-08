#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

sleep 8
sed -n '160,166p' scripts/isaac/build_core_world_prismatic_carrier_stand.py
python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py

env \
  PAYLOAD_MODE=cradle_free_box \
  PAYLOAD_MASS=8.0 \
  TORSO_MASS=36.0 \
  TORSO_Z=0.62 \
  ENABLE_HORIZONTAL_LEGS=1 \
  TARGET_X=-0.17 \
  GAIT_DRIVE_TARGET_X=-0.23 \
  STEP_LENGTH=0.07 \
  STEP_HEIGHT=0.05 \
  GAIT_PERIOD_STEPS=360 \
  SETTLE_STEPS=260 \
  RAMP_STEPS=260 \
  STANCE_HALF_LENGTH=0.65 \
  STANCE_HALF_WIDTH=0.24 \
  FOOT_LENGTH=0.65 \
  FOOT_WIDTH=0.18 \
  LEG_TARGET=-0.57 \
  LEG_LOWER=-0.82 \
  LEG_UPPER=-0.25 \
  LEG_STIFFNESS=32000 \
  LEG_DAMPING=3200 \
  LEG_MAX_FORCE=56000 \
  X_SLIDE_LIMIT=0.20 \
  X_SLIDE_STIFFNESS=30000 \
  X_SLIDE_DAMPING=3000 \
  X_SLIDE_MAX_FORCE=62000 \
  MOTION_MODE=prelift_quasistatic_step_cycle \
  PRELIFT_RESET_LIFT_FRACTION=0.35 \
  PRELIFT_RESET_LOWER_FRACTION=0.35 \
  STEPS=2800 \
  STAMP=20260705_prismatic_cradle_walklike_close_low_x045_z014_retry14c_retry2 \
  PAYLOAD_LOCAL_X=0.45 \
  PAYLOAD_LOCAL_Z=0.14 \
  bash scripts/isaac/run_core_world_prismatic_carrier_stand.sh
