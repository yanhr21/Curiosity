#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run low-CG cage robot-side batch on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-30}"
python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py

echo "[BATCH] lowcg_diag1 stance_translate target=0.03"
STAMP=20260705_lowcg_cage_robot_side_diag1_translate_3cm \
TARGET_X=0.030 \
STEPS=700 \
SETTLE_STEPS=350 \
RAMP_STEPS=300 \
bash scripts/isaac/run_core_world_prismatic_lowcg_cage_translate.sh

echo "[BATCH] lowcg_diag2 stance_translate target=0.06"
STAMP=20260705_lowcg_cage_robot_side_diag2_translate_6cm \
TARGET_X=0.060 \
STEPS=760 \
SETTLE_STEPS=350 \
RAMP_STEPS=360 \
bash scripts/isaac/run_core_world_prismatic_lowcg_cage_translate.sh

echo "[BATCH] lowcg_diag3 sync_inchworm target=0.03"
STAMP=20260705_lowcg_cage_robot_side_diag3_sync_3cm \
TARGET_X=0.030 \
STEPS=1000 \
SETTLE_STEPS=260 \
GAIT_PERIOD_STEPS=320 \
STEP_LENGTH=0.015 \
STEP_HEIGHT=0.020 \
bash scripts/isaac/run_core_world_prismatic_lowcg_cage_sync_inchworm.sh
