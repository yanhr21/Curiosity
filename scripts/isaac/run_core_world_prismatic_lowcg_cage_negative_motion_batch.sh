#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run low-CG negative-motion batch on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-30}"
python3 -m py_compile scripts/isaac/build_core_world_prismatic_carrier_stand.py

echo "[BATCH] lowcg_diag4 stance_translate target=-0.03"
STAMP=20260705_lowcg_cage_robot_side_diag4_translate_neg3cm \
TARGET_X=-0.030 \
STEPS=700 \
SETTLE_STEPS=350 \
RAMP_STEPS=300 \
bash scripts/isaac/run_core_world_prismatic_lowcg_cage_translate.sh

echo "[BATCH] lowcg_diag5 creep target=-0.03"
STAMP=20260705_lowcg_cage_robot_side_diag5_creep_neg3cm \
TARGET_X=-0.030 \
STEPS=900 \
SETTLE_STEPS=350 \
RAMP_STEPS=300 \
STEP_LENGTH=0.020 \
STEP_HEIGHT=0.020 \
GAIT_PERIOD_STEPS=480 \
bash scripts/isaac/run_core_world_prismatic_lowcg_cage_creep.sh
