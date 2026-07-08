#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
RUN_PREFIX="${RUN_PREFIX:-20260705_no_root_prismatic_posture_sweep}"
TARGET_X="${TARGET_X:-0.10}"
PAYLOAD_MASS="${PAYLOAD_MASS:-8.0}"
STEPS="${STEPS:-1200}"
MAX_FINAL_BOX_TARGET_DISTANCE_X="${MAX_FINAL_BOX_TARGET_DISTANCE_X:-0.025}"
MIN_BOX_TRAVEL="${MIN_BOX_TRAVEL:-0.08}"

cd "${ROOT_DIR}"

for posture in front_mid low_front chest_high; do
  stamp="${RUN_PREFIX}_${posture}_$(hostname)"
  echo "[SWEEP] posture=${posture} stamp=${stamp}"
  STAMP="${stamp}" \
  CARRY_POSTURE="${posture}" \
  STEPS="${STEPS}" \
  TARGET_X="${TARGET_X}" \
  PAYLOAD_MASS="${PAYLOAD_MASS}" \
  MOTION_MODE="quasistatic_stance_transfer" \
  QUASISTATIC_COMPENSATE_SETTLE_DRIFT="1" \
  X_SLIDE_LIMIT="${X_SLIDE_LIMIT:-0.20}" \
  bash scripts/isaac/run_direct_carry_task_no_root_prismatic_backend.sh

  summary="experiments/outputs/direct_carry_task_no_root_prismatic_backend/${stamp}/direct_carry_task_no_root_prismatic_backend_summary.json"
  python3 scripts/isaac/check_direct_carry_task_summary.py "${summary}" \
    --min-steps "${STEPS}" \
    --expect-controller-mode no_root_prismatic_legged_cradle \
    --expect-carry-posture "${posture}" \
    --expect-backend-support-mode no_root_prismatic_legged \
    --min-box-travel "${MIN_BOX_TRAVEL}" \
    --max-final-box-target-distance-x "${MAX_FINAL_BOX_TARGET_DISTANCE_X}" \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --require-root-shortcut-free \
    --max-anchor-world-joint-retarget-count 0 \
    --max-support-root-pose-write-count 0 \
    --require-non-success-claim
done
