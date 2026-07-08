#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run carry posture suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-20260706_direct_carry_posture_suite_64cm_8kg}"
SUITE_DIR="${ROOT_DIR}/experiments/outputs/direct_carry_posture_suite/${SUITE_STAMP}"
LOG_DIR="${ROOT_DIR}/logs/direct_carry_posture_suite"
mkdir -p "${SUITE_DIR}" "${LOG_DIR}"
cd "${ROOT_DIR}"

python3 -m py_compile \
  scripts/isaac/normalize_direct_carry_backend_summary.py \
  scripts/isaac/check_direct_carry_task_summary.py \
  scripts/isaac/summarize_direct_carry_posture_suite.py
bash -n scripts/isaac/run_direct_carry_task_physical_backend.sh

run_case() {
  local posture="$1"
  local stamp="${SUITE_STAMP}_${posture}"
  local output_dir="experiments/outputs/direct_carry_task_physical_backend/${stamp}"
  local backend_dir="${output_dir}/backend_anchored_cradle"
  local backend_summary="${backend_dir}/core_world_anchored_footstep_carrier_summary.json"
  local direct_summary="${output_dir}/direct_carry_task_physical_backend_summary.json"
  local backend_log="logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_${stamp}_backend_anchored_cradle.log"

  echo "[SUITE] posture=${posture} stamp=${stamp}"
  set +e
  STAMP="${stamp}" \
  OUTPUT_DIR="${output_dir}" \
  BACKEND_OUTPUT_DIR="${backend_dir}" \
  BACKEND_STAMP="${stamp}_backend_anchored_cradle" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE="${posture}" \
  TARGET_X=0.64 \
  PAYLOAD_MASS=8.0 \
  STEPS=3580 \
  STEP_LENGTH=0.016 \
  STANCE_STEPS=80 \
  SETTLE_STEPS=10 \
  RAIL_JOINT_COUNT=2 \
  RAIL_LOWER=-0.04 \
  RAIL_UPPER=0.10 \
  SUPPORT_FOOT_MASS=8.0 \
  SUPPORT_FOOT_X_LOWER=-0.17 \
  SUPPORT_FOOT_X_UPPER=0.17 \
  SUPPORT_FOOT_Z_LOWER=-0.005 \
  SUPPORT_FOOT_Z_UPPER=0.24 \
  SUPPORT_FOOT_STEP_HEIGHT=0.120 \
  SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=0.12 \
  SUPPORT_FOOT_STANCE_X=-0.130 \
  SUPPORT_FOOT_SWING_X=0.130 \
  SUPPORT_FOOT_CONTACT_Z_THRESHOLD=0.035 \
  SUPPORT_FOOT_DRIVE_STIFFNESS=24000.0 \
  SUPPORT_FOOT_DRIVE_DAMPING=3400.0 \
  SUPPORT_FOOT_DRIVE_MAX_FORCE=110000.0 \
  SUPPORT_FOOT_Z_DRIVE_STIFFNESS=36000.0 \
  SUPPORT_FOOT_Z_DRIVE_DAMPING=3200.0 \
  SUPPORT_FOOT_Z_DRIVE_MAX_FORCE=130000.0 \
  DRIVE_STIFFNESS=22000.0 \
  DRIVE_DAMPING=3500.0 \
  DRIVE_MAX_FORCE=80000.0 \
  STATIC_FRICTION=4.5 \
  DYNAMIC_FRICTION=4.0 \
  bash scripts/isaac/run_direct_carry_task_physical_backend.sh \
    2>&1 | tee "${LOG_DIR}/${stamp}.log"
  local wrapper_status=${PIPESTATUS[0]}
  set -e

  if [[ "${wrapper_status}" -ne 0 ]]; then
    echo "[WARN] run_direct wrapper exited ${wrapper_status} for ${posture}; attempting recovery if backend summary exists." >&2
  fi
  if [[ ! -f "${backend_summary}" ]]; then
    echo "[ERROR] Backend summary missing for ${posture}: ${backend_summary}" >&2
    exit "${wrapper_status}"
  fi
  if [[ ! -f "${direct_summary}" ]]; then
    python3 scripts/isaac/normalize_direct_carry_backend_summary.py \
      --backend-summary "${backend_summary}" \
      --backend-log "${backend_log}" \
      --controller-mode physical_alternating_anchor_feet_cradle \
      --carry-posture "${posture}" \
      --output-summary "${direct_summary}"
  fi

  python3 scripts/isaac/check_direct_carry_task_summary.py \
    "${direct_summary}" \
    --min-steps 3560 \
    --expect-controller-mode physical_alternating_anchor_feet_cradle \
    --expect-carry-posture "${posture}" \
    --expect-backend-support-mode dynamic_anchor \
    --expect-support-foot-mode xz_prismatic_to_anchor \
    --min-support-foot-joint-count 8 \
    --min-support-foot-z-joint-count 4 \
    --min-support-foot-x-joint-motion 0.35 \
    --min-support-foot-z-joint-motion 0.15 \
    --min-actual-support-foot-lift 0.02 \
    --min-drive-near-ground-foot-count 2 \
    --max-drive-near-ground-zero-steps 0 \
    --max-drive-near-ground-lt2-steps 0 \
    --min-commanded-stance-near-ground-foot-count 2 \
    --max-commanded-stance-near-ground-lt2-steps 0 \
    --min-support-polygon-margin 0.12 \
    --min-box-travel 0.52 \
    --max-final-box-target-distance-x 0.08 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --require-root-shortcut-free \
    --max-support-root-pose-write-count 0 \
    --max-anchor-world-joint-retarget-count 0 \
    --max-foot-pose-write-count 0 \
    --max-stance-anchor-pose-write-count 0 \
    --forbid-fixed-world-support \
    --require-non-success-claim
}

run_case front_mid
run_case low_front
run_case chest_high

python3 scripts/isaac/summarize_direct_carry_posture_suite.py \
  --summary "experiments/outputs/direct_carry_task_physical_backend/${SUITE_STAMP}_front_mid/direct_carry_task_physical_backend_summary.json" \
  --summary "experiments/outputs/direct_carry_task_physical_backend/${SUITE_STAMP}_low_front/direct_carry_task_physical_backend_summary.json" \
  --summary "experiments/outputs/direct_carry_task_physical_backend/${SUITE_STAMP}_chest_high/direct_carry_task_physical_backend_summary.json" \
  --output "${SUITE_DIR}/direct_carry_posture_suite_summary.json" \
  --min-postures 3 \
  --min-steps 3560 \
  --min-box-travel-x 0.52 \
  --max-target-distance-x 0.08 \
  --max-tilt 0.14 \
  --min-support-margin 0.12

echo "[INFO] Suite output: ${SUITE_DIR}"
