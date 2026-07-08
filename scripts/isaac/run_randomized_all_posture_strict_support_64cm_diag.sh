#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

STAMP="${STAMP:-20260705_randomized_all_posture_strict_support_64cm_seed7061}"
BOX_SEED="${BOX_SEED:-7061}"
ROOT_OUTPUT="experiments/outputs/randomized_all_posture_strict_support/${STAMP}"
mkdir -p "${ROOT_OUTPUT}"

run_case() {
  local posture="$1"
  local output_dir="${ROOT_OUTPUT}/${posture}"
  local backend_dir="${output_dir}/backend_anchored_cradle"
  local backend_summary="${backend_dir}/core_world_anchored_footstep_carrier_summary.json"
  local direct_summary="${output_dir}/direct_carry_task_physical_backend_summary.json"
  local backend_stamp="${STAMP}_${posture}_backend_anchored_cradle"
  local backend_log="logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_${backend_stamp}.log"

  echo "[CASE] posture=${posture} box_seed=${BOX_SEED}"
  mkdir -p "${output_dir}" "${backend_dir}"
  set +e
  STAMP="${STAMP}_${posture}" \
  OUTPUT_DIR="${output_dir}" \
  BACKEND_OUTPUT_DIR="${backend_dir}" \
  BACKEND_STAMP="${backend_stamp}" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE="${posture}" \
  TARGET_X=0.64 \
  PAYLOAD_MASS=8.0 \
  RANDOMIZE_PAYLOAD=1 \
  BOX_SEED="${BOX_SEED}" \
  PAYLOAD_MASS_MIN=4.0 \
  PAYLOAD_MASS_MAX=12.0 \
  PAYLOAD_SIZE_JITTER=0.10 \
  PAYLOAD_COM_OFFSET_RANGE_X=0.04 \
  PAYLOAD_COM_OFFSET_RANGE_Y=0.03 \
  PAYLOAD_COM_OFFSET_RANGE_Z=0.03 \
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
  ENABLE_SUPPORT_FOOT_CONTACT_REPORT=1 \
  SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD=0.0 \
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
    --enable-support-foot-contact-report \
    --support-foot-contact-report-threshold 0.0
  local wrapper_status=$?
  set -e
  if [[ "${wrapper_status}" -ne 0 ]]; then
    echo "[WARN] wrapper exited ${wrapper_status} for ${posture}; attempting summary recovery." >&2
  fi
  if [[ ! -f "${direct_summary}" ]]; then
    if [[ ! -f "${backend_summary}" ]]; then
      echo "[ERROR] Missing backend summary for ${posture}: ${backend_summary}" >&2
      exit "${wrapper_status}"
    fi
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
    --require-box-randomized \
    --expect-box-seed "${BOX_SEED}" \
    --expect-support-foot-mode xz_prismatic_to_anchor \
    --min-support-foot-joint-count 8 \
    --min-support-foot-z-joint-count 4 \
    --min-support-foot-x-joint-motion 0.35 \
    --min-support-foot-z-joint-motion 0.15 \
    --min-actual-support-foot-lift 0.02 \
    --min-drive-near-ground-foot-count 2 \
    --max-drive-near-ground-zero-steps 0 \
    --max-drive-near-ground-lt2-steps 0 \
    --require-support-foot-contact-report-evidence \
    --min-drive-contact-report-foot-count 2 \
    --max-drive-contact-report-zero-steps 0 \
    --max-drive-contact-report-lt2-steps 0 \
    --min-commanded-stance-contact-report-foot-count 2 \
    --max-commanded-stance-contact-report-lt2-steps 0 \
    --min-commanded-stance-near-ground-foot-count 2 \
    --max-commanded-stance-near-ground-lt2-steps 0 \
    --min-box-travel 0.52 \
    --max-final-box-target-distance-x 0.18 \
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

python3 scripts/isaac/summarize_randomized_all_posture_carry.py \
  --root "${ROOT_OUTPUT}" \
  --box-seed "${BOX_SEED}" \
  --output "${ROOT_OUTPUT}/randomized_all_posture_strict_support_summary.json"

echo "[INFO] Randomized all-posture output: ${ROOT_OUTPUT}"
