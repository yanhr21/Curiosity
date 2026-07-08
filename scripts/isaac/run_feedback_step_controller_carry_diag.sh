#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation/checking on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_feedback_step_controller_carry}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/feedback_step_controller_carry/${STAMP}}"
CHECK_REPORT="${OUTPUT_DIR}/feedback_step_controller_check.json"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

echo "[CONFIG] feedback-step carry diagnostic STAMP=${STAMP} OUTPUT_DIR=${OUTPUT_DIR}"

CASE_ENV=(
  "STAMP=${STAMP}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "SUPPORT_MODE=${SUPPORT_MODE:-alternating_anchor_feet}"
  "CARRY_POSTURE=${CARRY_POSTURE:-front_mid}"
  "CONTROLLER_MODE=${CONTROLLER_MODE:-physical_alternating_anchor_feet_cradle}"
  "STEPS=${STEPS:-3580}"
  "TARGET_X=${TARGET_X:-0.64}"
  "STEP_LENGTH=${STEP_LENGTH:-0.016}"
  "STANCE_STEPS=${STANCE_STEPS:-80}"
  "SETTLE_STEPS=${SETTLE_STEPS:-10}"
  "STOP_THRESHOLD=${STOP_THRESHOLD:-0.002}"
  "SUPPORT_FOOT_STANCE_X=${SUPPORT_FOOT_STANCE_X:--0.130}"
  "SUPPORT_FOOT_SWING_X=${SUPPORT_FOOT_SWING_X:-0.130}"
  "SUPPORT_FOOT_STEP_HEIGHT=${SUPPORT_FOOT_STEP_HEIGHT:-0.100}"
  "SUPPORT_FOOT_CONTACT_Z_THRESHOLD=${SUPPORT_FOOT_CONTACT_Z_THRESHOLD:-0.055}"
  "ENABLE_SUPPORT_FOOT_CONTACT_REPORT=${ENABLE_SUPPORT_FOOT_CONTACT_REPORT:-1}"
  "SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD=${SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD:-0.0}"
  "SUPPORT_FOOT_EFFORT_CONTACT_THRESHOLD=${SUPPORT_FOOT_EFFORT_CONTACT_THRESHOLD:-0.001}"
  "SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=${SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION:-0.18}"
  "SUPPORT_FOOT_CONTINUITY_GRACE_STEPS=${SUPPORT_FOOT_CONTINUITY_GRACE_STEPS:-12}"
  "SUPPORT_FOOT_MASS=${SUPPORT_FOOT_MASS:-8.0}"
  "SUPPORT_FOOT_X_LOWER=${SUPPORT_FOOT_X_LOWER:--0.18}"
  "SUPPORT_FOOT_X_UPPER=${SUPPORT_FOOT_X_UPPER:-0.18}"
  "SUPPORT_FOOT_Z_LOWER=${SUPPORT_FOOT_Z_LOWER:--0.005}"
  "SUPPORT_FOOT_Z_UPPER=${SUPPORT_FOOT_Z_UPPER:-0.24}"
  "SUPPORT_FOOT_DRIVE_STIFFNESS=${SUPPORT_FOOT_DRIVE_STIFFNESS:-24000.0}"
  "SUPPORT_FOOT_DRIVE_DAMPING=${SUPPORT_FOOT_DRIVE_DAMPING:-3400.0}"
  "SUPPORT_FOOT_DRIVE_MAX_FORCE=${SUPPORT_FOOT_DRIVE_MAX_FORCE:-110000.0}"
  "SUPPORT_FOOT_Z_DRIVE_STIFFNESS=${SUPPORT_FOOT_Z_DRIVE_STIFFNESS:-36000.0}"
  "SUPPORT_FOOT_Z_DRIVE_DAMPING=${SUPPORT_FOOT_Z_DRIVE_DAMPING:-3200.0}"
  "SUPPORT_FOOT_Z_DRIVE_MAX_FORCE=${SUPPORT_FOOT_Z_DRIVE_MAX_FORCE:-130000.0}"
  "FEEDBACK_STEP_CONTROLLER=${FEEDBACK_STEP_CONTROLLER:-1}"
  "FEEDBACK_STEP_X_GAIN=${FEEDBACK_STEP_X_GAIN:-0.015}"
  "FEEDBACK_STEP_X_LIMIT=${FEEDBACK_STEP_X_LIMIT:-0.008}"
  "FEEDBACK_STEP_TILT_GAIN=${FEEDBACK_STEP_TILT_GAIN:-0.05}"
  "FEEDBACK_STEP_TILT_LIMIT=${FEEDBACK_STEP_TILT_LIMIT:-0.005}"
  "RANDOMIZE_PAYLOAD=${RANDOMIZE_PAYLOAD:-1}"
  "BOX_SEED=${BOX_SEED:-7076}"
  "PAYLOAD_MASS_MIN=${PAYLOAD_MASS_MIN:-4.0}"
  "PAYLOAD_MASS_MAX=${PAYLOAD_MASS_MAX:-12.0}"
  "PAYLOAD_SIZE_JITTER=${PAYLOAD_SIZE_JITTER:-0.10}"
  "PAYLOAD_COM_OFFSET_RANGE_X=${PAYLOAD_COM_OFFSET_RANGE_X:-0.04}"
  "PAYLOAD_COM_OFFSET_RANGE_Y=${PAYLOAD_COM_OFFSET_RANGE_Y:-0.04}"
  "PAYLOAD_COM_OFFSET_RANGE_Z=${PAYLOAD_COM_OFFSET_RANGE_Z:-0.03}"
  "RAIL_JOINT_COUNT=${RAIL_JOINT_COUNT:-2}"
  "RAIL_LOWER=${RAIL_LOWER:--0.04}"
  "RAIL_UPPER=${RAIL_UPPER:-0.10}"
  "DRIVE_STIFFNESS=${DRIVE_STIFFNESS:-22000.0}"
  "DRIVE_DAMPING=${DRIVE_DAMPING:-3500.0}"
  "DRIVE_MAX_FORCE=${DRIVE_MAX_FORCE:-80000.0}"
  "STATIC_FRICTION=${STATIC_FRICTION:-4.5}"
  "DYNAMIC_FRICTION=${DYNAMIC_FRICTION:-4.0}"
  "DEVICE=${DEVICE:-cpu}"
)
env "${CASE_ENV[@]}" bash scripts/isaac/run_direct_carry_task_physical_backend.sh \
  --enable-support-foot-contact-report \
  --support-foot-contact-report-threshold "${SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD:-0.0}"

"${ISAAC_VENV}/bin/python" scripts/isaac/check_direct_carry_task_summary.py \
  "${OUTPUT_DIR}/direct_carry_task_physical_backend_summary.json" \
  --min-steps "${STEPS:-3580}" \
  --expect-controller-mode "${CONTROLLER_MODE:-physical_alternating_anchor_feet_cradle}" \
  --expect-carry-posture "${CARRY_POSTURE:-front_mid}" \
  --expect-backend-support-mode "dynamic_anchor" \
  --require-box-randomized \
  --expect-box-seed "${BOX_SEED:-7076}" \
  --min-box-travel "${MIN_BOX_TRAVEL:-0.50}" \
  --min-post-settle-box-travel-x "${MIN_POST_SETTLE_BOX_TRAVEL_X:-0.50}" \
  --max-final-post-settle-box-target-distance-x "${MAX_FINAL_POST_SETTLE_BOX_TARGET_DISTANCE_X:-0.20}" \
  --max-post-settle-box-travel-loss-after-peak "${MAX_POST_SETTLE_BOX_TRAVEL_LOSS_AFTER_PEAK:-0.05}" \
  --max-fall-events "${MAX_FALL_EVENTS:-0}" \
  --max-box-drop-events "${MAX_BOX_DROP_EVENTS:-0}" \
  --require-root-shortcut-free \
  --max-anchor-world-joint-retarget-count 0 \
  --max-support-root-pose-write-count 0 \
  --max-foot-pose-write-count 0 \
  --max-stance-anchor-pose-write-count 0 \
  --expect-support-foot-mode "xz_prismatic_to_anchor" \
  --require-feedback-step-controller \
  --min-feedback-step-applied-steps "${MIN_FEEDBACK_STEP_APPLIED_STEPS:-100}" \
  --max-rail-joint-motion "${MAX_RAIL_JOINT_MOTION:-0.025}" \
  --min-support-foot-joint-count 8 \
  --min-support-foot-x-joint-motion "${MIN_SUPPORT_FOOT_X_JOINT_MOTION:-0.20}" \
  --min-support-foot-z-joint-count 4 \
  --min-support-foot-z-joint-motion "${MIN_SUPPORT_FOOT_Z_JOINT_MOTION:-0.04}" \
  --min-actual-support-foot-lift "${MIN_ACTUAL_SUPPORT_FOOT_LIFT:-0.03}" \
  --min-drive-near-ground-foot-count "${MIN_DRIVE_NEAR_GROUND_FOOT_COUNT:-2}" \
  --max-drive-near-ground-lt2-steps "${MAX_DRIVE_NEAR_GROUND_LT2_STEPS:-0}" \
  --require-support-foot-contact-report-evidence \
  --min-drive-contact-report-foot-count "${MIN_DRIVE_CONTACT_REPORT_FOOT_COUNT:-2}" \
  --max-drive-contact-report-lt2-steps "${MAX_DRIVE_CONTACT_REPORT_LT2_STEPS:-0}" \
  --min-commanded-stance-contact-report-foot-count "${MIN_COMMANDED_STANCE_CONTACT_REPORT_FOOT_COUNT:-2}" \
  --max-commanded-stance-contact-report-lt2-steps "${MAX_COMMANDED_STANCE_CONTACT_REPORT_LT2_STEPS:-0}" \
  --require-support-foot-effort-evidence \
  --min-drive-effort-supported-foot-count "${MIN_DRIVE_EFFORT_SUPPORTED_FOOT_COUNT:-2}" \
  --max-drive-effort-supported-lt2-steps "${MAX_DRIVE_EFFORT_SUPPORTED_LT2_STEPS:-0}" \
  --min-commanded-stance-effort-supported-foot-count "${MIN_COMMANDED_STANCE_EFFORT_SUPPORTED_FOOT_COUNT:-2}" \
  --max-commanded-stance-effort-supported-lt2-steps "${MAX_COMMANDED_STANCE_EFFORT_SUPPORTED_LT2_STEPS:-0}" \
  --min-commanded-stance-near-ground-foot-count "${MIN_COMMANDED_STANCE_NEAR_GROUND_FOOT_COUNT:-2}" \
  --max-commanded-stance-near-ground-lt2-steps "${MAX_COMMANDED_STANCE_NEAR_GROUND_LT2_STEPS:-0}" \
  --min-support-polygon-margin "${MIN_SUPPORT_POLYGON_MARGIN:-0.0}" \
  --max-abs-anchor-travel-x "${MAX_ABS_ANCHOR_TRAVEL_X:-0.80}" \
  --max-abs-support-foot-travel-x "${MAX_ABS_SUPPORT_FOOT_TRAVEL_X:-0.90}" \
  --forbid-fixed-world-support \
  --require-non-success-claim \
  > "${CHECK_REPORT}"

cat "${CHECK_REPORT}"
echo "[INFO] Output: ${OUTPUT_DIR}"
echo "[INFO] Check report: ${CHECK_REPORT}"
