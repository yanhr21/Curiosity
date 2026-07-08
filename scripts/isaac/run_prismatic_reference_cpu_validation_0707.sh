#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
STAMP="${STAMP:-20260707_prismatic_reference_probe_adaptive_10kg_mid_cpu}"
OUTPUT_DIR="${ROOT_DIR}/experiments/outputs/core_world_prismatic_carrier_stand/${STAMP}"
SUMMARY_PATH="${OUTPUT_DIR}/core_world_prismatic_carrier_stand_summary.json"
CHECK_PATH="${OUTPUT_DIR}/reference_check.json"

export STAMP
export OUTPUT_DIR
export DEVICE="${DEVICE:-cpu}"
export PAYLOAD_MODE="${PAYLOAD_MODE:-cradle_free_box}"
export PAYLOAD_MASS="${PAYLOAD_MASS:-10.0}"
export PAYLOAD_LOCAL_X="${PAYLOAD_LOCAL_X:-0.50}"
export PAYLOAD_LOCAL_Z="${PAYLOAD_LOCAL_Z:-0.16}"
export TARGET_X="${TARGET_X:--0.17}"
export GAIT_DRIVE_TARGET_X="${GAIT_DRIVE_TARGET_X:--0.42}"
export GUARDED_STOP_TARGET_X="${GUARDED_STOP_TARGET_X:--0.17}"
export MOTION_MODE="${MOTION_MODE:-guarded_prelift_quasistatic_step_cycle}"
export ENABLE_HORIZONTAL_LEGS="${ENABLE_HORIZONTAL_LEGS:-1}"
export STEPS="${STEPS:-760}"
export SETTLE_STEPS="${SETTLE_STEPS:-260}"
export RAMP_STEPS="${RAMP_STEPS:-260}"
export STEP_LENGTH="${STEP_LENGTH:-0.10}"
export STEP_HEIGHT="${STEP_HEIGHT:-0.05}"
export GAIT_PERIOD_STEPS="${GAIT_PERIOD_STEPS:-300}"
export SWING_FRACTION="${SWING_FRACTION:-0.22}"
export STANCE_HALF_LENGTH="${STANCE_HALF_LENGTH:-0.65}"
export STANCE_HALF_WIDTH="${STANCE_HALF_WIDTH:-0.24}"
export FOOT_LENGTH="${FOOT_LENGTH:-0.34}"
export FOOT_WIDTH="${FOOT_WIDTH:-0.18}"
export FOOT_HEIGHT="${FOOT_HEIGHT:-0.055}"
export LEG_TARGET="${LEG_TARGET:--0.57}"
export LEG_LOWER="${LEG_LOWER:--0.82}"
export LEG_UPPER="${LEG_UPPER:--0.25}"
export LEG_STIFFNESS="${LEG_STIFFNESS:-36000}"
export LEG_DAMPING="${LEG_DAMPING:-3600}"
export LEG_MAX_FORCE="${LEG_MAX_FORCE:-68000}"
export X_SLIDE_LIMIT="${X_SLIDE_LIMIT:-0.28}"
export X_SLIDE_STIFFNESS="${X_SLIDE_STIFFNESS:-36000}"
export X_SLIDE_DAMPING="${X_SLIDE_DAMPING:-3600}"
export X_SLIDE_MAX_FORCE="${X_SLIDE_MAX_FORCE:-76000}"
export GATED_STEP_MAX_TRAVEL_LOSS="${GATED_STEP_MAX_TRAVEL_LOSS:-0.015}"
export GATED_STEP_RECOVERY_PHASE="${GATED_STEP_RECOVERY_PHASE:-0.43}"
export GUARDED_STEP_TARGET_TOLERANCE="${GUARDED_STEP_TARGET_TOLERANCE:-0.018}"
export ENABLE_ACTIVE_PROBE="${ENABLE_ACTIVE_PROBE:-1}"
export ACTIVE_PROBE_STEPS="${ACTIVE_PROBE_STEPS:-80}"
export ACTIVE_PROBE_LIFT_AMPLITUDE="${ACTIVE_PROBE_LIFT_AMPLITUDE:-0.030}"
export ENABLE_PROBE_ADAPTIVE_GAIT="${ENABLE_PROBE_ADAPTIVE_GAIT:-1}"
export ENABLE_PROBE_ADAPTIVE_POSTURE="${ENABLE_PROBE_ADAPTIVE_POSTURE:-1}"
export PROBE_ADAPTIVE_MEDIUM_RISK_THRESHOLD="${PROBE_ADAPTIVE_MEDIUM_RISK_THRESHOLD:-0.25}"
export PROBE_ADAPTIVE_HIGH_RISK_THRESHOLD="${PROBE_ADAPTIVE_HIGH_RISK_THRESHOLD:-0.75}"
export PROBE_ADAPTIVE_MEDIUM_GAIT_DRIVE_SCALE="${PROBE_ADAPTIVE_MEDIUM_GAIT_DRIVE_SCALE:-0.95}"
export PROBE_ADAPTIVE_HIGH_GAIT_DRIVE_SCALE="${PROBE_ADAPTIVE_HIGH_GAIT_DRIVE_SCALE:-0.85}"
export PROBE_ADAPTIVE_MEDIUM_POSTURE_LEG_TARGET_OFFSET="${PROBE_ADAPTIVE_MEDIUM_POSTURE_LEG_TARGET_OFFSET:-0.012}"
export PROBE_ADAPTIVE_HIGH_POSTURE_LEG_TARGET_OFFSET="${PROBE_ADAPTIVE_HIGH_POSTURE_LEG_TARGET_OFFSET:-0.024}"

mkdir -p "${OUTPUT_DIR}"

bash "${ROOT_DIR}/scripts/isaac/run_core_world_prismatic_carrier_stand.sh"

python3 "${ROOT_DIR}/scripts/isaac/check_prismatic_carrier_stand_summary.py" \
  "${SUMMARY_PATH}" \
  --expect-payload-mode cradle_free_box \
  --expect-motion-mode guarded_prelift_quasistatic_step_cycle \
  --require-articulated-carrier \
  --require-foot-contact-drive \
  --require-active-probe \
  --require-probe-belief \
  --require-no-hidden-probe-gt \
  --min-active-probe-steps 80 \
  --require-probe-adaptive-gait-decision \
  --require-probe-adaptive-posture-decision \
  --max-fall-events 0 \
  --max-box-drop-events 0 \
  --max-root-pose-writes 0 \
  --max-root-velocity-writes 0 \
  --max-body-root-pose-writes 0 \
  --max-body-root-velocity-commands 0 \
  --max-box-pose-writes 0 \
  --min-abs-post-settle-payload-travel-x 0.15 \
  --max-final-post-settle-payload-target-distance-x 0.02 \
  --max-payload-relative-offset-error 0.08 \
  --max-post-settle-payload-relative-offset-error 0.012 \
  --min-payload-z 0.45 \
  --max-tilt 0.20 \
  > "${CHECK_PATH}"

echo "[INFO] Summary: ${SUMMARY_PATH}"
echo "[INFO] Check: ${CHECK_PATH}"
