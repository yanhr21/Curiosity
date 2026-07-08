#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/direct_carry_task_no_root_prismatic_backend/${STAMP}}"
BACKEND_OUTPUT_DIR="${BACKEND_OUTPUT_DIR:-${OUTPUT_DIR}/backend_prismatic_legged}"
BACKEND_STAMP="${BACKEND_STAMP:-${STAMP}_backend_prismatic_legged}"
BACKEND_LOG="${ROOT_DIR}/logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_${BACKEND_STAMP}.log"
CARRY_POSTURE="${CARRY_POSTURE:-front_mid}"
MOTION_MODE="${MOTION_MODE:-feedback_sync_inchworm}"
CONTROLLER_MODE="${CONTROLLER_MODE:-no_root_prismatic_legged_cradle}"

mkdir -p "${OUTPUT_DIR}" "${BACKEND_OUTPUT_DIR}"
cd "${ROOT_DIR}"

case "${CARRY_POSTURE}" in
  front_mid)
    DEFAULT_PAYLOAD_LOCAL_X="0.50"
    DEFAULT_PAYLOAD_LOCAL_Z="0.18"
    DEFAULT_TORSO_Z="0.62"
    ;;
  low_front)
    DEFAULT_PAYLOAD_LOCAL_X="0.54"
    DEFAULT_PAYLOAD_LOCAL_Z="0.12"
    DEFAULT_TORSO_Z="0.66"
    ;;
  chest_high)
    DEFAULT_PAYLOAD_LOCAL_X="0.44"
    DEFAULT_PAYLOAD_LOCAL_Z="0.24"
    DEFAULT_TORSO_Z="0.62"
    ;;
  *)
    echo "Unknown CARRY_POSTURE=${CARRY_POSTURE}; expected front_mid, low_front, or chest_high" >&2
    exit 5
    ;;
esac

echo "[CONFIG] STAMP=${STAMP} CONTROLLER_MODE=${CONTROLLER_MODE} BACKEND=no_root_prismatic_legged CARRY_POSTURE=${CARRY_POSTURE} MOTION_MODE=${MOTION_MODE} TARGET_X=${TARGET_X:-0.10} PAYLOAD_MASS=${PAYLOAD_MASS:-8.0}"

STAMP="${BACKEND_STAMP}" \
OUTPUT_DIR="${BACKEND_OUTPUT_DIR}" \
STEPS="${STEPS:-1200}" \
PAYLOAD_MODE="${PAYLOAD_MODE:-cradle_free_box}" \
PAYLOAD_MASS="${PAYLOAD_MASS:-8.0}" \
TORSO_MASS="${TORSO_MASS:-36.0}" \
TORSO_Z="${TORSO_Z:-${DEFAULT_TORSO_Z}}" \
PAYLOAD_LOCAL_X="${PAYLOAD_LOCAL_X:-${DEFAULT_PAYLOAD_LOCAL_X}}" \
PAYLOAD_LOCAL_Z="${PAYLOAD_LOCAL_Z:-${DEFAULT_PAYLOAD_LOCAL_Z}}" \
MOTION_MODE="${MOTION_MODE}" \
ENABLE_HORIZONTAL_LEGS="${ENABLE_HORIZONTAL_LEGS:-1}" \
TARGET_X="${TARGET_X:-0.10}" \
STEP_LENGTH="${STEP_LENGTH:-0.06}" \
STEP_HEIGHT="${STEP_HEIGHT:-0.08}" \
FOOT_CONTACT_Z_THRESHOLD="${FOOT_CONTACT_Z_THRESHOLD:-0.050}" \
ENABLE_STANCE_FOOT_LATCH="${ENABLE_STANCE_FOOT_LATCH:-0}" \
STANCE_FOOT_LATCH_LIFT_THRESHOLD="${STANCE_FOOT_LATCH_LIFT_THRESHOLD:-0.010}" \
GAIT_PERIOD_STEPS="${GAIT_PERIOD_STEPS:-180}" \
SYNC_CYCLE_PAUSE_FRACTION="${SYNC_CYCLE_PAUSE_FRACTION:-0.20}" \
FEEDBACK_TILT_HOLD_THRESHOLD="${FEEDBACK_TILT_HOLD_THRESHOLD:-0.18}" \
FEEDBACK_PAYLOAD_ERROR_HOLD_THRESHOLD="${FEEDBACK_PAYLOAD_ERROR_HOLD_THRESHOLD:-0.10}" \
GATED_STEP_MAX_TRAVEL_LOSS="${GATED_STEP_MAX_TRAVEL_LOSS:-0.015}" \
GATED_STEP_RECOVERY_PHASE="${GATED_STEP_RECOVERY_PHASE:-0.43}" \
GATED_STEP_LOSS_REBASELINE_STEPS="${GATED_STEP_LOSS_REBASELINE_STEPS:-0}" \
PRELIFT_RESET_LIFT_FRACTION="${PRELIFT_RESET_LIFT_FRACTION:-0.30}" \
PRELIFT_RESET_LOWER_FRACTION="${PRELIFT_RESET_LOWER_FRACTION:-0.30}" \
PRELIFT_STANCE_OVERDRIVE="${PRELIFT_STANCE_OVERDRIVE:-1.0}" \
GUARDED_STEP_TARGET_TOLERANCE="${GUARDED_STEP_TARGET_TOLERANCE:-0.018}" \
QUASISTATIC_COMPENSATE_SETTLE_DRIFT="${QUASISTATIC_COMPENSATE_SETTLE_DRIFT:-0}" \
SETTLE_STEPS="${SETTLE_STEPS:-120}" \
RAMP_STEPS="${RAMP_STEPS:-180}" \
LEG_TARGET="${LEG_TARGET:--0.57}" \
LEG_LOWER="${LEG_LOWER:--0.82}" \
LEG_UPPER="${LEG_UPPER:--0.25}" \
LEG_STIFFNESS="${LEG_STIFFNESS:-18000.0}" \
LEG_DAMPING="${LEG_DAMPING:-1800.0}" \
LEG_MAX_FORCE="${LEG_MAX_FORCE:-25000.0}" \
X_SLIDE_LIMIT="${X_SLIDE_LIMIT:-0.12}" \
X_SLIDE_STIFFNESS="${X_SLIDE_STIFFNESS:-9000.0}" \
X_SLIDE_DAMPING="${X_SLIDE_DAMPING:-900.0}" \
X_SLIDE_MAX_FORCE="${X_SLIDE_MAX_FORCE:-12000.0}" \
SWING_X_FORCE_SCALE="${SWING_X_FORCE_SCALE:-1.0}" \
STATIC_FRICTION="${STATIC_FRICTION:-3.0}" \
DYNAMIC_FRICTION="${DYNAMIC_FRICTION:-2.5}" \
DEVICE="${DEVICE:-cpu}" \
bash scripts/isaac/run_core_world_prismatic_carrier_stand.sh \
  "$@"

python3 scripts/isaac/normalize_direct_carry_backend_summary.py \
  --backend-summary "${BACKEND_OUTPUT_DIR}/core_world_prismatic_carrier_stand_summary.json" \
  --backend-log "${BACKEND_LOG}" \
  --backend-name "core_world_prismatic_carrier_stand" \
  --backend-support-mode "no_root_prismatic_legged" \
  --controller-mode "${CONTROLLER_MODE}" \
  --carry-posture "${CARRY_POSTURE}" \
  --non-success-reason "no_root_prismatic_legged_backend_is_not_final_robot_controller" \
  --output-summary "${OUTPUT_DIR}/direct_carry_task_no_root_prismatic_backend_summary.json"

echo "[INFO] Output: ${OUTPUT_DIR}"
