#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run MuJoCo simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/mujoco_quadruped_freebox}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/mujoco_quadruped_freebox/${STAMP}}"
STEPS="${STEPS:-3000}"
BOX_MASS="${BOX_MASS:-2.0}"
TARGET_SPEED="${TARGET_SPEED:-0.18}"
TARGET_HEIGHT="${TARGET_HEIGHT:-0.56}"
ACTUATOR_KP="${ACTUATOR_KP:-80.0}"
ACTUATOR_KV="${ACTUATOR_KV:-8.0}"
ASSIST_MODE="${ASSIST_MODE:-body_force}"
MAX_ASSIST_FORCE_X="${MAX_ASSIST_FORCE_X:-115.0}"
MAX_ASSIST_FORCE_Z="${MAX_ASSIST_FORCE_Z:-340.0}"
MAX_ASSIST_TORQUE="${MAX_ASSIST_TORQUE:-240.0}"
TRAY_HALF_LENGTH="${TRAY_HALF_LENGTH:-0.38}"
TRAY_HALF_WIDTH="${TRAY_HALF_WIDTH:-0.24}"
WALL_HEIGHT="${WALL_HEIGHT:-0.16}"
STOP_AFTER_BOX_TRAVEL="${STOP_AFTER_BOX_TRAVEL:-}"
HOLD_TARGET_SPEED="${HOLD_TARGET_SPEED:-0.0}"
RETENTION_FORCE_MODE="${RETENTION_FORCE_MODE:-none}"
RETENTION_KP_X="${RETENTION_KP_X:-0.0}"
RETENTION_KD_X="${RETENTION_KD_X:-0.0}"
RETENTION_KP_Y="${RETENTION_KP_Y:-0.0}"
RETENTION_KD_Y="${RETENTION_KD_Y:-0.0}"
RETENTION_KP_Z="${RETENTION_KP_Z:-0.0}"
RETENTION_KD_Z="${RETENTION_KD_Z:-0.0}"
RETENTION_MAX_FORCE_X="${RETENTION_MAX_FORCE_X:-0.0}"
RETENTION_MAX_FORCE_Y="${RETENTION_MAX_FORCE_Y:-0.0}"
RETENTION_MAX_FORCE_Z="${RETENTION_MAX_FORCE_Z:-0.0}"
LEG_DRIVE_MODE="${LEG_DRIVE_MODE:-sinusoid}"
GAIT_FREQUENCY_HZ="${GAIT_FREQUENCY_HZ:-1.6}"
STANCE_DUTY="${STANCE_DUTY:-0.68}"
STRIDE_LENGTH="${STRIDE_LENGTH:-0.16}"
STANCE_FOOT_Z_DOWN="${STANCE_FOOT_Z_DOWN:-0.43}"
SWING_FOOT_Z_DOWN="${SWING_FOOT_Z_DOWN:-0.32}"
FOOT_ROLL_Z_GAIN="${FOOT_ROLL_Z_GAIN:-0.0}"
HIP_ROLL_BASE="${HIP_ROLL_BASE:-0.0}"
HIP_ROLL_FEEDBACK_GAIN="${HIP_ROLL_FEEDBACK_GAIN:-0.0}"
HOLD_STANCE_FOOT_Z_DOWN="${HOLD_STANCE_FOOT_Z_DOWN:-}"
HOLD_HIP_ROLL_BASE="${HOLD_HIP_ROLL_BASE:-}"
HOLD_HIP_ROLL_FEEDBACK_GAIN="${HOLD_HIP_ROLL_FEEDBACK_GAIN:-}"
HOLD_FOOT_ROLL_Z_GAIN="${HOLD_FOOT_ROLL_Z_GAIN:-}"
HOLD_FRONT_FOOT_X="${HOLD_FRONT_FOOT_X:-}"
HOLD_REAR_FOOT_X="${HOLD_REAR_FOOT_X:-}"
HOLD_PITCH_FOOT_X_GAIN="${HOLD_PITCH_FOOT_X_GAIN:-0.0}"
HOLD_CAPTURE_POINT_FOOT_PLACEMENT="${HOLD_CAPTURE_POINT_FOOT_PLACEMENT:-0}"
HOLD_CAPTURE_TIME_CONSTANT="${HOLD_CAPTURE_TIME_CONSTANT:-0.18}"
HOLD_CAPTURE_X_GAIN="${HOLD_CAPTURE_X_GAIN:-0.0}"
HOLD_CAPTURE_X_LIMIT="${HOLD_CAPTURE_X_LIMIT:-0.06}"
HOLD_CAPTURE_Y_HIP_GAIN="${HOLD_CAPTURE_Y_HIP_GAIN:-0.0}"
HOLD_CAPTURE_Y_FOOT_Z_GAIN="${HOLD_CAPTURE_Y_FOOT_Z_GAIN:-0.0}"
HOLD_CAPTURE_Y_LIMIT="${HOLD_CAPTURE_Y_LIMIT:-0.08}"
CLOSED_LOOP_FOOT_PLACEMENT="${CLOSED_LOOP_FOOT_PLACEMENT:-0}"
STRIDE_VELOCITY_GAIN="${STRIDE_VELOCITY_GAIN:-0.0}"
STRIDE_POSITION_GAIN="${STRIDE_POSITION_GAIN:-0.0}"
STRIDE_CLIP="${STRIDE_CLIP:-0.20}"
SUPPORT_CONTROLLER_MODE="${SUPPORT_CONTROLLER_MODE:-none}"
SUPPORT_FORCE_SCALE="${SUPPORT_FORCE_SCALE:-1.0}"
SUPPORT_FX_SCALE="${SUPPORT_FX_SCALE:-}"
HOLD_SUPPORT_FX_SCALE="${HOLD_SUPPORT_FX_SCALE:-}"
HOLD_SUPPORT_KP_VX_SCALE="${HOLD_SUPPORT_KP_VX_SCALE:-1.0}"
HOLD_SUPPORT_MAX_FX_SCALE="${HOLD_SUPPORT_MAX_FX_SCALE:-1.0}"
HOLD_SUPPORT_KD_Z_SCALE="${HOLD_SUPPORT_KD_Z_SCALE:-1.0}"
HOLD_SUPPORT_KD_ROLL_SCALE="${HOLD_SUPPORT_KD_ROLL_SCALE:-1.0}"
HOLD_SUPPORT_KD_PITCH_SCALE="${HOLD_SUPPORT_KD_PITCH_SCALE:-1.0}"
HOLD_SUPPORT_MAX_FOOT_FZ_SCALE="${HOLD_SUPPORT_MAX_FOOT_FZ_SCALE:-1.0}"
HOLD_SUPPORT_MAX_JOINT_TORQUE_SCALE="${HOLD_SUPPORT_MAX_JOINT_TORQUE_SCALE:-1.0}"
HOLD_SUPPORT_HEIGHT_OFFSET="${HOLD_SUPPORT_HEIGHT_OFFSET:-0.0}"
SUPPORT_COM_X_GAIN="${SUPPORT_COM_X_GAIN:-0.0}"
SUPPORT_COM_Y_GAIN="${SUPPORT_COM_Y_GAIN:-0.0}"
SUPPORT_COM_VX_GAIN="${SUPPORT_COM_VX_GAIN:-0.0}"
SUPPORT_COM_VY_GAIN="${SUPPORT_COM_VY_GAIN:-0.0}"
SUPPORT_COM_TARGET_X_OFFSET="${SUPPORT_COM_TARGET_X_OFFSET:-0.0}"
SUPPORT_COM_TARGET_Y_OFFSET="${SUPPORT_COM_TARGET_Y_OFFSET:-0.0}"
SUPPORT_COM_MAX_FZ_SHIFT="${SUPPORT_COM_MAX_FZ_SHIFT:-0.0}"
SUPPORT_COM_PRE_LATCH_SCALE="${SUPPORT_COM_PRE_LATCH_SCALE:-1.0}"
HOLD_SUPPORT_COM_SCALE="${HOLD_SUPPORT_COM_SCALE:-1.0}"
SUPPORT_FY_ROLL_GAIN="${SUPPORT_FY_ROLL_GAIN:-0.0}"
SUPPORT_FY_ROLL_RATE_GAIN="${SUPPORT_FY_ROLL_RATE_GAIN:-0.0}"
SUPPORT_FY_COM_Y_GAIN="${SUPPORT_FY_COM_Y_GAIN:-0.0}"
SUPPORT_FY_WORLD_Y_GAIN="${SUPPORT_FY_WORLD_Y_GAIN:-0.0}"
SUPPORT_FY_WORLD_VY_GAIN="${SUPPORT_FY_WORLD_VY_GAIN:-0.0}"
SUPPORT_FY_WORLD_Y_SOURCE="${SUPPORT_FY_WORLD_Y_SOURCE:-torso}"
SUPPORT_MAX_TOTAL_FY="${SUPPORT_MAX_TOTAL_FY:-0.0}"
SUPPORT_FY_PRE_LATCH_SCALE="${SUPPORT_FY_PRE_LATCH_SCALE:-1.0}"
HOLD_SUPPORT_FY_SCALE="${HOLD_SUPPORT_FY_SCALE:-1.0}"
SUPPORT_KP_Z="${SUPPORT_KP_Z:-2600.0}"
SUPPORT_KD_Z="${SUPPORT_KD_Z:-180.0}"
SUPPORT_KP_ROLL="${SUPPORT_KP_ROLL:-260.0}"
SUPPORT_KD_ROLL="${SUPPORT_KD_ROLL:-38.0}"
SUPPORT_KP_PITCH="${SUPPORT_KP_PITCH:-220.0}"
SUPPORT_KD_PITCH="${SUPPORT_KD_PITCH:-32.0}"
SUPPORT_KP_VX="${SUPPORT_KP_VX:-520.0}"
SUPPORT_MAX_TOTAL_FX="${SUPPORT_MAX_TOTAL_FX:-260.0}"
SUPPORT_MIN_FOOT_FZ="${SUPPORT_MIN_FOOT_FZ:-10.0}"
SUPPORT_MAX_FOOT_FZ="${SUPPORT_MAX_FOOT_FZ:-260.0}"
SUPPORT_MAX_JOINT_TORQUE="${SUPPORT_MAX_JOINT_TORQUE:-220.0}"
SUPPORT_LQR_HORIZON_STEPS="${SUPPORT_LQR_HORIZON_STEPS:-80}"
SUPPORT_LQR_Q_POS="${SUPPORT_LQR_Q_POS:-80.0}"
SUPPORT_LQR_Q_VEL="${SUPPORT_LQR_Q_VEL:-8.0}"
SUPPORT_LQR_R="${SUPPORT_LQR_R:-1.0}"
SUPPORT_LQR_MAX_FX="${SUPPORT_LQR_MAX_FX:-120.0}"
SUPPORT_LQR_MAX_FY="${SUPPORT_LQR_MAX_FY:-120.0}"
SUPPORT_LQR_POST_LATCH_ONLY="${SUPPORT_LQR_POST_LATCH_ONLY:-0}"

EXTRA_ARGS=()
if [[ -n "${STOP_AFTER_BOX_TRAVEL}" ]]; then
  EXTRA_ARGS+=(--stop-after-box-travel "${STOP_AFTER_BOX_TRAVEL}")
fi
if [[ -n "${HOLD_STANCE_FOOT_Z_DOWN}" ]]; then
  EXTRA_ARGS+=(--hold-stance-foot-z-down "${HOLD_STANCE_FOOT_Z_DOWN}")
fi
if [[ -n "${HOLD_HIP_ROLL_BASE}" ]]; then
  EXTRA_ARGS+=(--hold-hip-roll-base "${HOLD_HIP_ROLL_BASE}")
fi
if [[ -n "${HOLD_HIP_ROLL_FEEDBACK_GAIN}" ]]; then
  EXTRA_ARGS+=(--hold-hip-roll-feedback-gain "${HOLD_HIP_ROLL_FEEDBACK_GAIN}")
fi
if [[ -n "${HOLD_FOOT_ROLL_Z_GAIN}" ]]; then
  EXTRA_ARGS+=(--hold-foot-roll-z-gain "${HOLD_FOOT_ROLL_Z_GAIN}")
fi
if [[ -n "${HOLD_FRONT_FOOT_X}" ]]; then
  EXTRA_ARGS+=(--hold-front-foot-x "${HOLD_FRONT_FOOT_X}")
fi
if [[ -n "${HOLD_REAR_FOOT_X}" ]]; then
  EXTRA_ARGS+=(--hold-rear-foot-x "${HOLD_REAR_FOOT_X}")
fi
if [[ "${CLOSED_LOOP_FOOT_PLACEMENT}" == "1" ]]; then
  EXTRA_ARGS+=(--closed-loop-foot-placement)
fi
if [[ "${HOLD_CAPTURE_POINT_FOOT_PLACEMENT}" == "1" ]]; then
  EXTRA_ARGS+=(--hold-capture-point-foot-placement)
fi
if [[ -n "${SUPPORT_FX_SCALE}" ]]; then
  EXTRA_ARGS+=(--support-fx-scale "${SUPPORT_FX_SCALE}")
fi
if [[ -n "${HOLD_SUPPORT_FX_SCALE}" ]]; then
  EXTRA_ARGS+=(--hold-support-fx-scale "${HOLD_SUPPORT_FX_SCALE}")
fi
if [[ "${SUPPORT_LQR_POST_LATCH_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--support-lqr-post-latch-only)
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_PATH="${LOG_DIR}/mujoco_quadruped_freebox_${STAMP}.log"

cd "${ROOT_DIR}"
"${ISAAC_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/mujoco/run_quadruped_freebox_carry.py" \
  --steps "${STEPS}" \
  --box-mass "${BOX_MASS}" \
  --target-speed "${TARGET_SPEED}" \
  --target-height "${TARGET_HEIGHT}" \
  --actuator-kp "${ACTUATOR_KP}" \
  --actuator-kv "${ACTUATOR_KV}" \
  --assist-mode "${ASSIST_MODE}" \
  --max-assist-force-x "${MAX_ASSIST_FORCE_X}" \
  --max-assist-force-z "${MAX_ASSIST_FORCE_Z}" \
  --max-assist-torque "${MAX_ASSIST_TORQUE}" \
  --tray-half-length "${TRAY_HALF_LENGTH}" \
  --tray-half-width "${TRAY_HALF_WIDTH}" \
  --wall-height "${WALL_HEIGHT}" \
  --hold-target-speed "${HOLD_TARGET_SPEED}" \
  --retention-force-mode "${RETENTION_FORCE_MODE}" \
  --retention-kp-x "${RETENTION_KP_X}" \
  --retention-kd-x "${RETENTION_KD_X}" \
  --retention-kp-y "${RETENTION_KP_Y}" \
  --retention-kd-y "${RETENTION_KD_Y}" \
  --retention-kp-z "${RETENTION_KP_Z}" \
  --retention-kd-z "${RETENTION_KD_Z}" \
  --retention-max-force-x "${RETENTION_MAX_FORCE_X}" \
  --retention-max-force-y "${RETENTION_MAX_FORCE_Y}" \
  --retention-max-force-z "${RETENTION_MAX_FORCE_Z}" \
  --leg-drive-mode "${LEG_DRIVE_MODE}" \
  --gait-frequency-hz "${GAIT_FREQUENCY_HZ}" \
  --stance-duty "${STANCE_DUTY}" \
  --stride-length "${STRIDE_LENGTH}" \
  --stance-foot-z-down "${STANCE_FOOT_Z_DOWN}" \
  --swing-foot-z-down "${SWING_FOOT_Z_DOWN}" \
  --foot-roll-z-gain "${FOOT_ROLL_Z_GAIN}" \
  --hip-roll-base "${HIP_ROLL_BASE}" \
  --hip-roll-feedback-gain "${HIP_ROLL_FEEDBACK_GAIN}" \
  --hold-pitch-foot-x-gain "${HOLD_PITCH_FOOT_X_GAIN}" \
  --hold-capture-time-constant "${HOLD_CAPTURE_TIME_CONSTANT}" \
  --hold-capture-x-gain "${HOLD_CAPTURE_X_GAIN}" \
  --hold-capture-x-limit "${HOLD_CAPTURE_X_LIMIT}" \
  --hold-capture-y-hip-gain "${HOLD_CAPTURE_Y_HIP_GAIN}" \
  --hold-capture-y-foot-z-gain "${HOLD_CAPTURE_Y_FOOT_Z_GAIN}" \
  --hold-capture-y-limit "${HOLD_CAPTURE_Y_LIMIT}" \
  --stride-velocity-gain "${STRIDE_VELOCITY_GAIN}" \
  --stride-position-gain "${STRIDE_POSITION_GAIN}" \
  --stride-clip "${STRIDE_CLIP}" \
  --support-controller-mode "${SUPPORT_CONTROLLER_MODE}" \
  --support-force-scale "${SUPPORT_FORCE_SCALE}" \
  --hold-support-kp-vx-scale "${HOLD_SUPPORT_KP_VX_SCALE}" \
  --hold-support-max-fx-scale "${HOLD_SUPPORT_MAX_FX_SCALE}" \
  --hold-support-kd-z-scale "${HOLD_SUPPORT_KD_Z_SCALE}" \
  --hold-support-kd-roll-scale "${HOLD_SUPPORT_KD_ROLL_SCALE}" \
  --hold-support-kd-pitch-scale "${HOLD_SUPPORT_KD_PITCH_SCALE}" \
  --hold-support-max-foot-fz-scale "${HOLD_SUPPORT_MAX_FOOT_FZ_SCALE}" \
  --hold-support-max-joint-torque-scale "${HOLD_SUPPORT_MAX_JOINT_TORQUE_SCALE}" \
  --hold-support-height-offset "${HOLD_SUPPORT_HEIGHT_OFFSET}" \
  --support-com-x-gain "${SUPPORT_COM_X_GAIN}" \
  --support-com-y-gain "${SUPPORT_COM_Y_GAIN}" \
  --support-com-vx-gain "${SUPPORT_COM_VX_GAIN}" \
  --support-com-vy-gain "${SUPPORT_COM_VY_GAIN}" \
  --support-com-target-x-offset "${SUPPORT_COM_TARGET_X_OFFSET}" \
  --support-com-target-y-offset "${SUPPORT_COM_TARGET_Y_OFFSET}" \
  --support-com-max-fz-shift "${SUPPORT_COM_MAX_FZ_SHIFT}" \
  --support-com-pre-latch-scale "${SUPPORT_COM_PRE_LATCH_SCALE}" \
  --hold-support-com-scale "${HOLD_SUPPORT_COM_SCALE}" \
  --support-fy-roll-gain "${SUPPORT_FY_ROLL_GAIN}" \
  --support-fy-roll-rate-gain "${SUPPORT_FY_ROLL_RATE_GAIN}" \
  --support-fy-com-y-gain "${SUPPORT_FY_COM_Y_GAIN}" \
  --support-fy-world-y-gain "${SUPPORT_FY_WORLD_Y_GAIN}" \
  --support-fy-world-vy-gain "${SUPPORT_FY_WORLD_VY_GAIN}" \
  --support-fy-world-y-source "${SUPPORT_FY_WORLD_Y_SOURCE}" \
  --support-max-total-fy "${SUPPORT_MAX_TOTAL_FY}" \
  --support-fy-pre-latch-scale "${SUPPORT_FY_PRE_LATCH_SCALE}" \
  --hold-support-fy-scale "${HOLD_SUPPORT_FY_SCALE}" \
  --support-kp-z "${SUPPORT_KP_Z}" \
  --support-kd-z "${SUPPORT_KD_Z}" \
  --support-kp-roll "${SUPPORT_KP_ROLL}" \
  --support-kd-roll "${SUPPORT_KD_ROLL}" \
  --support-kp-pitch "${SUPPORT_KP_PITCH}" \
  --support-kd-pitch "${SUPPORT_KD_PITCH}" \
  --support-kp-vx "${SUPPORT_KP_VX}" \
  --support-max-total-fx "${SUPPORT_MAX_TOTAL_FX}" \
  --support-min-foot-fz "${SUPPORT_MIN_FOOT_FZ}" \
  --support-max-foot-fz "${SUPPORT_MAX_FOOT_FZ}" \
  --support-max-joint-torque "${SUPPORT_MAX_JOINT_TORQUE}" \
  --support-lqr-horizon-steps "${SUPPORT_LQR_HORIZON_STEPS}" \
  --support-lqr-q-pos "${SUPPORT_LQR_Q_POS}" \
  --support-lqr-q-vel "${SUPPORT_LQR_Q_VEL}" \
  --support-lqr-r "${SUPPORT_LQR_R}" \
  --support-lqr-max-fx "${SUPPORT_LQR_MAX_FX}" \
  --support-lqr-max-fy "${SUPPORT_LQR_MAX_FY}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
