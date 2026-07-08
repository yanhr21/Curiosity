#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_anchored_footstep_carrier}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_anchored_footstep_carrier/${STAMP}}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_DIR}/core_world_anchored_footstep_carrier_${STAMP}.log"
RAIL_LOWER_VALUE="${RAIL_LOWER:-}"
SUPPORT_FOOT_X_LOWER_VALUE="${SUPPORT_FOOT_X_LOWER:-}"
SUPPORT_FOOT_DRIVE_DIRECTION_SCALE_VALUE="${SUPPORT_FOOT_DRIVE_DIRECTION_SCALE:-}"
SUPPORT_FOOT_PLACEMENT_MODE_VALUE="${SUPPORT_FOOT_PLACEMENT_MODE:-}"
SUPPORT_FOOT_Z_LOWER_VALUE="${SUPPORT_FOOT_Z_LOWER:-}"
SUPPORT_FOOT_STANCE_X_VALUE="${SUPPORT_FOOT_STANCE_X:-}"
ONLINE_LOW_SUPPORT_STANCE_X_VALUE="${ONLINE_LOW_SUPPORT_STANCE_X:-}"
ONLINE_MEDIUM_SUPPORT_STANCE_X_VALUE="${ONLINE_MEDIUM_SUPPORT_STANCE_X:-}"
ONLINE_HIGH_SUPPORT_STANCE_X_VALUE="${ONLINE_HIGH_SUPPORT_STANCE_X:-}"
if [[ -z "${RAIL_LOWER_VALUE}" ]]; then RAIL_LOWER_VALUE="-0.10"; fi
if [[ -z "${SUPPORT_FOOT_X_LOWER_VALUE}" ]]; then SUPPORT_FOOT_X_LOWER_VALUE="-0.80"; fi
if [[ -z "${SUPPORT_FOOT_DRIVE_DIRECTION_SCALE_VALUE}" ]]; then SUPPORT_FOOT_DRIVE_DIRECTION_SCALE_VALUE="-1.0"; fi
if [[ -z "${SUPPORT_FOOT_PLACEMENT_MODE_VALUE}" ]]; then SUPPORT_FOOT_PLACEMENT_MODE_VALUE="alternating_fixed_x"; fi
if [[ -z "${SUPPORT_FOOT_Z_LOWER_VALUE}" ]]; then SUPPORT_FOOT_Z_LOWER_VALUE="-0.005"; fi
if [[ -z "${SUPPORT_FOOT_STANCE_X_VALUE}" ]]; then SUPPORT_FOOT_STANCE_X_VALUE="-0.080"; fi
if [[ -z "${ONLINE_LOW_SUPPORT_STANCE_X_VALUE}" ]]; then ONLINE_LOW_SUPPORT_STANCE_X_VALUE="-0.130"; fi
if [[ -z "${ONLINE_MEDIUM_SUPPORT_STANCE_X_VALUE}" ]]; then ONLINE_MEDIUM_SUPPORT_STANCE_X_VALUE="-0.115"; fi
if [[ -z "${ONLINE_HIGH_SUPPORT_STANCE_X_VALUE}" ]]; then ONLINE_HIGH_SUPPORT_STANCE_X_VALUE="-0.100"; fi
echo "[CONFIG] STAMP=${STAMP} STEPS=${STEPS:-720} TARGET_X=${TARGET_X:-0.24} STEP_LENGTH=${STEP_LENGTH:-0.06} PAYLOAD_MODE=${PAYLOAD_MODE:-fixed_joint_to_torso} PROBE_MODE=${PROBE_MODE:-horizontal_push_pull} PROBE_X_AMPLITUDE=${PROBE_X_AMPLITUDE:-0.0} PROBE_Z_AMPLITUDE=${PROBE_Z_AMPLITUDE:-0.0}"
CMD=(
  "${ISAAC_VENV}/bin/python"
  "${ROOT_DIR}/scripts/isaac/build_core_world_anchored_footstep_carrier.py"
  --viz none
  --experience "${EXPERIENCE}"
  --device "${DEVICE:-cpu}"
  --kit_args "${KIT_ARGS}"
  --steps "${STEPS:-720}"
  --target-x "${TARGET_X:-0.24}"
  --step-length "${STEP_LENGTH:-0.06}"
  --stance-steps "${STANCE_STEPS:-120}"
  --settle-steps "${SETTLE_STEPS:-60}"
  --probe-steps "${PROBE_STEPS:-0}"
  --probe-mode "${PROBE_MODE:-horizontal_push_pull}"
  --probe-x-amplitude "${PROBE_X_AMPLITUDE:-0.0}"
  --probe-z-amplitude "${PROBE_Z_AMPLITUDE:-0.0}"
  --belief-compliance-low-threshold "${BELIEF_COMPLIANCE_LOW_THRESHOLD:-0.08}"
  --belief-compliance-high-threshold "${BELIEF_COMPLIANCE_HIGH_THRESHOLD:-0.22}"
  --payload-mode "${PAYLOAD_MODE:-fixed_joint_to_torso}"
  --payload-mass "${PAYLOAD_MASS:-4.0}"
  --box-seed "${BOX_SEED:-0}"
  --payload-mass-range "${PAYLOAD_MASS_MIN:-4.0}" "${PAYLOAD_MASS_MAX:-12.0}"
  --payload-size-jitter "${PAYLOAD_SIZE_JITTER:-0.0}"
  --payload-com-offset-range "${PAYLOAD_COM_OFFSET_RANGE_X:-0.0}" "${PAYLOAD_COM_OFFSET_RANGE_Y:-0.0}" "${PAYLOAD_COM_OFFSET_RANGE_Z:-0.0}"
  --torso-mass "${TORSO_MASS:-36.0}"
  --torso-z "${TORSO_Z:-0.55}"
  --payload-local-x "${PAYLOAD_LOCAL_X:-0.20}"
  --payload-local-z "${PAYLOAD_LOCAL_Z:-0.04}"
  --cage-clearance-xy "${CAGE_CLEARANCE_XY:-0.025}"
  --cage-clearance-z "${CAGE_CLEARANCE_Z:-0.025}"
  --cage-wall-thickness "${CAGE_WALL_THICKNESS:-0.035}"
  --cage-deck-mass "${CAGE_DECK_MASS:-1.0}"
  --cage-wall-mass "${CAGE_WALL_MASS:-0.5}"
  --cage-lid-mass "${CAGE_LID_MASS:-0.3}"
  --grasp-enable-step "${GRASP_ENABLE_STEP:-30}"
  --grasp-shelf-clearance "${GRASP_SHELF_CLEARANCE:-0.003}"
  --tray-clearance-xy "${TRAY_CLEARANCE_XY:-0.030}"
  --tray-wall-height "${TRAY_WALL_HEIGHT:-0.090}"
  --tray-wall-thickness "${TRAY_WALL_THICKNESS:-0.025}"
  --tray-part-mass "${TRAY_PART_MASS:-0.4}"
  --clamp-open-gap "${CLAMP_OPEN_GAP:-0.060}"
  --clamp-closed-gap "${CLAMP_CLOSED_GAP:-0.006}"
  --clamp-pad-thickness "${CLAMP_PAD_THICKNESS:-0.035}"
  --clamp-pad-mass "${CLAMP_PAD_MASS:-0.8}"
  --clamp-close-start-step "${CLAMP_CLOSE_START_STEP:-40}"
  --clamp-close-steps "${CLAMP_CLOSE_STEPS:-120}"
  --clamp-drive-stiffness "${CLAMP_DRIVE_STIFFNESS:-2500.0}"
  --clamp-drive-damping "${CLAMP_DRIVE_DAMPING:-600.0}"
  --clamp-drive-max-force "${CLAMP_DRIVE_MAX_FORCE:-6000.0}"
  --x-cradle-open-gap "${X_CRADLE_OPEN_GAP:-0.060}"
  --x-cradle-closed-gap "${X_CRADLE_CLOSED_GAP:-0.006}"
  --drive-stiffness "${DRIVE_STIFFNESS:-22000.0}"
  --drive-damping "${DRIVE_DAMPING:-3500.0}"
  --drive-max-force "${DRIVE_MAX_FORCE:-60000.0}"
  --rail-lower "${RAIL_LOWER_VALUE}"
  --rail-upper "${RAIL_UPPER:-0.04}"
  --rail-joint-count "${RAIL_JOINT_COUNT:-1}"
  --rail-target-direction-scale "${RAIL_TARGET_DIRECTION_SCALE:-1.0}"
  --support-foot-mass "${SUPPORT_FOOT_MASS:-6.0}"
  --support-foot-x-lower "${SUPPORT_FOOT_X_LOWER_VALUE}"
  --support-foot-x-upper "${SUPPORT_FOOT_X_UPPER:-0.20}"
  --support-foot-drive-stiffness "${SUPPORT_FOOT_DRIVE_STIFFNESS:-18000.0}"
  --support-foot-drive-damping "${SUPPORT_FOOT_DRIVE_DAMPING:-3000.0}"
  --support-foot-drive-max-force "${SUPPORT_FOOT_DRIVE_MAX_FORCE:-80000.0}"
  --support-foot-drive-direction-scale "${SUPPORT_FOOT_DRIVE_DIRECTION_SCALE_VALUE}"
  --support-foot-placement-mode "${SUPPORT_FOOT_PLACEMENT_MODE_VALUE}"
  --support-foot-z-lower "${SUPPORT_FOOT_Z_LOWER_VALUE}"
  --support-foot-z-upper "${SUPPORT_FOOT_Z_UPPER:-0.120}"
  --support-foot-z-drive-stiffness "${SUPPORT_FOOT_Z_DRIVE_STIFFNESS:-14000.0}"
  --support-foot-z-drive-damping "${SUPPORT_FOOT_Z_DRIVE_DAMPING:-1800.0}"
  --support-foot-z-drive-max-force "${SUPPORT_FOOT_Z_DRIVE_MAX_FORCE:-70000.0}"
  --support-foot-step-height "${SUPPORT_FOOT_STEP_HEIGHT:-0.070}"
  --support-foot-stance-x "${SUPPORT_FOOT_STANCE_X_VALUE}"
  --support-foot-swing-x "${SUPPORT_FOOT_SWING_X:-0.080}"
  --support-foot-contact-z-threshold "${SUPPORT_FOOT_CONTACT_Z_THRESHOLD:-0.028}"
  --support-foot-contact-report-threshold "${SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD:-0.0}"
  --support-foot-effort-contact-threshold "${SUPPORT_FOOT_EFFORT_CONTACT_THRESHOLD:-1.0}"
  --support-foot-double-support-fraction "${SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION:-0.0}"
  --feedback-step-x-gain "${FEEDBACK_STEP_X_GAIN:-0.0}"
  --feedback-step-x-limit "${FEEDBACK_STEP_X_LIMIT:-0.030}"
  --feedback-step-tilt-gain "${FEEDBACK_STEP_TILT_GAIN:-0.0}"
  --feedback-step-tilt-limit "${FEEDBACK_STEP_TILT_LIMIT:-0.020}"
  --online-probe-adaptive-medium-threshold "${ONLINE_PROBE_ADAPTIVE_MEDIUM_THRESHOLD:-0.58}"
  --online-probe-adaptive-high-threshold "${ONLINE_PROBE_ADAPTIVE_HIGH_THRESHOLD:-0.75}"
  --online-low-support-step-height "${ONLINE_LOW_SUPPORT_STEP_HEIGHT:-0.120}"
  --online-low-support-double-support-fraction "${ONLINE_LOW_SUPPORT_DOUBLE_SUPPORT_FRACTION:-0.12}"
  --online-low-support-stance-x "${ONLINE_LOW_SUPPORT_STANCE_X_VALUE}"
  --online-low-support-swing-x "${ONLINE_LOW_SUPPORT_SWING_X:-0.130}"
  --online-medium-support-step-height "${ONLINE_MEDIUM_SUPPORT_STEP_HEIGHT:-0.100}"
  --online-medium-support-double-support-fraction "${ONLINE_MEDIUM_SUPPORT_DOUBLE_SUPPORT_FRACTION:-0.18}"
  --online-medium-support-stance-x "${ONLINE_MEDIUM_SUPPORT_STANCE_X_VALUE}"
  --online-medium-support-swing-x "${ONLINE_MEDIUM_SUPPORT_SWING_X:-0.115}"
  --online-high-support-step-height "${ONLINE_HIGH_SUPPORT_STEP_HEIGHT:-0.080}"
  --online-high-support-double-support-fraction "${ONLINE_HIGH_SUPPORT_DOUBLE_SUPPORT_FRACTION:-0.24}"
  --online-high-support-stance-x "${ONLINE_HIGH_SUPPORT_STANCE_X_VALUE}"
  --online-high-support-swing-x "${ONLINE_HIGH_SUPPORT_SWING_X:-0.100}"
  --online-low-hold-closure-fraction "${ONLINE_LOW_HOLD_CLOSURE_FRACTION:-0.45}"
  --online-medium-hold-closure-fraction "${ONLINE_MEDIUM_HOLD_CLOSURE_FRACTION:-0.75}"
  --online-high-hold-closure-fraction "${ONLINE_HIGH_HOLD_CLOSURE_FRACTION:-1.0}"
  --support-foot-continuity-grace-steps "${SUPPORT_FOOT_CONTINUITY_GRACE_STEPS:-0}"
  --stance-half-length "${STANCE_HALF_LENGTH:-0.28}"
  --stance-half-width "${STANCE_HALF_WIDTH:-0.22}"
  --stop-threshold "${STOP_THRESHOLD:-0.01}"
  --static-friction "${STATIC_FRICTION:-3.0}"
  --dynamic-friction "${DYNAMIC_FRICTION:-2.5}"
  --output-dir "${OUTPUT_DIR}"
  "$@"
)
case "${RANDOMIZE_PAYLOAD:-0}" in
  1|true) CMD+=(--randomize-payload) ;;
esac
case "${FEEDBACK_STEP_CONTROLLER:-0}" in
  1|true) CMD+=(--feedback-step-controller) ;;
esac
case "${ENABLE_ONLINE_PROBE_ADAPTIVE_SUPPORT:-0}" in
  1|true) CMD+=(--enable-online-probe-adaptive-support) ;;
esac
case "${ENABLE_ONLINE_PROBE_ADAPTIVE_HOLD:-0}" in
  1|true) CMD+=(--enable-online-probe-adaptive-hold) ;;
esac
case "${ENABLE_SUPPORT_FOOT_CONTACT_REPORT:-0}" in
  1|true) CMD+=(--enable-support-foot-contact-report) ;;
esac
case "${STANCE_FOOT_WORLD_LOCK:-0}" in
  1|true) CMD+=(--stance-foot-world-lock) ;;
esac
case "${FREEZE_LOCKED_STANCE_FOOT_TARGETS:-0}" in
  1|true) CMD+=(--freeze-locked-stance-foot-targets) ;;
esac
case "${FREEZE_COMMANDED_STANCE_FOOT_TARGETS:-0}" in
  1|true) CMD+=(--freeze-commanded-stance-foot-targets) ;;
esac
case "${PLANTED_STANCE_RAIL_PROPULSION:-0}" in
  1|true) CMD+=(--planted-stance-rail-propulsion) ;;
esac
case "${DEBUG_CORE_CMD:-0}" in
  1|true)
    printf '[DEBUG_CORE_CMD] argc=%s\n' "${#CMD[@]}"
    printf '[DEBUG_CORE_CMD_ARG] %q\n' "${CMD[@]}"
    ;;
esac
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
