#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-20260705_core_world_g1_front_probe_bumper_smoke}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_g1_box_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_box_scene/${STAMP}}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_DIR}/core_world_g1_box_scene_${STAMP}.log"
EXTRA_ARM_ARGS=(
  --arm-pose-mode "${ARM_POSE_MODE:-none}"
  --arm-pose-start-step "${ARM_POSE_START_STEP:-0}"
  --arm-pose-ramp-steps "${ARM_POSE_RAMP_STEPS:-120}"
)
for spec in \
  RIGHT_SHOULDER_PITCH:--right-shoulder-pitch \
  RIGHT_SHOULDER_ROLL:--right-shoulder-roll \
  RIGHT_SHOULDER_YAW:--right-shoulder-yaw \
  RIGHT_ELBOW:--right-elbow \
  RIGHT_WRIST_ROLL:--right-wrist-roll \
  RIGHT_WRIST_PITCH:--right-wrist-pitch \
  RIGHT_WRIST_YAW:--right-wrist-yaw \
  LEFT_SHOULDER_PITCH:--left-shoulder-pitch \
  LEFT_SHOULDER_ROLL:--left-shoulder-roll \
  LEFT_SHOULDER_YAW:--left-shoulder-yaw \
  LEFT_ELBOW:--left-elbow \
  LEFT_WRIST_ROLL:--left-wrist-roll \
  LEFT_WRIST_PITCH:--left-wrist-pitch \
  LEFT_WRIST_YAW:--left-wrist-yaw; do
  env_name="${spec%%:*}"
  arg_name="${spec#*:}"
  if [[ -n "${!env_name:-}" ]]; then
    EXTRA_ARM_ARGS+=("${arg_name}" "${!env_name}")
  fi
done
echo "[CONFIG] STAMP=${STAMP} G1 front-probe bumper smoke"
"${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_g1_box_scene.py \
  --viz none \
  --experience "${EXPERIENCE}" \
  --device "${DEVICE:-cpu}" \
  --kit_args "${KIT_ARGS}" \
  --steps "${STEPS:-360}" \
  --g1-usd "${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}" \
  --box-mass "${BOX_MASS:-2.0}" \
  --box-size "${BOX_SIZE_X:-0.45}" "${BOX_SIZE_Y:-0.30}" "${BOX_SIZE_Z:-0.30}" \
  --box-position "${BOX_POS_X:-0.75}" "${BOX_POS_Y:-0.0}" "${BOX_POS_Z:-0.15}" \
  --box-support-mode "${BOX_SUPPORT_MODE:-none}" \
  --box-support-size "${BOX_SUPPORT_SIZE_X:-0.75}" "${BOX_SUPPORT_SIZE_Y:-0.55}" "${BOX_SUPPORT_SIZE_Z:-0.65}" \
  --box-support-top-clearance "${BOX_SUPPORT_TOP_CLEARANCE:-0.0}" \
  --box-support-release-step "${BOX_SUPPORT_RELEASE_STEP:--1}" \
  --g1-root-position "${G1_ROOT_X:-0.0}" "${G1_ROOT_Y:-0.0}" "${G1_ROOT_Z:-0.78}" \
  --gait-mode "${GAIT_MODE:-stand}" \
  --gait-amplitude "${GAIT_AMPLITUDE:-0.0}" \
  --gait-frequency-hz "${GAIT_FREQUENCY_HZ:-0.7}" \
  --gait-start-step "${GAIT_START_STEP:-0}" \
  --gait-stop-step "${GAIT_STOP_STEP:--1}" \
  --gait-ramp-down-start-step "${GAIT_RAMP_DOWN_START_STEP:--1}" \
  --gait-ramp-down-end-step "${GAIT_RAMP_DOWN_END_STEP:--1}" \
  --gait-min-amplitude-scale "${GAIT_MIN_AMPLITUDE_SCALE:-0.0}" \
  --terminal-hold-start-step "${TERMINAL_HOLD_START_STEP:--1}" \
  --terminal-hold-hip-pitch-offset "${TERMINAL_HOLD_HIP_PITCH_OFFSET:-0.0}" \
  --terminal-hold-knee-offset "${TERMINAL_HOLD_KNEE_OFFSET:-0.0}" \
  --terminal-hold-ankle-pitch-offset "${TERMINAL_HOLD_ANKLE_PITCH_OFFSET:-0.0}" \
  --terminal-hold-waist-pitch-offset "${TERMINAL_HOLD_WAIST_PITCH_OFFSET:-0.0}" \
  --recovery-pitch-threshold "${RECOVERY_PITCH_THRESHOLD:-999.0}" \
  --recovery-pitch-rate-threshold "${RECOVERY_PITCH_RATE_THRESHOLD:-999.0}" \
  --recovery-hip-pitch-offset "${RECOVERY_HIP_PITCH_OFFSET:-0.0}" \
  --recovery-knee-offset "${RECOVERY_KNEE_OFFSET:-0.0}" \
  --recovery-ankle-pitch-offset "${RECOVERY_ANKLE_PITCH_OFFSET:-0.0}" \
  --recovery-waist-pitch-offset "${RECOVERY_WAIST_PITCH_OFFSET:-0.0}" \
  ${BALANCE_FEEDBACK_CONTROLLER:+--balance-feedback-controller} \
  --balance-pitch-gain "${BALANCE_PITCH_GAIN:-0.0}" \
  --balance-roll-gain "${BALANCE_ROLL_GAIN:-0.0}" \
  --balance-pitch-rate-gain "${BALANCE_PITCH_RATE_GAIN:-0.0}" \
  --balance-roll-rate-gain "${BALANCE_ROLL_RATE_GAIN:-0.0}" \
  --balance-adjustment-limit "${BALANCE_ADJUSTMENT_LIMIT:-0.25}" \
  --balance-pitch-target "${BALANCE_PITCH_TARGET:-0.0}" \
  --balance-roll-target "${BALANCE_ROLL_TARGET:-0.0}" \
  --balance-target-start-step "${BALANCE_TARGET_START_STEP:-0}" \
  --balance-target-end-step "${BALANCE_TARGET_END_STEP:--1}" \
  --balance-target-pulse-period-steps "${BALANCE_TARGET_PULSE_PERIOD_STEPS:-0}" \
  --balance-target-pulse-width-steps "${BALANCE_TARGET_PULSE_WIDTH_STEPS:-0}" \
  --balance-target-pulse-phase-step "${BALANCE_TARGET_PULSE_PHASE_STEP:-0}" \
  --balance-pitch-sign "${BALANCE_PITCH_SIGN:--1.0}" \
  --balance-roll-sign "${BALANCE_ROLL_SIGN:--1.0}" \
  --balance-start-step "${BALANCE_START_STEP:-0}" \
  --balance-pitch-activation-threshold "${BALANCE_PITCH_ACTIVATION_THRESHOLD:-0.0}" \
  --balance-roll-activation-threshold "${BALANCE_ROLL_ACTIVATION_THRESHOLD:-0.0}" \
  --balance-pitch-rate-activation-threshold "${BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD:-0.0}" \
  --balance-roll-rate-activation-threshold "${BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD:-0.0}" \
  --diagnostic-root-drive "${DIAGNOSTIC_ROOT_DRIVE:-none}" \
  --diagnostic-root-drive-start-step "${DIAGNOSTIC_ROOT_DRIVE_START_STEP:-0}" \
  --diagnostic-root-drive-stop-step "${DIAGNOSTIC_ROOT_DRIVE_STOP_STEP:--1}" \
  --diagnostic-root-drive-speed "${DIAGNOSTIC_ROOT_DRIVE_SPEED:-0.0}" \
  --diagnostic-root-drive-ramp-steps "${DIAGNOSTIC_ROOT_DRIVE_RAMP_STEPS:-120}" \
  --creep-hip-pitch-offset "${CREEP_HIP_PITCH_OFFSET:-0.12}" \
  --creep-knee-offset "${CREEP_KNEE_OFFSET:-0.04}" \
  --creep-ankle-pitch-offset "${CREEP_ANKLE_PITCH_OFFSET:--0.06}" \
  --creep-waist-pitch-offset "${CREEP_WAIST_PITCH_OFFSET:-0.04}" \
  --creep-stance-push-scale "${CREEP_STANCE_PUSH_SCALE:-0.18}" \
  --creep-lift-scale "${CREEP_LIFT_SCALE:-0.50}" \
  --creep-ankle-lift-scale "${CREEP_ANKLE_LIFT_SCALE:--0.30}" \
  "${EXTRA_ARM_ARGS[@]}" \
  --attach-box none \
  --torso-cradle "${TORSO_CRADLE:-none}" \
  --cradle-deck-size "${CRADLE_DECK_SIZE_X:-0.34}" "${CRADLE_DECK_SIZE_Y:-0.42}" "${CRADLE_DECK_SIZE_Z:-0.035}" \
  --cradle-deck-local-pos0 "${CRADLE_DECK_LOCAL_POS0_X:-0.30}" "${CRADLE_DECK_LOCAL_POS0_Y:-0.0}" "${CRADLE_DECK_LOCAL_POS0_Z:--0.02}" \
  --cradle-side-rail-height "${CRADLE_SIDE_RAIL_HEIGHT:-0.12}" \
  --cradle-end-stop-height "${CRADLE_END_STOP_HEIGHT:-0.16}" \
  --cradle-rail-thickness "${CRADLE_RAIL_THICKNESS:-0.025}" \
  --cradle-mass-scale "${CRADLE_MASS_SCALE:-1.0}" \
  --probe-mode "${PROBE_MODE:-front_bumper}" \
  --probe-start-step "${PROBE_START_STEP:-0}" \
  --probe-pad-size "${PROBE_PAD_SIZE_X:-0.10}" "${PROBE_PAD_SIZE_Y:-0.34}" "${PROBE_PAD_SIZE_Z:-0.10}" \
  --probe-pad-local-pos0 "${PROBE_PAD_LOCAL_POS0_X:-0.72}" "${PROBE_PAD_LOCAL_POS0_Y:-0.0}" "${PROBE_PAD_LOCAL_POS0_Z:--0.62}" \
  --probe-pad-mass "${PROBE_PAD_MASS:-0.05}" \
  --grasp-mode "${GRASP_MODE:-none}" \
  --grasp-body-path "${GRASP_BODY_PATH:-/World/G1/torso_link}" \
  --grasp-enable-step "${GRASP_ENABLE_STEP:-120}" \
  --grasp-lift-offset-z "${GRASP_LIFT_OFFSET_Z:-0.0}" \
  --fall-z "${FALL_Z:-0.45}" \
  --drop-z "${DROP_Z:-0.05}" \
  --output-dir "${OUTPUT_DIR}" \
  --stand-drive-preset arena \
  --stand-hip-pitch "${STAND_HIP_PITCH:--0.12}" \
  --stand-knee "${STAND_KNEE:-0.30}" \
  --stand-ankle-pitch "${STAND_ANKLE_PITCH:--0.15}" \
  --apply-arena-stand-gains \
  2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
