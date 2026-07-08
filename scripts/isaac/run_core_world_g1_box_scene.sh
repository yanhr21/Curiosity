#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_g1_box_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_box_scene/${STAMP}}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_DIR}/core_world_g1_box_scene_${STAMP}.log"
echo "[CONFIG] STAMP=${STAMP} core-world G1 box STEPS=${STEPS:-180} ATTACH_BOX=${ATTACH_BOX:-none} BOX_MASS=${BOX_MASS:-2.0}"
cmd=(
  "${ISAAC_VENV}/bin/python"
  "${ROOT_DIR}/scripts/isaac/build_core_world_g1_box_scene.py"
  --viz none
  --experience "${EXPERIENCE}"
  --device "${DEVICE:-cpu}"
  --kit_args "${KIT_ARGS}"
  --steps "${STEPS:-180}"
  --g1-usd "${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
  --box-mass "${BOX_MASS:-2.0}"
  --box-size "${BOX_SIZE_X:-0.45}" "${BOX_SIZE_Y:-0.30}" "${BOX_SIZE_Z:-0.30}"
  --box-position "${BOX_POS_X:-0.55}" "${BOX_POS_Y:-0.0}" "${BOX_POS_Z:-0.95}"
  --target-xy "${TARGET_X:-1.2}" "${TARGET_Y:-0.0}"
  --g1-root-position "${G1_ROOT_X:-0.0}" "${G1_ROOT_Y:-0.0}" "${G1_ROOT_Z:-0.78}"
  --g1-root-orientation-wxyz "${G1_ROOT_QW:-1.0}" "${G1_ROOT_QX:-0.0}" "${G1_ROOT_QY:-0.0}" "${G1_ROOT_QZ:-0.0}"
  --gait-mode "${GAIT_MODE:-stand}"
  --gait-amplitude "${GAIT_AMPLITUDE:-0.0}"
  --gait-frequency-hz "${GAIT_FREQUENCY_HZ:-0.7}"
  --gait-start-step "${GAIT_START_STEP:-0}"
  --gait-stop-step "${GAIT_STOP_STEP:--1}"
  --gait-ramp-down-start-step "${GAIT_RAMP_DOWN_START_STEP:--1}"
  --gait-ramp-down-end-step "${GAIT_RAMP_DOWN_END_STEP:--1}"
  --gait-min-amplitude-scale "${GAIT_MIN_AMPLITUDE_SCALE:-0.0}"
  --recovery-pitch-threshold "${RECOVERY_PITCH_THRESHOLD:-999.0}"
  --recovery-pitch-rate-threshold "${RECOVERY_PITCH_RATE_THRESHOLD:-999.0}"
  --recovery-hip-pitch-offset "${RECOVERY_HIP_PITCH_OFFSET:-0.0}"
  --recovery-knee-offset "${RECOVERY_KNEE_OFFSET:-0.0}"
  --recovery-ankle-pitch-offset "${RECOVERY_ANKLE_PITCH_OFFSET:-0.0}"
  --recovery-waist-pitch-offset "${RECOVERY_WAIST_PITCH_OFFSET:-0.0}"
  --terminal-hold-start-step "${TERMINAL_HOLD_START_STEP:--1}"
  --terminal-hold-box-target-travel "${TERMINAL_HOLD_BOX_TARGET_TRAVEL:--1.0}"
  --terminal-hold-robot-target-travel "${TERMINAL_HOLD_ROBOT_TARGET_TRAVEL:--1.0}"
  --terminal-hold-pitch-threshold "${TERMINAL_HOLD_PITCH_THRESHOLD:-999.0}"
  --terminal-hold-pitch-rate-threshold "${TERMINAL_HOLD_PITCH_RATE_THRESHOLD:-999.0}"
  --terminal-hold-hip-pitch-offset "${TERMINAL_HOLD_HIP_PITCH_OFFSET:-0.0}"
  --terminal-hold-knee-offset "${TERMINAL_HOLD_KNEE_OFFSET:-0.0}"
  --terminal-hold-ankle-pitch-offset "${TERMINAL_HOLD_ANKLE_PITCH_OFFSET:-0.0}"
  --terminal-hold-waist-pitch-offset "${TERMINAL_HOLD_WAIST_PITCH_OFFSET:-0.0}"
  --terminal-drive-gain-scale "${TERMINAL_DRIVE_GAIN_SCALE:--1.0}"
  --terminal-drive-force-scale "${TERMINAL_DRIVE_FORCE_SCALE:--1.0}"
  --balance-pitch-gain "${BALANCE_PITCH_GAIN:-0.0}"
  --balance-roll-gain "${BALANCE_ROLL_GAIN:-0.0}"
  --balance-pitch-rate-gain "${BALANCE_PITCH_RATE_GAIN:-0.0}"
  --balance-roll-rate-gain "${BALANCE_ROLL_RATE_GAIN:-0.0}"
  --balance-adjustment-limit "${BALANCE_ADJUSTMENT_LIMIT:-0.25}"
  --balance-pitch-target "${BALANCE_PITCH_TARGET:-0.0}"
  --balance-roll-target "${BALANCE_ROLL_TARGET:-0.0}"
  --balance-target-start-step "${BALANCE_TARGET_START_STEP:-0}"
  --balance-target-end-step "${BALANCE_TARGET_END_STEP:--1}"
  --balance-target-pulse-period-steps "${BALANCE_TARGET_PULSE_PERIOD_STEPS:-0}"
  --balance-target-pulse-width-steps "${BALANCE_TARGET_PULSE_WIDTH_STEPS:-0}"
  --balance-target-pulse-phase-step "${BALANCE_TARGET_PULSE_PHASE_STEP:-0}"
  --balance-pitch-sign "${BALANCE_PITCH_SIGN:--1.0}"
  --balance-roll-sign "${BALANCE_ROLL_SIGN:--1.0}"
  --balance-start-step "${BALANCE_START_STEP:-0}"
  --balance-pitch-activation-threshold "${BALANCE_PITCH_ACTIVATION_THRESHOLD:-0.0}"
  --balance-roll-activation-threshold "${BALANCE_ROLL_ACTIVATION_THRESHOLD:-0.0}"
  --balance-pitch-rate-activation-threshold "${BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD:-0.0}"
  --balance-roll-rate-activation-threshold "${BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD:-0.0}"
  --policy-start-step "${POLICY_START_STEP:-40}"
  --policy-control-decimation "${POLICY_CONTROL_DECIMATION:-4}"
  --agile-command "${AGILE_COMMAND_X:-0.25}" "${AGILE_COMMAND_Y:-0.0}" "${AGILE_COMMAND_YAW:-0.0}"
  --agile-height-command "${AGILE_HEIGHT_COMMAND:-0.72}"
  --agile-policy-backend "${AGILE_POLICY_BACKEND:-torch_checkpoint}"
  --agile-config "${AGILE_CONFIG:-${ROOT_DIR}/external/IsaacLab-Arena/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_agile.yaml}"
  --agile-onnx "${AGILE_ONNX:-${ROOT_DIR}/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx}"
  --agile-torch-checkpoint "${AGILE_TORCH_CHECKPOINT:-${ROOT_DIR}/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt}"
  --attach-box "${ATTACH_BOX:-none}"
  --attach-body-path "${ATTACH_BODY_PATH:-/World/G1/torso_link}"
  --attach-local-pos0 "${ATTACH_LOCAL_POS0_X:-0.24}" "${ATTACH_LOCAL_POS0_Y:-0.0}" "${ATTACH_LOCAL_POS0_Z:-0.08}"
  --torso-cradle "${TORSO_CRADLE:-none}"
  --cradle-deck-size "${CRADLE_DECK_SIZE_X:-0.34}" "${CRADLE_DECK_SIZE_Y:-0.42}" "${CRADLE_DECK_SIZE_Z:-0.035}"
  --cradle-deck-local-pos0 "${CRADLE_DECK_LOCAL_POS0_X:-0.30}" "${CRADLE_DECK_LOCAL_POS0_Y:-0.0}" "${CRADLE_DECK_LOCAL_POS0_Z:--0.02}"
  --cradle-side-rail-height "${CRADLE_SIDE_RAIL_HEIGHT:-0.12}"
  --cradle-end-stop-height "${CRADLE_END_STOP_HEIGHT:-0.16}"
  --cradle-rail-thickness "${CRADLE_RAIL_THICKNESS:-0.025}"
  --cradle-mass-scale "${CRADLE_MASS_SCALE:-1.0}"
  --probe-mode "${PROBE_MODE:-none}"
  --probe-start-step "${PROBE_START_STEP:-0}"
  --probe-end-step "${PROBE_END_STEP:--1}"
  --probe-pad-size "${PROBE_PAD_SIZE_X:-0.05}" "${PROBE_PAD_SIZE_Y:-0.36}" "${PROBE_PAD_SIZE_Z:-0.18}"
  --probe-pad-local-pos0 "${PROBE_PAD_LOCAL_POS0_X:-0.50}" "${PROBE_PAD_LOCAL_POS0_Y:-0.0}" "${PROBE_PAD_LOCAL_POS0_Z:-0.02}"
  --probe-pad-mass "${PROBE_PAD_MASS:-0.2}"
  --fall-z "${FALL_Z:-0.45}"
  --drop-z "${DROP_Z:-0.20}"
  --output-dir "${OUTPUT_DIR}"
  --stand-drive-preset "${STAND_DRIVE_PRESET:-arena}"
  --stand-gain-scale "${STAND_GAIN_SCALE:-1.0}"
  --stand-force-scale "${STAND_FORCE_SCALE:-1.0}"
)
if [[ "${DISABLE_USD_PELVIS_XFORM:-0}" == "1" ]]; then
  cmd+=(--disable-usd-pelvis-xform)
fi
if [[ "${DISABLE_SETUP_JOINT_STATE_WRITE:-0}" == "1" ]]; then
  cmd+=(--disable-setup-joint-state-write)
fi
if [[ -n "${STAND_HIP_PITCH:-}" ]]; then
  cmd+=(--stand-hip-pitch "${STAND_HIP_PITCH}")
fi
if [[ -n "${STAND_KNEE:-}" ]]; then
  cmd+=(--stand-knee "${STAND_KNEE}")
fi
if [[ -n "${STAND_ANKLE_PITCH:-}" ]]; then
  cmd+=(--stand-ankle-pitch "${STAND_ANKLE_PITCH}")
fi
if [[ -n "${STAND_HIP_ROLL:-}" ]]; then
  cmd+=(--stand-hip-roll "${STAND_HIP_ROLL}")
fi
if [[ -n "${STAND_ANKLE_ROLL:-}" ]]; then
  cmd+=(--stand-ankle-roll "${STAND_ANKLE_ROLL}")
fi
if [[ -n "${APPLY_ARENA_STAND_GAINS:-}" ]]; then
  cmd+=(--apply-arena-stand-gains)
fi
if [[ "${BALANCE_FEEDBACK_CONTROLLER:-0}" == "1" ]]; then
  cmd+=(--balance-feedback-controller)
fi
if [[ "${BOX_COLLISION_ENABLED:-1}" == "0" ]]; then
  cmd+=(--disable-box-collision)
fi
if [[ "${SPAWN_CARRY_BOX:-1}" == "0" ]]; then
  cmd+=(--disable-carry-box-spawn)
fi
if [[ "${CRADLE_COLLISION_ENABLED:-1}" == "0" ]]; then
  cmd+=(--disable-cradle-collision)
fi
if [[ "${PROBE_PAD_COLLISION_ENABLED:-1}" == "0" ]]; then
  cmd+=(--disable-probe-pad-collision)
fi
if [[ "${PROBE_COLLISION_WINDOW:-0}" == "1" ]]; then
  cmd+=(--probe-collision-window)
fi
if [[ "${REQUIRE_BOX_NO_DROP:-0}" == "1" ]]; then
  cmd+=(--require-box-no-drop)
fi
"${cmd[@]}" 2>&1 | tee "${LOG_PATH}"

echo "[INFO] Log: ${LOG_PATH}"
echo "[INFO] Output: ${OUTPUT_DIR}"
