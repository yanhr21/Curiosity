#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 AGILE policy suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_agile_policy_low_cradle)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"
AGILE_POLICY_BACKEND="${AGILE_POLICY_BACKEND:-onnx}"
AGILE_COMMAND_X="${AGILE_COMMAND_X:-0.10}"
AGILE_COMMAND_Y="${AGILE_COMMAND_Y:-0.0}"
AGILE_COMMAND_YAW="${AGILE_COMMAND_YAW:-0.0}"
AGILE_HEIGHT_COMMAND="${AGILE_HEIGHT_COMMAND:-0.72}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

env | sort | grep -E '^(AGILE_|ARM_|BALANCE_|CAPTURE_|CRADLE_|FREE_|G1_|GENERATE_|LARGERBOX_|MIN_|RECORD_|RUN_|STRICT|SUITE_|TARGET_)' \
  > "${SUITE_DIR}/agile_policy_low_cradle_env_snapshot.txt"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/agile_policy_low_cradle_status.tsv"
printf "case\tbuild_status\tcheck_status\toutput_dir\n" > "${status_file}"

base_python=(
  "${ISAAC_VENV}/bin/python"
  scripts/isaac/build_core_world_g1_box_scene.py
  --viz none
  --experience "${EXPERIENCE}"
  --device "${DEVICE}"
  --kit_args "${KIT_ARGS}"
  --g1-usd "${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
  --target-xy "${TARGET_X:-1.2}" "${TARGET_Y:-0.0}"
  --g1-root-position 0.0 0.0 0.78
  --g1-root-orientation-wxyz "${G1_ROOT_QW:-1.0}" "${G1_ROOT_QX:-0.0}" "${G1_ROOT_QY:-0.0}" "${G1_ROOT_QZ:-0.0}"
  --stand-hip-pitch -0.10
  --stand-knee 0.30
  --stand-ankle-pitch -0.20
  --apply-arena-stand-gains
  --stand-drive-preset "${STAND_DRIVE_PRESET:-isaaclab29dof}"
  --stand-gain-scale 1.0
  --stand-force-scale 1.0
  --gait-mode agile_policy
  --policy-start-step 40
  --policy-control-decimation 4
  --agile-command "${AGILE_COMMAND_X}" "${AGILE_COMMAND_Y}" "${AGILE_COMMAND_YAW}"
  --agile-height-command "${AGILE_HEIGHT_COMMAND}"
  --target-window-center "${TARGET_WINDOW_CENTER:--1.0}"
  --target-window-halfwidth "${TARGET_WINDOW_HALFWIDTH:--1.0}"
  --arm-pose-mode "${ARM_POSE_MODE:-none}"
  --arm-pose-start-step "${ARM_POSE_START_STEP:-0}"
  --arm-pose-ramp-steps "${ARM_POSE_RAMP_STEPS:-120}"
  --box-retention-rel-start "${BOX_RETENTION_REL_START:-0.10}"
  --box-retention-rel-stop "${BOX_RETENTION_REL_STOP:-0.28}"
  --box-retention-tilt-start "${BOX_RETENTION_TILT_START:-0.20}"
  --box-retention-tilt-stop "${BOX_RETENTION_TILT_STOP:-0.55}"
  --box-retention-hip-pitch-offset "${BOX_RETENTION_HIP_PITCH_OFFSET:--0.04}"
  --box-retention-knee-offset "${BOX_RETENTION_KNEE_OFFSET:-0.12}"
  --box-retention-ankle-pitch-offset "${BOX_RETENTION_ANKLE_PITCH_OFFSET:--0.06}"
  --box-retention-waist-pitch-offset "${BOX_RETENTION_WAIST_PITCH_OFFSET:--0.03}"
  --box-retention-shoulder-pitch-offset "${BOX_RETENTION_SHOULDER_PITCH_OFFSET:--0.10}"
  --box-retention-elbow-offset "${BOX_RETENTION_ELBOW_OFFSET:-0.16}"
  --box-retention-wrist-pitch-offset "${BOX_RETENTION_WRIST_PITCH_OFFSET:--0.04}"
  --probe-mode "${PROBE_MODE:-none}"
  --probe-start-step "${PROBE_START_STEP:-0}"
  --probe-end-step "${PROBE_END_STEP:--1}"
  --probe-pad-size "${PROBE_PAD_SIZE_X:-0.05}" "${PROBE_PAD_SIZE_Y:-0.36}" "${PROBE_PAD_SIZE_Z:-0.18}"
  --probe-pad-local-pos0 "${PROBE_PAD_LOCAL_X:-0.50}" "${PROBE_PAD_LOCAL_Y:-0.0}" "${PROBE_PAD_LOCAL_Z:-0.02}"
  --probe-pad-mass "${PROBE_PAD_MASS:-0.2}"
  --agile-command-stop-step "${AGILE_COMMAND_STOP_STEP:--1}"
  --agile-command-stop-box-target-travel "${AGILE_COMMAND_STOP_BOX_TARGET_TRAVEL:--1.0}"
  --agile-command-stop-robot-target-travel "${AGILE_COMMAND_STOP_ROBOT_TARGET_TRAVEL:--1.0}"
  --agile-command-stop-target-window-min-step "${AGILE_COMMAND_STOP_TARGET_WINDOW_MIN_STEP:--1}"
  --agile-command-hold-scale "${AGILE_COMMAND_HOLD_SCALE:-0.0}"
  --agile-command-hold-adaptive-min-scale "${AGILE_COMMAND_HOLD_ADAPTIVE_MIN_SCALE:-0.0}"
  --agile-command-hold-adaptive-max-scale "${AGILE_COMMAND_HOLD_ADAPTIVE_MAX_SCALE:-1.0}"
  --agile-command-hold-adaptive-tilt-start "${AGILE_COMMAND_HOLD_ADAPTIVE_TILT_START:-0.20}"
  --agile-command-hold-adaptive-tilt-stop "${AGILE_COMMAND_HOLD_ADAPTIVE_TILT_STOP:-0.65}"
  --agile-command-hold-adaptive-rate-start "${AGILE_COMMAND_HOLD_ADAPTIVE_RATE_START:-2.0}"
  --agile-command-hold-adaptive-rate-stop "${AGILE_COMMAND_HOLD_ADAPTIVE_RATE_STOP:-8.0}"
  --agile-command-hold-adaptive-rel-start "${AGILE_COMMAND_HOLD_ADAPTIVE_REL_START:-0.16}"
  --agile-command-hold-adaptive-rel-stop "${AGILE_COMMAND_HOLD_ADAPTIVE_REL_STOP:-0.35}"
  --agile-command-hold-adaptive-box-tilt-start "${AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_START:-0.16}"
  --agile-command-hold-adaptive-box-tilt-stop "${AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_STOP:-0.45}"
  --agile-command-hold-adaptive-box-tilt-rate-start "${AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_START:-2.0}"
  --agile-command-hold-adaptive-box-tilt-rate-stop "${AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT_RATE_STOP:-8.0}"
  --agile-command-hold-adaptive-scale-smoothing "${AGILE_COMMAND_HOLD_ADAPTIVE_SCALE_SMOOTHING:-0.15}"
  --agile-command-hold-lateral-gain "${AGILE_COMMAND_HOLD_LATERAL_GAIN:-0.0}"
  --agile-command-hold-lateral-limit "${AGILE_COMMAND_HOLD_LATERAL_LIMIT:-0.05}"
  --agile-command-hold-lateral-sign "${AGILE_COMMAND_HOLD_LATERAL_SIGN:-1.0}"
  --agile-command-hold-lateral-error-start "${AGILE_COMMAND_HOLD_LATERAL_ERROR_START:-0.0}"
  --agile-command-hold-lateral-max-tilt "${AGILE_COMMAND_HOLD_LATERAL_MAX_TILT:-999.0}"
  --agile-command-hold-lateral-max-box-tilt "${AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT:-999.0}"
  --agile-command-hold-yaw-gain "${AGILE_COMMAND_HOLD_YAW_GAIN:-0.0}"
  --agile-command-hold-yaw-limit "${AGILE_COMMAND_HOLD_YAW_LIMIT:-0.20}"
  --agile-command-hold-yaw-sign "${AGILE_COMMAND_HOLD_YAW_SIGN:-1.0}"
  --agile-command-box-progress-start-step "${AGILE_COMMAND_BOX_PROGRESS_START_STEP:-0}"
  --agile-command-box-progress-target "${AGILE_COMMAND_BOX_PROGRESS_TARGET:--1.0}"
  --agile-command-box-progress-deadband "${AGILE_COMMAND_BOX_PROGRESS_DEADBAND:-0.05}"
  --agile-command-box-progress-gain "${AGILE_COMMAND_BOX_PROGRESS_GAIN:-0.08}"
  --agile-command-box-progress-max-forward "${AGILE_COMMAND_BOX_PROGRESS_MAX_FORWARD:-0.10}"
  --agile-command-box-progress-max-reverse "${AGILE_COMMAND_BOX_PROGRESS_MAX_REVERSE:-0.03}"
  --agile-command-box-progress-max-tilt "${AGILE_COMMAND_BOX_PROGRESS_MAX_TILT:-999.0}"
  --agile-command-box-progress-max-box-tilt "${AGILE_COMMAND_BOX_PROGRESS_MAX_BOX_TILT:-999.0}"
  --agile-command-box-lateral-deadband "${AGILE_COMMAND_BOX_LATERAL_DEADBAND:-0.08}"
  --agile-command-box-lateral-gain "${AGILE_COMMAND_BOX_LATERAL_GAIN:-0.02}"
  --agile-command-box-lateral-limit "${AGILE_COMMAND_BOX_LATERAL_LIMIT:-0.004}"
  --agile-command-box-lateral-sign "${AGILE_COMMAND_BOX_LATERAL_SIGN:-1.0}"
  --agile-command-hold-terminal-box-target-travel "${AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL:--1.0}"
  --agile-command-hold-terminal-min-robot-target-travel "${AGILE_COMMAND_HOLD_TERMINAL_MIN_ROBOT_TARGET_TRAVEL:--1.0}"
  --agile-command-hold-terminal-min-step "${AGILE_COMMAND_HOLD_TERMINAL_MIN_STEP:--1}"
  --agile-command-hold-terminal-scale "${AGILE_COMMAND_HOLD_TERMINAL_SCALE:-0.0}"
  --agile-command-hold-final-box-target-travel "${AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL:--1.0}"
  --agile-command-hold-final-min-robot-target-travel "${AGILE_COMMAND_HOLD_FINAL_MIN_ROBOT_TARGET_TRAVEL:--1.0}"
  --agile-command-hold-final-min-step "${AGILE_COMMAND_HOLD_FINAL_MIN_STEP:--1}"
  --agile-command-hold-final-scale "${AGILE_COMMAND_HOLD_FINAL_SCALE:--1.0}"
  --agile-command-hold-final-brake-command-x "${AGILE_COMMAND_HOLD_FINAL_BRAKE_COMMAND_X:-0.0}"
  --agile-command-hold-final-brake-delay-steps "${AGILE_COMMAND_HOLD_FINAL_BRAKE_DELAY_STEPS:-0}"
  --agile-command-hold-final-brake-steps "${AGILE_COMMAND_HOLD_FINAL_BRAKE_STEPS:-0}"
  --agile-command-hold-final-freeze-max-tilt "${AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_TILT:-0.25}"
  --agile-command-hold-final-freeze-max-box-tilt "${AGILE_COMMAND_HOLD_FINAL_FREEZE_MAX_BOX_TILT:-0.35}"
  --agile-command-hold-final-stand-delay-steps "${AGILE_COMMAND_HOLD_FINAL_STAND_DELAY_STEPS:-0}"
  --agile-command-hold-mode "${AGILE_COMMAND_HOLD_MODE:-policy_command}"
  --agile-command-hold-stand-blend-rate "${AGILE_COMMAND_HOLD_STAND_BLEND_RATE:-0.04}"
  --agile-command-hold-policy-then-stand-delay-steps "${AGILE_COMMAND_HOLD_POLICY_THEN_STAND_DELAY_STEPS:-80}"
  --agile-command-hold-rescue-forward-pitch-threshold "${AGILE_COMMAND_HOLD_RESCUE_FORWARD_PITCH_THRESHOLD:--999.0}"
  --agile-command-hold-rescue-abs-roll-threshold "${AGILE_COMMAND_HOLD_RESCUE_ABS_ROLL_THRESHOLD:-999.0}"
  --agile-command-hold-rescue-blend-rate "${AGILE_COMMAND_HOLD_RESCUE_BLEND_RATE:-0.04}"
  --agile-policy-backend "${AGILE_POLICY_BACKEND}"
  --agile-config "${AGILE_CONFIG:-${ROOT_DIR}/external/IsaacLab-Arena/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_agile.yaml}"
  --agile-onnx "${AGILE_ONNX:-${ROOT_DIR}/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx}"
  --agile-torch-checkpoint "${AGILE_TORCH_CHECKPOINT:-${ROOT_DIR}/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt}"
)

if [[ "${CAPTURE_RGB:-0}" == "1" ]]; then
  base_python+=(
    --enable_cameras
    --capture-rgb
    --render
    --capture-rgb-every-n-steps "${CAPTURE_RGB_EVERY_N_STEPS:-10}"
    --capture-rgb-resolution "${CAPTURE_RGB_WIDTH:-1280}" "${CAPTURE_RGB_HEIGHT:-720}"
    --capture-rgb-rt-subframes "${CAPTURE_RGB_RT_SUBFRAMES:-4}"
    --capture-camera-position "${CAPTURE_CAMERA_X:-1.8}" "${CAPTURE_CAMERA_Y:--2.4}" "${CAPTURE_CAMERA_Z:-1.25}"
    --capture-camera-look-at "${CAPTURE_LOOK_AT_X:--0.45}" "${CAPTURE_LOOK_AT_Y:-0.0}" "${CAPTURE_LOOK_AT_Z:-0.82}"
  )
fi

if [[ "${RECORD_REPLAY_CSV:-0}" == "1" ]]; then
  base_python+=(
    --record-replay-csv
    --record-replay-every-n-steps "${RECORD_REPLAY_EVERY_N_STEPS:-10}"
  )
fi
if [[ "${BOX_RETENTION_POSTURE_CONTROLLER:-0}" == "1" ]]; then
  base_python+=(--box-retention-posture-controller)
fi

if [[ -n "${AGILE_COMMAND_HOLD_STAND_HIP_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-stand-hip-pitch "${AGILE_COMMAND_HOLD_STAND_HIP_PITCH}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_STAND_KNEE:-}" ]]; then
  base_python+=(--agile-command-hold-stand-knee "${AGILE_COMMAND_HOLD_STAND_KNEE}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_STAND_ANKLE_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-stand-ankle-pitch "${AGILE_COMMAND_HOLD_STAND_ANKLE_PITCH}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_STAND_HIP_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-stand-hip-roll "${AGILE_COMMAND_HOLD_STAND_HIP_ROLL}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_STAND_ANKLE_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-stand-ankle-roll "${AGILE_COMMAND_HOLD_STAND_ANKLE_ROLL}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_STAND_WAIST_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-stand-waist-pitch "${AGILE_COMMAND_HOLD_STAND_WAIST_PITCH}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_RESCUE_HIP_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-hip-pitch "${AGILE_COMMAND_HOLD_RESCUE_HIP_PITCH}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_RESCUE_KNEE:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-knee "${AGILE_COMMAND_HOLD_RESCUE_KNEE}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_RESCUE_ANKLE_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-ankle-pitch "${AGILE_COMMAND_HOLD_RESCUE_ANKLE_PITCH}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_RESCUE_HIP_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-hip-roll "${AGILE_COMMAND_HOLD_RESCUE_HIP_ROLL}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_RESCUE_ANKLE_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-ankle-roll "${AGILE_COMMAND_HOLD_RESCUE_ANKLE_ROLL}")
fi
if [[ -n "${AGILE_COMMAND_HOLD_RESCUE_WAIST_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-waist-pitch "${AGILE_COMMAND_HOLD_RESCUE_WAIST_PITCH}")
fi

if [[ "${AGILE_COMMAND_HOLD_RESCUE_ENABLE:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-rescue-enable)
fi
if [[ "${AGILE_COMMAND_HOLD_ADAPTIVE_SCALE:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-adaptive-scale)
fi
if [[ "${AGILE_COMMAND_HOLD_ADAPTIVE_BOX_TILT:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-adaptive-box-tilt)
fi
if [[ "${AGILE_COMMAND_HOLD_LATERAL_CORRECTION:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-lateral-correction)
fi
if [[ "${AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-lateral-terminal-only)
fi
if [[ "${AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-lateral-use-excess-error)
fi
if [[ "${AGILE_COMMAND_BOX_PROGRESS_CONTROLLER:-0}" == "1" ]]; then
  base_python+=(--agile-command-box-progress-controller)
fi
if [[ "${AGILE_COMMAND_BOX_LATERAL_CONTROLLER:-0}" == "1" ]]; then
  base_python+=(--agile-command-box-lateral-controller)
fi
if [[ "${AGILE_COMMAND_HOLD_YAW_CORRECTION:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-yaw-correction)
fi
if [[ "${AGILE_COMMAND_BOX_PROGRESS_CONTROLLER:-0}" == "1" ]]; then
  base_python+=(--agile-command-box-progress-controller)
fi
if [[ "${AGILE_COMMAND_STOP_TARGET_WINDOW:-0}" == "1" ]]; then
  base_python+=(--agile-command-stop-target-window)
fi
if [[ "${AGILE_COMMAND_BOX_PROGRESS_SCALE_ON_HOLD:-0}" == "1" ]]; then
  base_python+=(--agile-command-box-progress-scale-on-hold)
fi
if [[ "${AGILE_COMMAND_BOX_LATERAL_CONTROLLER:-0}" == "1" ]]; then
  base_python+=(--agile-command-box-lateral-controller)
fi
if [[ "${AGILE_COMMAND_BOX_LATERAL_SCALE_ON_HOLD:-0}" == "1" ]]; then
  base_python+=(--agile-command-box-lateral-scale-on-hold)
fi
if [[ "${AGILE_COMMAND_HOLD_TERMINAL_LATCH:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-terminal-latch)
fi
if [[ "${AGILE_COMMAND_HOLD_FINAL_LATCH:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-final-latch)
fi
if [[ "${AGILE_COMMAND_HOLD_FINAL_ZERO_CORRECTIONS:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-final-zero-corrections)
fi
if [[ "${AGILE_COMMAND_HOLD_FINAL_RESET_POLICY_STATE:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-final-reset-policy-state)
fi
if [[ "${AGILE_COMMAND_HOLD_FINAL_FREEZE_IN_TARGET_WINDOW:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-final-freeze-in-target-window)
fi
if [[ "${AGILE_COMMAND_HOLD_RESCUE_OVERRIDES_FINAL_FREEZE:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-rescue-overrides-final-freeze)
fi
if [[ "${AGILE_COMMAND_HOLD_FINAL_STAND:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-final-stand)
fi
if [[ "${DISABLE_PROBE_PAD_COLLISION:-0}" == "1" ]]; then
  base_python+=(--disable-probe-pad-collision)
fi
if [[ "${PROBE_COLLISION_WINDOW:-0}" == "1" ]]; then
  base_python+=(--probe-collision-window)
fi

if [[ "${BALANCE_FEEDBACK_CONTROLLER:-0}" == "1" ]]; then
  base_python+=(
    --balance-feedback-controller
    --balance-pitch-gain "${BALANCE_PITCH_GAIN:-0.0}"
    --balance-roll-gain "${BALANCE_ROLL_GAIN:-0.0}"
    --balance-pitch-rate-gain "${BALANCE_PITCH_RATE_GAIN:-0.0}"
    --balance-roll-rate-gain "${BALANCE_ROLL_RATE_GAIN:-0.0}"
    --balance-adjustment-limit "${BALANCE_ADJUSTMENT_LIMIT:-0.25}"
    --balance-feedback-base "${BALANCE_FEEDBACK_BASE:-stand}"
    --balance-roll-left-ankle-scale "${BALANCE_ROLL_LEFT_ANKLE_SCALE:-1.0}"
    --balance-roll-right-ankle-scale "${BALANCE_ROLL_RIGHT_ANKLE_SCALE:-1.0}"
    --balance-roll-left-hip-scale "${BALANCE_ROLL_LEFT_HIP_SCALE:--0.5}"
    --balance-roll-right-hip-scale "${BALANCE_ROLL_RIGHT_HIP_SCALE:--0.5}"
    --balance-pitch-target "${BALANCE_PITCH_TARGET:-0.0}"
    --balance-roll-target "${BALANCE_ROLL_TARGET:-0.0}"
    --balance-roll-target-lateral-source "${BALANCE_ROLL_TARGET_LATERAL_SOURCE:-robot}"
    --balance-roll-target-lateral-gain "${BALANCE_ROLL_TARGET_LATERAL_GAIN:-0.0}"
    --balance-roll-target-lateral-limit "${BALANCE_ROLL_TARGET_LATERAL_LIMIT:-0.0}"
    --balance-roll-target-lateral-deadband "${BALANCE_ROLL_TARGET_LATERAL_DEADBAND:-0.0}"
    --balance-roll-target-lateral-sign "${BALANCE_ROLL_TARGET_LATERAL_SIGN:-1.0}"
    --balance-roll-target-lateral-start-after-hold-steps "${BALANCE_ROLL_TARGET_LATERAL_START_AFTER_HOLD_STEPS:-0}"
    --balance-roll-target-lateral-ramp-steps "${BALANCE_ROLL_TARGET_LATERAL_RAMP_STEPS:-0}"
    --balance-roll-target-lateral-max-tilt "${BALANCE_ROLL_TARGET_LATERAL_MAX_TILT:-999.0}"
    --balance-roll-target-lateral-max-box-tilt "${BALANCE_ROLL_TARGET_LATERAL_MAX_BOX_TILT:-999.0}"
    --balance-target-start-step "${BALANCE_TARGET_START_STEP:-0}"
    --balance-target-end-step "${BALANCE_TARGET_END_STEP:--1}"
    --balance-pitch-sign "${BALANCE_PITCH_SIGN:--1.0}"
    --balance-roll-sign "${BALANCE_ROLL_SIGN:--1.0}"
    --balance-start-step "${BALANCE_START_STEP:-0}"
    --balance-pitch-activation-threshold "${BALANCE_PITCH_ACTIVATION_THRESHOLD:-0.0}"
    --balance-roll-activation-threshold "${BALANCE_ROLL_ACTIVATION_THRESHOLD:-0.0}"
    --balance-pitch-rate-activation-threshold "${BALANCE_PITCH_RATE_ACTIVATION_THRESHOLD:-0.0}"
    --balance-roll-rate-activation-threshold "${BALANCE_ROLL_RATE_ACTIVATION_THRESHOLD:-0.0}"
  )
fi
if [[ "${BALANCE_START_ON_AGILE_HOLD:-0}" == "1" ]]; then
  base_python+=(--balance-start-on-agile-hold)
fi
if [[ "${BALANCE_ROLL_TARGET_FROM_LATERAL:-0}" == "1" ]]; then
  base_python+=(--balance-roll-target-from-lateral)
fi

if [[ "${AGILE_COMMAND_HOLD_RESET_POLICY_STATE:-0}" == "1" ]]; then
  base_python+=(--agile-command-hold-reset-policy-state)
fi

if [[ -n "${AGILE_HOLD_STAND_HIP_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-stand-hip-pitch "${AGILE_HOLD_STAND_HIP_PITCH}")
fi
if [[ -n "${AGILE_HOLD_STAND_KNEE:-}" ]]; then
  base_python+=(--agile-command-hold-stand-knee "${AGILE_HOLD_STAND_KNEE}")
fi
if [[ -n "${AGILE_HOLD_STAND_ANKLE_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-stand-ankle-pitch "${AGILE_HOLD_STAND_ANKLE_PITCH}")
fi
if [[ -n "${AGILE_HOLD_STAND_HIP_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-stand-hip-roll "${AGILE_HOLD_STAND_HIP_ROLL}")
fi
if [[ -n "${AGILE_HOLD_STAND_ANKLE_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-stand-ankle-roll "${AGILE_HOLD_STAND_ANKLE_ROLL}")
fi
if [[ -n "${AGILE_HOLD_STAND_WAIST_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-stand-waist-pitch "${AGILE_HOLD_STAND_WAIST_PITCH}")
fi

if [[ -n "${AGILE_HOLD_RESCUE_HIP_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-hip-pitch "${AGILE_HOLD_RESCUE_HIP_PITCH}")
fi
if [[ -n "${AGILE_HOLD_RESCUE_KNEE:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-knee "${AGILE_HOLD_RESCUE_KNEE}")
fi
if [[ -n "${AGILE_HOLD_RESCUE_ANKLE_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-ankle-pitch "${AGILE_HOLD_RESCUE_ANKLE_PITCH}")
fi
if [[ -n "${AGILE_HOLD_RESCUE_HIP_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-hip-roll "${AGILE_HOLD_RESCUE_HIP_ROLL}")
fi
if [[ -n "${AGILE_HOLD_RESCUE_ANKLE_ROLL:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-ankle-roll "${AGILE_HOLD_RESCUE_ANKLE_ROLL}")
fi
if [[ -n "${AGILE_HOLD_RESCUE_WAIST_PITCH:-}" ]]; then
  base_python+=(--agile-command-hold-rescue-waist-pitch "${AGILE_HOLD_RESCUE_WAIST_PITCH}")
fi

free_cradle_extra_args=()
if [[ "${CRADLE_TOP_LID_ENABLED:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-top-lid)
fi
if [[ "${CRADLE_TOP_LID_ENABLE_ON_HOLD:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-top-lid-enable-on-hold)
fi
if [[ "${CRADLE_CHEST_PAD_ENABLED:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad)
fi
if [[ "${CRADLE_CHEST_PAD_SPAWN_ON_TRIGGER:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad-spawn-on-trigger)
fi
if [[ "${CRADLE_CHEST_PAD_ENABLE_ON_HOLD:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad-enable-on-hold)
fi
if [[ "${CRADLE_CHEST_PAD_ENABLE_ON_TERMINAL_HOLD:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad-enable-on-terminal-hold)
fi
if [[ "${CRADLE_CHEST_PAD_ENABLE_ON_FINAL_HOLD:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad-enable-on-final-hold)
fi
if [[ "${CRADLE_CHEST_PAD_ENABLE_ON_TARGET_WINDOW:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad-enable-on-target-window)
fi
if [[ "${CRADLE_CHEST_PAD_ENABLE_ON_BOX_TILT:-0}" == "1" ]]; then
  free_cradle_extra_args+=(--cradle-chest-pad-enable-on-box-tilt)
fi
free_cradle_extra_args+=(
  --cradle-side-rail-height "${CRADLE_SIDE_RAIL_HEIGHT:-0.07}"
  --cradle-end-stop-height "${CRADLE_END_STOP_HEIGHT:-0.08}"
  --cradle-rail-thickness "${CRADLE_RAIL_THICKNESS:-0.018}"
  --cradle-mass-scale "${CRADLE_MASS_SCALE:-0.40}"
  --cradle-top-lid-local-z "${CRADLE_TOP_LID_LOCAL_Z:-0.12}"
  --cradle-top-lid-thickness "${CRADLE_TOP_LID_THICKNESS:-0.018}"
  --cradle-top-lid-x-scale "${CRADLE_TOP_LID_X_SCALE:-1.0}"
  --cradle-top-lid-y-scale "${CRADLE_TOP_LID_Y_SCALE:-1.0}"
  --cradle-chest-pad-local-pos0 "${CRADLE_CHEST_PAD_LOCAL_X:-0.10}" "${CRADLE_CHEST_PAD_LOCAL_Y:-0.0}" "${CRADLE_CHEST_PAD_LOCAL_Z:-0.08}"
  --cradle-chest-pad-size "${CRADLE_CHEST_PAD_SIZE_X:-0.035}" "${CRADLE_CHEST_PAD_SIZE_Y:-0.34}" "${CRADLE_CHEST_PAD_SIZE_Z:-0.20}"
  --cradle-chest-pad-mass-scale "${CRADLE_CHEST_PAD_MASS_SCALE:-1.0}"
  --cradle-chest-pad-target-window-min-step "${CRADLE_CHEST_PAD_TARGET_WINDOW_MIN_STEP:--1}"
  --cradle-chest-pad-box-tilt-threshold "${CRADLE_CHEST_PAD_BOX_TILT_THRESHOLD:-999.0}"
  --cradle-chest-pad-box-tilt-min-step "${CRADLE_CHEST_PAD_BOX_TILT_MIN_STEP:--1}"
)

run_case() {
  local case_id="$1"
  local steps="$2"
  local check_kind="$3"
  local min_robot_travel="$4"
  local min_box_travel="$5"
  local max_tilt="$6"
  local max_final_rel="$7"
  shift 7
  local out="${SUITE_DIR}/${case_id}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[AGILE-SUITE] ${case_id} kind=${check_kind} steps=${steps} backend=${AGILE_POLICY_BACKEND}"

  local build_status=0
  set +e
  "${base_python[@]}" \
    --steps "${steps}" \
    --output-dir "${out}" \
    "$@" 2>&1 | tee "${build_log}"
  build_status=${PIPESTATUS[0]}
  set -e

  local check_status=99
  if [[ -f "${out}/core_world_g1_box_scene_summary.json" ]]; then
    local check_args=(
      python3 scripts/isaac/check_core_world_g1_box_scene_summary.py
      "${out}/core_world_g1_box_scene_summary.json"
      --min-steps "${steps}"
      --expect-gait-mode agile_policy
      --min-joint-count 40
      --max-fall-events 0
      --max-box-drop-events 0
      --min-robot-z 0.45
      --max-tilt "${max_tilt}"
      --min-final-robot-target-directed-travel "${min_robot_travel}"
      --max-root-pose-write-count-rollout 0
      --max-root-velocity-write-count-rollout 0
      --max-box-pose-write-count-rollout 0
      --require-diagnostic-claim
    )
    local case_max_box_tilt=""
    local case_max_robot_lateral=""
    local case_max_box_lateral=""
    local case_max_final_robot_lateral=""
    local case_max_final_box_lateral=""
    local case_max_final_robot_travel=""
    local case_max_final_box_travel=""
    if [[ "${check_kind}" == "nobox" ]]; then
      case_max_box_tilt="${NOBOX_MAX_BOX_TILT:-${MAX_BOX_TILT:-}}"
      case_max_robot_lateral="${NOBOX_MAX_ROBOT_LATERAL_ERROR:-${MAX_ROBOT_LATERAL_ERROR:-}}"
      case_max_box_lateral="${NOBOX_MAX_BOX_LATERAL_ERROR:-${MAX_BOX_LATERAL_ERROR:-}}"
      case_max_final_robot_lateral="${NOBOX_MAX_FINAL_ROBOT_LATERAL_ERROR:-${MAX_FINAL_ROBOT_LATERAL_ERROR:-}}"
      case_max_final_box_lateral="${NOBOX_MAX_FINAL_BOX_LATERAL_ERROR:-${MAX_FINAL_BOX_LATERAL_ERROR:-}}"
      case_max_final_robot_travel="${NOBOX_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-${MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-}}"
      case_max_final_box_travel="${NOBOX_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-${MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-}}"
    elif [[ "${check_kind}" == "fixed" ]]; then
      case_max_box_tilt="${FIXED_MAX_BOX_TILT:-${MAX_BOX_TILT:-}}"
      case_max_robot_lateral="${FIXED_MAX_ROBOT_LATERAL_ERROR:-${MAX_ROBOT_LATERAL_ERROR:-}}"
      case_max_box_lateral="${FIXED_MAX_BOX_LATERAL_ERROR:-${MAX_BOX_LATERAL_ERROR:-}}"
      case_max_final_robot_lateral="${FIXED_MAX_FINAL_ROBOT_LATERAL_ERROR:-${MAX_FINAL_ROBOT_LATERAL_ERROR:-}}"
      case_max_final_box_lateral="${FIXED_MAX_FINAL_BOX_LATERAL_ERROR:-${MAX_FINAL_BOX_LATERAL_ERROR:-}}"
      case_max_final_robot_travel="${FIXED_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-${MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-}}"
      case_max_final_box_travel="${FIXED_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-${MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-}}"
    elif [[ "${check_kind}" == "free_cradle" ]]; then
      case_max_box_tilt="${FREE_MAX_BOX_TILT:-${MAX_BOX_TILT:-}}"
      case_max_robot_lateral="${FREE_MAX_ROBOT_LATERAL_ERROR:-${MAX_ROBOT_LATERAL_ERROR:-}}"
      case_max_box_lateral="${FREE_MAX_BOX_LATERAL_ERROR:-${MAX_BOX_LATERAL_ERROR:-}}"
      case_max_final_robot_lateral="${FREE_MAX_FINAL_ROBOT_LATERAL_ERROR:-${MAX_FINAL_ROBOT_LATERAL_ERROR:-}}"
      case_max_final_box_lateral="${FREE_MAX_FINAL_BOX_LATERAL_ERROR:-${MAX_FINAL_BOX_LATERAL_ERROR:-}}"
      case_max_final_robot_travel="${FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-${MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-}}"
      case_max_final_box_travel="${FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-${MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-}}"
    fi
    if [[ -n "${case_max_box_tilt}" ]]; then
      check_args+=(--max-box-tilt "${case_max_box_tilt}")
    fi
    if [[ -n "${MAX_FINAL_HOLD_FALL_EVENTS:-}" ]]; then
      check_args+=(--max-final-hold-fall-events "${MAX_FINAL_HOLD_FALL_EVENTS}")
    fi
    if [[ -n "${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-}" ]]; then
      check_args+=(--max-final-hold-box-drop-events "${MAX_FINAL_HOLD_BOX_DROP_EVENTS}")
    fi
    if [[ -n "${MAX_FINAL_STAND_FALL_EVENTS:-}" ]]; then
      check_args+=(--max-final-stand-fall-events "${MAX_FINAL_STAND_FALL_EVENTS}")
    fi
    if [[ -n "${MAX_FINAL_STAND_BOX_DROP_EVENTS:-}" ]]; then
      check_args+=(--max-final-stand-box-drop-events "${MAX_FINAL_STAND_BOX_DROP_EVENTS}")
    fi
    if [[ -n "${case_max_robot_lateral}" ]]; then
      check_args+=(--max-robot-target-lateral-error "${case_max_robot_lateral}")
    fi
    if [[ -n "${case_max_box_lateral}" ]]; then
      check_args+=(--max-box-target-lateral-error "${case_max_box_lateral}")
    fi
    if [[ -n "${case_max_final_robot_lateral}" ]]; then
      check_args+=(--max-final-robot-target-lateral-error "${case_max_final_robot_lateral}")
    fi
    if [[ -n "${case_max_final_box_lateral}" ]]; then
      check_args+=(--max-final-box-target-lateral-error "${case_max_final_box_lateral}")
    fi
    if [[ -n "${case_max_final_robot_travel}" ]]; then
      check_args+=(--max-final-robot-target-directed-travel "${case_max_final_robot_travel}")
    fi
    if [[ -n "${case_max_final_box_travel}" ]]; then
      check_args+=(--max-final-box-target-directed-travel "${case_max_final_box_travel}")
    fi
    if [[ -n "${MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS:-}" ]]; then
      check_args+=(--min-agile-command-hold-final-active-steps "${MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS}")
    fi
    if [[ -n "${MIN_AGILE_COMMAND_HOLD_ACTIVE_STEPS:-}" ]]; then
      check_args+=(--min-agile-command-hold-active-steps "${MIN_AGILE_COMMAND_HOLD_ACTIVE_STEPS}")
    fi
    if [[ "${REQUIRE_AGILE_COMMAND_STOP_TARGET_WINDOW_LATCHED:-0}" == "1" ]]; then
      check_args+=(--require-agile-command-stop-target-window-latched)
    fi
    if [[ -n "${MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS:-}" ]]; then
      check_args+=(
        --min-agile-command-hold-final-stand-active-steps \
        "${MIN_AGILE_COMMAND_HOLD_FINAL_STAND_ACTIVE_STEPS}"
      )
    fi
    if [[ -n "${MAX_FINAL_HOLD_COMMAND_X:-}" ]]; then
      check_args+=(--max-final-hold-command-x "${MAX_FINAL_HOLD_COMMAND_X}")
    fi
    if [[ -n "${MAX_FINAL_HOLD_COMMAND_Y:-}" ]]; then
      check_args+=(--max-final-hold-command-y "${MAX_FINAL_HOLD_COMMAND_Y}")
    fi
    if [[ -n "${MAX_FINAL_HOLD_COMMAND_YAW:-}" ]]; then
      check_args+=(--max-final-hold-command-yaw "${MAX_FINAL_HOLD_COMMAND_YAW}")
    fi
    if [[ -n "${MIN_FINAL_HOLD_ROBOT_Z:-}" ]]; then
      check_args+=(--min-final-hold-robot-z "${MIN_FINAL_HOLD_ROBOT_Z}")
    fi
    if [[ -n "${MIN_FINAL_HOLD_BOX_Z:-}" ]]; then
      check_args+=(--min-final-hold-box-z "${MIN_FINAL_HOLD_BOX_Z}")
    fi
    if [[ -n "${MAX_FINAL_HOLD_TILT:-}" ]]; then
      check_args+=(--max-final-hold-tilt "${MAX_FINAL_HOLD_TILT}")
    fi
    if [[ -n "${MAX_FINAL_HOLD_BOX_TILT:-}" ]]; then
      check_args+=(--max-final-hold-box-tilt "${MAX_FINAL_HOLD_BOX_TILT}")
    fi
    if [[ -n "${MIN_FINAL_STAND_ROBOT_Z:-}" ]]; then
      check_args+=(--min-final-stand-robot-z "${MIN_FINAL_STAND_ROBOT_Z}")
    fi
    if [[ -n "${MIN_FINAL_STAND_BOX_Z:-}" ]]; then
      check_args+=(--min-final-stand-box-z "${MIN_FINAL_STAND_BOX_Z}")
    fi
    if [[ -n "${MAX_FINAL_STAND_TILT:-}" ]]; then
      check_args+=(--max-final-stand-tilt "${MAX_FINAL_STAND_TILT}")
    fi
    if [[ -n "${MAX_FINAL_STAND_BOX_TILT:-}" ]]; then
      check_args+=(--max-final-stand-box-tilt "${MAX_FINAL_STAND_BOX_TILT}")
    fi
    if [[ -n "${MIN_TARGET_WINDOW_ROBOT_STABLE_STEPS:-}" ]]; then
      check_args+=(--min-target-window-robot-stable-steps "${MIN_TARGET_WINDOW_ROBOT_STABLE_STEPS}")
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOX_STABLE_STEPS:-}" ]]; then
      check_args+=(--min-target-window-box-stable-steps "${MIN_TARGET_WINDOW_BOX_STABLE_STEPS}")
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-}" ]]; then
      check_args+=(--min-target-window-both-stable-steps "${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS}")
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-longest-streak-steps \
        "${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-streak-at-end-steps \
        "${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-final-hold-stable-steps \
        "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-final-hold-longest-streak-steps \
        "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-final-hold-streak-at-end-steps \
        "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STABLE_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-final-stand-stable-steps \
        "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STABLE_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_LONGEST_STREAK_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-final-stand-longest-streak-steps \
        "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_LONGEST_STREAK_STEPS}"
      )
    fi
    if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STREAK_AT_END_STEPS:-}" ]]; then
      check_args+=(
        --min-target-window-both-final-stand-streak-at-end-steps \
        "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STREAK_AT_END_STEPS}"
      )
    fi
    if [[ "${check_kind}" == "nobox" ]]; then
      check_args+=(--expect-carry-box-spawned false)
    elif [[ "${check_kind}" == "fixed" ]]; then
      local fixed_collision_expected="true"
      if [[ "${FIXED_BOX_COLLISION_ENABLED:-1}" == "0" ]]; then
        fixed_collision_expected="false"
      fi
      check_args+=(
        --expect-carry-box-spawned true
        --expect-attach-box fixed_torso
        --expect-box-collision-enabled "${fixed_collision_expected}"
        --min-box-z 0.20
        --min-final-box-target-directed-travel "${min_box_travel}"
        --max-final-box-robot-relative-offset-error "${max_final_rel}"
      )
    elif [[ "${check_kind}" == "free_cradle" ]]; then
      check_args+=(
        --expect-carry-box-spawned true
        --expect-attach-box none
        --expect-torso-cradle front_tray
        --expect-box-collision-enabled true
        --expect-cradle-collision-enabled true
        --min-cradle-piece-count 5
        --min-box-z 0.20
        --min-final-box-target-directed-travel "${min_box_travel}"
        --max-final-box-robot-relative-offset-error "${max_final_rel}"
      )
    else
      echo "Unknown check kind: ${check_kind}" >&2
      exit 2
    fi

    set +e
    "${check_args[@]}" > "${check_log}"
    check_status=$?
    set -e
    cat "${check_log}"
  else
    echo "{\"status\":\"fail\",\"failures\":[\"summary missing\"]}" > "${check_log}"
  fi

  printf "%s\t%s\t%s\t%s\n" "${case_id}" "${build_status}" "${check_status}" "${out}" >> "${status_file}"
  if [[ "${STRICT}" == "1" && ( "${build_status}" != "0" || "${check_status}" != "0" ) ]]; then
    echo "[AGILE-SUITE] strict mode stopping after ${case_id}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

if [[ "${RUN_NOBOX:-1}" == "1" ]]; then
  run_case agile_nobox_walk "${NOBOX_STEPS:-420}" nobox "${NOBOX_MIN_ROBOT_TRAVEL:-0.05}" 0.00 "${NOBOX_MAX_TILT:-0.85}" 0.00 \
    --disable-carry-box-spawn
fi

if [[ "${RUN_FIXED:-1}" == "1" ]]; then
  fixed_extra_args=()
  if [[ "${FIXED_BOX_COLLISION_ENABLED:-1}" == "0" ]]; then
    fixed_extra_args+=(--disable-box-collision)
  fi
  run_case agile_fixed_payload_walk "${FIXED_STEPS:-420}" fixed "${FIXED_MIN_ROBOT_TRAVEL:-0.04}" "${FIXED_MIN_BOX_TRAVEL:-0.04}" "${FIXED_MAX_TILT:-0.90}" "${FIXED_MAX_FINAL_REL:-0.08}" \
    --box-mass "${FIXED_BOX_MASS:-0.25}" \
    --box-size "${FIXED_BOX_SIZE_X:-0.10}" "${FIXED_BOX_SIZE_Y:-0.08}" "${FIXED_BOX_SIZE_Z:-0.06}" \
    --box-position "${FIXED_BOX_POS_X:-0.0}" "${FIXED_BOX_POS_Y:-0.0}" "${FIXED_BOX_POS_Z:-0.88}" \
    --attach-box fixed_torso \
    --attach-body-path /World/G1/torso_link \
    --attach-local-pos0 "${FIXED_ATTACH_LOCAL_X:-0.0}" "${FIXED_ATTACH_LOCAL_Y:-0.0}" "${FIXED_ATTACH_LOCAL_Z:-0.08}" \
    --torso-cradle none \
    "${fixed_extra_args[@]}"
fi

if [[ "${RUN_FREE:-1}" == "1" ]]; then
  run_case agile_low_cradle_freebox_walk "${FREE_STEPS:-520}" free_cradle "${FREE_MIN_ROBOT_TRAVEL:-0.04}" "${FREE_MIN_BOX_TRAVEL:-0.04}" "${FREE_MAX_TILT:-0.95}" "${FREE_MAX_FINAL_REL:-0.20}" \
    --box-mass "${FREE_BOX_MASS:-0.25}" \
    --box-size "${FREE_BOX_SIZE_X:-0.10}" "${FREE_BOX_SIZE_Y:-0.08}" "${FREE_BOX_SIZE_Z:-0.06}" \
    --box-position "${FREE_BOX_POS_X:-0.34}" "${FREE_BOX_POS_Y:-0.0}" "${FREE_BOX_POS_Z:-0.90}" \
    --attach-box none \
    --torso-cradle front_tray \
    --require-box-no-drop \
    --cradle-deck-size 0.24 0.26 0.025 \
    --cradle-deck-local-pos0 "${FREE_CRADLE_LOCAL_X:-0.34}" "${FREE_CRADLE_LOCAL_Y:-0.0}" "${FREE_CRADLE_LOCAL_Z:-0.05}" \
    "${free_cradle_extra_args[@]}"
fi

if [[ "${GENERATE_FINAL_HOLD_COMPARISON:-0}" == "1" ]]; then
  comparison_json="${SUITE_DIR}/g1_final_hold_comparison.json"
  comparison_md="${SUITE_DIR}/g1_final_hold_comparison.md"
  comparison_args=(
    scripts/isaac/summarize_core_world_g1_final_hold_comparison.py
    --json-output "${comparison_json}"
    --markdown-output "${comparison_md}"
  )
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-}" ]]; then
    comparison_args+=(--min-target-window-both-stable-steps "${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_COMMAND_X:-}" ]]; then
    comparison_args+=(--max-final-hold-command-x "${MAX_FINAL_HOLD_COMMAND_X}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_COMMAND_Y:-}" ]]; then
    comparison_args+=(--max-final-hold-command-y "${MAX_FINAL_HOLD_COMMAND_Y}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_COMMAND_YAW:-}" ]]; then
    comparison_args+=(--max-final-hold-command-yaw "${MAX_FINAL_HOLD_COMMAND_YAW}")
  fi
  if [[ -n "${MIN_FINAL_HOLD_ROBOT_Z:-}" ]]; then
    comparison_args+=(--min-final-hold-robot-z "${MIN_FINAL_HOLD_ROBOT_Z}")
  fi
  if [[ -n "${MIN_FINAL_HOLD_BOX_Z:-}" ]]; then
    comparison_args+=(--min-final-hold-box-z "${MIN_FINAL_HOLD_BOX_Z}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_TILT:-}" ]]; then
    comparison_args+=(--max-final-hold-tilt "${MAX_FINAL_HOLD_TILT}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_BOX_TILT:-}" ]]; then
    comparison_args+=(--max-final-hold-box-tilt "${MAX_FINAL_HOLD_BOX_TILT}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_FALL_EVENTS:-}" ]]; then
    comparison_args+=(--max-final-hold-fall-events "${MAX_FINAL_HOLD_FALL_EVENTS}")
  fi
  if [[ -n "${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-}" ]]; then
    comparison_args+=(--max-final-hold-box-drop-events "${MAX_FINAL_HOLD_BOX_DROP_EVENTS}")
  fi
  if [[ -n "${MIN_FINAL_STAND_ROBOT_Z:-}" ]]; then
    comparison_args+=(--min-final-stand-robot-z "${MIN_FINAL_STAND_ROBOT_Z}")
  fi
  if [[ -n "${MIN_FINAL_STAND_BOX_Z:-}" ]]; then
    comparison_args+=(--min-final-stand-box-z "${MIN_FINAL_STAND_BOX_Z}")
  fi
  if [[ -n "${MAX_FINAL_STAND_TILT:-}" ]]; then
    comparison_args+=(--max-final-stand-tilt "${MAX_FINAL_STAND_TILT}")
  fi
  if [[ -n "${MAX_FINAL_STAND_BOX_TILT:-}" ]]; then
    comparison_args+=(--max-final-stand-box-tilt "${MAX_FINAL_STAND_BOX_TILT}")
  fi
  if [[ -n "${MAX_FINAL_STAND_FALL_EVENTS:-}" ]]; then
    comparison_args+=(--max-final-stand-fall-events "${MAX_FINAL_STAND_FALL_EVENTS}")
  fi
  if [[ -n "${MAX_FINAL_STAND_BOX_DROP_EVENTS:-}" ]]; then
    comparison_args+=(--max-final-stand-box-drop-events "${MAX_FINAL_STAND_BOX_DROP_EVENTS}")
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-longest-streak-steps \
      "${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-streak-at-end-steps \
      "${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-final-hold-stable-steps \
      "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-final-hold-longest-streak-steps \
      "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-final-hold-streak-at-end-steps \
      "${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STABLE_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-final-stand-stable-steps \
      "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STABLE_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_LONGEST_STREAK_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-final-stand-longest-streak-steps \
      "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_LONGEST_STREAK_STEPS}"
    )
  fi
  if [[ -n "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STREAK_AT_END_STEPS:-}" ]]; then
    comparison_args+=(
      --min-target-window-both-final-stand-streak-at-end-steps \
      "${MIN_TARGET_WINDOW_BOTH_FINAL_STAND_STREAK_AT_END_STEPS}"
    )
  fi
  set +e
  python3 "${comparison_args[@]}" > "${SUITE_DIR}/g1_final_hold_comparison.stdout.md"
  comparison_status=$?
  set -e
  echo "[AGILE-SUITE] final-hold comparison status=${comparison_status} json=${comparison_json} md=${comparison_md}"
  if [[ "${STRICT}" == "1" && "${comparison_status}" != "0" ]]; then
    exit "${comparison_status}"
  fi
fi

echo "[AGILE-SUITE] status file: ${status_file}"
