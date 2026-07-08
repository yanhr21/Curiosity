#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 pulsed-creep diagnostics on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py
bash -n scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh

run_diag() {
  local stamp="$1"
  local pitch_target="$2"
  local pulse_period="$3"
  local pulse_width="$4"
  local gait_amp="$5"
  local stance_push="$6"
  local balance_gain="$7"
  local balance_limit="$8"

  echo "[BATCH] ${stamp} pitch_target=${pitch_target} pulse=${pulse_width}/${pulse_period} amp=${gait_amp} push=${stance_push} gain=${balance_gain} limit=${balance_limit}"
  STAMP="${stamp}" \
  STEPS=620 \
  BOX_MASS=0.5 \
  BOX_SIZE_X=0.40 \
  BOX_SIZE_Y=0.26 \
  BOX_SIZE_Z=0.24 \
  BOX_POS_X=0.25 \
  BOX_POS_Y=0.0 \
  BOX_POS_Z=0.82 \
  DROP_Z=0.20 \
  PROBE_MODE=none \
  GRASP_MODE=none \
  TORSO_CRADLE=front_tray \
  CRADLE_DECK_SIZE_X=0.46 \
  CRADLE_DECK_SIZE_Y=0.36 \
  CRADLE_DECK_SIZE_Z=0.025 \
  CRADLE_DECK_LOCAL_POS0_X=0.25 \
  CRADLE_DECK_LOCAL_POS0_Y=0.0 \
  CRADLE_DECK_LOCAL_POS0_Z=-0.16 \
  CRADLE_SIDE_RAIL_HEIGHT=0.12 \
  CRADLE_END_STOP_HEIGHT=0.14 \
  CRADLE_RAIL_THICKNESS=0.025 \
  CRADLE_MASS_SCALE=0.15 \
  GAIT_MODE=targeted_creep \
  GAIT_AMPLITUDE="${gait_amp}" \
  GAIT_FREQUENCY_HZ=0.7 \
  GAIT_START_STEP=140 \
  CREEP_HIP_PITCH_OFFSET=0.08 \
  CREEP_KNEE_OFFSET=0.03 \
  CREEP_ANKLE_PITCH_OFFSET=-0.04 \
  CREEP_WAIST_PITCH_OFFSET=0.02 \
  CREEP_STANCE_PUSH_SCALE="${stance_push}" \
  CREEP_LIFT_SCALE=0.35 \
  CREEP_ANKLE_LIFT_SCALE=-0.18 \
  BALANCE_FEEDBACK_CONTROLLER=1 \
  BALANCE_PITCH_GAIN="${balance_gain}" \
  BALANCE_PITCH_RATE_GAIN=0.02 \
  BALANCE_ADJUSTMENT_LIMIT="${balance_limit}" \
  BALANCE_PITCH_SIGN=1.0 \
  BALANCE_PITCH_TARGET="${pitch_target}" \
  BALANCE_TARGET_START_STEP=140 \
  BALANCE_TARGET_END_STEP=560 \
  BALANCE_TARGET_PULSE_PERIOD_STEPS="${pulse_period}" \
  BALANCE_TARGET_PULSE_WIDTH_STEPS="${pulse_width}" \
  BALANCE_TARGET_PULSE_PHASE_STEP=140 \
  STAND_HIP_PITCH=-0.12 \
  STAND_KNEE=0.30 \
  STAND_ANKLE_PITCH=-0.15 \
  bash scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh

  python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
    "experiments/outputs/core_world_g1_box_scene/${stamp}/core_world_g1_box_scene_summary.json" \
    --min-steps 620 \
    --expect-attach-box none \
    --expect-torso-cradle front_tray \
    --expect-probe-mode none \
    --expect-grasp-mode none \
    --expect-carry-box-spawned true \
    --min-cradle-piece-count 5 \
    --min-joint-count 40 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --min-robot-z 0.70 \
    --min-box-z 0.70 \
    --max-tilt 0.25 \
    --max-root-pose-write-count-rollout 0 \
    --max-root-velocity-write-count-rollout 0 \
    --max-box-pose-write-count-rollout 0 \
    --min-balance-target-active-steps 80 \
    --min-max-box-target-directed-travel 0.08 \
    --require-diagnostic-claim || true
}

run_diag 20260706_core_world_g1_front_tray_freebox_pulsecreep_retry10a 0.020 120 40 0.045 0.10 0.45 0.12
run_diag 20260706_core_world_g1_front_tray_freebox_pulsecreep_retry10b 0.015 100 50 0.060 0.14 0.40 0.10
run_diag 20260706_core_world_g1_front_tray_freebox_pulsecreep_retry10c 0.012 90 45 0.070 0.18 0.35 0.08
