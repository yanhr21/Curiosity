#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run terminal-drive validation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-30}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

run_diag() {
  local diag="$1"
  local trigger_mode="$2"
  local trigger_value="$3"
  local terminal_gain="$4"
  local terminal_force="$5"
  local stamp="20260705_core_world_g1_terminal_drive_${diag}_${trigger_mode}${trigger_value}_tg${terminal_gain}_tf${terminal_force}"
  local output_dir="experiments/outputs/core_world_g1_box_scene/${stamp}"
  local terminal_args=()
  case "${trigger_mode}" in
    pitch)
      terminal_args=(--terminal-hold-pitch-threshold "${trigger_value}")
      ;;
    travel)
      terminal_args=(--terminal-hold-box-target-travel "${trigger_value}")
      ;;
    step)
      terminal_args=(--terminal-hold-start-step "${trigger_value}")
      ;;
    *)
      echo "Unknown trigger mode: ${trigger_mode}" >&2
      exit 2
      ;;
  esac
  echo "[BATCH] ${diag} trigger=${trigger_mode}:${trigger_value} terminal_drive=${terminal_gain}/${terminal_force}"
  "${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_g1_box_scene.py \
    --viz none \
    --experience "${EXPERIENCE}" \
    --device "${DEVICE:-cpu}" \
    --kit_args "${KIT_ARGS}" \
    --steps 700 \
    --g1-usd "${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}" \
    --box-mass 0.25 \
    --box-size 0.10 0.08 0.06 \
    --box-position 0.44 0.0 0.95 \
    --g1-root-position 0.0 0.0 0.78 \
    --g1-root-orientation-wxyz 1.0 0.0 0.0 0.0 \
    --stand-hip-pitch -0.12 \
    --stand-knee 0.30 \
    --stand-ankle-pitch -0.15 \
    --apply-arena-stand-gains \
    --stand-drive-preset arena \
    --stand-gain-scale 1.0 \
    --stand-force-scale 1.0 \
    --gait-mode staged_march \
    --gait-amplitude 0.11 \
    --gait-frequency-hz 0.7 \
    --gait-ramp-down-start-step 100 \
    --gait-ramp-down-end-step 210 \
    --gait-min-amplitude-scale 0.0 \
    --recovery-pitch-threshold 999.0 \
    --recovery-pitch-rate-threshold 999.0 \
    "${terminal_args[@]}" \
    --terminal-hold-hip-pitch-offset 0.12 \
    --terminal-hold-knee-offset -0.10 \
    --terminal-hold-ankle-pitch-offset -0.12 \
    --terminal-hold-waist-pitch-offset 0.10 \
    --terminal-drive-gain-scale "${terminal_gain}" \
    --terminal-drive-force-scale "${terminal_force}" \
    --attach-box none \
    --torso-cradle front_tray \
    --require-box-no-drop \
    --cradle-deck-size 0.24 0.26 0.025 \
    --cradle-deck-local-pos0 0.44 0.0 0.10 \
    --cradle-side-rail-height 0.07 \
    --cradle-end-stop-height 0.08 \
    --cradle-rail-thickness 0.018 \
    --cradle-mass-scale 0.90 \
    --output-dir "${output_dir}"

  python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
    "${output_dir}/core_world_g1_box_scene_summary.json" \
    --min-steps 700 \
    --expect-attach-box none \
    --expect-torso-cradle front_tray \
    --expect-carry-box-spawned true \
    --min-cradle-piece-count 5 \
    --min-joint-count 40 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --min-robot-z 0.45 \
    --min-box-z 0.20 \
    --max-tilt 0.85 \
    --max-root-pose-write-count-rollout 0 \
    --max-root-velocity-write-count-rollout 0 \
    --max-box-pose-write-count-rollout 0 \
    --min-final-box-target-directed-travel 0.10 \
    --require-diagnostic-claim || true
}

run_diag diag93 pitch 0.20 2.0 2.0
run_diag diag94 travel 0.18 2.0 2.0
run_diag diag95 step 320 2.0 2.0
run_diag diag96 pitch 0.20 2.0 4.0
