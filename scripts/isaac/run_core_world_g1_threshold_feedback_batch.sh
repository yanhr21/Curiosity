#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run threshold-feedback Isaac batch on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-35}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

for spec in \
  diag63:0.20:0.35:0.00 \
  diag64:0.30:0.35:0.00 \
  diag65:0.20:0.70:0.03 \
  diag66:0.30:0.70:0.03
do
  diag="${spec%%:*}"
  rest="${spec#*:}"
  pitch_threshold="${rest%%:*}"
  rest="${rest#*:}"
  pitch_gain="${rest%%:*}"
  pitch_rate_gain="${rest##*:}"
  stamp="20260705_core_world_g1_min_cradle_amp016_mass1_threshold_${diag}_pitch${pitch_threshold}_gain${pitch_gain}_direct_retry2"
  output_dir="experiments/outputs/core_world_g1_box_scene/${stamp}"
  echo "[BATCH] Running ${diag} pitch_threshold=${pitch_threshold} gain=${pitch_gain} rate_gain=${pitch_rate_gain}"
  "${ISAAC_VENV}/bin/python" scripts/isaac/build_core_world_g1_box_scene.py \
    --viz none \
    --experience "${EXPERIENCE}" \
    --device "${DEVICE:-cpu}" \
    --kit_args "${KIT_ARGS}" \
    --steps 420 \
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
    --gait-mode open_loop_march \
    --gait-amplitude 0.16 \
    --gait-frequency-hz 0.7 \
    --balance-feedback-controller \
    --balance-start-step 0 \
    --balance-pitch-sign 1.0 \
    --balance-pitch-gain "${pitch_gain}" \
    --balance-pitch-rate-gain "${pitch_rate_gain}" \
    --balance-adjustment-limit 0.25 \
    --balance-pitch-activation-threshold "${pitch_threshold}" \
    --balance-roll-activation-threshold 99.0 \
    --balance-pitch-rate-activation-threshold 99.0 \
    --balance-roll-rate-activation-threshold 99.0 \
    --attach-box none \
    --torso-cradle front_tray \
    --require-box-no-drop \
    --cradle-deck-size 0.24 0.26 0.025 \
    --cradle-deck-local-pos0 0.44 0.0 0.10 \
    --cradle-side-rail-height 0.07 \
    --cradle-end-stop-height 0.08 \
    --cradle-rail-thickness 0.018 \
    --cradle-mass-scale 1.0 \
    --output-dir "${output_dir}"

  python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
    "${output_dir}/core_world_g1_box_scene_summary.json" \
    --min-steps 420 \
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
done
