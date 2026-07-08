#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 direct-carry posture suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_direct_carry_postures)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_direct_carry_posture_suite/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
STEPS="${STEPS:-420}"
MIN_FINAL_BOX_TARGET_TRAVEL="${MIN_FINAL_BOX_TARGET_TRAVEL:-0.08}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/posture_suite_status.tsv"
printf "posture\tbuild_status\tcheck_status\toutput_dir\n" > "${status_file}"

base_python=(
  "${ISAAC_VENV}/bin/python"
  scripts/isaac/build_core_world_g1_box_scene.py
  --viz none
  --experience "${EXPERIENCE}"
  --device "${DEVICE}"
  --kit_args "${KIT_ARGS}"
  --steps "${STEPS}"
  --g1-usd "${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
  --g1-root-position 0.0 0.0 0.78
  --g1-root-orientation-wxyz 1.0 0.0 0.0 0.0
  --stand-hip-pitch -0.12
  --stand-knee 0.30
  --stand-ankle-pitch -0.15
  --apply-arena-stand-gains
  --stand-drive-preset arena
  --stand-gain-scale 1.0
  --stand-force-scale 1.0
  --box-mass "${BOX_MASS:-0.25}"
  --box-size "${BOX_SIZE_X:-0.10}" "${BOX_SIZE_Y:-0.08}" "${BOX_SIZE_Z:-0.06}"
  --attach-box none
  --torso-cradle front_tray
  --require-box-no-drop
  --gait-mode staged_march
  --gait-frequency-hz 0.7
  --gait-ramp-down-start-step "${GAIT_RAMP_DOWN_START_STEP:-90}"
  --gait-ramp-down-end-step "${GAIT_RAMP_DOWN_END_STEP:-190}"
  --gait-min-amplitude-scale 0.0
  --recovery-pitch-threshold 0.10
  --recovery-pitch-rate-threshold 999.0
  --recovery-hip-pitch-offset -0.10
  --recovery-knee-offset 0.12
  --recovery-ankle-pitch-offset 0.12
  --recovery-waist-pitch-offset -0.08
)

run_posture() {
  local posture="$1"
  local box_x="$2"
  local box_z="$3"
  local deck_x="$4"
  local deck_z="$5"
  local deck_len="$6"
  local deck_width="$7"
  local deck_height="$8"
  local side_rail_height="$9"
  local end_stop_height="${10}"
  local rail_thickness="${11}"
  local cradle_mass_scale="${12}"
  local gait_amp="${13}"
  local out="${SUITE_DIR}/${posture}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[POSTURE] ${posture} box=(${box_x},0,${box_z}) deck=(${deck_x},0,${deck_z}) mass_scale=${cradle_mass_scale} amp=${gait_amp}"

  local build_status=0
  set +e
  "${base_python[@]}" \
    --box-position "${box_x}" 0.0 "${box_z}" \
    --cradle-deck-size "${deck_len}" "${deck_width}" "${deck_height}" \
    --cradle-deck-local-pos0 "${deck_x}" 0.0 "${deck_z}" \
    --cradle-side-rail-height "${side_rail_height}" \
    --cradle-end-stop-height "${end_stop_height}" \
    --cradle-rail-thickness "${rail_thickness}" \
    --cradle-mass-scale "${cradle_mass_scale}" \
    --gait-amplitude "${gait_amp}" \
    --output-dir "${out}" 2>&1 | tee "${build_log}"
  build_status=${PIPESTATUS[0]}
  set -e

  local check_status=99
  if [[ -f "${out}/core_world_g1_box_scene_summary.json" ]]; then
    set +e
    python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
      "${out}/core_world_g1_box_scene_summary.json" \
      --min-steps "${STEPS}" \
      --expect-carry-box-spawned true \
      --expect-attach-box none \
      --expect-torso-cradle front_tray \
      --expect-gait-mode staged_march \
      --expect-box-collision-enabled true \
      --expect-cradle-collision-enabled true \
      --min-cradle-piece-count 5 \
      --min-joint-count 40 \
      --max-fall-events 0 \
      --max-box-drop-events 0 \
      --min-robot-z 0.45 \
      --min-box-z 0.20 \
      --max-tilt 0.85 \
      --max-box-robot-relative-offset-error "${MAX_BOX_ROBOT_REL_ERROR:-0.75}" \
      --min-final-box-target-directed-travel "${MIN_FINAL_BOX_TARGET_TRAVEL}" \
      --max-root-pose-write-count-rollout 0 \
      --max-root-velocity-write-count-rollout 0 \
      --max-box-pose-write-count-rollout 0 \
      --require-diagnostic-claim > "${check_log}"
    check_status=$?
    set -e
    cat "${check_log}"
  else
    echo "{\"status\":\"fail\",\"failures\":[\"summary missing\"]}" > "${check_log}"
  fi

  printf "%s\t%s\t%s\t%s\n" "${posture}" "${build_status}" "${check_status}" "${out}" >> "${status_file}"
  if [[ "${STRICT}" == "1" && ( "${build_status}" != "0" || "${check_status}" != "0" ) ]]; then
    echo "[POSTURE] strict mode stopping after ${posture}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

run_posture front_mid 0.44 0.95 0.44 0.10 0.24 0.26 0.025 0.07 0.08 0.018 0.95 0.10
run_posture close_chest 0.36 0.98 0.36 0.13 0.22 0.26 0.025 0.08 0.09 0.018 0.85 0.09
run_posture low_front 0.46 0.90 0.46 0.05 0.24 0.26 0.025 0.07 0.08 0.018 0.75 0.08
run_posture extended_front 0.52 0.95 0.52 0.10 0.28 0.28 0.025 0.07 0.08 0.018 0.75 0.08

echo "[POSTURE] status file: ${status_file}"
