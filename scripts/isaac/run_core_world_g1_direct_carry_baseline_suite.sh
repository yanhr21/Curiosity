#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 direct-carry baseline suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_direct_carry_baseline)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_direct_carry_baseline_suite/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/suite_status.tsv"
printf "stage\tstamp\tbuild_status\tcheck_status\toutput_dir\n" > "${status_file}"

base_python=(
  "${ISAAC_VENV}/bin/python"
  scripts/isaac/build_core_world_g1_box_scene.py
  --viz none
  --experience "${EXPERIENCE}"
  --device "${DEVICE}"
  --kit_args "${KIT_ARGS}"
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
)

run_case() {
  local stage="$1"
  shift
  local stamp="${SUITE_STAMP}_${stage}"
  local out="${SUITE_DIR}/${stage}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[SUITE] ${stage} output=${out}"

  local build_status=0
  set +e
  "${base_python[@]}" "$@" --output-dir "${out}" 2>&1 | tee "${build_log}"
  build_status=${PIPESTATUS[0]}
  set -e

  local check_status=99
  if [[ -f "${out}/core_world_g1_box_scene_summary.json" ]]; then
    set +e
    python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
      "${out}/core_world_g1_box_scene_summary.json" \
      "${CHECK_ARGS[@]}" > "${check_log}"
    check_status=$?
    set -e
    cat "${check_log}"
  else
    echo "{\"status\":\"fail\",\"failures\":[\"summary missing\"]}" > "${check_log}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "${stage}" "${stamp}" "${build_status}" "${check_status}" "${out}" >> "${status_file}"
  if [[ "${STRICT}" == "1" && ( "${build_status}" != "0" || "${check_status}" != "0" ) ]]; then
    echo "[SUITE] strict mode stopping after ${stage}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

CHECK_ARGS=(
  --min-steps 360
  --expect-carry-box-spawned false
  --expect-gait-mode stand
  --min-joint-count 40
  --require-stand-drive-gains
  --min-stand-drive-gain-count 20
  --max-fall-events 0
  --max-box-drop-events 0
  --min-robot-z 0.70
  --max-tilt 0.05
  --max-root-pose-write-count-rollout 0
  --max-root-velocity-write-count-rollout 0
  --max-box-pose-write-count-rollout 0
  --require-diagnostic-claim
)
run_case stage0_nobox_stand \
  --steps 360 \
  --disable-carry-box-spawn \
  --gait-mode stand

CHECK_ARGS=(
  --min-steps 360
  --expect-carry-box-spawned true
  --expect-attach-box fixed_torso
  --expect-gait-mode stand
  --expect-box-collision-enabled true
  --min-joint-count 40
  --max-fall-events 0
  --max-box-drop-events 0
  --min-robot-z 0.70
  --min-box-z 0.70
  --max-tilt 0.08
  --max-box-robot-relative-offset-error 0.02
  --max-final-box-robot-relative-offset-error 0.02
  --max-root-pose-write-count-rollout 0
  --max-root-velocity-write-count-rollout 0
  --max-box-pose-write-count-rollout 0
  --require-diagnostic-claim
)
run_case stage1_fixed_payload_stand \
  --steps 360 \
  --box-mass "${FIXED_PAYLOAD_MASS:-2.0}" \
  --box-size 0.22 0.16 0.12 \
  --box-position 0.24 0.0 0.88 \
  --attach-box fixed_torso \
  --attach-local-pos0 0.24 0.0 0.08 \
  --gait-mode stand

CHECK_ARGS=(
  --min-steps 360
  --expect-carry-box-spawned true
  --expect-attach-box none
  --expect-torso-cradle front_tray
  --expect-gait-mode stand
  --expect-box-collision-enabled true
  --expect-cradle-collision-enabled true
  --min-cradle-piece-count 5
  --min-joint-count 40
  --max-fall-events 0
  --max-box-drop-events 0
  --min-robot-z 0.70
  --min-box-z 0.90
  --max-tilt 0.08
  --max-box-robot-relative-offset-error 0.08
  --max-final-box-robot-relative-offset-error 0.05
  --max-root-pose-write-count-rollout 0
  --max-root-velocity-write-count-rollout 0
  --max-box-pose-write-count-rollout 0
  --require-diagnostic-claim
)
run_case stage2_freebox_cradle_stand \
  --steps 360 \
  --box-mass "${FREEBOX_MASS:-0.25}" \
  --box-size 0.10 0.08 0.06 \
  --box-position "${STAND_FREEBOX_POS_X:-0.34}" 0.0 "${STAND_FREEBOX_POS_Z:-0.90}" \
  --attach-box none \
  --torso-cradle front_tray \
  --require-box-no-drop \
  --cradle-deck-size 0.24 0.26 0.025 \
  --cradle-deck-local-pos0 "${STAND_CRADLE_LOCAL_X:-0.34}" 0.0 "${STAND_CRADLE_LOCAL_Z:-0.05}" \
  --cradle-side-rail-height 0.07 \
  --cradle-end-stop-height 0.08 \
  --cradle-rail-thickness 0.018 \
  --cradle-mass-scale "${STAND_CRADLE_MASS_SCALE:-0.40}" \
  --gait-mode stand

CHECK_ARGS=(
  --min-steps 420
  --expect-carry-box-spawned true
  --expect-attach-box none
  --expect-torso-cradle front_tray
  --expect-gait-mode staged_march
  --expect-box-collision-enabled true
  --expect-cradle-collision-enabled true
  --min-cradle-piece-count 5
  --min-joint-count 40
  --max-fall-events 0
  --max-box-drop-events 0
  --min-robot-z 0.45
  --min-box-z 0.20
  --max-tilt 0.85
  --max-box-robot-relative-offset-error 0.70
  --min-final-box-target-directed-travel "${SHORT_CARRY_MIN_FINAL_BOX_TARGET_TRAVEL:-0.10}"
  --max-root-pose-write-count-rollout 0
  --max-root-velocity-write-count-rollout 0
  --max-box-pose-write-count-rollout 0
  --require-diagnostic-claim
)
run_case stage3_freebox_short_carry \
  --steps 420 \
  --box-mass "${FREEBOX_MASS:-0.25}" \
  --box-size 0.10 0.08 0.06 \
  --box-position "${SHORT_CARRY_FREEBOX_POS_X:-0.36}" 0.0 "${SHORT_CARRY_FREEBOX_POS_Z:-0.95}" \
  --attach-box none \
  --torso-cradle front_tray \
  --require-box-no-drop \
  --cradle-deck-size 0.24 0.26 0.025 \
  --cradle-deck-local-pos0 "${SHORT_CARRY_CRADLE_LOCAL_X:-0.36}" 0.0 "${SHORT_CARRY_CRADLE_LOCAL_Z:-0.10}" \
  --cradle-side-rail-height 0.07 \
  --cradle-end-stop-height 0.08 \
  --cradle-rail-thickness 0.018 \
  --cradle-mass-scale "${SHORT_CARRY_CRADLE_MASS_SCALE:-0.50}" \
  --gait-mode staged_march \
  --gait-amplitude "${SHORT_CARRY_GAIT_AMPLITUDE:-0.10}" \
  --gait-frequency-hz 0.7 \
  --gait-ramp-down-start-step "${SHORT_CARRY_RAMP_START:-90}" \
  --gait-ramp-down-end-step "${SHORT_CARRY_RAMP_END:-190}" \
  --gait-min-amplitude-scale 0.0 \
  --recovery-pitch-threshold 0.10 \
  --recovery-pitch-rate-threshold 999.0 \
  --recovery-hip-pitch-offset -0.10 \
  --recovery-knee-offset 0.12 \
  --recovery-ankle-pitch-offset 0.12 \
  --recovery-waist-pitch-offset -0.08

CHECK_ARGS=(
  --min-steps 700
  --expect-carry-box-spawned true
  --expect-attach-box none
  --expect-torso-cradle front_tray
  --expect-gait-mode staged_march
  --expect-box-collision-enabled true
  --expect-cradle-collision-enabled true
  --min-cradle-piece-count 5
  --min-joint-count 40
  --max-fall-events 0
  --max-box-drop-events 0
  --min-robot-z 0.45
  --min-box-z 0.20
  --max-tilt 0.85
  --max-box-robot-relative-offset-error 0.70
  --min-final-box-target-directed-travel "${LONG_HOLD_MIN_FINAL_BOX_TARGET_TRAVEL:-0.10}"
  --max-root-pose-write-count-rollout 0
  --max-root-velocity-write-count-rollout 0
  --max-box-pose-write-count-rollout 0
  --require-diagnostic-claim
)
run_case stage4_freebox_long_hold_validation \
  --steps 700 \
  --box-mass "${FREEBOX_MASS:-0.25}" \
  --box-size 0.10 0.08 0.06 \
  --box-position "${LONG_HOLD_FREEBOX_POS_X:-0.36}" 0.0 "${LONG_HOLD_FREEBOX_POS_Z:-0.95}" \
  --attach-box none \
  --torso-cradle front_tray \
  --require-box-no-drop \
  --cradle-deck-size 0.24 0.26 0.025 \
  --cradle-deck-local-pos0 "${LONG_HOLD_CRADLE_LOCAL_X:-0.36}" 0.0 "${LONG_HOLD_CRADLE_LOCAL_Z:-0.10}" \
  --cradle-side-rail-height 0.07 \
  --cradle-end-stop-height 0.08 \
  --cradle-rail-thickness 0.018 \
  --cradle-mass-scale "${LONG_HOLD_CRADLE_MASS_SCALE:-0.50}" \
  --gait-mode staged_march \
  --gait-amplitude "${LONG_HOLD_GAIT_AMPLITUDE:-0.10}" \
  --gait-frequency-hz 0.7 \
  --gait-ramp-down-start-step "${LONG_HOLD_RAMP_START:-90}" \
  --gait-ramp-down-end-step "${LONG_HOLD_RAMP_END:-190}" \
  --gait-min-amplitude-scale 0.0 \
  --recovery-pitch-threshold 0.10 \
  --recovery-pitch-rate-threshold 999.0 \
  --recovery-hip-pitch-offset -0.10 \
  --recovery-knee-offset 0.12 \
  --recovery-ankle-pitch-offset 0.12 \
  --recovery-waist-pitch-offset -0.08

echo "[SUITE] status file: ${status_file}"
