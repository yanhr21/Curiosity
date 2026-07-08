#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 targeted-creep stop tuning on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_targeted_creep_stop_tune)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_targeted_creep_stop_tune/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
STEPS="${STEPS:-560}"
MIN_FINAL_BOX_TARGET_TRAVEL="${MIN_FINAL_BOX_TARGET_TRAVEL:-0.10}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/targeted_creep_stop_tune_status.tsv"
printf "case\tbuild_status\tcheck_status\toutput_dir\n" > "${status_file}"

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
  --cradle-deck-size 0.24 0.26 0.025
  --cradle-side-rail-height 0.07
  --cradle-end-stop-height 0.08
  --cradle-rail-thickness 0.018
  --gait-mode targeted_creep
  --gait-frequency-hz 0.7
  --creep-hip-pitch-offset 0.12
  --creep-knee-offset 0.04
  --creep-ankle-pitch-offset -0.06
  --creep-waist-pitch-offset 0.04
  --creep-ankle-lift-scale -0.30
)

run_case() {
  local case_id="$1"
  local box_x="$2"
  local box_z="$3"
  local deck_x="$4"
  local deck_z="$5"
  local cradle_mass_scale="$6"
  local gait_amp="$7"
  local creep_push="$8"
  local creep_lift="$9"
  local gait_stop="${10}"
  local balance_start="${11}"
  local balance_gain="${12}"
  local balance_limit="${13}"
  local out="${SUITE_DIR}/${case_id}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[CREEP-STOP] ${case_id} box=(${box_x},0,${box_z}) deck=(${deck_x},0,${deck_z}) amp=${gait_amp} push=${creep_push} stop=${gait_stop} balance_start=${balance_start}"

  local extra_args=()
  if [[ "${gait_stop}" != "none" ]]; then
    extra_args+=(--gait-stop-step "${gait_stop}")
  fi
  if [[ "${balance_start}" != "none" ]]; then
    extra_args+=(
      --balance-feedback-controller
      --balance-start-step "${balance_start}"
      --balance-pitch-gain "${balance_gain}"
      --balance-pitch-rate-gain 0.01
      --balance-adjustment-limit "${balance_limit}"
      --balance-pitch-activation-threshold 0.035
      --balance-pitch-rate-activation-threshold 0.20
    )
  fi

  local build_status=0
  set +e
  "${base_python[@]}" \
    --box-position "${box_x}" 0.0 "${box_z}" \
    --cradle-deck-local-pos0 "${deck_x}" 0.0 "${deck_z}" \
    --cradle-mass-scale "${cradle_mass_scale}" \
    --gait-amplitude "${gait_amp}" \
    --creep-stance-push-scale "${creep_push}" \
    --creep-lift-scale "${creep_lift}" \
    "${extra_args[@]}" \
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
      --expect-gait-mode targeted_creep \
      --expect-box-collision-enabled true \
      --expect-cradle-collision-enabled true \
      --min-cradle-piece-count 5 \
      --min-joint-count 40 \
      --max-fall-events 0 \
      --max-box-drop-events 0 \
      --min-robot-z 0.45 \
      --min-box-z 0.20 \
      --max-tilt "${MAX_TILT:-0.20}" \
      --max-box-robot-relative-offset-error "${MAX_REL_ERROR:-0.12}" \
      --max-final-box-robot-relative-offset-error "${MAX_FINAL_REL_ERROR:-0.08}" \
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

  printf "%s\t%s\t%s\t%s\n" "${case_id}" "${build_status}" "${check_status}" "${out}" >> "${status_file}"
  if [[ "${STRICT}" == "1" && ( "${build_status}" != "0" || "${check_status}" != "0" ) ]]; then
    echo "[CREEP-STOP] strict mode stopping after ${case_id}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

run_case close_push022 0.36 0.95 0.36 0.10 0.50 0.16 0.22 0.55 none none 0.0 0.0
run_case close_push024 0.36 0.95 0.36 0.10 0.50 0.16 0.24 0.55 none none 0.0 0.0
run_case close_push028_stop400 0.36 0.95 0.36 0.10 0.50 0.16 0.28 0.55 400 none 0.0 0.0
run_case close_push028_stop420 0.36 0.95 0.36 0.10 0.50 0.16 0.28 0.55 420 none 0.0 0.0
run_case close_push028_stop400_latefb 0.36 0.95 0.36 0.10 0.50 0.16 0.28 0.55 400 380 0.16 0.06
run_case close_push030_stop400_latefb 0.36 0.95 0.36 0.10 0.50 0.16 0.30 0.55 400 380 0.16 0.06
run_case low_push032 0.34 0.90 0.34 0.05 0.40 0.16 0.32 0.55 none none 0.0 0.0

echo "[CREEP-STOP] status file: ${status_file}"
