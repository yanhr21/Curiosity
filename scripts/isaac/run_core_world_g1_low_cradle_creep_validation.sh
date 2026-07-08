#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 low-cradle creep validation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_low_cradle_creep_validation)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_low_cradle_creep_validation/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/low_cradle_creep_validation_status.tsv"
printf "case\tbuild_status\tcheck_status\toutput_dir\n" > "${status_file}"

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
  --box-mass "${BOX_MASS:-0.25}"
  --box-size "${BOX_SIZE_X:-0.10}" "${BOX_SIZE_Y:-0.08}" "${BOX_SIZE_Z:-0.06}"
  --box-position "${BOX_POS_X:-0.34}" 0.0 "${BOX_POS_Z:-0.90}"
  --attach-box none
  --torso-cradle front_tray
  --require-box-no-drop
  --cradle-deck-size 0.24 0.26 0.025
  --cradle-deck-local-pos0 "${CRADLE_LOCAL_X:-0.34}" 0.0 "${CRADLE_LOCAL_Z:-0.05}"
  --cradle-side-rail-height 0.07
  --cradle-end-stop-height 0.08
  --cradle-rail-thickness 0.018
  --cradle-mass-scale "${CRADLE_MASS_SCALE:-0.40}"
  --gait-mode targeted_creep
  --gait-frequency-hz "${GAIT_FREQUENCY_HZ:-0.70}"
  --gait-amplitude "${GAIT_AMPLITUDE:-0.16}"
  --creep-hip-pitch-offset 0.12
  --creep-knee-offset 0.04
  --creep-ankle-pitch-offset -0.06
  --creep-waist-pitch-offset 0.04
  --creep-stance-push-scale "${CREEP_STANCE_PUSH_SCALE:-0.32}"
  --creep-lift-scale "${CREEP_LIFT_SCALE:-0.55}"
  --creep-ankle-lift-scale -0.30
)

run_case() {
  local case_id="$1"
  local steps="$2"
  local min_final_travel="$3"
  local max_tilt="$4"
  local max_rel="$5"
  local max_final_rel="$6"
  local out="${SUITE_DIR}/${case_id}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[LOW-CREEP] ${case_id} steps=${steps} min_final_travel=${min_final_travel}"

  local build_status=0
  set +e
  "${base_python[@]}" \
    --steps "${steps}" \
    --output-dir "${out}" 2>&1 | tee "${build_log}"
  build_status=${PIPESTATUS[0]}
  set -e

  local check_status=99
  if [[ -f "${out}/core_world_g1_box_scene_summary.json" ]]; then
    set +e
    python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
      "${out}/core_world_g1_box_scene_summary.json" \
      --min-steps "${steps}" \
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
      --max-tilt "${max_tilt}" \
      --max-box-robot-relative-offset-error "${max_rel}" \
      --max-final-box-robot-relative-offset-error "${max_final_rel}" \
      --min-final-box-target-directed-travel "${min_final_travel}" \
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
    echo "[LOW-CREEP] strict mode stopping after ${case_id}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

run_case low_push032_700 700 "${MIN_FINAL_TRAVEL_700:-0.18}" "${MAX_TILT_700:-0.25}" "${MAX_REL_700:-0.14}" "${MAX_FINAL_REL_700:-0.10}"
run_case low_push032_1000 1000 "${MIN_FINAL_TRAVEL_1000:-0.24}" "${MAX_TILT_1000:-0.35}" "${MAX_REL_1000:-0.20}" "${MAX_FINAL_REL_1000:-0.16}"

echo "[LOW-CREEP] status file: ${status_file}"
