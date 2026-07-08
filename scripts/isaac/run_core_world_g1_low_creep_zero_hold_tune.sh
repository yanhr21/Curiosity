#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 low-creep zero-hold tuning on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_low_creep_zero_hold_tune)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_low_creep_zero_hold_tune/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
STEPS="${STEPS:-700}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/low_creep_zero_hold_tune_status.tsv"
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
  --box-position 0.34 0.0 0.90
  --attach-box none
  --torso-cradle front_tray
  --require-box-no-drop
  --cradle-deck-size 0.24 0.26 0.025
  --cradle-deck-local-pos0 0.34 0.0 0.05
  --cradle-side-rail-height 0.07
  --cradle-end-stop-height 0.08
  --cradle-rail-thickness 0.018
  --cradle-mass-scale 0.40
  --gait-mode targeted_creep
  --gait-frequency-hz 0.70
  --gait-amplitude 0.16
  --creep-hip-pitch-offset 0.12
  --creep-knee-offset 0.04
  --creep-ankle-pitch-offset -0.06
  --creep-waist-pitch-offset 0.04
  --creep-stance-push-scale 0.32
  --creep-lift-scale 0.55
  --creep-ankle-lift-scale -0.30
  --terminal-hold-start-step -1
  --terminal-hold-hip-pitch-offset 0.0
  --terminal-hold-knee-offset 0.0
  --terminal-hold-ankle-pitch-offset 0.0
  --terminal-hold-waist-pitch-offset 0.0
)

run_case() {
  local case_id="$1"
  local hold_travel="$2"
  local min_final_travel="$3"
  local max_tilt="$4"
  local max_rel="$5"
  local max_final_rel="$6"
  local out="${SUITE_DIR}/${case_id}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[ZERO-HOLD] ${case_id} hold_travel=${hold_travel}"

  local build_status=0
  set +e
  "${base_python[@]}" \
    --terminal-hold-box-target-travel "${hold_travel}" \
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
    echo "[ZERO-HOLD] strict mode stopping after ${case_id}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

run_case hold008_zero 0.08 0.08 0.25 0.14 0.10
run_case hold010_zero 0.10 0.10 0.28 0.16 0.12
run_case hold012_zero 0.12 0.12 0.32 0.18 0.14
run_case hold014_zero 0.14 0.14 0.36 0.20 0.16

echo "[ZERO-HOLD] status file: ${status_file}"
