#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 AGILE no-box smoke on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_core_world_g1_agile_policy_nobox_smoke)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_nobox_smoke/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
STRICT="${STRICT:-0}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-20}"
AGILE_POLICY_BACKEND="${AGILE_POLICY_BACKEND:-onnx}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP}"
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

status_file="${SUITE_DIR}/agile_policy_nobox_smoke_status.tsv"
printf "case\tbuild_status\tcheck_status\toutput_dir\n" > "${status_file}"

base_python=(
  "${ISAAC_VENV}/bin/python"
  scripts/isaac/build_core_world_g1_box_scene.py
  --viz none
  --experience "${EXPERIENCE}"
  --device "${DEVICE}"
  --kit_args "${KIT_ARGS}"
  --steps "${STEPS:-320}"
  --g1-usd "${G1_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
  --target-xy "${TARGET_X:-1.2}" "${TARGET_Y:-0.0}"
  --g1-root-position 0.0 0.0 0.78
  --g1-root-orientation-wxyz "${G1_ROOT_QW:-1.0}" "${G1_ROOT_QX:-0.0}" "${G1_ROOT_QY:-0.0}" "${G1_ROOT_QZ:-0.0}"
  --stand-hip-pitch -0.10
  --stand-knee 0.30
  --stand-ankle-pitch -0.20
  --apply-arena-stand-gains
  --stand-drive-preset isaaclab29dof
  --stand-gain-scale 1.0
  --stand-force-scale 1.0
  --gait-mode agile_policy
  --policy-start-step 40
  --policy-control-decimation 4
  --agile-height-command "${AGILE_HEIGHT_COMMAND:-0.72}"
  --agile-policy-backend "${AGILE_POLICY_BACKEND}"
  --agile-config "${AGILE_CONFIG:-${ROOT_DIR}/external/IsaacLab-Arena/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_agile.yaml}"
  --agile-onnx "${AGILE_ONNX:-${ROOT_DIR}/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx}"
  --agile-torch-checkpoint "${AGILE_TORCH_CHECKPOINT:-${ROOT_DIR}/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt}"
  --disable-carry-box-spawn
)

run_case() {
  local case_id="$1"
  local command_x="$2"
  local min_robot_travel="$3"
  local out="${SUITE_DIR}/${case_id}"
  local build_log="${out}/build.log"
  local check_log="${out}/check.json"
  mkdir -p "${out}"
  echo "[AGILE-NOBOX] ${case_id} command_x=${command_x} backend=${AGILE_POLICY_BACKEND}"

  local build_status=0
  set +e
  "${base_python[@]}" \
    --agile-command "${command_x}" 0.0 0.0 \
    --output-dir "${out}" 2>&1 | tee "${build_log}"
  build_status=${PIPESTATUS[0]}
  set -e

  local check_status=99
  if [[ -f "${out}/core_world_g1_box_scene_summary.json" ]]; then
    set +e
    python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
      "${out}/core_world_g1_box_scene_summary.json" \
      --min-steps "${STEPS:-320}" \
      --expect-carry-box-spawned false \
      --expect-gait-mode agile_policy \
      --min-joint-count 40 \
      --max-fall-events 0 \
      --max-box-drop-events 0 \
      --min-robot-z 0.45 \
      --max-tilt "${MAX_TILT:-0.85}" \
      --min-final-robot-target-directed-travel "${min_robot_travel}" \
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
    echo "[AGILE-NOBOX] strict mode stopping after ${case_id}: build=${build_status} check=${check_status}" >&2
    exit 1
  fi
}

if [[ "${INCLUDE_POSITIVE_COMMANDS:-1}" == "1" ]]; then
  run_case onnx_cmd010_isaaclab_gains 0.10 0.03
  run_case onnx_cmd005_isaaclab_gains 0.05 0.015
fi

if [[ "${INCLUDE_NEGATIVE_COMMANDS:-0}" == "1" ]]; then
  run_case onnx_cmdneg010_isaaclab_gains -0.10 0.03
  run_case onnx_cmdneg005_isaaclab_gains -0.05 0.015
fi

echo "[AGILE-NOBOX] status file: ${status_file}"
