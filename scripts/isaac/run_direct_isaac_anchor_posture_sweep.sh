#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT_DIR}/experiments/outputs/direct_isaac_anchor_posture_sweep/${STAMP}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/direct_isaac_anchor_posture_sweep}"
EXPERIENCE="${EXPERIENCE:-${ROOT_DIR}/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"

mkdir -p "${SWEEP_ROOT}" "${LOG_DIR}"
cd "${ROOT_DIR}"

BOX_SEED="${BOX_SEED:-17}"
PAYLOAD_MASS_MIN="${PAYLOAD_MASS_MIN:-4.0}"
PAYLOAD_MASS_MAX="${PAYLOAD_MASS_MAX:-12.0}"
PAYLOAD_SIZE_JITTER="${PAYLOAD_SIZE_JITTER:-0.12}"
PAYLOAD_COM_OFFSET_RANGE_X="${PAYLOAD_COM_OFFSET_RANGE_X:-0.04}"
PAYLOAD_COM_OFFSET_RANGE_Y="${PAYLOAD_COM_OFFSET_RANGE_Y:-0.03}"
PAYLOAD_COM_OFFSET_RANGE_Z="${PAYLOAD_COM_OFFSET_RANGE_Z:-0.03}"

run_candidate() {
  local name="$1"
  local payload_x="$2"
  local payload_z="$3"
  local output_dir="${SWEEP_ROOT}/${name}"
  local candidate_stamp="${STAMP}_${name}"
  local log_path="${LOG_DIR}/direct_isaac_anchor_posture_sweep_${candidate_stamp}.log"
  mkdir -p "${output_dir}"
  echo "[CANDIDATE] ${name} payload_local=(${payload_x}, 0, ${payload_z})"
  local cmd=(
    "${ISAAC_VENV}/bin/python"
    "${ROOT_DIR}/scripts/isaac/build_core_world_anchored_footstep_carrier.py"
    --viz none
    --experience "${EXPERIENCE}"
    --device "${DEVICE:-cpu}"
    --kit_args "${KIT_ARGS}"
    --steps "${STEPS:-300}"
    --target-x "${TARGET_X:-0.08}"
    --step-length "${STEP_LENGTH:-0.04}"
    --stance-steps "${STANCE_STEPS:-90}"
    --settle-steps "${SETTLE_STEPS:-50}"
    --probe-steps "${PROBE_STEPS:-80}"
    --probe-mode "${PROBE_MODE:-horizontal_push_pull}"
    --probe-x-amplitude "${PROBE_X_AMPLITUDE:-0.02}"
    --probe-z-amplitude "${PROBE_Z_AMPLITUDE:-0.0}"
    --payload-mode "${PAYLOAD_MODE:-fixed_joint_to_torso}"
    --payload-mass "${PAYLOAD_MASS:-4.0}"
    --box-seed "${BOX_SEED}"
    --randomize-payload
    --payload-mass-range "${PAYLOAD_MASS_MIN}" "${PAYLOAD_MASS_MAX}"
    --payload-size-jitter "${PAYLOAD_SIZE_JITTER}"
    --payload-com-offset-range "${PAYLOAD_COM_OFFSET_RANGE_X}" "${PAYLOAD_COM_OFFSET_RANGE_Y}" "${PAYLOAD_COM_OFFSET_RANGE_Z}"
    --payload-local-x "${payload_x}"
    --payload-local-z "${payload_z}"
    --support-foot-mode "${SUPPORT_FOOT_MODE:-static_markers}"
    --fix-anchor-to-world \
    --cumulative-cycle-target \
    --rail-joint-count "${RAIL_JOINT_COUNT:-1}" \
    --rail-lower "${RAIL_LOWER:--0.10}" \
    --rail-upper "${RAIL_UPPER:-0.08}" \
    --output-dir "${output_dir}"
  )
  "${cmd[@]}" 2>&1 | tee "${log_path}"
}

run_candidate "front_mid" "0.20" "0.04"
run_candidate "chest_close" "0.12" "0.02"
run_candidate "low_close" "0.14" "-0.06"
run_candidate "extended_front" "0.26" "0.02"

"${PYTHON:-python3}" scripts/isaac/summarize_anchor_posture_sweep.py \
  --sweep-dir "${SWEEP_ROOT}" \
  --output "${SWEEP_ROOT}/direct_isaac_anchor_posture_sweep_summary.json"

echo "[INFO] Sweep summary: ${SWEEP_ROOT}/direct_isaac_anchor_posture_sweep_summary.json"
