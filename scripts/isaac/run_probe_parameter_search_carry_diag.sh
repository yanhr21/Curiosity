#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

STAMP="${STAMP:-20260705_probe_parameter_search_carry_seed7067}"
BOX_SEED="${BOX_SEED:-7067}"
ROOT_OUTPUT="experiments/outputs/probe_parameter_search_carry/${STAMP}"
LOG_DIR="logs/probe_parameter_search_carry"
mkdir -p "${ROOT_OUTPUT}" "${ROOT_OUTPUT}/candidates" "${LOG_DIR}"

PAYLOAD_MASS_MIN="${PAYLOAD_MASS_MIN:-4.0}"
PAYLOAD_MASS_MAX="${PAYLOAD_MASS_MAX:-12.0}"
PAYLOAD_SIZE_JITTER="${PAYLOAD_SIZE_JITTER:-0.10}"
PAYLOAD_COM_OFFSET_RANGE_X="${PAYLOAD_COM_OFFSET_RANGE_X:-0.04}"
PAYLOAD_COM_OFFSET_RANGE_Y="${PAYLOAD_COM_OFFSET_RANGE_Y:-0.03}"
PAYLOAD_COM_OFFSET_RANGE_Z="${PAYLOAD_COM_OFFSET_RANGE_Z:-0.03}"

common_backend_env=(
  SUPPORT_MODE=alternating_anchor_feet
  PAYLOAD_MASS=8.0
  RANDOMIZE_PAYLOAD=1
  BOX_SEED="${BOX_SEED}"
  PAYLOAD_MASS_MIN="${PAYLOAD_MASS_MIN}"
  PAYLOAD_MASS_MAX="${PAYLOAD_MASS_MAX}"
  PAYLOAD_SIZE_JITTER="${PAYLOAD_SIZE_JITTER}"
  PAYLOAD_COM_OFFSET_RANGE_X="${PAYLOAD_COM_OFFSET_RANGE_X}"
  PAYLOAD_COM_OFFSET_RANGE_Y="${PAYLOAD_COM_OFFSET_RANGE_Y}"
  PAYLOAD_COM_OFFSET_RANGE_Z="${PAYLOAD_COM_OFFSET_RANGE_Z}"
  RAIL_JOINT_COUNT=2
  RAIL_LOWER=-0.04
  RAIL_UPPER=0.10
  SUPPORT_FOOT_MASS=8.0
  SUPPORT_FOOT_X_LOWER=-0.18
  SUPPORT_FOOT_X_UPPER=0.18
  SUPPORT_FOOT_Z_LOWER=-0.005
  SUPPORT_FOOT_Z_UPPER=0.24
  SUPPORT_FOOT_STEP_HEIGHT=0.120
  SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=0.12
  SUPPORT_FOOT_CONTACT_Z_THRESHOLD=0.035
  SUPPORT_FOOT_DRIVE_STIFFNESS=24000.0
  SUPPORT_FOOT_DRIVE_DAMPING=3400.0
  SUPPORT_FOOT_DRIVE_MAX_FORCE=110000.0
  SUPPORT_FOOT_Z_DRIVE_STIFFNESS=36000.0
  SUPPORT_FOOT_Z_DRIVE_DAMPING=3200.0
  SUPPORT_FOOT_Z_DRIVE_MAX_FORCE=130000.0
  DRIVE_STIFFNESS=22000.0
  DRIVE_DAMPING=3500.0
  DRIVE_MAX_FORCE=80000.0
  STATIC_FRICTION=4.5
  DYNAMIC_FRICTION=4.0
)

run_direct_case() {
  local case_stamp="$1"
  local output_dir="$2"
  shift 2
  mkdir -p "${output_dir}"
  env \
    STAMP="${case_stamp}" \
    OUTPUT_DIR="${output_dir}" \
    BACKEND_OUTPUT_DIR="${output_dir}/backend_anchored_cradle" \
    BACKEND_STAMP="${case_stamp}_backend_anchored_cradle" \
    "${common_backend_env[@]}" \
    "$@" \
    bash scripts/isaac/run_direct_carry_task_physical_backend.sh
}

PROBE_STAMP="${STAMP}_probe"
PROBE_OUTPUT="${ROOT_OUTPUT}/probe"
set +e
run_direct_case "${PROBE_STAMP}" "${PROBE_OUTPUT}" \
  CARRY_POSTURE=front_mid \
  TARGET_X=0.08 \
  STEPS=720 \
  STEP_LENGTH=0.016 \
  STANCE_STEPS=80 \
  SETTLE_STEPS=10 \
  SUPPORT_FOOT_STANCE_X=-0.130 \
  SUPPORT_FOOT_SWING_X=0.130 \
  PROBE_STEPS=160 \
  PROBE_MODE=vertical_micro_lift \
  PROBE_X_AMPLITUDE=0.0 \
  PROBE_Z_AMPLITUDE=0.030 \
  2>&1 | tee "${LOG_DIR}/${STAMP}_probe.log"
probe_status=${PIPESTATUS[0]}
set -e
echo "${probe_status}" > "${PROBE_OUTPUT}/wrapper_status.txt"
if [[ "${probe_status}" -ne 0 ]]; then
  echo "[WARN] probe wrapper exited ${probe_status}; continuing only if summary exists." >&2
fi

PROBE_SUMMARY="${PROBE_OUTPUT}/direct_carry_task_physical_backend_summary.json"
if [[ ! -f "${PROBE_SUMMARY}" ]]; then
  echo "[ERROR] Missing probe summary: ${PROBE_SUMMARY}" >&2
  exit 10
fi

CANDIDATE_SET="${CANDIDATE_SET:-default}"
case "${CANDIDATE_SET}" in
  default)
    declare -a CANDIDATES=(
      "front_mid_nominal front_mid 80 0.016 -0.130 0.130 0.55 0.20 0.04 0.120 0.12 0.28 0.22"
      "low_front_slow low_front 96 0.014 -0.130 0.130 0.58 0.24 -0.02 0.120 0.12 0.28 0.22"
      "chest_high_slowest chest_high 112 0.012 -0.135 0.135 0.55 0.16 0.12 0.120 0.12 0.28 0.22"
      "front_mid_wide_slow front_mid 96 0.014 -0.145 0.145 0.55 0.20 0.04 0.120 0.12 0.28 0.24"
      "low_front_wide_slowest low_front 112 0.012 -0.145 0.145 0.58 0.24 -0.02 0.120 0.12 0.28 0.24"
    )
    ;;
  expanded)
    declare -a CANDIDATES=(
      "front_mid_nominal front_mid 80 0.016 -0.130 0.130 0.55 0.20 0.04 0.120 0.12 0.28 0.22"
      "front_mid_lower_slow front_mid 96 0.014 -0.130 0.130 0.53 0.21 0.01 0.110 0.16 0.28 0.22"
      "front_mid_high_clearance front_mid 88 0.015 -0.135 0.135 0.57 0.19 0.07 0.130 0.16 0.30 0.23"
      "low_front_slow low_front 96 0.014 -0.130 0.130 0.58 0.24 -0.02 0.120 0.12 0.28 0.22"
      "low_front_cautious low_front 112 0.012 -0.125 0.125 0.57 0.23 -0.03 0.100 0.18 0.30 0.23"
      "chest_high_slowest chest_high 112 0.012 -0.135 0.135 0.55 0.16 0.12 0.120 0.12 0.28 0.22"
      "chest_high_cautious chest_high 128 0.010 -0.125 0.125 0.56 0.15 0.11 0.100 0.18 0.30 0.22"
      "front_mid_wide_slow front_mid 96 0.014 -0.145 0.145 0.55 0.20 0.04 0.120 0.12 0.30 0.24"
      "low_front_wide_slowest low_front 112 0.012 -0.145 0.145 0.58 0.24 -0.02 0.120 0.12 0.30 0.24"
    )
    ;;
  *)
    echo "[ERROR] Unknown CANDIDATE_SET=${CANDIDATE_SET}; expected default or expanded." >&2
    exit 6
    ;;
esac

for spec in "${CANDIDATES[@]}"; do
  read -r candidate_id posture stance_steps step_length stance_x swing_x torso_z payload_x payload_z step_height double_support stance_half_length stance_half_width <<< "${spec}"
  candidate_stamp="${STAMP}_${candidate_id}"
  candidate_output="${ROOT_OUTPUT}/candidates/${candidate_id}"
  backend_summary="${candidate_output}/backend_anchored_cradle/core_world_anchored_footstep_carrier_summary.json"
  direct_summary="${candidate_output}/direct_carry_task_physical_backend_summary.json"
  backend_log="logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_${candidate_stamp}_backend_anchored_cradle.log"
  mkdir -p "${candidate_output}"
  python3 - "${candidate_output}/candidate_config.json" "${candidate_id}" "${posture}" "${stance_steps}" "${step_length}" "${stance_x}" "${swing_x}" "${torso_z}" "${payload_x}" "${payload_z}" "${step_height}" "${double_support}" "${stance_half_length}" "${stance_half_width}" <<'PY'
import json
import sys
(
    path,
    candidate_id,
    posture,
    stance_steps,
    step_length,
    stance_x,
    swing_x,
    torso_z,
    payload_x,
    payload_z,
    step_height,
    double_support,
    stance_half_length,
    stance_half_width,
) = sys.argv[1:15]
data = {
    "candidate_id": candidate_id,
    "carry_posture": posture,
    "stance_steps": int(stance_steps),
    "step_length_m": float(step_length),
    "support_foot_stance_x_m": float(stance_x),
    "support_foot_swing_x_m": float(swing_x),
    "torso_z_m": float(torso_z),
    "payload_local_x_m": float(payload_x),
    "payload_local_z_m": float(payload_z),
    "support_foot_step_height_m": float(step_height),
    "support_foot_double_support_fraction": float(double_support),
    "stance_half_length_m": float(stance_half_length),
    "stance_half_width_m": float(stance_half_width),
}
open(path, "w").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  echo "[CANDIDATE] id=${candidate_id} posture=${posture} stance_steps=${stance_steps} step_length=${step_length} stance_x=${stance_x} swing_x=${swing_x} torso_z=${torso_z} payload_x=${payload_x} payload_z=${payload_z} step_height=${step_height} double_support=${double_support} stance_half_length=${stance_half_length} stance_half_width=${stance_half_width}"
  set +e
  run_direct_case "${candidate_stamp}" "${candidate_output}" \
    CARRY_POSTURE="${posture}" \
    TARGET_X=0.64 \
    STEPS=3580 \
    STEP_LENGTH="${step_length}" \
    STANCE_STEPS="${stance_steps}" \
    SETTLE_STEPS=10 \
    SUPPORT_FOOT_STANCE_X="${stance_x}" \
    SUPPORT_FOOT_SWING_X="${swing_x}" \
    TORSO_Z="${torso_z}" \
    PAYLOAD_LOCAL_X="${payload_x}" \
    PAYLOAD_LOCAL_Z="${payload_z}" \
    SUPPORT_FOOT_STEP_HEIGHT="${step_height}" \
    SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION="${double_support}" \
    STANCE_HALF_LENGTH="${stance_half_length}" \
    STANCE_HALF_WIDTH="${stance_half_width}" \
    PROBE_STEPS=0 \
    PROBE_X_AMPLITUDE=0.0 \
    PROBE_Z_AMPLITUDE=0.0 \
    2>&1 | tee "${LOG_DIR}/${STAMP}_${candidate_id}.log"
  wrapper_status=${PIPESTATUS[0]}
  set -e
  echo "${wrapper_status}" > "${candidate_output}/wrapper_status.txt"
  if [[ "${wrapper_status}" -ne 0 ]]; then
    echo "[WARN] candidate ${candidate_id} wrapper exited ${wrapper_status}; attempting summary recovery." >&2
  fi
  if [[ ! -f "${direct_summary}" && -f "${backend_summary}" ]]; then
    python3 scripts/isaac/normalize_direct_carry_backend_summary.py \
      --backend-summary "${backend_summary}" \
      --backend-log "${backend_log}" \
      --controller-mode physical_alternating_anchor_feet_cradle \
      --carry-posture "${posture}" \
      --output-summary "${direct_summary}"
  fi
  if [[ ! -f "${direct_summary}" ]]; then
    echo "[WARN] candidate ${candidate_id} missing direct summary after recovery." >&2
    echo "99" > "${candidate_output}/check_status.txt"
    continue
  fi
  set +e
  python3 scripts/isaac/check_direct_carry_task_summary.py \
    "${direct_summary}" \
    --min-steps 3560 \
    --expect-controller-mode physical_alternating_anchor_feet_cradle \
    --expect-carry-posture "${posture}" \
    --expect-backend-support-mode dynamic_anchor \
    --require-box-randomized \
    --expect-box-seed "${BOX_SEED}" \
    --expect-support-foot-mode xz_prismatic_to_anchor \
    --min-support-foot-joint-count 8 \
    --min-support-foot-z-joint-count 4 \
    --min-support-foot-x-joint-motion 0.35 \
    --min-support-foot-z-joint-motion 0.15 \
    --min-actual-support-foot-lift 0.02 \
    --min-drive-near-ground-foot-count 2 \
    --max-drive-near-ground-zero-steps 0 \
    --max-drive-near-ground-lt2-steps 0 \
    --min-commanded-stance-near-ground-foot-count 2 \
    --max-commanded-stance-near-ground-lt2-steps 0 \
    --min-box-travel 0.52 \
    --max-final-box-target-distance-x 0.18 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --require-root-shortcut-free \
    --max-support-root-pose-write-count 0 \
    --max-anchor-world-joint-retarget-count 0 \
    --max-foot-pose-write-count 0 \
    --max-stance-anchor-pose-write-count 0 \
    --forbid-fixed-world-support \
    --require-non-success-claim \
    > "${candidate_output}/strict_check_report.json"
  check_status=$?
  set -e
  echo "${check_status}" > "${candidate_output}/check_status.txt"
  if [[ "${check_status}" -ne 0 ]]; then
    echo "[WARN] candidate ${candidate_id} strict checker failed; preserving result and continuing." >&2
  fi
done

python3 scripts/isaac/summarize_probe_parameter_search_carry.py \
  --root "${ROOT_OUTPUT}" \
  --probe-summary "${PROBE_SUMMARY}" \
  --box-seed "${BOX_SEED}" \
  --output "${ROOT_OUTPUT}/probe_parameter_search_carry_summary.json"

echo "[INFO] Probe parameter-search output: ${ROOT_OUTPUT}"
