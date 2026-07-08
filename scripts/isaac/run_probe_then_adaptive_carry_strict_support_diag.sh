#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

STAMP="${STAMP:-20260705_probe_then_adaptive_carry_strict_support_seed7055}"
ROOT_OUTPUT="experiments/outputs/probe_then_adaptive_carry/${STAMP}"
LOG_DIR="logs/probe_then_adaptive_carry"
mkdir -p "${ROOT_OUTPUT}" "${LOG_DIR}"

BOX_SEED="${BOX_SEED:-7055}"
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
  SUPPORT_FOOT_X_LOWER=-0.17
  SUPPORT_FOOT_X_UPPER=0.17
  SUPPORT_FOOT_Z_LOWER=-0.005
  SUPPORT_FOOT_Z_UPPER=0.24
  SUPPORT_FOOT_STEP_HEIGHT=0.120
  SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=0.12
  SUPPORT_FOOT_STANCE_X=-0.130
  SUPPORT_FOOT_SWING_X=0.130
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
  PROBE_STEPS=160 \
  PROBE_MODE=vertical_micro_lift \
  PROBE_X_AMPLITUDE=0.0 \
  PROBE_Z_AMPLITUDE=0.030 \
  2>&1 | tee "${LOG_DIR}/${STAMP}_probe.log"
probe_status=${PIPESTATUS[0]}
set -e
if [[ "${probe_status}" -ne 0 ]]; then
  echo "[WARN] probe wrapper exited ${probe_status}; continuing only if summary exists." >&2
fi

PROBE_SUMMARY="${PROBE_OUTPUT}/direct_carry_task_physical_backend_summary.json"
if [[ ! -f "${PROBE_SUMMARY}" ]]; then
  echo "[ERROR] Missing probe summary: ${PROBE_SUMMARY}" >&2
  exit 10
fi

read -r SELECTED_POSTURE SELECTED_STANCE_STEPS SELECTED_STEP_LENGTH SELECTION_RULE < <(
  python3 - "${PROBE_SUMMARY}" <<'PY'
import json
import sys
summary = json.loads(open(sys.argv[1]).read())
risk = summary.get("probe_risk_score")
try:
    risk = float(risk)
except (TypeError, ValueError):
    risk = 1.0
bucket = summary.get("probe_load_risk_bucket") or "unknown"
if risk < 0.33:
    print("front_mid 80 0.016 risk_lt_0.33_nominal_front_mid")
elif risk < 0.66:
    print("low_front 96 0.014 risk_0.33_to_0.66_lower_and_slower")
else:
    print("chest_high 112 0.012 risk_ge_0.66_chest_supported_slowest")
print(f"[SELECTOR] risk={risk:.6f} bucket={bucket}", file=sys.stderr)
PY
)

echo "[SELECTOR] selected_posture=${SELECTED_POSTURE} stance_steps=${SELECTED_STANCE_STEPS} step_length=${SELECTED_STEP_LENGTH} rule=${SELECTION_RULE}"

CARRY_STAMP="${STAMP}_carry_${SELECTED_POSTURE}"
CARRY_OUTPUT="${ROOT_OUTPUT}/carry_${SELECTED_POSTURE}"
set +e
run_direct_case "${CARRY_STAMP}" "${CARRY_OUTPUT}" \
  CARRY_POSTURE="${SELECTED_POSTURE}" \
  TARGET_X=0.64 \
  STEPS=3580 \
  STEP_LENGTH="${SELECTED_STEP_LENGTH}" \
  STANCE_STEPS="${SELECTED_STANCE_STEPS}" \
  SETTLE_STEPS=10 \
  PROBE_STEPS=0 \
  PROBE_X_AMPLITUDE=0.0 \
  PROBE_Z_AMPLITUDE=0.0 \
  2>&1 | tee "${LOG_DIR}/${STAMP}_carry_${SELECTED_POSTURE}.log"
carry_status=${PIPESTATUS[0]}
set -e
if [[ "${carry_status}" -ne 0 ]]; then
  echo "[WARN] carry wrapper exited ${carry_status}; checking whether backend/direct summaries exist." >&2
fi

CARRY_SUMMARY="${CARRY_OUTPUT}/direct_carry_task_physical_backend_summary.json"
if [[ ! -f "${CARRY_SUMMARY}" ]]; then
  BACKEND_SUMMARY="${CARRY_OUTPUT}/backend_anchored_cradle/core_world_anchored_footstep_carrier_summary.json"
  BACKEND_LOG="logs/core_world_anchored_footstep_carrier/core_world_anchored_footstep_carrier_${CARRY_STAMP}_backend_anchored_cradle.log"
  if [[ ! -f "${BACKEND_SUMMARY}" ]]; then
    echo "[ERROR] Missing carry summary and backend summary: ${CARRY_SUMMARY}" >&2
    exit 11
  fi
  python3 scripts/isaac/normalize_direct_carry_backend_summary.py \
    --backend-summary "${BACKEND_SUMMARY}" \
    --backend-log "${BACKEND_LOG}" \
    --controller-mode physical_alternating_anchor_feet_cradle \
    --carry-posture "${SELECTED_POSTURE}" \
    --output-summary "${CARRY_SUMMARY}"
fi

python3 scripts/isaac/check_direct_carry_task_summary.py \
  "${CARRY_SUMMARY}" \
  --min-steps 3560 \
  --expect-controller-mode physical_alternating_anchor_feet_cradle \
  --expect-carry-posture "${SELECTED_POSTURE}" \
  --expect-backend-support-mode dynamic_anchor \
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
  --require-non-success-claim

python3 scripts/isaac/summarize_probe_then_adaptive_carry.py \
  --probe-summary "${PROBE_SUMMARY}" \
  --carry-summary "${CARRY_SUMMARY}" \
  --selected-posture "${SELECTED_POSTURE}" \
  --selected-stance-steps "${SELECTED_STANCE_STEPS}" \
  --selected-step-length "${SELECTED_STEP_LENGTH}" \
  --selection-rule "${SELECTION_RULE}" \
  --output "${ROOT_OUTPUT}/probe_then_adaptive_carry_summary.json"

echo "[INFO] Adaptive diagnostic output: ${ROOT_OUTPUT}"
