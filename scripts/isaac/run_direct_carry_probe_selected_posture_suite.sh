#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run probe-selected carry suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-20260706_direct_carry_probe_selected_posture_suite_64cm_8kg}"
SUITE_DIR="${ROOT_DIR}/experiments/outputs/direct_carry_probe_selected_posture_suite/${SUITE_STAMP}"
LOG_DIR="${ROOT_DIR}/logs/direct_carry_probe_selected_posture_suite"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
mkdir -p "${SUITE_DIR}" "${LOG_DIR}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m py_compile \
  scripts/isaac/select_direct_carry_posture_from_probe.py \
  scripts/isaac/normalize_direct_carry_backend_summary.py \
  scripts/isaac/check_direct_carry_task_summary.py \
  scripts/isaac/summarize_direct_carry_posture_suite.py
bash -n scripts/isaac/run_direct_carry_task_physical_backend.sh

SUMMARY_ARGS=()
SELECTION_REPORTS=()
LAST_PROBE_SUMMARY=""

run_probe() {
  local condition="$1"
  local probe_mode="$2"
  local probe_x="$3"
  local probe_z="$4"
  local stamp="${SUITE_STAMP}_${condition}_probe"
  local output_dir="experiments/outputs/direct_carry_task_physical_backend/${stamp}"
  local direct_summary="${output_dir}/direct_carry_task_physical_backend_summary.json"

  echo "[PROBE] condition=${condition} mode=${probe_mode} x=${probe_x} z=${probe_z}"
  STAMP="${stamp}" \
  OUTPUT_DIR="${output_dir}" \
  BACKEND_OUTPUT_DIR="${output_dir}/backend_anchored_cradle" \
  BACKEND_STAMP="${stamp}_backend_anchored_cradle" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE=front_mid \
  TARGET_X=0.08 \
  PAYLOAD_MASS=8.0 \
  STEPS=360 \
  STEP_LENGTH=0.016 \
  STANCE_STEPS=80 \
  SETTLE_STEPS=10 \
  PROBE_STEPS=60 \
  PROBE_MODE="${probe_mode}" \
  PROBE_X_AMPLITUDE="${probe_x}" \
  PROBE_Z_AMPLITUDE="${probe_z}" \
  RAIL_JOINT_COUNT=2 \
  RAIL_LOWER=-0.04 \
  RAIL_UPPER=0.10 \
  SUPPORT_FOOT_MASS=8.0 \
  SUPPORT_FOOT_X_LOWER=-0.17 \
  SUPPORT_FOOT_X_UPPER=0.17 \
  SUPPORT_FOOT_Z_LOWER=-0.005 \
  SUPPORT_FOOT_Z_UPPER=0.24 \
  SUPPORT_FOOT_STEP_HEIGHT=0.120 \
  SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=0.12 \
  SUPPORT_FOOT_STANCE_X=-0.130 \
  SUPPORT_FOOT_SWING_X=0.130 \
  SUPPORT_FOOT_CONTACT_Z_THRESHOLD=0.035 \
  SUPPORT_FOOT_DRIVE_STIFFNESS=24000.0 \
  SUPPORT_FOOT_DRIVE_DAMPING=3400.0 \
  SUPPORT_FOOT_DRIVE_MAX_FORCE=110000.0 \
  SUPPORT_FOOT_Z_DRIVE_STIFFNESS=36000.0 \
  SUPPORT_FOOT_Z_DRIVE_DAMPING=3200.0 \
  SUPPORT_FOOT_Z_DRIVE_MAX_FORCE=130000.0 \
  DRIVE_STIFFNESS=22000.0 \
  DRIVE_DAMPING=3500.0 \
  DRIVE_MAX_FORCE=80000.0 \
  STATIC_FRICTION=4.5 \
  DYNAMIC_FRICTION=4.0 \
  bash scripts/isaac/run_direct_carry_task_physical_backend.sh \
    2>&1 | tee "${LOG_DIR}/${stamp}.log"

  python3 scripts/isaac/check_direct_carry_task_summary.py \
    "${direct_summary}" \
    --min-steps 300 \
    --expect-controller-mode physical_alternating_anchor_feet_cradle \
    --expect-carry-posture front_mid \
    --min-probe-steps 60 \
    --require-probe-belief \
    --forbid-probe-hidden-ground-truth \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --require-root-shortcut-free \
    --max-support-root-pose-write-count 0 \
    --max-anchor-world-joint-retarget-count 0 \
    --max-foot-pose-write-count 0 \
    --max-stance-anchor-pose-write-count 0 \
    --forbid-fixed-world-support \
    --require-non-success-claim

  LAST_PROBE_SUMMARY="${direct_summary}"
}

run_selected_carry() {
  local condition="$1"
  local posture="$2"
  local stamp="${SUITE_STAMP}_${condition}_carry_${posture}"
  local output_dir="experiments/outputs/direct_carry_task_physical_backend/${stamp}"
  local direct_summary="${output_dir}/direct_carry_task_physical_backend_summary.json"

  echo "[CARRY] condition=${condition} selected_posture=${posture}"
  STAMP="${stamp}" \
  OUTPUT_DIR="${output_dir}" \
  BACKEND_OUTPUT_DIR="${output_dir}/backend_anchored_cradle" \
  BACKEND_STAMP="${stamp}_backend_anchored_cradle" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE="${posture}" \
  TARGET_X=0.64 \
  PAYLOAD_MASS=8.0 \
  STEPS=3580 \
  STEP_LENGTH=0.016 \
  STANCE_STEPS=80 \
  SETTLE_STEPS=10 \
  RAIL_JOINT_COUNT=2 \
  RAIL_LOWER=-0.04 \
  RAIL_UPPER=0.10 \
  SUPPORT_FOOT_MASS=8.0 \
  SUPPORT_FOOT_X_LOWER=-0.17 \
  SUPPORT_FOOT_X_UPPER=0.17 \
  SUPPORT_FOOT_Z_LOWER=-0.005 \
  SUPPORT_FOOT_Z_UPPER=0.24 \
  SUPPORT_FOOT_STEP_HEIGHT=0.120 \
  SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION=0.12 \
  SUPPORT_FOOT_STANCE_X=-0.130 \
  SUPPORT_FOOT_SWING_X=0.130 \
  SUPPORT_FOOT_CONTACT_Z_THRESHOLD=0.035 \
  SUPPORT_FOOT_DRIVE_STIFFNESS=24000.0 \
  SUPPORT_FOOT_DRIVE_DAMPING=3400.0 \
  SUPPORT_FOOT_DRIVE_MAX_FORCE=110000.0 \
  SUPPORT_FOOT_Z_DRIVE_STIFFNESS=36000.0 \
  SUPPORT_FOOT_Z_DRIVE_DAMPING=3200.0 \
  SUPPORT_FOOT_Z_DRIVE_MAX_FORCE=130000.0 \
  DRIVE_STIFFNESS=22000.0 \
  DRIVE_DAMPING=3500.0 \
  DRIVE_MAX_FORCE=80000.0 \
  STATIC_FRICTION=4.5 \
  DYNAMIC_FRICTION=4.0 \
  bash scripts/isaac/run_direct_carry_task_physical_backend.sh \
    2>&1 | tee "${LOG_DIR}/${stamp}.log"

  python3 scripts/isaac/check_direct_carry_task_summary.py \
    "${direct_summary}" \
    --min-steps 3560 \
    --expect-controller-mode physical_alternating_anchor_feet_cradle \
    --expect-carry-posture "${posture}" \
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
    --min-support-polygon-margin 0.12 \
    --min-box-travel 0.52 \
    --max-final-box-target-distance-x 0.08 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --require-root-shortcut-free \
    --max-support-root-pose-write-count 0 \
    --max-anchor-world-joint-retarget-count 0 \
    --max-foot-pose-write-count 0 \
    --max-stance-anchor-pose-write-count 0 \
    --forbid-fixed-world-support \
    --require-non-success-claim

  SUMMARY_ARGS+=(--summary "${direct_summary}")
}

run_condition() {
  local condition="$1"
  local probe_mode="$2"
  local probe_x="$3"
  local probe_z="$4"
  run_probe "${condition}" "${probe_mode}" "${probe_x}" "${probe_z}"
  local probe_summary="${LAST_PROBE_SUMMARY}"
  local selection_report="${SUITE_DIR}/${condition}_selection.json"
  "${PYTHON_BIN}" scripts/isaac/select_direct_carry_posture_from_probe.py \
    --probe-summary "${probe_summary}" \
    --output "${selection_report}" \
    --medium-threshold 0.58 \
    --high-threshold 0.75 \
    --low-posture front_reach \
    --medium-posture close_mid \
    --high-posture chest_high
  local posture
  posture="$(jq -r '.selected_carry_posture' "${selection_report}")"
  SELECTION_REPORTS+=("${selection_report}")
  run_selected_carry "${condition}" "${posture}"
}

run_condition vertical_probe vertical_micro_lift 0.0 0.030
run_condition horizontal_probe horizontal_push_pull 0.050 0.0

python3 scripts/isaac/summarize_direct_carry_posture_suite.py \
  "${SUMMARY_ARGS[@]}" \
  --output "${SUITE_DIR}/probe_selected_carry_summary.json" \
  --min-postures 1 \
  --min-steps 3560 \
  --min-box-travel-x 0.52 \
  --max-target-distance-x 0.08 \
  --max-tilt 0.14 \
  --min-support-margin 0.12

{
  echo "suite_stamp=${SUITE_STAMP}"
  echo "selection_reports=${SELECTION_REPORTS[*]}"
  echo "carry_summary=${SUITE_DIR}/probe_selected_carry_summary.json"
} > "${SUITE_DIR}/manifest.txt"

cat "${SUITE_DIR}/manifest.txt"
