#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run online probe-adaptive hold suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-20260706_direct_carry_online_probe_adaptive_hold_64cm_8kg}"
SUITE_DIR="${ROOT_DIR}/experiments/outputs/direct_carry_online_probe_adaptive_hold_suite/${SUITE_STAMP}"
LOG_DIR="${ROOT_DIR}/logs/direct_carry_online_probe_adaptive_hold_suite"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
mkdir -p "${SUITE_DIR}" "${LOG_DIR}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m py_compile \
  scripts/isaac/build_core_world_anchored_footstep_carrier.py \
  scripts/isaac/normalize_direct_carry_backend_summary.py \
  scripts/isaac/check_direct_carry_task_summary.py \
  scripts/isaac/summarize_direct_carry_posture_suite.py
bash -n \
  scripts/isaac/run_core_world_anchored_footstep_carrier.sh \
  scripts/isaac/run_direct_carry_task_physical_backend.sh

SUMMARY_ARGS=()

run_case() {
  local condition="$1"
  local probe_mode="$2"
  local probe_x="$3"
  local probe_z="$4"
  local expected_bucket="$5"
  local expected_profile="$6"
  local expected_closure="$7"
  local expected_collision_enabled="$8"
  local stamp="${SUITE_STAMP}_${condition}"
  local output_dir="experiments/outputs/direct_carry_task_physical_backend/${stamp}"
  local direct_summary="${output_dir}/direct_carry_task_physical_backend_summary.json"

  echo "[ONLINE_HOLD_SUITE] condition=${condition} probe_mode=${probe_mode}"
  STAMP="${stamp}" \
  OUTPUT_DIR="${output_dir}" \
  BACKEND_OUTPUT_DIR="${output_dir}/backend_anchored_adaptive_cradle" \
  BACKEND_STAMP="${stamp}_backend_anchored_adaptive_cradle" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE=front_mid \
  PAYLOAD_MODE=cradle_free_box \
  TARGET_X=0.64 \
  PAYLOAD_MASS=8.0 \
  STEPS=3640 \
  STEP_LENGTH=0.016 \
  STANCE_STEPS=80 \
  SETTLE_STEPS=10 \
  PROBE_STEPS=60 \
  PROBE_MODE="${probe_mode}" \
  PROBE_X_AMPLITUDE="${probe_x}" \
  PROBE_Z_AMPLITUDE="${probe_z}" \
  ENABLE_ONLINE_PROBE_ADAPTIVE_SUPPORT=1 \
  ENABLE_ONLINE_PROBE_ADAPTIVE_HOLD=1 \
  ONLINE_PROBE_ADAPTIVE_MEDIUM_THRESHOLD=0.58 \
  ONLINE_PROBE_ADAPTIVE_HIGH_THRESHOLD=0.75 \
  ONLINE_LOW_SUPPORT_STEP_HEIGHT=0.120 \
  ONLINE_LOW_SUPPORT_DOUBLE_SUPPORT_FRACTION=0.12 \
  ONLINE_LOW_SUPPORT_STANCE_X=-0.130 \
  ONLINE_LOW_SUPPORT_SWING_X=0.130 \
  ONLINE_MEDIUM_SUPPORT_STEP_HEIGHT=0.100 \
  ONLINE_MEDIUM_SUPPORT_DOUBLE_SUPPORT_FRACTION=0.18 \
  ONLINE_MEDIUM_SUPPORT_STANCE_X=-0.115 \
  ONLINE_MEDIUM_SUPPORT_SWING_X=0.115 \
  ONLINE_HIGH_SUPPORT_STEP_HEIGHT=0.080 \
  ONLINE_HIGH_SUPPORT_DOUBLE_SUPPORT_FRACTION=0.24 \
  ONLINE_HIGH_SUPPORT_STANCE_X=-0.100 \
  ONLINE_HIGH_SUPPORT_SWING_X=0.100 \
  ONLINE_LOW_HOLD_CLOSURE_FRACTION=0.75 \
  ONLINE_MEDIUM_HOLD_CLOSURE_FRACTION=1.0 \
  ONLINE_HIGH_HOLD_CLOSURE_FRACTION=1.0 \
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
  CLAMP_CLOSE_START_STEP=70 \
  CLAMP_CLOSE_STEPS=120 \
  CLAMP_DRIVE_STIFFNESS=3500.0 \
  CLAMP_DRIVE_DAMPING=700.0 \
  CLAMP_DRIVE_MAX_FORCE=8000.0 \
  CLAMP_OPEN_GAP=0.060 \
  CLAMP_CLOSED_GAP=0.006 \
  DRIVE_STIFFNESS=22000.0 \
  DRIVE_DAMPING=3500.0 \
  DRIVE_MAX_FORCE=80000.0 \
  STATIC_FRICTION=4.5 \
  DYNAMIC_FRICTION=4.0 \
  bash scripts/isaac/run_direct_carry_task_physical_backend.sh \
    2>&1 | tee "${LOG_DIR}/${stamp}.log"

  python3 scripts/isaac/check_direct_carry_task_summary.py \
    "${direct_summary}" \
    --min-steps 3620 \
    --expect-controller-mode physical_alternating_anchor_feet_cradle \
    --expect-carry-posture front_mid \
    --expect-backend-support-mode dynamic_anchor \
    --expect-support-foot-mode xz_prismatic_to_anchor \
    --min-probe-steps 60 \
    --require-probe-belief \
    --forbid-probe-hidden-ground-truth \
    --require-online-probe-adaptive-support \
    --forbid-online-probe-adaptive-hidden-ground-truth \
    --expect-online-probe-adaptive-support-bucket "${expected_bucket}" \
    --require-online-probe-adaptive-hold \
    --require-online-probe-adaptive-hold-collision-available \
    --expect-online-probe-adaptive-hold-collision-enabled "${expected_collision_enabled}" \
    --min-online-probe-adaptive-hold-collision-updates 1 \
    --forbid-online-probe-adaptive-hold-hidden-ground-truth \
    --expect-online-probe-adaptive-hold-bucket "${expected_bucket}" \
    --expect-online-probe-adaptive-hold-profile "${expected_profile}" \
    --expect-online-probe-adaptive-hold-closure-fraction "${expected_closure}" \
    --min-support-foot-joint-count 8 \
    --min-support-foot-z-joint-count 4 \
    --min-support-foot-x-joint-motion 0.25 \
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

run_case vertical_probe vertical_micro_lift 0.0 0.030 medium reinforced_contact_closure 1.0 1
run_case horizontal_probe horizontal_push_pull 0.050 0.0 low light_contact_closure 0.75 0

python3 scripts/isaac/summarize_direct_carry_posture_suite.py \
  "${SUMMARY_ARGS[@]}" \
  --output "${SUITE_DIR}/online_probe_adaptive_hold_summary.json" \
  --min-postures 1 \
  --min-steps 3620 \
  --min-box-travel-x 0.52 \
  --max-target-distance-x 0.08 \
  --max-tilt 0.14 \
  --min-support-margin 0.12

{
  echo "suite_stamp=${SUITE_STAMP}"
  echo "summary=${SUITE_DIR}/online_probe_adaptive_hold_summary.json"
} > "${SUITE_DIR}/manifest.txt"

cat "${SUITE_DIR}/manifest.txt"
