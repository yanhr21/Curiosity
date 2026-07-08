#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run task-runner checking/export on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-20260705_direct_carry_task_runner_check}"
SUMMARY="${SUMMARY:?Set SUMMARY to a direct_carry_task_physical_backend_summary.json path}"
CARRY_POSTURE="${CARRY_POSTURE:-front_mid}"
BOX_SEED="${BOX_SEED:-7078}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/direct_carry_task_runner_checks/${STAMP}}"
CHECK_REPORT="${CHECK_REPORT:-${OUTPUT_DIR}/direct_carry_task_runner_check.json}"
EPISODE_TABLE="${EPISODE_TABLE:-${OUTPUT_DIR}/direct_carry_task_runner_episode_table.jsonl}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

PROBE_ARGS=()
if [[ "${REQUIRE_PROBE_BELIEF:-0}" == "1" ]]; then
  PROBE_ARGS=(
    --min-probe-steps "${MIN_PROBE_STEPS:-1}"
    --require-probe-belief
    --forbid-probe-hidden-ground-truth
    --min-probe-box-travel-x "${MIN_PROBE_BOX_TRAVEL_X:-0.001}"
  )
fi

PLACEMENT_ARGS=()
if [[ "${REQUIRE_DIRECTIONAL_FOOT_PLACEMENT:-0}" == "1" ]]; then
  PLACEMENT_ARGS=(
    --expect-support-foot-placement-mode "${SUPPORT_FOOT_PLACEMENT_MODE:-alternating_directional_x}"
    --require-support-foot-placement-controller
    --require-directional-foot-placement
  )
fi

ABS_TRAVEL_ARGS=()
if [[ -n "${MIN_ABS_BOX_TRAVEL_X:-}" ]]; then
  ABS_TRAVEL_ARGS+=(--min-abs-box-travel-x "${MIN_ABS_BOX_TRAVEL_X}")
fi
if [[ -n "${MIN_ABS_POST_SETTLE_BOX_TRAVEL_X:-}" ]]; then
  ABS_TRAVEL_ARGS+=(--min-abs-post-settle-box-travel-x "${MIN_ABS_POST_SETTLE_BOX_TRAVEL_X}")
fi

FOOT_SLIP_AUDIT_ARGS=()
if [[ -n "${MAX_NEAR_GROUND_FOOT_SPEED:-}" ]]; then
  FOOT_SLIP_AUDIT_ARGS+=(--max-near-ground-foot-speed "${MAX_NEAR_GROUND_FOOT_SPEED}")
fi
if [[ -n "${MAX_NEAR_GROUND_FOOT_SLIP:-}" ]]; then
  FOOT_SLIP_AUDIT_ARGS+=(--max-near-ground-foot-slip "${MAX_NEAR_GROUND_FOOT_SLIP}")
fi

FIXED_WORLD_SUPPORT_ARGS=(--forbid-fixed-world-support)
if [[ "${REQUIRE_STANCE_FOOT_WORLD_LOCK:-0}" == "1" || "${REQUIRE_STANCE_FOOT_WORLD_LOCK:-false}" == "true" ]]; then
  FIXED_WORLD_SUPPORT_ARGS=(
    --require-stance-foot-world-lock
    --min-stance-foot-world-lock-switches "${MIN_STANCE_FOOT_WORLD_LOCK_SWITCHES:-1}"
  )
  if [[ "${REQUIRE_FREEZE_LOCKED_STANCE_FOOT_TARGETS:-0}" == "1" || "${REQUIRE_FREEZE_LOCKED_STANCE_FOOT_TARGETS:-false}" == "true" ]]; then
    FIXED_WORLD_SUPPORT_ARGS+=(--require-freeze-locked-stance-foot-targets)
  fi
  if [[ "${REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS:-0}" == "1" || "${REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS:-false}" == "true" ]]; then
    FIXED_WORLD_SUPPORT_ARGS+=(--require-freeze-commanded-stance-foot-targets)
  fi
  if [[ "${REQUIRE_PLANTED_STANCE_RAIL_PROPULSION:-0}" == "1" || "${REQUIRE_PLANTED_STANCE_RAIL_PROPULSION:-false}" == "true" ]]; then
    FIXED_WORLD_SUPPORT_ARGS+=(--require-planted-stance-rail-propulsion)
  fi
elif [[ "${REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS:-0}" == "1" || "${REQUIRE_FREEZE_COMMANDED_STANCE_FOOT_TARGETS:-false}" == "true" ]]; then
  FIXED_WORLD_SUPPORT_ARGS+=(--require-freeze-commanded-stance-foot-targets)
fi
if [[ ! ( "${REQUIRE_STANCE_FOOT_WORLD_LOCK:-0}" == "1" || "${REQUIRE_STANCE_FOOT_WORLD_LOCK:-false}" == "true" ) ]]; then
  if [[ "${REQUIRE_PLANTED_STANCE_RAIL_PROPULSION:-0}" == "1" || "${REQUIRE_PLANTED_STANCE_RAIL_PROPULSION:-false}" == "true" ]]; then
    FIXED_WORLD_SUPPORT_ARGS+=(--require-planted-stance-rail-propulsion)
  fi
fi

"${ISAAC_VENV}/bin/python" scripts/isaac/check_direct_carry_task_summary.py \
  "${SUMMARY}" \
  --min-steps "${MIN_STEPS:-3580}" \
  --expect-controller-mode "${CONTROLLER_MODE:-physical_alternating_anchor_feet_cradle}" \
  --expect-carry-posture "${CARRY_POSTURE}" \
  --expect-backend-support-mode "dynamic_anchor" \
  --require-box-randomized \
  --expect-box-seed "${BOX_SEED}" \
  "${PROBE_ARGS[@]}" \
  --min-box-travel "${MIN_BOX_TRAVEL:-0.50}" \
  "${ABS_TRAVEL_ARGS[@]}" \
  --min-post-settle-box-travel-x "${MIN_POST_SETTLE_BOX_TRAVEL_X:-0.50}" \
  --max-final-post-settle-box-target-distance-x "${MAX_FINAL_POST_SETTLE_BOX_TARGET_DISTANCE_X:-0.20}" \
  --max-post-settle-box-travel-loss-after-peak "${MAX_POST_SETTLE_BOX_TRAVEL_LOSS_AFTER_PEAK:-0.05}" \
  --max-fall-events "${MAX_FALL_EVENTS:-0}" \
  --max-box-drop-events "${MAX_BOX_DROP_EVENTS:-0}" \
  --require-root-shortcut-free \
  --max-anchor-world-joint-retarget-count 0 \
  --max-support-root-pose-write-count 0 \
  --max-foot-pose-write-count 0 \
  --max-stance-anchor-pose-write-count 0 \
  --expect-support-foot-mode "xz_prismatic_to_anchor" \
  "${PLACEMENT_ARGS[@]}" \
  --require-feedback-step-controller \
  --min-feedback-step-applied-steps "${MIN_FEEDBACK_STEP_APPLIED_STEPS:-100}" \
  --max-rail-joint-motion "${MAX_RAIL_JOINT_MOTION:-0.025}" \
  --min-support-foot-joint-count 8 \
  --min-support-foot-x-joint-motion "${MIN_SUPPORT_FOOT_X_JOINT_MOTION:-0.20}" \
  --min-support-foot-z-joint-count 4 \
  --min-support-foot-z-joint-motion "${MIN_SUPPORT_FOOT_Z_JOINT_MOTION:-0.04}" \
  --min-actual-support-foot-lift "${MIN_ACTUAL_SUPPORT_FOOT_LIFT:-0.03}" \
  --min-drive-near-ground-foot-count "${MIN_DRIVE_NEAR_GROUND_FOOT_COUNT:-2}" \
  --max-drive-near-ground-lt2-steps "${MAX_DRIVE_NEAR_GROUND_LT2_STEPS:-0}" \
  --require-support-foot-contact-report-evidence \
  --min-drive-contact-report-foot-count "${MIN_DRIVE_CONTACT_REPORT_FOOT_COUNT:-2}" \
  --max-drive-contact-report-lt2-steps "${MAX_DRIVE_CONTACT_REPORT_LT2_STEPS:-0}" \
  --min-commanded-stance-contact-report-foot-count "${MIN_COMMANDED_STANCE_CONTACT_REPORT_FOOT_COUNT:-2}" \
  --max-commanded-stance-contact-report-lt2-steps "${MAX_COMMANDED_STANCE_CONTACT_REPORT_LT2_STEPS:-0}" \
  --require-support-foot-effort-evidence \
  --min-drive-effort-supported-foot-count "${MIN_DRIVE_EFFORT_SUPPORTED_FOOT_COUNT:-2}" \
  --max-drive-effort-supported-lt2-steps "${MAX_DRIVE_EFFORT_SUPPORTED_LT2_STEPS:-0}" \
  --min-commanded-stance-effort-supported-foot-count "${MIN_COMMANDED_STANCE_EFFORT_SUPPORTED_FOOT_COUNT:-2}" \
  --max-commanded-stance-effort-supported-lt2-steps "${MAX_COMMANDED_STANCE_EFFORT_SUPPORTED_LT2_STEPS:-0}" \
  --min-commanded-stance-near-ground-foot-count "${MIN_COMMANDED_STANCE_NEAR_GROUND_FOOT_COUNT:-2}" \
  --max-commanded-stance-near-ground-lt2-steps "${MAX_COMMANDED_STANCE_NEAR_GROUND_LT2_STEPS:-0}" \
  --min-support-polygon-margin "${MIN_SUPPORT_POLYGON_MARGIN:-0.0}" \
  "${FOOT_SLIP_AUDIT_ARGS[@]}" \
  --max-abs-anchor-travel-x "${MAX_ABS_ANCHOR_TRAVEL_X:-0.80}" \
  --max-abs-support-foot-travel-x "${MAX_ABS_SUPPORT_FOOT_TRAVEL_X:-0.90}" \
  "${FIXED_WORLD_SUPPORT_ARGS[@]}" \
  --require-non-success-claim \
  > "${CHECK_REPORT}"

"${ISAAC_VENV}/bin/python" scripts/isaac/export_direct_carry_task_episode_table.py \
  --summary "${SUMMARY}" \
  --output "${EPISODE_TABLE}"

cat "${CHECK_REPORT}"
echo "[INFO] Task-runner check report: ${CHECK_REPORT}"
echo "[INFO] Task-runner episode table: ${EPISODE_TABLE}"
