#!/usr/bin/env bash
set -uo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run project Python checker on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

CHECK=(python3 scripts/isaac/check_prismatic_carrier_stand_summary.py)
COMMON_ARGS=(
  --expect-motion-mode guarded_prelift_quasistatic_step_cycle
  --expect-payload-mode cradle_free_box
  --max-fall-events 0
  --max-box-drop-events 0
  --max-root-pose-writes 0
  --max-root-velocity-writes 0
  --max-root-angular-velocity-writes 0
  --max-body-root-pose-writes 0
  --max-body-root-velocity-commands 0
  --max-box-pose-writes 0
  --max-payload-pose-writes 0
  --max-tilt 0.13
  --min-payload-z 0.70
  --max-payload-relative-offset-error 0.12
  --min-abs-post-settle-payload-travel-x 0.15
  --max-final-post-settle-payload-target-distance-x 0.03
  --min-joint-count 8
  --min-joint-motion 0.25
  --require-articulated-carrier
  --require-foot-contact-drive
  --require-active-probe
  --require-probe-belief
  --require-no-hidden-probe-gt
  --min-active-probe-steps 60
  --require-probe-adaptive-gait-decision
  --max-nonfinite-events 0
)

overall_status=0

run_check() {
  local stamp="$1"
  local expected_bucket="$2"
  local expected_scale="$3"
  echo "===== ${stamp} ====="
  "${CHECK[@]}" \
    "experiments/outputs/core_world_prismatic_carrier_stand/${stamp}/core_world_prismatic_carrier_stand_summary.json" \
    --log "logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_${stamp}.log" \
    "${COMMON_ARGS[@]}" \
    --expect-probe-adaptive-risk-bucket "${expected_bucket}" \
    --expect-probe-adaptive-gait-drive-scale "${expected_scale}"
  local status=$?
  echo "status=${status}"
  if [[ "${status}" -ne 0 ]]; then
    overall_status=1
  fi
}

run_check 20260706_prismatic_cradle_probe_adaptive_standard10_mid_retry23a low 1.0
run_check 20260706_prismatic_cradle_probe_adaptive_tall10_mid_retry23b medium 0.98

exit "${overall_status}"
