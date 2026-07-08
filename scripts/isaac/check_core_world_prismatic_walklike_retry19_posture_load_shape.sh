#!/usr/bin/env bash
set -uo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run project Python checker on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

CHECK=(python3 scripts/isaac/check_prismatic_carrier_stand_summary.py)
ARGS=(
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
  --max-nonfinite-events 0
)

stamps=(
  20260705_prismatic_cradle_walklike_posture_load_mid_x050_z016_mass12_retry19a
  20260705_prismatic_cradle_walklike_posture_load_high_x050_z018_mass12_retry19b
  20260705_prismatic_cradle_walklike_posture_shape_mid_x050_z016_tall_retry19c
  20260705_prismatic_cradle_walklike_posture_shape_high_x050_z018_tall_retry19d
)

overall_status=0
for stamp in "${stamps[@]}"; do
  echo "===== ${stamp} ====="
  "${CHECK[@]}" \
    "experiments/outputs/core_world_prismatic_carrier_stand/${stamp}/core_world_prismatic_carrier_stand_summary.json" \
    --log "logs/core_world_prismatic_carrier_stand/core_world_prismatic_carrier_stand_${stamp}.log" \
    "${ARGS[@]}"
  status=$?
  echo "status=${status}"
  if [[ "${status}" -ne 0 ]]; then
    overall_status=1
  fi
done

exit "${overall_status}"
