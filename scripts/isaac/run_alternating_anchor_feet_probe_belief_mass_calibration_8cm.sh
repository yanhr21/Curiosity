#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

run_case() {
  local stamp="$1"
  local mass="$2"
  local max_final_dist="$3"

  STAMP="${stamp}" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE=front_mid \
  TARGET_X=0.08 \
  PAYLOAD_MASS="${mass}" \
  BOX_SEED=0 \
  RANDOMIZE_PAYLOAD=0 \
  PROBE_STEPS=60 \
  PROBE_X_AMPLITUDE=0.020 \
  BELIEF_COMPLIANCE_LOW_THRESHOLD=0.08 \
  BELIEF_COMPLIANCE_HIGH_THRESHOLD=0.22 \
  STEPS=860 \
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
  SUPPORT_FOOT_STANCE_X=-0.130 \
  SUPPORT_FOOT_SWING_X=0.130 \
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
  bash scripts/isaac/run_direct_carry_task_physical_backend.sh

  python3 scripts/isaac/check_direct_carry_task_summary.py \
    "experiments/outputs/direct_carry_task_physical_backend/${stamp}/direct_carry_task_physical_backend_summary.json" \
    --min-steps 840 \
    --expect-controller-mode physical_alternating_anchor_feet_cradle \
    --expect-backend-support-mode dynamic_anchor \
    --expect-support-foot-mode xz_prismatic_to_anchor \
    --min-support-foot-joint-count 8 \
    --min-support-foot-z-joint-count 4 \
    --min-support-foot-x-joint-motion 0.20 \
    --min-support-foot-z-joint-motion 0.08 \
    --min-actual-support-foot-lift 0.02 \
    --min-box-travel 0.055 \
    --max-final-box-target-distance-x "${max_final_dist}" \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --require-root-shortcut-free \
    --max-support-root-pose-write-count 0 \
    --max-anchor-world-joint-retarget-count 0 \
    --max-foot-pose-write-count 0 \
    --max-stance-anchor-pose-write-count 0 \
    --forbid-fixed-world-support \
    --require-non-success-claim
}

run_case 20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_8cm_6kg 6.0 0.050
run_case 20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_8cm_10kg 10.0 0.055

python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/outputs/direct_carry_task_physical_backend")
cases = [
    "20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_8cm_6kg",
    "20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_8cm_10kg",
]
rows = []
for case in cases:
    data = json.loads((root / case / "direct_carry_task_physical_backend_summary.json").read_text())
    rows.append({
        "case": case,
        "box_mass_kg": data.get("box_mass_kg"),
        "probe_compliance_proxy": data.get("probe_compliance_proxy"),
        "probe_lag_proxy": data.get("probe_lag_proxy"),
        "probe_risk_score": data.get("probe_risk_score"),
        "probe_load_risk_bucket": data.get("probe_load_risk_bucket"),
        "max_probe_box_relative_error_m": data.get("max_probe_box_relative_error_m"),
        "final_box_target_distance_x_m": data.get("final_box_target_distance_x_m"),
        "fall_events": data.get("fall_events"),
        "box_drop_events": data.get("box_drop_events"),
    })
out_dir = root / "20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_mass_calibration_8cm"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "mass_calibration_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
print(json.dumps(rows, indent=2, sort_keys=True))
PY
