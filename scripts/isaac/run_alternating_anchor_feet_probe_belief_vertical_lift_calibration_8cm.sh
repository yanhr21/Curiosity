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

  STAMP="${stamp}" \
  SUPPORT_MODE=alternating_anchor_feet \
  CARRY_POSTURE=front_mid \
  TARGET_X=0.08 \
  PAYLOAD_MASS="${mass}" \
  BOX_SEED=0 \
  RANDOMIZE_PAYLOAD=0 \
  PROBE_STEPS=60 \
  PROBE_MODE=vertical_micro_lift \
  PROBE_X_AMPLITUDE=0.0 \
  PROBE_Z_AMPLITUDE=0.030 \
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
  SUPPORT_FOOT_Z_LOWER=-0.08 \
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
  bash scripts/isaac/run_direct_carry_task_physical_backend.sh \
    --probe-mode vertical_micro_lift \
    --probe-x-amplitude 0.0 \
    --probe-z-amplitude 0.030

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
    --max-final-box-target-distance-x 0.055 \
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

run_case 20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_vertical_lift_8cm_6kg 6.0
run_case 20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_vertical_lift_8cm_10kg 10.0

python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/outputs/direct_carry_task_physical_backend")
cases = [
    "20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_vertical_lift_8cm_6kg",
    "20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_vertical_lift_8cm_10kg",
]
rows = []
for case in cases:
    data = json.loads((root / case / "direct_carry_task_physical_backend_summary.json").read_text())
    rows.append({
        "case": case,
        "box_mass_kg": data.get("box_mass_kg"),
        "probe_mode": data.get("probe_mode"),
        "probe_z_amplitude_m": data.get("probe_z_amplitude_m"),
        "probe_joint_effort_available": data.get("probe_joint_effort_available"),
        "probe_joint_effort_read_error_count": data.get("probe_joint_effort_read_error_count"),
        "probe_joint_effort_first_error": data.get("probe_joint_effort_first_error"),
        "max_probe_torso_travel_z_m": data.get("max_probe_torso_travel_z_m"),
        "max_probe_box_travel_z_m": data.get("max_probe_box_travel_z_m"),
        "final_probe_box_lag_z_m": data.get("final_probe_box_lag_z_m"),
        "max_probe_support_foot_z_tracking_error_m": data.get("max_probe_support_foot_z_tracking_error_m"),
        "mean_probe_support_foot_z_tracking_error_m": data.get("mean_probe_support_foot_z_tracking_error_m"),
        "probe_support_foot_z_tracking_proxy": data.get("probe_support_foot_z_tracking_proxy"),
        "max_probe_support_foot_z_measured_effort": data.get("max_probe_support_foot_z_measured_effort"),
        "mean_probe_support_foot_z_measured_effort": data.get("mean_probe_support_foot_z_measured_effort"),
        "probe_support_foot_z_effort_proxy": data.get("probe_support_foot_z_effort_proxy"),
        "probe_compliance_proxy": data.get("probe_compliance_proxy"),
        "probe_lag_proxy": data.get("probe_lag_proxy"),
        "probe_risk_score": data.get("probe_risk_score"),
        "probe_load_risk_bucket": data.get("probe_load_risk_bucket"),
        "final_box_target_distance_x_m": data.get("final_box_target_distance_x_m"),
        "fall_events": data.get("fall_events"),
        "box_drop_events": data.get("box_drop_events"),
    })
out_dir = root / "20260705_direct_physical_backend_alternating_anchor_feet_probe_belief_vertical_lift_calibration_8cm"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "vertical_lift_calibration_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
print(json.dumps(rows, indent=2, sort_keys=True))
PY
