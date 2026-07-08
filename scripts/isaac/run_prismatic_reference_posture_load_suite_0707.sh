#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-20260707_prismatic_reference_posture_load_suite}"
SUITE_DIR="${ROOT_DIR}/experiments/outputs/core_world_prismatic_carrier_stand/${SUITE_STAMP}"
SUMMARY_JSON="${SUITE_DIR}/prismatic_reference_posture_load_suite_summary.json"
mkdir -p "${SUITE_DIR}"

run_case() {
  local label="$1"
  shift
  local case_stamp="${SUITE_STAMP}_${label}"
  local case_dir="${ROOT_DIR}/experiments/outputs/core_world_prismatic_carrier_stand/${case_stamp}"
  local summary="${case_dir}/core_world_prismatic_carrier_stand_summary.json"
  local check="${case_dir}/reference_check_corrected.json"
  local status="pass"

  echo "[SUITE] Running ${label}"
  (
    export STAMP="${case_stamp}"
    export STEPS="${STEPS:-2880}"
    export FOOT_LENGTH="${FOOT_LENGTH:-0.65}"
    export GATED_STEP_MAX_TRAVEL_LOSS="${GATED_STEP_MAX_TRAVEL_LOSS:-0.04}"
    export GATED_STEP_RECOVERY_PHASE="${GATED_STEP_RECOVERY_PHASE:-0.35}"
    export GUARDED_STEP_TARGET_TOLERANCE="${GUARDED_STEP_TARGET_TOLERANCE:-0.03}"
    while (($#)); do
      export "$1"
      shift
    done
    bash "${ROOT_DIR}/scripts/isaac/run_prismatic_reference_cpu_validation_0707.sh"
  ) || status="fail"

  if [[ -s "${summary}" && ! -s "${check}" ]]; then
    python3 "${ROOT_DIR}/scripts/isaac/check_prismatic_carrier_stand_summary.py" \
      "${summary}" \
      --expect-payload-mode cradle_free_box \
      --expect-motion-mode guarded_prelift_quasistatic_step_cycle \
      --require-articulated-carrier \
      --require-foot-contact-drive \
      --require-active-probe \
      --require-probe-belief \
      --require-no-hidden-probe-gt \
      --min-active-probe-steps 80 \
      --require-probe-adaptive-gait-decision \
      --require-probe-adaptive-posture-decision \
      --max-fall-events 0 \
      --max-box-drop-events 0 \
      --max-root-pose-writes 0 \
      --max-root-velocity-writes 0 \
      --max-body-root-pose-writes 0 \
      --max-body-root-velocity-commands 0 \
      --max-box-pose-writes 0 \
      --min-abs-post-settle-payload-travel-x 0.15 \
      --max-final-post-settle-payload-target-distance-x 0.025 \
      --max-payload-relative-offset-error 0.08 \
      --max-post-settle-payload-relative-offset-error 0.018 \
      --min-payload-z 0.45 \
      --max-tilt 0.24 \
      > "${check}" || status="fail"
  fi

  echo "${label}|${status}|${summary}|${check}" >> "${SUITE_DIR}/case_status.tsv"
}

: > "${SUITE_DIR}/case_status.tsv"
run_case "mid_10kg_nominal" \
  "PAYLOAD_MASS=10.0" "PAYLOAD_SIZE_X=0.34" "PAYLOAD_SIZE_Y=0.24" "PAYLOAD_SIZE_Z=0.24" \
  "PAYLOAD_LOCAL_X=0.50" "PAYLOAD_LOCAL_Z=0.16"
run_case "near_chest_12kg_high" \
  "PAYLOAD_MASS=12.0" "PAYLOAD_SIZE_X=0.34" "PAYLOAD_SIZE_Y=0.24" "PAYLOAD_SIZE_Z=0.24" \
  "PAYLOAD_LOCAL_X=0.38" "PAYLOAD_LOCAL_Z=0.22" "LEG_TARGET=-0.59"
run_case "long_reach_8kg_low" \
  "PAYLOAD_MASS=8.0" "PAYLOAD_SIZE_X=0.34" "PAYLOAD_SIZE_Y=0.24" "PAYLOAD_SIZE_Z=0.24" \
  "PAYLOAD_LOCAL_X=0.62" "PAYLOAD_LOCAL_Z=0.12" "LEG_TARGET=-0.56"
run_case "bulky_10kg_mid" \
  "PAYLOAD_MASS=10.0" "PAYLOAD_SIZE_X=0.44" "PAYLOAD_SIZE_Y=0.30" "PAYLOAD_SIZE_Z=0.28" \
  "PAYLOAD_LOCAL_X=0.54" "PAYLOAD_LOCAL_Z=0.18" "CRADLE_CLEARANCE_X=0.035" "CRADLE_CLEARANCE_Y=0.050"

python3 - "${SUITE_DIR}/case_status.tsv" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
cases = []
for line in status_path.read_text().splitlines():
    label, shell_status, summary_path, check_path = line.split("|")
    item = {
        "label": label,
        "shell_status": shell_status,
        "summary": summary_path,
        "check": check_path,
        "check_status": "missing",
        "failures": ["missing check"],
    }
    check = Path(check_path)
    if check.is_file() and check.stat().st_size > 0:
        report = json.loads(check.read_text())
        item.update(
            {
                "check_status": report.get("status"),
                "failures": report.get("failures", []),
                "completed_steps": report.get("completed_steps"),
                "fall_events": report.get("fall_events"),
                "box_drop_events": report.get("box_drop_events"),
                "final_post_settle_payload_target_distance_x_m": report.get(
                    "final_post_settle_payload_target_distance_x_m"
                ),
                "max_abs_post_settle_payload_travel_x_m": report.get(
                    "max_abs_post_settle_payload_travel_x_m"
                ),
                "max_post_settle_payload_relative_offset_error_m": report.get(
                    "max_post_settle_payload_relative_offset_error_m"
                ),
                "max_tilt_rad": report.get("max_tilt_rad"),
                "active_probe_observed_load_risk_bucket": report.get(
                    "active_probe_observed_load_risk_bucket"
                ),
                "probe_adaptive_posture_strategy": report.get(
                    "probe_adaptive_posture_strategy"
                ),
            }
        )
    cases.append(item)
status = "pass" if cases and all(c["check_status"] == "pass" for c in cases) else "fail"
out = {
    "scene_type": "prismatic_reference_posture_load_suite",
    "success_claim": "multi_case_prismatic_scaffold_only_not_humanoid_or_final_success",
    "status": status,
    "case_count": len(cases),
    "passing_case_count": sum(c["check_status"] == "pass" for c in cases),
    "cases": cases,
}
out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if status == "pass" else 1)
PY
