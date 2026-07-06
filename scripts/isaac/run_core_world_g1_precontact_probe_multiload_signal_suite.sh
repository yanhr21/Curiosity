#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 precontact probe multiload suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_g1_precontact_probe_multiload_signal)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_precontact_probe_multiload_signal/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
MASSES_CSV="${MASSES_CSV:-0.25,0.50,0.75}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

bash -n scripts/isaac/run_core_world_g1_safe_probe_signal_bracket.sh

status_file="${SUITE_DIR}/precontact_probe_multiload_status.tsv"
printf "label\tmass_kg\tstatus\tsummary_path\n" > "${status_file}"

IFS=',' read -r -a masses <<< "${MASSES_CSV}"
for mass in "${masses[@]}"; do
  label="mass$(printf '%s' "${mass}" | tr '.' 'p')"
  case_stamp="${SUITE_STAMP}_${label}"
  summary_path="${ROOT_DIR}/experiments/outputs/core_world_g1_safe_probe_signal_bracket/${case_stamp}/safe_probe_signal_bracket_summary.json"
  echo "[PRECONTACT-PROBE-LOAD] label=${label} mass=${mass} stamp=${case_stamp}"
  set +e
  DEVICE="${DEVICE}" \
  SUITE_STAMP="${case_stamp}" \
  PROBE_COLLISION_WINDOW_MODE=0 \
  PROBE_TEST_MASS="${mass}" \
  CASE_SPECS="${CASE_SPECS:-small_x042:0.42:0.020:0.18:40:120:0.03}" \
  COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
  bash scripts/isaac/run_core_world_g1_safe_probe_signal_bracket.sh
  status=$?
  set -e
  printf "%s\t%s\t%s\t%s\n" "${label}" "${mass}" "${status}" "${summary_path}" >> "${status_file}"
done

python3 - <<PY
import json
from pathlib import Path

suite_dir = Path("${SUITE_DIR}")
cases = []
for line in (suite_dir / "precontact_probe_multiload_status.tsv").read_text().strip().splitlines()[1:]:
    label, mass, status, summary_path = line.split("\t")
    path = Path(summary_path)
    report = json.loads(path.read_text()) if path.is_file() else {}
    inner = (report.get("cases") or [{}])[0]
    cases.append({
        "label": label,
        "mass_kg": float(mass),
        "exit_status": int(status),
        "summary_path": summary_path,
        "suite_status": report.get("status"),
        "safe_signal_cases": report.get("safe_signal_cases", []),
        "case_label": inner.get("label"),
        "completed_steps": inner.get("completed_steps"),
        "summary_error": inner.get("summary_error"),
        "probe_active_steps": inner.get("probe_active_steps"),
        "probe_box_moved": inner.get("probe_box_moved"),
        "max_probe_box_target_directed_travel_m": inner.get("max_probe_box_target_directed_travel_m"),
        "final_probe_box_target_directed_travel_m": inner.get("final_probe_box_target_directed_travel_m"),
        "max_probe_box_travel_xy_m": inner.get("max_probe_box_travel_xy_m"),
        "fall_events": inner.get("fall_events"),
        "box_drop_events": inner.get("box_drop_events"),
        "max_tilt_rad": inner.get("max_tilt_rad"),
        "max_box_tilt_rad": inner.get("max_box_tilt_rad"),
        "root_pose_write_count_rollout": inner.get("root_pose_write_count_rollout"),
        "root_velocity_write_count_rollout": inner.get("root_velocity_write_count_rollout"),
        "box_pose_write_count_rollout": inner.get("box_pose_write_count_rollout"),
    })

passed = [
    case for case in cases
    if case["exit_status"] == 0
    and case.get("suite_status") == "pass"
    and case.get("summary_error") is None
    and int(case.get("fall_events") or 0) == 0
    and int(case.get("box_drop_events") or 0) == 0
    and sum(int(case.get(k) or 0) for k in (
        "root_pose_write_count_rollout",
        "root_velocity_write_count_rollout",
        "box_pose_write_count_rollout",
    )) == 0
]
motions = [float(case.get("max_probe_box_target_directed_travel_m") or 0.0) for case in cases]
report = {
    "scene_type": "core_world_g1_precontact_probe_multiload_signal",
    "success_claim": "probe_signal_multiload_diagnostic_not_unknown_load_adaptation_or_carrying_success",
    "suite_stamp": "${SUITE_STAMP}",
    "status": "pass" if len(passed) == len(cases) and cases else "fail",
    "cases_total": len(cases),
    "cases_passed": len(passed),
    "motion_range_m": (max(motions) - min(motions)) if motions else None,
    "cases": cases,
}
out = suite_dir / "precontact_probe_multiload_signal_summary.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["status"] == "pass" else 1)
PY
