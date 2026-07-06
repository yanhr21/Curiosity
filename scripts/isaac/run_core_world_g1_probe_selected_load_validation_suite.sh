#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 probe-selected load validation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_g1_probe_selected_load_validation)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_probe_selected_load_validation/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
MASSES_CSV="${MASSES_CSV:-0.25,0.50,0.75}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

python3 -m py_compile \
  scripts/isaac/select_core_world_g1_carry_posture_from_probe.py
bash -n scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh

IFS=',' read -r -a masses <<< "${MASSES_CSV}"
status_file="${SUITE_DIR}/probe_selected_load_validation_status.tsv"
printf "case\tmass_kg\tstatus\tpipeline_summary\n" > "${status_file}"

for mass in "${masses[@]}"; do
  label="mass$(printf '%s' "${mass}" | tr '.' 'p')"
  pipeline_stamp="${SUITE_STAMP}_${label}"
  output_root="${SUITE_DIR}/${label}"
  echo "[PROBE-LOAD] case=${label} mass=${mass} pipeline_stamp=${pipeline_stamp}"
  set +e
  DEVICE="${DEVICE}" \
  PIPELINE_STAMP="${pipeline_stamp}" \
  OUTPUT_ROOT="${output_root}" \
  FREE_BOX_MASS="${mass}" \
  PROBE_FREE_STEPS="${PROBE_FREE_STEPS:-260}" \
  PROBE_COLLISION_WINDOW="${PROBE_COLLISION_WINDOW:-1}" \
  PROBE_START_STEP="${PROBE_START_STEP:-40}" \
  PROBE_END_STEP="${PROBE_END_STEP:-80}" \
  PROBE_PAD_LOCAL_X="${PROBE_PAD_LOCAL_X:-0.42}" \
  PROBE_PAD_LOCAL_Y="${PROBE_PAD_LOCAL_Y:-0.0}" \
  PROBE_PAD_LOCAL_Z="${PROBE_PAD_LOCAL_Z:-0.02}" \
  PROBE_PAD_SIZE_X="${PROBE_PAD_SIZE_X:-0.025}" \
  PROBE_PAD_SIZE_Y="${PROBE_PAD_SIZE_Y:-0.20}" \
  PROBE_PAD_SIZE_Z="${PROBE_PAD_SIZE_Z:-0.10}" \
  PROBE_PAD_MASS="${PROBE_PAD_MASS:-0.05}" \
  MAX_PROBE_FALL_EVENTS="${MAX_PROBE_FALL_EVENTS:-0}" \
  MAX_PROBE_BOX_DROP_EVENTS="${MAX_PROBE_BOX_DROP_EVENTS:-0}" \
  MAX_PROBE_TILT="${MAX_PROBE_TILT:-0.85}" \
  MAX_PROBE_BOX_TILT="${MAX_PROBE_BOX_TILT:-0.95}" \
  MIN_PROBE_COMPLETED_STEPS="${MIN_PROBE_COMPLETED_STEPS:-${PROBE_FREE_STEPS:-260}}" \
  SELECTED_FREE_STEPS="${SELECTED_FREE_STEPS:-819}" \
  COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}" \
  bash scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh
  status=$?
  set -e
  summary="${output_root}/g1_probe_selected_pipeline_summary.json"
  printf "%s\t%s\t%s\t%s\n" "${label}" "${mass}" "${status}" "${summary}" >> "${status_file}"
done

python3 - <<PY
import json
from pathlib import Path

suite_dir = Path("${SUITE_DIR}")
cases = []
for line in (suite_dir / "probe_selected_load_validation_status.tsv").read_text().strip().splitlines()[1:]:
    label, mass, status, summary_path = line.split("\t")
    summary = None
    path = Path(summary_path)
    if path.is_file():
        summary = json.loads(path.read_text())
    selected_check = (summary or {}).get("selected_check") or {}
    selected_summary = (summary or {}).get("selected_summary") or {}
    probe_summary = (summary or {}).get("probe_summary") or {}
    selection_report = (summary or {}).get("selection_report") or {}
    cases.append({
        "label": label,
        "mass_kg": float(mass),
        "exit_status": int(status),
        "summary_path": summary_path,
        "selected_posture": (summary or {}).get("selected_posture"),
        "selection_uses_hidden_ground_truth": (summary or {}).get("selection_uses_hidden_ground_truth"),
        "selection_exit_status": (summary or {}).get("selection_exit_status"),
        "validation_skipped_reason": (summary or {}).get("validation_skipped_reason"),
        "probe_active_steps": probe_summary.get("probe_active_steps"),
        "probe_box_travel_m": probe_summary.get("final_probe_box_travel_m"),
        "probe_box_target_directed_travel_m": probe_summary.get("final_probe_box_target_directed_travel_m"),
        "selector_decision": selection_report.get("selected_posture"),
        "validation_status": selected_check.get("status"),
        "validation_failures": selected_check.get("failures", []),
        "completed_steps": selected_summary.get("completed_steps"),
        "fall_events": selected_summary.get("fall_events"),
        "box_drop_events": selected_summary.get("box_drop_events"),
        "final_robot_target_directed_travel_m": selected_summary.get("final_robot_target_directed_travel_m"),
        "final_box_target_directed_travel_m": selected_summary.get("final_box_target_directed_travel_m"),
        "target_window_both_streak_at_end_steps": selected_summary.get("target_window_both_streak_at_end_steps"),
        "rollout_write_count_total": sum(int(selected_summary.get(k) or 0) for k in (
            "root_pose_write_count_rollout",
            "root_velocity_write_count_rollout",
            "box_pose_write_count_rollout",
        )),
    })

passed = [
    case for case in cases
    if case["exit_status"] == 0
    and case.get("validation_status") == "pass"
    and int(case.get("fall_events") or 0) == 0
    and int(case.get("box_drop_events") or 0) == 0
    and int(case.get("rollout_write_count_total") or 0) == 0
]
report = {
    "scene_type": "core_world_g1_probe_selected_load_validation",
    "success_claim": "active_probe_selection_diagnostic_not_final_autonomous_carrying_success",
    "status": "pass" if len(passed) == len(cases) and cases else "fail",
    "suite_stamp": "${SUITE_STAMP}",
    "cases_total": len(cases),
    "cases_passed": len(passed),
    "cases": cases,
}
out = suite_dir / "probe_selected_load_validation_summary.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["status"] == "pass" else 1)
PY
