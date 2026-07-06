#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 safe-probe signal bracket on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-$(date +%Y%m%d_g1_safe_probe_signal_bracket)}"
SUITE_DIR="${SUITE_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_safe_probe_signal_bracket/${SUITE_STAMP}}"
DEVICE="${DEVICE:-cpu}"
PROBE_TEST_MASS="${PROBE_TEST_MASS:-0.50}"
MIN_SAFE_SIGNAL_M="${MIN_SAFE_SIGNAL_M:-0.002}"
MAX_SAFE_PROBE_TILT="${MAX_SAFE_PROBE_TILT:-0.95}"
MAX_SAFE_PROBE_BOX_TILT="${MAX_SAFE_PROBE_BOX_TILT:-1.05}"
COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP:-5}"
PROBE_COLLISION_WINDOW_MODE="${PROBE_COLLISION_WINDOW_MODE:-1}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

python3 -m py_compile scripts/isaac/build_core_world_g1_box_scene.py
bash -n scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh

status_file="${SUITE_DIR}/safe_probe_signal_bracket_status.tsv"
printf "label\tstatus\tsummary_path\n" > "${status_file}"

case_specs="${CASE_SPECS:-base_end80:0.42:0.025:0.20:40:80:0.05
end120:0.42:0.025:0.20:40:120:0.05
x046_end120:0.46:0.030:0.22:40:120:0.05
x048_wide_end120:0.48:0.040:0.26:40:120:0.06}"

while IFS=: read -r label pad_x pad_size_x pad_size_y start_step end_step pad_mass; do
  [[ -n "${label}" ]] || continue
  case_stamp="${SUITE_STAMP}_${label}"
  summary_path="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${case_stamp}/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json"
  echo "[SAFE-PROBE-BRACKET] label=${label} pad_x=${pad_x} size_x=${pad_size_x} size_y=${pad_size_y} window=${start_step}-${end_step}"
  set +e
  DEVICE="${DEVICE}" \
  SUITE_STAMP="${case_stamp}" \
  LARGERBOX_STRICT_MODE=lowcarry \
  STRICT=0 \
  RUN_NOBOX=0 \
  RUN_FIXED=0 \
  RUN_FREE=1 \
  FREE_STEPS="${PROBE_FREE_STEPS:-180}" \
  FREE_BOX_MASS="${PROBE_TEST_MASS}" \
  FREE_MIN_ROBOT_TRAVEL=0.0 \
  FREE_MIN_BOX_TRAVEL=0.0 \
  FREE_MAX_TILT=2.0 \
  FREE_MAX_BOX_TILT=2.0 \
  FREE_MAX_FINAL_REL=2.0 \
  TARGET_WINDOW_CENTER=-1.0 \
  TARGET_WINDOW_HALFWIDTH=-1.0 \
  PROBE_MODE=front_bumper \
  PROBE_COLLISION_WINDOW="${PROBE_COLLISION_WINDOW_MODE}" \
  PROBE_START_STEP="${start_step}" \
  PROBE_END_STEP="${end_step}" \
  PROBE_PAD_LOCAL_X="${pad_x}" \
  PROBE_PAD_LOCAL_Y=0.0 \
  PROBE_PAD_LOCAL_Z=0.02 \
  PROBE_PAD_SIZE_X="${pad_size_x}" \
  PROBE_PAD_SIZE_Y="${pad_size_y}" \
  PROBE_PAD_SIZE_Z=0.10 \
  PROBE_PAD_MASS="${pad_mass}" \
  COMPUTE_SIDE_STARTUP_SLEEP="${COMPUTE_SIDE_STARTUP_SLEEP}" \
  bash scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh
  status=$?
  set -e
  printf "%s\t%s\t%s\n" "${label}" "${status}" "${summary_path}" >> "${status_file}"
done <<< "${case_specs}"

python3 - <<PY
import json
from pathlib import Path

suite_dir = Path("${SUITE_DIR}")
cases = []
safe_signal_cases = []
for line in (suite_dir / "safe_probe_signal_bracket_status.tsv").read_text().strip().splitlines()[1:]:
    label, status, summary_path = line.split("\t")
    path = Path(summary_path)
    summary = json.loads(path.read_text()) if path.is_file() else {}
    case = {
        "label": label,
        "exit_status": int(status),
        "summary_path": summary_path,
        "summary_exists": path.is_file(),
        "summary_status": summary.get("status"),
        "summary_error": summary.get("error"),
        "completed_steps": summary.get("completed_steps"),
        "probe_collision_window_enabled": summary.get("probe_collision_window_enabled"),
        "probe_collision_enabled_step": summary.get("probe_collision_enabled_step"),
        "probe_collision_disabled_step": summary.get("probe_collision_disabled_step"),
        "probe_collision_update_count": summary.get("probe_collision_update_count"),
        "probe_collision_update_error": summary.get("probe_collision_update_error"),
        "probe_active_steps": summary.get("probe_active_steps"),
        "probe_box_moved": summary.get("probe_box_moved"),
        "final_probe_box_travel_xy_m": summary.get("final_probe_box_travel_xy_m"),
        "max_probe_box_travel_xy_m": summary.get("max_probe_box_travel_xy_m"),
        "final_probe_box_target_directed_travel_m": summary.get("final_probe_box_target_directed_travel_m"),
        "max_probe_box_target_directed_travel_m": summary.get("max_probe_box_target_directed_travel_m"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "max_box_tilt_rad": summary.get("max_box_tilt_rad"),
        "root_pose_write_count_rollout": summary.get("root_pose_write_count_rollout"),
        "root_velocity_write_count_rollout": summary.get("root_velocity_write_count_rollout"),
        "box_pose_write_count_rollout": summary.get("box_pose_write_count_rollout"),
    }
    cases.append(case)
    if (
        path.is_file()
        and summary.get("error") is None
        and int(case.get("completed_steps") or 0) >= int("${PROBE_FREE_STEPS:-180}")
        and int(case.get("fall_events") or 0) == 0
        and int(case.get("box_drop_events") or 0) == 0
        and float(case.get("max_tilt_rad") or 0.0) <= float("${MAX_SAFE_PROBE_TILT}")
        and float(case.get("max_box_tilt_rad") or 0.0) <= float("${MAX_SAFE_PROBE_BOX_TILT}")
        and float(case.get("max_probe_box_target_directed_travel_m") or 0.0) >= float("${MIN_SAFE_SIGNAL_M}")
        and sum(int(case.get(k) or 0) for k in (
            "root_pose_write_count_rollout",
            "root_velocity_write_count_rollout",
            "box_pose_write_count_rollout",
        )) == 0
    ):
        safe_signal_cases.append(case["label"])

report = {
    "scene_type": "core_world_g1_safe_probe_signal_bracket",
    "success_claim": "probe_signal_diagnostic_not_unknown_load_adaptation_or_carrying_success",
    "suite_stamp": "${SUITE_STAMP}",
    "probe_test_mass_kg": float("${PROBE_TEST_MASS}"),
    "probe_collision_window_mode": int("${PROBE_COLLISION_WINDOW_MODE}"),
    "min_safe_signal_m": float("${MIN_SAFE_SIGNAL_M}"),
    "max_safe_probe_tilt": float("${MAX_SAFE_PROBE_TILT}"),
    "max_safe_probe_box_tilt": float("${MAX_SAFE_PROBE_BOX_TILT}"),
    "status": "pass" if safe_signal_cases else "fail",
    "safe_signal_cases": safe_signal_cases,
    "cases": cases,
}
out = suite_dir / "safe_probe_signal_bracket_summary.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if safe_signal_cases else 1)
PY
