#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 low-carry post-rollsign decision on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
cd "${ROOT_DIR}"

ROLLSIGN_STAMP="${ROLLSIGN_STAMP:-20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_mild_rollpos_targethold819_strict_targetnegx1}"
BASELINE_STAMP="${BASELINE_STAMP:-20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_mild_targethold819_strict_targetnegx1}"
FOLLOWUP_PREFIX="${FOLLOWUP_PREFIX:-$(date +%Y%m%d_g1_lowcarry_after_rollsign)}"

ROLLSIGN_CHECK="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${ROLLSIGN_STAMP}/agile_low_cradle_freebox_walk/check.json"
BASELINE_CHECK="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${BASELINE_STAMP}/agile_low_cradle_freebox_walk/check.json"
DECISION_DIR="${ROOT_DIR}/experiments/reports/g1_lowcarry_after_rollsign"
mkdir -p "${DECISION_DIR}"
DECISION_JSON="${DECISION_DIR}/${FOLLOWUP_PREFIX}_decision.json"

CASE_SET="$(
  python3 - "${ROLLSIGN_CHECK}" "${BASELINE_CHECK}" "${DECISION_JSON}" <<'PY'
import json
import math
import sys
from pathlib import Path

rollsign_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
decision_path = Path(sys.argv[3])


def load(path: Path) -> dict:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_missing"] = False
    data["_path"] = str(path)
    return data


rollsign = load(rollsign_path)
baseline = load(baseline_path)


def num(data: dict, key: str, default: float = math.inf) -> float:
    value = data.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def abs_num(data: dict, key: str, default: float = math.inf) -> float:
    return abs(num(data, key, default))


if rollsign.get("_missing") or baseline.get("_missing"):
    case = "contact"
    reason = "missing_check_json"
else:
    base_falls = num(baseline, "fall_events")
    base_drops = num(baseline, "box_drop_events")
    base_robot_lat = abs_num(baseline, "final_robot_target_lateral_error_m")
    base_box_lat = abs_num(baseline, "final_box_target_lateral_error_m")
    base_rel = num(baseline, "final_box_robot_relative_offset_error_m")
    base_tilt = num(baseline, "max_tilt_rad")
    base_box_tilt = num(baseline, "max_box_tilt_rad")

    roll_falls = num(rollsign, "fall_events")
    roll_drops = num(rollsign, "box_drop_events")
    roll_robot_lat = abs_num(rollsign, "final_robot_target_lateral_error_m")
    roll_box_lat = abs_num(rollsign, "final_box_target_lateral_error_m")
    roll_rel = num(rollsign, "final_box_robot_relative_offset_error_m")
    roll_tilt = num(rollsign, "max_tilt_rad")
    roll_box_tilt = num(rollsign, "max_box_tilt_rad")

    retention_ok = roll_drops <= base_drops
    falls_not_worse = roll_falls <= base_falls
    lateral_better = roll_robot_lat < base_robot_lat and roll_box_lat < base_box_lat
    rel_not_worse = roll_rel <= base_rel
    tilt_better = roll_tilt < base_tilt and roll_box_tilt < base_box_tilt

    if retention_ok and falls_not_worse and (lateral_better or tilt_better) and rel_not_worse:
        case = "roll"
        reason = "rollsign_improved"
    else:
        case = "contact"
        reason = "rollsign_not_enough"

decision = {
    "case_set": case,
    "reason": reason,
    "rollsign_path": str(rollsign_path),
    "baseline_path": str(baseline_path),
    "rollsign_missing": bool(rollsign.get("_missing")),
    "baseline_missing": bool(baseline.get("_missing")),
    "rollsign_status": rollsign.get("status"),
    "baseline_status": baseline.get("status"),
    "rollsign_fall_events": rollsign.get("fall_events"),
    "baseline_fall_events": baseline.get("fall_events"),
    "rollsign_box_drop_events": rollsign.get("box_drop_events"),
    "baseline_box_drop_events": baseline.get("box_drop_events"),
    "rollsign_final_robot_lateral": rollsign.get("final_robot_target_lateral_error_m"),
    "baseline_final_robot_lateral": baseline.get("final_robot_target_lateral_error_m"),
    "rollsign_final_box_lateral": rollsign.get("final_box_target_lateral_error_m"),
    "baseline_final_box_lateral": baseline.get("final_box_target_lateral_error_m"),
    "rollsign_final_rel": rollsign.get("final_box_robot_relative_offset_error_m"),
    "baseline_final_rel": baseline.get("final_box_robot_relative_offset_error_m"),
    "rollsign_max_tilt": rollsign.get("max_tilt_rad"),
    "baseline_max_tilt": baseline.get("max_tilt_rad"),
    "rollsign_max_box_tilt": rollsign.get("max_box_tilt_rad"),
    "baseline_max_box_tilt": baseline.get("max_box_tilt_rad"),
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(case)
PY
)"

echo "Selected CASE_SET=${CASE_SET}"
echo "Decision JSON: ${DECISION_JSON}"

export CASE_SET
export BASE_STAMP_PREFIX="${FOLLOWUP_PREFIX}"
scripts/isaac/run_core_world_g1_lowcarry_followup_decision_suite.sh
