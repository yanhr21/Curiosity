#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run G1 probe-selected validation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
PIPELINE_STAMP="${PIPELINE_STAMP:-20260706_g1_probe_selected_targetwindow_validation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/outputs/core_world_g1_probe_selected_targetwindow/${PIPELINE_STAMP}}"

PROBE_STAMP="${PIPELINE_STAMP}_probe"
SELECTED_STAMP="${PIPELINE_STAMP}_selected"
PROBE_SUMMARY="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${PROBE_STAMP}/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json"
SELECTION_JSON="${OUTPUT_ROOT}/g1_probe_posture_selection.json"
SELECTION_ENV="${OUTPUT_ROOT}/g1_probe_posture_selection.env"

mkdir -p "${OUTPUT_ROOT}"

echo "[G1-PROBE-SELECT] probe_stamp=${PROBE_STAMP}"
env \
  SUITE_STAMP="${PROBE_STAMP}" \
  LARGERBOX_STRICT_MODE="${PROBE_LARGERBOX_STRICT_MODE:-lowcarry}" \
  RUN_NOBOX=0 \
  RUN_FIXED=0 \
  RUN_FREE=1 \
  FREE_STEPS="${PROBE_FREE_STEPS:-260}" \
  FREE_MIN_ROBOT_TRAVEL=0.0 \
  FREE_MIN_BOX_TRAVEL=0.0 \
  FREE_MAX_TILT="${PROBE_FREE_MAX_TILT:-0.95}" \
  FREE_MAX_BOX_TILT="${PROBE_FREE_MAX_BOX_TILT:-0.95}" \
  FREE_MAX_FINAL_REL="${PROBE_FREE_MAX_FINAL_REL:-1.0}" \
  PROBE_MODE=front_bumper \
  PROBE_START_STEP="${PROBE_START_STEP:-40}" \
  PROBE_END_STEP="${PROBE_END_STEP:--1}" \
  PROBE_COLLISION_WINDOW="${PROBE_COLLISION_WINDOW:-0}" \
  PROBE_PAD_LOCAL_X="${PROBE_PAD_LOCAL_X:-0.50}" \
  PROBE_PAD_LOCAL_Y="${PROBE_PAD_LOCAL_Y:-0.0}" \
  PROBE_PAD_LOCAL_Z="${PROBE_PAD_LOCAL_Z:-0.02}" \
  PROBE_PAD_SIZE_X="${PROBE_PAD_SIZE_X:-0.05}" \
  PROBE_PAD_SIZE_Y="${PROBE_PAD_SIZE_Y:-0.36}" \
  PROBE_PAD_SIZE_Z="${PROBE_PAD_SIZE_Z:-0.18}" \
  PROBE_PAD_MASS="${PROBE_PAD_MASS:-0.2}" \
  bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"

if [[ ! -f "${PROBE_SUMMARY}" ]]; then
  echo "Missing probe summary: ${PROBE_SUMMARY}" >&2
  exit 1
fi

echo "[G1-PROBE-SELECT] selecting posture from ${PROBE_SUMMARY}"
selector_args=(
  --probe-summary "${PROBE_SUMMARY}"
  --output "${SELECTION_JSON}"
  --output-env "${SELECTION_ENV}"
  --resistant-probe-travel-threshold "${RESISTANT_PROBE_TRAVEL_THRESHOLD:-0.015}"
  --tall-box-threshold "${TALL_BOX_THRESHOLD:-0.09}"
  --wide-box-threshold "${WIDE_BOX_THRESHOLD:-0.12}"
)
if [[ -n "${HIGH_PROBE_TRAVEL_THRESHOLD:-}" ]]; then
  selector_args+=(--high-probe-travel-threshold "${HIGH_PROBE_TRAVEL_THRESHOLD}")
fi
if [[ -n "${MAX_PROBE_FALL_EVENTS:-}" ]]; then
  selector_args+=(--max-probe-fall-events "${MAX_PROBE_FALL_EVENTS}")
fi
if [[ -n "${MAX_PROBE_BOX_DROP_EVENTS:-}" ]]; then
  selector_args+=(--max-probe-box-drop-events "${MAX_PROBE_BOX_DROP_EVENTS}")
fi
if [[ -n "${MAX_PROBE_TILT:-}" ]]; then
  selector_args+=(--max-probe-tilt "${MAX_PROBE_TILT}")
fi
if [[ -n "${MAX_PROBE_BOX_TILT:-}" ]]; then
  selector_args+=(--max-probe-box-tilt "${MAX_PROBE_BOX_TILT}")
fi
if [[ -n "${PROBE_TILT_RISK_THRESHOLD:-}" ]]; then
  selector_args+=(--probe-tilt-risk-threshold "${PROBE_TILT_RISK_THRESHOLD}")
fi
if [[ -n "${PROBE_BOX_TILT_RISK_THRESHOLD:-}" ]]; then
  selector_args+=(--probe-box-tilt-risk-threshold "${PROBE_BOX_TILT_RISK_THRESHOLD}")
fi
if [[ -n "${PROBE_RELATIVE_OFFSET_RISK_THRESHOLD:-}" ]]; then
  selector_args+=(--probe-relative-offset-risk-threshold "${PROBE_RELATIVE_OFFSET_RISK_THRESHOLD}")
fi
if [[ -n "${MIN_PROBE_COMPLETED_STEPS:-}" ]]; then
  selector_args+=(--min-probe-completed-steps "${MIN_PROBE_COMPLETED_STEPS}")
fi
set +e
python3 "${ROOT_DIR}/scripts/isaac/select_core_world_g1_carry_posture_from_probe.py" "${selector_args[@]}"
selection_status=$?
set -e

if [[ "${selection_status}" != "0" ]]; then
  PIPELINE_SUMMARY="${OUTPUT_ROOT}/g1_probe_selected_pipeline_summary.json"
  python3 - <<PY
import json
from pathlib import Path

def load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())

report = {
    "scene_type": "core_world_g1_probe_selected_targetwindow_pipeline",
    "success_claim": "diagnostic_pipeline_not_final_autonomous_or_learned_success",
    "pipeline_stamp": "${PIPELINE_STAMP}",
    "probe_stamp": "${PROBE_STAMP}",
    "selected_validation_stamp": None,
    "selected_posture": None,
    "selection_uses_hidden_ground_truth": False,
    "selection_exit_status": int("${selection_status}"),
    "validation_exit_status": -1,
    "validation_skipped_reason": "probe_selection_failed_or_probe_safety_gate_failed",
    "probe_summary_path": "${PROBE_SUMMARY}",
    "selection_report_path": "${SELECTION_JSON}",
    "selected_summary_path": None,
    "selected_check_path": None,
    "probe_summary": load("${PROBE_SUMMARY}"),
    "selection_report": load("${SELECTION_JSON}"),
    "selected_summary": None,
    "selected_check": None,
}
out = Path("${PIPELINE_SUMMARY}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps({
    "pipeline_summary": str(out),
    "selection_exit_status": report["selection_exit_status"],
    "validation_skipped_reason": report["validation_skipped_reason"],
}, indent=2, sort_keys=True))
PY
  exit "${selection_status}"
fi

# shellcheck disable=SC1090
source "${SELECTION_ENV}"

if [[ "${LARGERBOX_STRICT_MODE}" == "lowcarry" ]]; then
  export FREE_STEPS="${SELECTED_FREE_STEPS:-819}"
  export MIN_TARGET_WINDOW_BOTH_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-80}"
  export MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-50}"
  export MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-40}"
  export MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STABLE_STEPS:-80}"
  export MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_LONGEST_STREAK_STEPS:-50}"
  export MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_FINAL_HOLD_STREAK_AT_END_STEPS:-40}"
  export MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS="${MIN_AGILE_COMMAND_HOLD_FINAL_ACTIVE_STEPS:-399}"
  export MAX_FINAL_HOLD_COMMAND_X="${MAX_FINAL_HOLD_COMMAND_X:-0.001}"
  export MAX_FINAL_HOLD_COMMAND_Y="${MAX_FINAL_HOLD_COMMAND_Y:-0.003}"
  export MAX_FINAL_HOLD_COMMAND_YAW="${MAX_FINAL_HOLD_COMMAND_YAW:-0.001}"
  export MIN_FINAL_HOLD_ROBOT_Z="${MIN_FINAL_HOLD_ROBOT_Z:-0.45}"
  export MIN_FINAL_HOLD_BOX_Z="${MIN_FINAL_HOLD_BOX_Z:-0.45}"
  export MAX_FINAL_HOLD_TILT="${MAX_FINAL_HOLD_TILT:-0.35}"
  export MAX_FINAL_HOLD_BOX_TILT="${MAX_FINAL_HOLD_BOX_TILT:-0.45}"
  export MAX_FINAL_HOLD_FALL_EVENTS="${MAX_FINAL_HOLD_FALL_EVENTS:-0}"
  export MAX_FINAL_HOLD_BOX_DROP_EVENTS="${MAX_FINAL_HOLD_BOX_DROP_EVENTS:-0}"
else
  export FREE_STEPS="${SELECTED_FREE_STEPS:-1000}"
  export MIN_TARGET_WINDOW_BOTH_STABLE_STEPS="${MIN_TARGET_WINDOW_BOTH_STABLE_STEPS:-100}"
  export MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS="${MIN_TARGET_WINDOW_BOTH_LONGEST_STREAK_STEPS:-100}"
  export MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS="${MIN_TARGET_WINDOW_BOTH_STREAK_AT_END_STEPS:-100}"
fi

export SUITE_STAMP="${SELECTED_STAMP}_${LARGERBOX_STRICT_MODE}"
export FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_ROBOT_TARGET_DIRECTED_TRAVEL:-2.35}"
export FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL="${FREE_MAX_FINAL_BOX_TARGET_DIRECTED_TRAVEL:-2.35}"
export TARGET_WINDOW_CENTER="${TARGET_WINDOW_CENTER:-2.0}"
export TARGET_WINDOW_HALFWIDTH="${TARGET_WINDOW_HALFWIDTH:-0.35}"

echo "[G1-PROBE-SELECT] selected=${LARGERBOX_STRICT_MODE} validation_stamp=${SUITE_STAMP}"
set +e
bash "${ROOT_DIR}/scripts/isaac/run_core_world_g1_largerbox_strict_suite.sh"
validation_status=$?
set -e

SELECTED_SUMMARY="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${SUITE_STAMP}/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json"
SELECTED_CHECK="${ROOT_DIR}/experiments/outputs/core_world_g1_agile_policy_low_cradle/${SUITE_STAMP}/agile_low_cradle_freebox_walk/check.json"
PIPELINE_SUMMARY="${OUTPUT_ROOT}/g1_probe_selected_pipeline_summary.json"

python3 - <<PY
import json
from pathlib import Path

def load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())

probe_summary_path = "${PROBE_SUMMARY}"
selection_path = "${SELECTION_JSON}"
selected_summary_path = "${SELECTED_SUMMARY}"
selected_check_path = "${SELECTED_CHECK}"

report = {
    "scene_type": "core_world_g1_probe_selected_targetwindow_pipeline",
    "success_claim": "diagnostic_pipeline_not_final_autonomous_or_learned_success",
    "pipeline_stamp": "${PIPELINE_STAMP}",
    "probe_stamp": "${PROBE_STAMP}",
    "selected_validation_stamp": "${SUITE_STAMP}",
    "selected_posture": "${LARGERBOX_STRICT_MODE}",
    "selection_uses_hidden_ground_truth": False,
    "selection_exit_status": int("${selection_status}"),
    "validation_exit_status": int("${validation_status}"),
    "validation_skipped_reason": None,
    "probe_summary_path": probe_summary_path,
    "selection_report_path": selection_path,
    "selected_summary_path": selected_summary_path,
    "selected_check_path": selected_check_path,
    "probe_summary": load(probe_summary_path),
    "selection_report": load(selection_path),
    "selected_summary": load(selected_summary_path),
    "selected_check": load(selected_check_path),
}
out = Path("${PIPELINE_SUMMARY}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps({
    "pipeline_summary": str(out),
    "selected_posture": report["selected_posture"],
    "validation_exit_status": report["validation_exit_status"],
}, indent=2, sort_keys=True))
PY

exit "${validation_status}"
