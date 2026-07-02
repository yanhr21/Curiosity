#!/usr/bin/env bash
set -euo pipefail

# Evaluate the validation-selected threshold-repaired curiosity checkpoint on
# held-out Phase07 cells. This is post-selection evaluation only: held-out cells
# were not used to choose the threshold. It is not training and not a success
# claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase07_closed_loop_threshold_repair_v1.json}"
REPAIR_SUMMARY="${REPAIR_SUMMARY:-$ROOT/experiments/outputs/phase07_closed_loop_threshold_repair_v1_20260628_summary.json}"
RUN_TAG="${RUN_TAG:-phase07_threshold_repaired_heldout_eval_v1_20260628}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_METHOD_SUFFIX="${EVAL_METHOD_SUFFIX:-curiosity_threshold_repaired}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

if [[ ! -x "$NEWTON_VENV/bin/python" || ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing required local env under envs/." >&2
  exit 3
fi
if [[ ! -f "$REPAIR_SUMMARY" ]]; then
  echo "ERROR: missing threshold repair summary: $REPAIR_SUMMARY" >&2
  exit 4
fi

readarray -t config_values < <("$NEWTON_VENV/bin/python" - "$ROOT" "$CONFIG" "$REPAIR_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
summary = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if summary.get("validation_only") is not True:
    raise SystemExit("repair summary is not validation_only")
if summary.get("selected_threshold") is None:
    raise SystemExit("missing selected_threshold")
print(root / config["current_checkpoint"])
print(summary["selected_threshold"])
PY
)
checkpoint="${config_values[0]}"
selected_threshold="${config_values[1]}"
if [[ -n "${SELECTED_THRESHOLD_OVERRIDE:-}" ]]; then
  selected_threshold="$SELECTED_THRESHOLD_OVERRIDE"
  echo "SELECTED_THRESHOLD_OVERRIDE_APPLIED=$selected_threshold"
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing checkpoint: $checkpoint" >&2
  exit 5
fi

echo "PHASE07_THRESHOLD_REPAIRED_HELDOUT_EVAL_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "CHECKPOINT=$checkpoint"
echo "SELECTED_THRESHOLD=$selected_threshold"
echo "REPAIR_SUMMARY=$REPAIR_SUMMARY"
echo "EVAL_METHOD_SUFFIX=$EVAL_METHOD_SUFFIX"
echo "NOTE=heldout_eval_after_validation_threshold_selection_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

validate_action_bridge_npz() {
  local eval_tag="$1"
  "$NEWTON_VENV/bin/python" - "$ROOT" "$eval_tag" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
run_tag = sys.argv[2]
npz_path = root / "experiments" / "outputs" / f"{run_tag}.npz"
required = [
    "candidate.action.eef_delta_x",
    "candidate.action.eef_delta_y",
    "candidate.action.eef_delta_z",
    "candidate.action.eef_delta_roll",
    "candidate.action.eef_delta_pitch",
    "candidate.action.eef_delta_yaw",
    "candidate.action.gripper",
    "candidate.action.eef_delta_xyzrpy_gripper",
]
data = np.load(npz_path)
missing = [key for key in required if key not in data.files]
payload = {
    "classification": "phase07_candidate_action_bridge_validation_v1",
    "run_tag": run_tag,
    "npz": str(npz_path.relative_to(root)),
    "required_fields": required,
    "missing_fields": missing,
    "status": "pass" if not missing else "fail",
    "not_mainstream_success_claim": True,
}
out = root / "experiments" / "outputs" / f"{run_tag}_candidate_action_bridge_validation.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if missing:
    raise SystemExit(1)
PY
}

run_eval_cell() {
  local cell="$1"
  local mass=""
  local friction=""
  local fill=""
  local nominal_visual_fill=""
  local visual_fill_cue=""
  case "$cell" in
    empty_high_misleading)
      mass="0.08"; friction="1.2"; fill="empty"; nominal_visual_fill="0.0"; visual_fill_cue="misleading_fill_cue";;
    full_low_hidden)
      mass="0.35"; friction="0.35"; fill="full"; nominal_visual_fill="1.0"; visual_fill_cue="hidden_fill_cue";;
    three_quarter_low_misleading)
      mass="0.29"; friction="0.35"; fill="three_quarter"; nominal_visual_fill="0.75"; visual_fill_cue="misleading_fill_cue";;
    *)
      echo "ERROR: unknown held-out cell: $cell" >&2
      exit 10
      ;;
  esac
  local eval_tag="phase07_eval_${cell}_${EVAL_METHOD_SUFFIX}_20260628"
  echo "=== PHASE07_THRESHOLD_REPAIRED_HELDOUT_CELL_START cell=$cell eval_tag=$eval_tag ==="
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="cube" \
  TRACKED_OBJECT="existing_cup_asset" \
  CONTROLLER_MODE="lift_hold_learned_residual" \
  FINAL_HOLD_DURATION="3.0" \
  LIFT_HEIGHT_MIN="0.12" \
  HOLD_DURATION_MIN="2.0" \
  DROP_HEIGHT_LOSS="0.05" \
  PHYSICS_VARIANT_LABEL="phase07_threshold_repaired_eval_${cell}" \
  BODY_MASS_SCALE="1.0" \
  SHAPE_FRICTION_SCALE="1.0" \
  OBJECT_MASS_KG="$mass" \
  OBJECT_FRICTION_MU="$friction" \
  FILL_LABEL="$fill" \
  NOMINAL_VISUAL_FILL="$nominal_visual_fill" \
  VISUAL_FILL_CUE="$visual_fill_cue" \
  VISUAL_FILL_CUE_RENDERED="0" \
  FEEDBACK_MIN_CONTACT_COUNT="58" \
  FEEDBACK_ACCEL_THRESHOLD="6.5" \
  FEEDBACK_HEIGHT_DROP_THRESHOLD="0.015" \
  FEEDBACK_INITIAL_LIFT_DURATION_SCALE="1.65" \
  FEEDBACK_LIFT_DURATION_SCALE_MAX="1.05" \
  FEEDBACK_HOLD_HEIGHT_STEP="0.0005" \
  FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="0.005" \
  FEEDBACK_STABILIZATION_STEP="0.05" \
  FEEDBACK_STABILIZATION_MAX="0.3" \
  PRE_RECORD_WARMUP_STEPS="15" \
  RESIDUAL_ADAPTER_CHECKPOINT="$checkpoint" \
  RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="$selected_threshold" \
  NUM_STEPS="360" \
  SAMPLE_STEPS="0,90,180,270,359" \
  VIDEO_FRAME_STRIDE="1" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase07_${EVAL_METHOD_SUFFIX}" \
  MASS_LABEL="$fill" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"

  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  validate_action_bridge_npz "$eval_tag"
  echo "=== PHASE07_THRESHOLD_REPAIRED_HELDOUT_CELL_END cell=$cell eval_tag=$eval_tag ==="
}

run_eval_cell "empty_high_misleading"
run_eval_cell "full_low_hidden"
run_eval_cell "three_quarter_low_misleading"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$selected_threshold" "$REPAIR_SUMMARY" "$EVAL_METHOD_SUFFIX" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
selected_threshold = float(sys.argv[3])
repair_summary = Path(sys.argv[4])
eval_method_suffix = sys.argv[5]
cells = ["empty_high_misleading", "full_low_hidden", "three_quarter_low_misleading"]
baseline_tags = {
    "empty_high_misleading": "phase07_eval_empty_high_misleading_no_adaptation_rerun_20260627",
    "full_low_hidden": "phase07_eval_full_low_hidden_no_adaptation_20260627",
    "three_quarter_low_misleading": "phase07_eval_three_quarter_low_misleading_no_adaptation_20260627",
}

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def metric_row(tag):
    metrics = load(root / "experiments" / "outputs" / f"{tag}_metrics.json")
    rows = metrics.get("rows") or []
    return rows[0] if rows else {}

def score(row):
    return (
        1.0 if row.get("status") == "success" and row.get("object_not_dropped") is True else 0.0,
        float(row.get("hold_duration_s", 0.0)),
        float(row.get("lift_height_m", 0.0)),
        -float(row.get("max_slip_m", 1e9)),
        -float(row.get("contact_loss_frames", 1e9)),
        -float(row.get("max_object_accel_m_s2", 1e9)),
    )

per_cell = {}
comparison_pass = True
for cell in cells:
    eval_tag = f"phase07_eval_{cell}_{eval_method_suffix}_20260628"
    row = metric_row(eval_tag)
    base = metric_row(baseline_tags[cell])
    row_score = score(row)
    base_score = score(base)
    beats = row_score > base_score
    safety_regressions = []
    for key in ("max_slip_m", "contact_loss_frames", "drop_height_loss_m", "max_object_accel_m_s2"):
        if float(row.get(key, 0.0)) > float(base.get(key, 0.0)):
            safety_regressions.append(f"{key}_regression")
    passed = beats and not safety_regressions
    comparison_pass = comparison_pass and passed
    per_cell[cell] = {
        "eval_run_tag": eval_tag,
        "baseline_run_tag": baseline_tags[cell],
        "selected_threshold": selected_threshold,
        "metric_row": row,
        "baseline_metric_row": base,
        "score_tuple": list(row_score),
        "baseline_score_tuple": list(base_score),
        "beats_no_adaptation": beats,
        "safety_regressions": safety_regressions,
        "status": "pass" if passed else "fail",
        "summary": f"experiments/outputs/{eval_tag}_summary.json",
        "metrics": f"experiments/outputs/{eval_tag}_metrics.json",
        "rollout_video": load(root / "experiments" / "outputs" / f"{eval_tag}_summary.json").get("rollout_video"),
    }

payload = {
    "classification": "phase07_threshold_repaired_heldout_eval_summary_v1",
    "status": "pass" if comparison_pass else "open_not_satisfied",
    "run_tag": run_tag,
    "selected_threshold": selected_threshold,
    "eval_method_suffix": eval_method_suffix,
    "threshold_source": str(repair_summary.relative_to(root)),
    "held_out_cells": cells,
    "per_cell": per_cell,
    "beats_no_adaptation_all_cells_without_safety_regression": comparison_pass,
    "not_training": True,
    "not_success_claim": True,
    "not_official_trex_method": True,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
report = root / "experiments" / "reports" / "2026-06-28_phase07_threshold_repaired_heldout_eval_v1.md"
lines = [
    "# Phase07 Threshold-Repaired Held-Out Eval V1",
    "",
    "Post-selection held-out evaluation using the validation-selected threshold. This is not training and not a success claim.",
    "",
    f"- selected threshold: `{selected_threshold}`",
    f"- status: `{payload['status']}`",
    f"- beats no-adaptation on all cells without safety regression: `{comparison_pass}`",
    "",
    "## Cells",
    "",
]
for cell, item in per_cell.items():
    lines.append(
        f"- `{cell}`: status `{item['status']}`, beats_no_adaptation `{item['beats_no_adaptation']}`, "
        f"safety_regressions `{item['safety_regressions']}`, video `{item['rollout_video']}`"
    )
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE07_THRESHOLD_REPAIRED_HELDOUT_EVAL_END"
