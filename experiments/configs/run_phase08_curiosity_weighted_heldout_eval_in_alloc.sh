#!/usr/bin/env bash
set -euo pipefail

# Evaluate only the Phase08 curiosity-weighted residual checkpoint on held-out
# cells, then compare against the already completed retry1 baseline set:
# no-adaptation, guarded feedback, and advantage-gated residual.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
RUN_TAG="${RUN_TAG:-phase08_curiosity_weighted_heldout_eval_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_curiosity_weighted_eval}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"
BASELINE_EVAL_TAG_PREFIX="${BASELINE_EVAL_TAG_PREFIX:-phase08_advantage_retry1_eval}"
BASELINE_EVAL_TAG_SUFFIX="${BASELINE_EVAL_TAG_SUFFIX:-20260628}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-28_phase08_curiosity_weighted_heldout_eval_v1.md}"
DEVICE="${DEVICE:-cuda:0}"
CURIOSITY_CHECKPOINT="${CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase08_curiosity_weighted_residual_adapter_trainer_v1_20260628/phase08_curiosity_weighted_residual_adapter_v1_train_20260628.pt}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.5}"
CURIOSITY_CONTROLLER_MODE="${CURIOSITY_CONTROLLER_MODE:-lift_hold_learned_residual}"

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
if [[ ! -f "$CURIOSITY_CHECKPOINT" ]]; then
  echo "ERROR: missing curiosity-weighted checkpoint: $CURIOSITY_CHECKPOINT" >&2
  exit 4
fi

echo "PHASE08_CURIOSITY_WEIGHTED_HELDOUT_EVAL_START"
echo "RUN_TAG=$RUN_TAG"
echo "EVAL_TAG_PREFIX=$EVAL_TAG_PREFIX"
echo "BASELINE_EVAL_TAG_PREFIX=$BASELINE_EVAL_TAG_PREFIX"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "CURIOSITY_CHECKPOINT=$CURIOSITY_CHECKPOINT"
echo "ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD"
echo "CURIOSITY_CONTROLLER_MODE=$CURIOSITY_CONTROLLER_MODE"
echo "NOTE=evaluation_only_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,210p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

validate_npz_fields() {
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
    "candidate.action.eef_delta_xyzrpy_gripper",
    "candidate.task.grasp_offset_delta_x",
    "candidate.task.grasp_offset_delta_y",
    "candidate.task.grasp_offset_delta_z",
    "newton.panda.rigid_contact_count",
    "newton.panda.object_body_q",
]
data = np.load(npz_path)
missing = [key for key in required if key not in data.files]
payload = {
    "classification": "phase08_curiosity_weighted_npz_field_validation_v1",
    "run_tag": run_tag,
    "npz": str(npz_path.relative_to(root)),
    "required_fields": required,
    "missing_fields": missing,
    "status": "pass" if not missing else "fail",
    "not_success_claim": True,
}
out = root / "experiments" / "outputs" / f"{run_tag}_npz_field_validation.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if missing:
    raise SystemExit(1)
PY
}

run_curiosity_cell() {
  local cell="$1"
  local grasp_offset_delta="$2"
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_curiosity_weighted_residual_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE08_CURIOSITY_WEIGHTED_CELL_START cell=$cell eval_tag=$eval_tag ==="
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="pen" \
  TRACKED_OBJECT="official_object" \
  CONTROLLER_MODE="$CURIOSITY_CONTROLLER_MODE" \
  FINAL_HOLD_DURATION="3.0" \
  LIFT_HEIGHT_MIN="0.12" \
  HOLD_DURATION_MIN="2.8" \
  DROP_HEIGHT_LOSS="0.05" \
  PHYSICS_VARIANT_LABEL="phase08_curiosity_weighted_heldout_${cell}" \
  BODY_MASS_SCALE="1.0" \
  SHAPE_FRICTION_SCALE="1.0" \
  OBJECT_MASS_KG="0.08" \
  OBJECT_FRICTION_MU="0.45" \
  GRASP_OFFSET_DELTA_XYZ="$grasp_offset_delta" \
  FILL_LABEL="$cell" \
  NOMINAL_VISUAL_FILL="0.5" \
  VISUAL_FILL_CUE="hidden_fill_cue" \
  VISUAL_FILL_CUE_RENDERED="0" \
  FEEDBACK_MIN_CONTACT_COUNT="68" \
  FEEDBACK_ACCEL_THRESHOLD="4.5" \
  FEEDBACK_HEIGHT_DROP_THRESHOLD="0.008" \
  FEEDBACK_INITIAL_LIFT_DURATION_SCALE="1.9" \
  FEEDBACK_LIFT_DURATION_SCALE_MAX="1.55" \
  FEEDBACK_HOLD_HEIGHT_STEP="0.0007" \
  FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="0.012" \
  FEEDBACK_STABILIZATION_STEP="0.35" \
  FEEDBACK_STABILIZATION_MAX="1.8" \
  PRE_RECORD_WARMUP_STEPS="15" \
  RESIDUAL_ADAPTER_CHECKPOINT="$CURIOSITY_CHECKPOINT" \
  RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="$ACTIVE_THRESHOLD" \
  NUM_STEPS="450" \
  SAMPLE_STEPS="0,112,225,337,449" \
  VIDEO_FRAME_STRIDE="1" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_direct_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase08_curiosity_weighted_residual" \
  MASS_LABEL="heldout" \
  FRICTION_LABEL="0.45" \
  POSE_SEED="$grasp_offset_delta" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  validate_npz_fields "$eval_tag"
  echo "=== PHASE08_CURIOSITY_WEIGHTED_CELL_END cell=$cell eval_tag=$eval_tag ==="
}

run_curiosity_cell "pen_end_bias_heldout_center" "-0.0255,0.0135,0"
run_curiosity_cell "pen_end_bias_heldout_high_y" "-0.0250,0.0165,0"
run_curiosity_cell "pen_end_bias_heldout_low_x" "-0.0275,0.0125,0"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$ACTIVE_THRESHOLD" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" "$BASELINE_EVAL_TAG_PREFIX" "$BASELINE_EVAL_TAG_SUFFIX" "$REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
active_threshold = float(sys.argv[3])
cur_prefix = sys.argv[4]
cur_suffix = sys.argv[5]
base_prefix = sys.argv[6]
base_suffix = sys.argv[7]
report_path = Path(sys.argv[8])
cells = ["pen_end_bias_heldout_center", "pen_end_bias_heldout_high_y", "pen_end_bias_heldout_low_x"]
baseline_methods = ["no_adaptation", "guarded_feedback", "advantage_gated_residual"]
candidate_method = "curiosity_weighted_residual"
methods = [*baseline_methods, candidate_method]

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def tag_for(cell, method):
    if method == candidate_method:
        return f"{cur_prefix}_{cell}_{method}_{cur_suffix}"
    return f"{base_prefix}_{cell}_{method}_{base_suffix}"

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
candidate_beats_strongest_all = True
candidate_beats_advantage_all = True
for cell in cells:
    per_method = {}
    for method in methods:
        tag = tag_for(cell, method)
        row = metric_row(tag)
        summary = load(root / "experiments" / "outputs" / f"{tag}_summary.json")
        per_method[method] = {
            "eval_run_tag": tag,
            "metric_row": row,
            "score_tuple": list(score(row)),
            "summary": f"experiments/outputs/{tag}_summary.json",
            "metrics": f"experiments/outputs/{tag}_metrics.json",
            "rollout_video": summary.get("rollout_video"),
            "contact_sheet": summary.get("contact_sheet"),
        }
    candidate_score = tuple(per_method[candidate_method]["score_tuple"])
    baseline_scores = {method: tuple(per_method[method]["score_tuple"]) for method in baseline_methods}
    strongest_name, strongest_score = max(baseline_scores.items(), key=lambda item: item[1])
    safety_regressions = {}
    for baseline_name in baseline_methods:
        regressions = []
        for key in ("max_slip_m", "contact_loss_frames", "drop_height_loss_m", "max_object_accel_m_s2"):
            candidate = float(per_method[candidate_method]["metric_row"].get(key, 0.0))
            baseline = float(per_method[baseline_name]["metric_row"].get(key, 0.0))
            if candidate > baseline:
                regressions.append(f"{key}_regression")
        safety_regressions[baseline_name] = regressions
    beats_by_baseline = {
        method: candidate_score > score_value and not safety_regressions[method]
        for method, score_value in baseline_scores.items()
    }
    beats_strongest = candidate_score > strongest_score and not safety_regressions[strongest_name]
    candidate_beats_strongest_all = candidate_beats_strongest_all and beats_strongest
    candidate_beats_advantage_all = candidate_beats_advantage_all and beats_by_baseline["advantage_gated_residual"]
    per_cell[cell] = {
        "methods": per_method,
        "active_threshold": active_threshold,
        "strongest_baseline": strongest_name,
        "curiosity_weighted_beats_by_baseline_without_safety_regression": beats_by_baseline,
        "curiosity_weighted_beats_strongest_baseline_without_safety_regression": beats_strongest,
        "safety_regressions_vs_baselines": safety_regressions,
    }

status = "pass_candidate_needs_manual_visual_and_mainstream_gate" if candidate_beats_strongest_all else "open_not_satisfied"
payload = {
    "classification": "phase08_curiosity_weighted_heldout_eval_summary_v1",
    "run_tag": run_tag,
    "status": status,
    "not_training": True,
    "not_final_curiosity_success_claim": True,
    "active_threshold": active_threshold,
    "candidate_method": candidate_method,
    "baseline_methods": baseline_methods,
    "held_out_cells": cells,
    "methods": methods,
    "per_cell": per_cell,
    "curiosity_weighted_beats_strongest_baseline_all_cells_without_safety_regression": candidate_beats_strongest_all,
    "curiosity_weighted_beats_advantage_gated_residual_all_cells_without_safety_regression": candidate_beats_advantage_all,
    "manual_visual_inspection": "pending_direct_agent_check",
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Phase08 Curiosity-Weighted Held-Out Eval V1",
    "",
    "Evaluation only. Not training and not a final curiosity success claim.",
    "",
    f"- status: `{status}`",
    f"- active threshold: `{active_threshold}`",
    f"- curiosity-weighted beats strongest baseline all cells without safety regression: `{candidate_beats_strongest_all}`",
    f"- curiosity-weighted beats advantage-gated residual all cells without safety regression: `{candidate_beats_advantage_all}`",
    "",
    "## Cells",
    "",
]
for cell, item in per_cell.items():
    lines.append(f"- `{cell}`: strongest baseline `{item['strongest_baseline']}`, curiosity beats strongest `{item['curiosity_weighted_beats_strongest_baseline_without_safety_regression']}`")
    for method, details in item["methods"].items():
        row = details["metric_row"]
        lines.append(f"  - `{method}`: status `{row.get('status')}`, hold `{row.get('hold_duration_s')}`, lift `{row.get('lift_height_m')}`, accel `{row.get('max_object_accel_m_s2')}`, video `{details.get('rollout_video')}`")
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE08_CURIOSITY_WEIGHTED_HELDOUT_EVAL_END"
