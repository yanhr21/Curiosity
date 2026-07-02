#!/usr/bin/env bash
set -euo pipefail

# Evaluate the Phase08 advantage-gated residual checkpoint on held-out pen
# contact-patch cells. This is closed-loop evaluation only: not training, not
# curiosity success, and not a substitute for the final curiosity gate.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
RUN_TAG="${RUN_TAG:-phase08_advantage_gated_heldout_eval_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_advantage_eval}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-28_phase08_advantage_gated_heldout_eval_v1.md}"
DEVICE="${DEVICE:-cuda:0}"
ADVANTAGE_CHECKPOINT="${ADVANTAGE_CHECKPOINT:-$ROOT/checkpoints/phase08_advantage_gated_residual_adapter_trainer_v1_20260628/phase08_advantage_gated_residual_adapter_v1_train_20260628.pt}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.5}"

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
if [[ ! -f "$ADVANTAGE_CHECKPOINT" ]]; then
  echo "ERROR: missing advantage-gated checkpoint: $ADVANTAGE_CHECKPOINT" >&2
  exit 4
fi

echo "PHASE08_ADVANTAGE_GATED_HELDOUT_EVAL_START"
echo "RUN_TAG=$RUN_TAG"
echo "EVAL_TAG_PREFIX=$EVAL_TAG_PREFIX"
echo "EVAL_TAG_SUFFIX=$EVAL_TAG_SUFFIX"
echo "REPORT_PATH=$REPORT_PATH"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "ADVANTAGE_CHECKPOINT=$ADVANTAGE_CHECKPOINT"
echo "ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD"
echo "NOTE=heldout_closed_loop_evaluation_not_training_not_curiosity_success_claim"
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
    "classification": "phase08_advantage_heldout_npz_field_validation_v1",
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

run_eval_cell() {
  local method="$1"
  local cell="$2"
  local controller_mode="$3"
  local grasp_offset_delta="$4"
  local checkpoint="${5:-}"
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_${method}_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE08_ADVANTAGE_HELDOUT_CELL_START method=$method cell=$cell eval_tag=$eval_tag ==="

  if [[ "$controller_mode" == "lift_hold_learned_residual" ]]; then
    RUN_TAG="$eval_tag" \
    NEWTON_VENV="$NEWTON_VENV" \
    TRAINER_VENV="$TRAINER_VENV" \
    SCENE="pen" \
    TRACKED_OBJECT="official_object" \
    CONTROLLER_MODE="$controller_mode" \
    FINAL_HOLD_DURATION="3.0" \
    LIFT_HEIGHT_MIN="0.12" \
    HOLD_DURATION_MIN="2.8" \
    DROP_HEIGHT_LOSS="0.05" \
    PHYSICS_VARIANT_LABEL="phase08_advantage_heldout_${cell}_${method}" \
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
    RESIDUAL_ADAPTER_CHECKPOINT="$checkpoint" \
    RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="$ACTIVE_THRESHOLD" \
    NUM_STEPS="450" \
    SAMPLE_STEPS="0,112,225,337,449" \
    VIDEO_FRAME_STRIDE="1" \
    VIDEO_FPS="12" \
    DEVICE="$DEVICE" \
    NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
      bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
        2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"
  else
    RUN_TAG="$eval_tag" \
    NEWTON_VENV="$NEWTON_VENV" \
    TRAINER_VENV="$TRAINER_VENV" \
    SCENE="pen" \
    TRACKED_OBJECT="official_object" \
    CONTROLLER_MODE="$controller_mode" \
    FINAL_HOLD_DURATION="3.0" \
    LIFT_HEIGHT_MIN="0.12" \
    HOLD_DURATION_MIN="2.8" \
    DROP_HEIGHT_LOSS="0.05" \
    PHYSICS_VARIANT_LABEL="phase08_advantage_heldout_${cell}_${method}" \
    BODY_MASS_SCALE="1.0" \
    SHAPE_FRICTION_SCALE="1.0" \
    OBJECT_MASS_KG="0.08" \
    OBJECT_FRICTION_MU="0.45" \
    GRASP_OFFSET_DELTA_XYZ="$grasp_offset_delta" \
    FILL_LABEL="$cell" \
    NOMINAL_VISUAL_FILL="0.5" \
    VISUAL_FILL_CUE="hidden_fill_cue" \
    VISUAL_FILL_CUE_RENDERED="0" \
    PRE_RECORD_WARMUP_STEPS="15" \
    NUM_STEPS="450" \
    SAMPLE_STEPS="0,112,225,337,449" \
    VIDEO_FRAME_STRIDE="1" \
    VIDEO_FPS="12" \
    DEVICE="$DEVICE" \
    NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
      bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
        2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"
  fi

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase08_advantage_${method}" \
  MASS_LABEL="heldout" \
  FRICTION_LABEL="0.45" \
  POSE_SEED="$grasp_offset_delta" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  validate_npz_fields "$eval_tag"
  echo "=== PHASE08_ADVANTAGE_HELDOUT_CELL_END method=$method cell=$cell eval_tag=$eval_tag ==="
}

run_cell_set() {
  local cell="$1"
  local grasp_offset_delta="$2"
  run_eval_cell "no_adaptation" "$cell" "lift_hold" "$grasp_offset_delta"
  run_eval_cell "guarded_feedback" "$cell" "lift_hold_feedback" "$grasp_offset_delta"
  run_eval_cell "advantage_gated_residual" "$cell" "lift_hold_learned_residual" "$grasp_offset_delta" "$ADVANTAGE_CHECKPOINT"
}

run_cell_set "pen_end_bias_heldout_center" "-0.0255,0.0135,0"
run_cell_set "pen_end_bias_heldout_high_y" "-0.0250,0.0165,0"
run_cell_set "pen_end_bias_heldout_low_x" "-0.0275,0.0125,0"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$ACTIVE_THRESHOLD" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" "$REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
active_threshold = float(sys.argv[3])
prefix = sys.argv[4]
suffix = sys.argv[5]
report_path = Path(sys.argv[6])
cells = ["pen_end_bias_heldout_center", "pen_end_bias_heldout_high_y", "pen_end_bias_heldout_low_x"]
methods = ["no_adaptation", "guarded_feedback", "advantage_gated_residual"]

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
beats_no_adaptation_all = True
beats_guarded_feedback_all = True
beats_strongest_baseline_all = True
for cell in cells:
    per_method = {}
    for method in methods:
        tag = f"{prefix}_{cell}_{method}_{suffix}"
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
    trained_score = tuple(per_method["advantage_gated_residual"]["score_tuple"])
    no_adapt_score = tuple(per_method["no_adaptation"]["score_tuple"])
    guarded_score = tuple(per_method["guarded_feedback"]["score_tuple"])
    strongest_name = "guarded_feedback" if guarded_score >= no_adapt_score else "no_adaptation"
    strongest_score = max(no_adapt_score, guarded_score)
    safety_regressions = {}
    for baseline_name in ("no_adaptation", "guarded_feedback"):
        regressions = []
        for key in ("max_slip_m", "contact_loss_frames", "drop_height_loss_m", "max_object_accel_m_s2"):
            trained = float(per_method["advantage_gated_residual"]["metric_row"].get(key, 0.0))
            baseline = float(per_method[baseline_name]["metric_row"].get(key, 0.0))
            if trained > baseline:
                regressions.append(f"{key}_regression")
        safety_regressions[baseline_name] = regressions
    beats_no_adapt = trained_score > no_adapt_score and not safety_regressions["no_adaptation"]
    beats_guarded = trained_score > guarded_score and not safety_regressions["guarded_feedback"]
    strongest_regressions = safety_regressions[strongest_name]
    beats_strongest = trained_score > strongest_score and not strongest_regressions
    beats_no_adaptation_all = beats_no_adaptation_all and beats_no_adapt
    beats_guarded_feedback_all = beats_guarded_feedback_all and beats_guarded
    beats_strongest_baseline_all = beats_strongest_baseline_all and beats_strongest
    per_cell[cell] = {
        "methods": per_method,
        "active_threshold": active_threshold,
        "strongest_baseline": strongest_name,
        "advantage_gated_beats_no_adaptation_without_safety_regression": beats_no_adapt,
        "advantage_gated_beats_guarded_feedback_without_safety_regression": beats_guarded,
        "advantage_gated_beats_strongest_baseline_without_safety_regression": beats_strongest,
        "safety_regressions_vs_baselines": safety_regressions,
    }

status = (
    "pass_candidate_needs_manual_visual_and_curiosity_mainstream_gates"
    if beats_strongest_baseline_all
    else "open_not_satisfied"
)
payload = {
    "classification": "phase08_advantage_gated_heldout_eval_summary_v1",
    "run_tag": run_tag,
    "status": status,
    "not_training": True,
    "not_curiosity_success_claim": True,
    "active_threshold": active_threshold,
    "eval_tag_prefix": prefix,
    "eval_tag_suffix": suffix,
    "held_out_cells": cells,
    "methods": methods,
    "per_cell": per_cell,
    "advantage_gated_beats_no_adaptation_all_cells_without_safety_regression": beats_no_adaptation_all,
    "advantage_gated_beats_guarded_feedback_all_cells_without_safety_regression": beats_guarded_feedback_all,
    "advantage_gated_beats_strongest_baseline_all_cells_without_safety_regression": beats_strongest_baseline_all,
    "manual_visual_inspection": "pending_direct_agent_check",
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Phase08 Advantage-Gated Held-Out Eval V1",
    "",
    "Evaluation only. Not training and not a final curiosity success claim.",
    "",
    f"- status: `{status}`",
    f"- active threshold: `{active_threshold}`",
    f"- advantage-gated beats no-adaptation all cells without safety regression: `{beats_no_adaptation_all}`",
    f"- advantage-gated beats guarded-feedback all cells without safety regression: `{beats_guarded_feedback_all}`",
    f"- advantage-gated beats strongest baseline all cells without safety regression: `{beats_strongest_baseline_all}`",
    "",
    "## Cells",
    "",
]
for cell, item in per_cell.items():
    lines.append(f"- `{cell}`: strongest baseline `{item['strongest_baseline']}`, trained beats strongest `{item['advantage_gated_beats_strongest_baseline_without_safety_regression']}`")
    for method, details in item["methods"].items():
        row = details["metric_row"]
        lines.append(f"  - `{method}`: status `{row.get('status')}`, hold `{row.get('hold_duration_s')}`, lift `{row.get('lift_height_m')}`, accel `{row.get('max_object_accel_m_s2')}`, video `{details.get('rollout_video')}`")
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE08_ADVANTAGE_GATED_HELDOUT_EVAL_END"
