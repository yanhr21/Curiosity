#!/usr/bin/env bash
set -euo pipefail

# Diagnostic/source collection only: refine validation offsets near the
# positive pen_end_bias region before advantage-gated training.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
RUN_TAG="${RUN_TAG:-phase08_pen_validation_refine_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_pen_validation_refine}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/visuals

echo "PHASE08_PEN_VALIDATION_REFINE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NOTE=validation_refinement_source_collection_not_training_not_success_claim"

run_cell() {
  local cell="$1"
  local method="$2"
  local controller_mode="$3"
  local grasp_offset_delta="$4"
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_${method}_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE08_PEN_VALIDATION_CELL_START method=$method cell=$cell eval_tag=$eval_tag ==="
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
  PHYSICS_VARIANT_LABEL="phase08_pen_validation_${cell}_${method}" \
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
  NUM_STEPS="450" \
  SAMPLE_STEPS="0,112,225,337,449" \
  VIDEO_FRAME_STRIDE="1" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase08_pen_validation_${method}" \
  MASS_LABEL="validation" \
  FRICTION_LABEL="0.45" \
  POSE_SEED="$grasp_offset_delta" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  echo "=== PHASE08_PEN_VALIDATION_CELL_END method=$method cell=$cell eval_tag=$eval_tag ==="
}

run_pair() {
  local cell="$1"
  local offset="$2"
  run_cell "$cell" "no_adaptation" "lift_hold" "$offset"
  run_cell "$cell" "guarded_feedback" "lift_hold_feedback" "$offset"
}

run_pair "pen_end_bias_val_c" "-0.025,0.014,0"
run_pair "pen_end_bias_val_d" "-0.026,0.015,0"
run_pair "pen_end_bias_val_e" "-0.0245,0.0145,0"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
prefix = sys.argv[3]
suffix = sys.argv[4]
cells = ["pen_end_bias_val_c", "pen_end_bias_val_d", "pen_end_bias_val_e"]
rows = []
paired = {}
for cell in cells:
    for method in ["no_adaptation", "guarded_feedback"]:
        tag = f"{prefix}_{cell}_{method}_{suffix}"
        summary_path = root / "experiments" / "outputs" / f"{tag}_summary.json"
        metrics_path = root / "experiments" / "outputs" / f"{tag}_metrics.json"
        summary = json.loads(summary_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        metric_row = (metrics.get("rows") or [{}])[0]
        world = summary["task_metrics"]["per_world"][0]
        rows.append({
            "split": "validation",
            "cell": cell,
            "method": method,
            "metrics_status": metric_row.get("status"),
            "hold_duration_s": world.get("longest_hold_s"),
            "lift_height_m": world.get("max_lift"),
            "drop_from_max_m": world.get("drop_from_max"),
            "max_xy_drift_m": world.get("max_xy_drift"),
            "max_object_accel_m_s2": metric_row.get("max_object_accel_m_s2"),
            "max_slip_m": metric_row.get("max_slip_m"),
            "scripted_feedback_trigger_count": summary.get("scripted_feedback", {}).get("final_trigger_count"),
            "summary": str(summary_path.relative_to(root)),
            "metrics": str(metrics_path.relative_to(root)),
            "contact_sheet": summary.get("contact_sheet"),
            "rollout_video": summary.get("rollout_video"),
        })
for cell in cells:
    base = next(row for row in rows if row["cell"] == cell and row["method"] == "no_adaptation")
    intervention = next(row for row in rows if row["cell"] == cell and row["method"] == "guarded_feedback")
    hold_gain = (intervention["hold_duration_s"] or 0.0) - (base["hold_duration_s"] or 0.0)
    lift_gain = (intervention["lift_height_m"] or 0.0) - (base["lift_height_m"] or 0.0)
    accel_ok = (intervention["max_object_accel_m_s2"] or 0.0) <= (base["max_object_accel_m_s2"] or 0.0)
    paired[cell] = {
        "split": "validation",
        "hold_gain_s": hold_gain,
        "lift_gain_m": lift_gain,
        "accel_non_regression": accel_ok,
        "baseline_run_tag": f"{prefix}_{cell}_no_adaptation_{suffix}",
        "intervention_run_tag": f"{prefix}_{cell}_guarded_feedback_{suffix}",
        "candidate_for_advantage_gate": hold_gain >= 0.0 and lift_gain >= 0.0 and accel_ok,
    }
payload = {
    "classification": "phase08_pen_validation_refine_v1",
    "status": "complete_diagnostic_not_training_not_success_claim",
    "run_tag": run_tag,
    "rows": rows,
    "paired": paired,
    "success_claim": False,
    "training_started": False,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE08_PEN_VALIDATION_REFINE_END"
