#!/usr/bin/env bash
set -euo pipefail

# Diagnostic/source collection only: collect paired no-adaptation and guarded
# feedback rollouts around the positive pen_end_bias contact-patch cell. The
# output is intended for the strict advantage-gated residual preflight; it is
# not training and not a success claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
RUN_TAG="${RUN_TAG:-phase08_pen_end_bias_pair_collection_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_pen_end_bias_pair}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

echo "PHASE08_PEN_END_BIAS_PAIR_COLLECTION_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NOTE=paired_train_validation_source_collection_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,190p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

run_pair_cell() {
  local split="$1"
  local cell="$2"
  local method="$3"
  local controller_mode="$4"
  local grasp_offset_delta="$5"
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_${method}_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE08_PEN_PAIR_CELL_START split=$split method=$method cell=$cell eval_tag=$eval_tag ==="
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
  PHYSICS_VARIANT_LABEL="phase08_pen_pair_${cell}_${method}" \
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
  BASELINE_NAME="phase08_pen_pair_${method}" \
  MASS_LABEL="$split" \
  FRICTION_LABEL="0.45" \
  POSE_SEED="$grasp_offset_delta" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  echo "=== PHASE08_PEN_PAIR_CELL_END split=$split method=$method cell=$cell eval_tag=$eval_tag ==="
}

run_cell_pair() {
  local split="$1"
  local cell="$2"
  local grasp_offset_delta="$3"
  run_pair_cell "$split" "$cell" "no_adaptation" "lift_hold" "$grasp_offset_delta"
  run_pair_cell "$split" "$cell" "guarded_feedback" "lift_hold_feedback" "$grasp_offset_delta"
}

run_cell_pair "train" "pen_end_bias_train_b" "-0.024,0.010,0"
run_cell_pair "train" "pen_end_bias_train_c" "-0.026,0.012,0"
run_cell_pair "train" "pen_end_bias_train_d" "-0.025,0.015,0"
run_cell_pair "validation" "pen_end_bias_val_a" "-0.027,0.014,0"
run_cell_pair "validation" "pen_end_bias_val_b" "-0.023,0.013,0"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
prefix = sys.argv[3]
suffix = sys.argv[4]
cells = [
    ("train", "pen_end_bias_train_b"),
    ("train", "pen_end_bias_train_c"),
    ("train", "pen_end_bias_train_d"),
    ("validation", "pen_end_bias_val_a"),
    ("validation", "pen_end_bias_val_b"),
]
methods = ["no_adaptation", "guarded_feedback"]
rows = []
paired = {}
for split, cell in cells:
    for method in methods:
        tag = f"{prefix}_{cell}_{method}_{suffix}"
        summary_path = root / "experiments" / "outputs" / f"{tag}_summary.json"
        metrics_path = root / "experiments" / "outputs" / f"{tag}_metrics.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metric_row = (metrics.get("rows") or [{}])[0]
        world = (summary.get("task_metrics") or {}).get("per_world", [{}])[0]
        rows.append({
            "split": split,
            "cell": cell,
            "method": method,
            "status": summary.get("status"),
            "metrics_status": metric_row.get("status"),
            "success_all_worlds": (summary.get("task_metrics") or {}).get("success_all_worlds"),
            "hold_duration_s": world.get("longest_hold_s"),
            "lift_height_m": world.get("max_lift"),
            "drop_from_max_m": world.get("drop_from_max"),
            "max_xy_drift_m": world.get("max_xy_drift"),
            "max_object_accel_m_s2": metric_row.get("max_object_accel_m_s2"),
            "max_slip_m": metric_row.get("max_slip_m"),
            "scripted_feedback_trigger_count": (summary.get("scripted_feedback") or {}).get("final_trigger_count"),
            "grasp_perturbation_adapter": summary.get("grasp_perturbation_adapter"),
            "summary": str(summary_path.relative_to(root)),
            "metrics": str(metrics_path.relative_to(root)),
            "contact_sheet": summary.get("contact_sheet"),
            "rollout_video": summary.get("rollout_video"),
        })
for split, cell in cells:
    base = next(row for row in rows if row["cell"] == cell and row["method"] == "no_adaptation")
    intervention = next(row for row in rows if row["cell"] == cell and row["method"] == "guarded_feedback")
    hold_gain = (intervention.get("hold_duration_s") or 0.0) - (base.get("hold_duration_s") or 0.0)
    lift_gain = (intervention.get("lift_height_m") or 0.0) - (base.get("lift_height_m") or 0.0)
    accel_ok = (intervention.get("max_object_accel_m_s2") or 0.0) <= (base.get("max_object_accel_m_s2") or 0.0)
    paired[cell] = {
        "split": split,
        "hold_gain_s": hold_gain,
        "lift_gain_m": lift_gain,
        "accel_non_regression": accel_ok,
        "baseline_run_tag": f"{prefix}_{cell}_no_adaptation_{suffix}",
        "intervention_run_tag": f"{prefix}_{cell}_guarded_feedback_{suffix}",
        "candidate_for_advantage_gate": hold_gain >= 0.0 and lift_gain >= 0.0 and accel_ok,
    }
payload = {
    "classification": "phase08_pen_end_bias_pair_collection_v1",
    "status": "complete_diagnostic_not_training_not_success_claim",
    "run_tag": run_tag,
    "rows": rows,
    "paired": paired,
    "purpose": "collect train/validation paired evidence for strict advantage-gated residual preflight",
    "success_claim": False,
    "training_started": False,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE08_PEN_END_BIAS_PAIR_COLLECTION_END"
