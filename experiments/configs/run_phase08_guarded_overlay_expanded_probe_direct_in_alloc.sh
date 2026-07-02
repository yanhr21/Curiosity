#!/usr/bin/env bash
set -euo pipefail

# Train/validation source probe only. This expands around the one accepted
# guarded-overlay source cell from the direct probe. It is not training and
# not a success claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
RUN_TAG="${RUN_TAG:-phase08_guarded_overlay_expanded_probe_direct_v1_20260629}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_guarded_overlay_expanded_probe_direct}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260629}"
OVERLAY_CHECKPOINT="${OVERLAY_CHECKPOINT:-$ROOT/checkpoints/phase08_selective_anchor_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_selective_anchor_curiosity_repair_v1_train_20260629.pt}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-0.5}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-29_phase08_guarded_overlay_expanded_probe_direct_v1.md}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

echo "PHASE08_GUARDED_OVERLAY_EXPANDED_PROBE_DIRECT_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "OVERLAY_CHECKPOINT=$OVERLAY_CHECKPOINT"
echo "NOTE=train_validation_source_probe_not_training_not_success_claim"

run_cell() {
  local split="$1"
  local cell="$2"
  local method="$3"
  local controller_mode="$4"
  local grasp_offset_delta="$5"
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_${method}_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE08_GUARDED_OVERLAY_EXPANDED_CELL_START split=$split method=$method cell=$cell eval_tag=$eval_tag offset=$grasp_offset_delta ==="
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
  PHYSICS_VARIANT_LABEL="phase08_guarded_overlay_expanded_${cell}_${method}" \
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
  RESIDUAL_ADAPTER_CHECKPOINT="$OVERLAY_CHECKPOINT" \
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
  BASELINE_NAME="phase08_guarded_overlay_expanded_${method}" \
  MASS_LABEL="$split" \
  FRICTION_LABEL="0.45" \
  POSE_SEED="$grasp_offset_delta" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  echo "=== PHASE08_GUARDED_OVERLAY_EXPANDED_CELL_END split=$split method=$method cell=$cell eval_tag=$eval_tag ==="
}

run_pair() {
  local split="$1"
  local cell="$2"
  local offset="$3"
  run_cell "$split" "$cell" "guarded_feedback" "lift_hold_feedback" "$offset"
  run_cell "$split" "$cell" "guarded_overlay" "lift_hold_feedback_residual_overlay" "$offset"
}

# No held-out cells. These are near-neighbor train/validation probes around
# the previous accepted train_c offset and the adjacent rejected boundaries.
run_pair "train" "pen_end_bias_overlay_train_c0" "-0.026,0.012,0"
run_pair "train" "pen_end_bias_overlay_train_c1" "-0.0265,0.012,0"
run_pair "train" "pen_end_bias_overlay_train_c2" "-0.026,0.0115,0"
run_pair "train" "pen_end_bias_overlay_train_c3" "-0.0265,0.0115,0"
run_pair "validation" "pen_end_bias_overlay_val_c0" "-0.026,0.0125,0"
run_pair "validation" "pen_end_bias_overlay_val_c1" "-0.0265,0.0125,0"
run_pair "validation" "pen_end_bias_overlay_val_c2" "-0.026,0.011,0"
run_pair "validation" "pen_end_bias_overlay_val_c3" "-0.0255,0.012,0"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" "$REPORT_PATH" "$OVERLAY_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
prefix = sys.argv[3]
suffix = sys.argv[4]
report_path = Path(sys.argv[5])
checkpoint = sys.argv[6]
cells = [
    ("train", "pen_end_bias_overlay_train_c0"),
    ("train", "pen_end_bias_overlay_train_c1"),
    ("train", "pen_end_bias_overlay_train_c2"),
    ("train", "pen_end_bias_overlay_train_c3"),
    ("validation", "pen_end_bias_overlay_val_c0"),
    ("validation", "pen_end_bias_overlay_val_c1"),
    ("validation", "pen_end_bias_overlay_val_c2"),
    ("validation", "pen_end_bias_overlay_val_c3"),
]
methods = ["guarded_feedback", "guarded_overlay"]

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

rows = []
paired = {}
for split, cell in cells:
    for method in methods:
        tag = f"{prefix}_{cell}_{method}_{suffix}"
        summary_path = root / "experiments" / "outputs" / f"{tag}_summary.json"
        metrics_path = root / "experiments" / "outputs" / f"{tag}_metrics.json"
        summary = load(summary_path)
        metrics = load(metrics_path)
        metric_row = (metrics.get("rows") or [{}])[0]
        world = (summary.get("task_metrics") or {}).get("per_world", [{}])[0]
        rows.append({
            "split": split,
            "cell": cell,
            "method": method,
            "metrics_status": metric_row.get("status"),
            "hold_duration_s": world.get("longest_hold_s"),
            "lift_height_m": world.get("max_lift"),
            "drop_from_max_m": world.get("drop_from_max"),
            "max_xy_drift_m": world.get("max_xy_drift"),
            "max_object_accel_m_s2": metric_row.get("max_object_accel_m_s2"),
            "max_slip_m": metric_row.get("max_slip_m"),
            "trigger_count": (summary.get("scripted_feedback") or {}).get("final_trigger_count"),
            "summary": str(summary_path.relative_to(root)),
            "metrics": str(metrics_path.relative_to(root)),
            "contact_sheet": summary.get("contact_sheet"),
            "rollout_video": summary.get("rollout_video"),
        })

for split, cell in cells:
    base = next(row for row in rows if row["cell"] == cell and row["method"] == "guarded_feedback")
    overlay = next(row for row in rows if row["cell"] == cell and row["method"] == "guarded_overlay")
    hold_gain = (overlay.get("hold_duration_s") or 0.0) - (base.get("hold_duration_s") or 0.0)
    lift_gain = (overlay.get("lift_height_m") or 0.0) - (base.get("lift_height_m") or 0.0)
    slip_non_regression = (overlay.get("max_slip_m") or 0.0) <= (base.get("max_slip_m") or 0.0)
    accel_non_regression = (overlay.get("max_object_accel_m_s2") or 0.0) <= (base.get("max_object_accel_m_s2") or 0.0)
    paired[cell] = {
        "split": split,
        "hold_gain_s": hold_gain,
        "lift_gain_m": lift_gain,
        "slip_non_regression": slip_non_regression,
        "accel_non_regression": accel_non_regression,
        "baseline_run_tag": f"{prefix}_{cell}_guarded_feedback_{suffix}",
        "overlay_run_tag": f"{prefix}_{cell}_guarded_overlay_{suffix}",
        "candidate_for_overlay_training": hold_gain >= 0.0 and lift_gain >= 0.0 and slip_non_regression and accel_non_regression,
    }

accepted = [cell for cell, item in paired.items() if item["candidate_for_overlay_training"]]
accepted_train = [cell for cell in accepted if paired[cell]["split"] == "train"]
accepted_validation = [cell for cell in accepted if paired[cell]["split"] == "validation"]
payload = {
    "classification": "phase08_guarded_overlay_expanded_probe_direct_v1",
    "run_tag": run_tag,
    "status": "pass_source_candidates_found" if accepted else "open_no_overlay_source_candidates",
    "not_training": True,
    "not_final_curiosity_success_claim": True,
    "overlay_checkpoint": checkpoint,
    "rows": rows,
    "paired": paired,
    "accepted_overlay_source_cells": accepted,
    "accepted_overlay_source_count": len(accepted),
    "accepted_train_overlay_source_count": len(accepted_train),
    "accepted_validation_overlay_source_count": len(accepted_validation),
    "held_out_cells_used": [],
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Phase08 Guarded Overlay Expanded Probe Direct V1",
    "",
    "Train/validation source probe only. Not training and not a success claim.",
    "",
    f"- status: `{payload['status']}`",
    f"- accepted overlay source count: `{len(accepted)}`",
    f"- accepted train count: `{len(accepted_train)}`",
    f"- accepted validation count: `{len(accepted_validation)}`",
    f"- overlay checkpoint: `{checkpoint}`",
    "",
    "## Cells",
    "",
]
for cell, item in paired.items():
    lines.append(
        f"- `{cell}` ({item['split']}): accepted `{item['candidate_for_overlay_training']}`, "
        f"hold gain `{item['hold_gain_s']}`, lift gain `{item['lift_gain_m']}`, "
        f"slip ok `{item['slip_non_regression']}`, accel ok `{item['accel_non_regression']}`"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE08_GUARDED_OVERLAY_EXPANDED_PROBE_DIRECT_END"
