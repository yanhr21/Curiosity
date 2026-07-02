#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: probe official Newton non-cup object families (especially
# pen) to find a harder distribution where no-adaptation leaves room for
# closed-loop intervention. Not training and not a success claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
RUN_TAG="${RUN_TAG:-phase08_object_family_probe_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase08_object_probe}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

echo "PHASE08_OBJECT_FAMILY_PROBE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NOTE=object_family_source_selection_diagnostic_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,190p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

run_probe_cell() {
  local method="$1"
  local cell="$2"
  local controller_mode="$3"
  local scene="$4"
  local tracked_object="$5"
  local mass="$6"
  local friction="$7"
  local fill="$8"
  local nominal_visual_fill="$9"
  local visual_fill_cue="${10}"
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_${method}_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE08_OBJECT_PROBE_CELL_START method=$method cell=$cell scene=$scene eval_tag=$eval_tag ==="
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="$scene" \
  TRACKED_OBJECT="$tracked_object" \
  CONTROLLER_MODE="$controller_mode" \
  FINAL_HOLD_DURATION="3.0" \
  LIFT_HEIGHT_MIN="0.12" \
  HOLD_DURATION_MIN="2.8" \
  DROP_HEIGHT_LOSS="0.05" \
  PHYSICS_VARIANT_LABEL="phase08_object_probe_${cell}_${method}" \
  BODY_MASS_SCALE="1.0" \
  SHAPE_FRICTION_SCALE="1.0" \
  OBJECT_MASS_KG="$mass" \
  OBJECT_FRICTION_MU="$friction" \
  FILL_LABEL="$fill" \
  NOMINAL_VISUAL_FILL="$nominal_visual_fill" \
  VISUAL_FILL_CUE="$visual_fill_cue" \
  VISUAL_FILL_CUE_RENDERED="0" \
  FEEDBACK_MIN_CONTACT_COUNT="35" \
  FEEDBACK_ACCEL_THRESHOLD="6.5" \
  FEEDBACK_HEIGHT_DROP_THRESHOLD="0.015" \
  FEEDBACK_INITIAL_LIFT_DURATION_SCALE="1.65" \
  FEEDBACK_LIFT_DURATION_SCALE_MAX="1.35" \
  FEEDBACK_HOLD_HEIGHT_STEP="0.0005" \
  FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="0.008" \
  FEEDBACK_STABILIZATION_STEP="0.2" \
  FEEDBACK_STABILIZATION_MAX="1.2" \
  PRE_RECORD_WARMUP_STEPS="15" \
  NUM_STEPS="420" \
  SAMPLE_STEPS="0,105,210,315,419" \
  VIDEO_FRAME_STRIDE="1" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase08_object_probe_${method}" \
  MASS_LABEL="$fill" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  echo "=== PHASE08_OBJECT_PROBE_CELL_END method=$method cell=$cell eval_tag=$eval_tag ==="
}

run_probe_cell "no_adaptation" "pen_nominal_medium" "lift_hold" "pen" "official_object" "0.08" "0.7" "pen" "0.5" "hidden_fill_cue"
run_probe_cell "scripted_feedback" "pen_nominal_medium" "lift_hold_feedback" "pen" "official_object" "0.08" "0.7" "pen" "0.5" "hidden_fill_cue"
run_probe_cell "no_adaptation" "pen_low_misleading" "lift_hold" "pen" "official_object" "0.08" "0.35" "pen" "0.5" "misleading_fill_cue"
run_probe_cell "scripted_feedback" "pen_low_misleading" "lift_hold_feedback" "pen" "official_object" "0.08" "0.35" "pen" "0.5" "misleading_fill_cue"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
prefix = sys.argv[3]
suffix = sys.argv[4]
cells = ["pen_nominal_medium", "pen_low_misleading"]
methods = ["no_adaptation", "scripted_feedback"]
rows = []
for cell in cells:
    for method in methods:
        tag = f"{prefix}_{cell}_{method}_{suffix}"
        summary_path = root / "experiments" / "outputs" / f"{tag}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        world = (summary.get("task_metrics") or {}).get("per_world", [{}])[0]
        rows.append({
            "cell": cell,
            "method": method,
            "status": summary.get("status"),
            "success_all_worlds": (summary.get("task_metrics") or {}).get("success_all_worlds"),
            "hold_duration_s": world.get("longest_hold_s"),
            "lift_height_m": world.get("max_lift"),
            "drop_from_max_m": world.get("drop_from_max"),
            "max_xy_drift_m": world.get("max_xy_drift"),
            "scripted_feedback_trigger_count": (summary.get("scripted_feedback") or {}).get("final_trigger_count"),
            "summary": str(summary_path.relative_to(root)),
            "contact_sheet": summary.get("contact_sheet"),
            "rollout_video": summary.get("rollout_video"),
        })
payload = {
    "classification": "phase08_object_family_probe_v1",
    "status": "complete_diagnostic_not_training_not_success_claim",
    "run_tag": run_tag,
    "rows": rows,
    "purpose": "find official non-cup object candidates where no-adaptation leaves room for closed-loop intervention",
    "success_claim": False,
    "training_started": False,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE08_OBJECT_FAMILY_PROBE_END"
