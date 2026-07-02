#!/usr/bin/env bash
set -euo pipefail

# Evaluate Phase07 V2 policies on held-out cells with matched rollout settings.
# This is evaluation only, not training and not a success claim.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
RUN_TAG="${RUN_TAG:-phase07_v2_heldout_eval_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase07_v2_eval}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-28_phase07_v2_heldout_eval_v1.md}"
DEVICE="${DEVICE:-cuda:0}"
NO_CURIOSITY_CHECKPOINT="${NO_CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase07_v2_residual_adapter_trainer_v1_20260628/phase07_v2_residual_adapter_v1_train_20260628.pt}"
CURIOSITY_CHECKPOINT="${CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase07_v2_curiosity_weighted_residual_adapter_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_v1_train_20260628.pt}"
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
if [[ ! -f "$NO_CURIOSITY_CHECKPOINT" ]]; then
  echo "ERROR: missing no-curiosity checkpoint: $NO_CURIOSITY_CHECKPOINT" >&2
  exit 4
fi
if [[ ! -f "$CURIOSITY_CHECKPOINT" ]]; then
  echo "ERROR: missing curiosity checkpoint: $CURIOSITY_CHECKPOINT" >&2
  exit 5
fi

echo "PHASE07_V2_HELDOUT_EVAL_START"
echo "RUN_TAG=$RUN_TAG"
echo "EVAL_TAG_PREFIX=$EVAL_TAG_PREFIX"
echo "EVAL_TAG_SUFFIX=$EVAL_TAG_SUFFIX"
echo "REPORT_PATH=$REPORT_PATH"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "NO_CURIOSITY_CHECKPOINT=$NO_CURIOSITY_CHECKPOINT"
echo "CURIOSITY_CHECKPOINT=$CURIOSITY_CHECKPOINT"
echo "ACTIVE_THRESHOLD=$ACTIVE_THRESHOLD"
echo "NOTE=evaluation_only_not_training_not_success_claim"
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
  local method="$1"
  local cell="$2"
  local controller_mode="$3"
  local checkpoint="$4"
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
  local eval_tag="${EVAL_TAG_PREFIX}_${cell}_${method}_${EVAL_TAG_SUFFIX}"
  echo "=== PHASE07_V2_HELDOUT_CELL_START method=$method cell=$cell eval_tag=$eval_tag ==="
  if [[ "$controller_mode" == "lift_hold_learned_residual" ]]; then
    RUN_TAG="$eval_tag" \
    NEWTON_VENV="$NEWTON_VENV" \
    TRAINER_VENV="$TRAINER_VENV" \
    SCENE="cube" \
    TRACKED_OBJECT="existing_cup_asset" \
    CONTROLLER_MODE="$controller_mode" \
    FINAL_HOLD_DURATION="3.0" \
    LIFT_HEIGHT_MIN="0.12" \
    HOLD_DURATION_MIN="2.8" \
    DROP_HEIGHT_LOSS="0.05" \
    PHYSICS_VARIANT_LABEL="phase07_v2_eval_${cell}_${method}" \
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
    FEEDBACK_LIFT_DURATION_SCALE_MAX="1.25" \
    FEEDBACK_HOLD_HEIGHT_STEP="0.0005" \
    FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="0.005" \
    FEEDBACK_STABILIZATION_STEP="0.15" \
    FEEDBACK_STABILIZATION_MAX="0.9" \
    PRE_RECORD_WARMUP_STEPS="15" \
    RESIDUAL_ADAPTER_CHECKPOINT="$checkpoint" \
    RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="$ACTIVE_THRESHOLD" \
    NUM_STEPS="420" \
    SAMPLE_STEPS="0,105,210,315,419" \
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
    SCENE="cube" \
    TRACKED_OBJECT="existing_cup_asset" \
    CONTROLLER_MODE="$controller_mode" \
    FINAL_HOLD_DURATION="3.0" \
    LIFT_HEIGHT_MIN="0.12" \
    HOLD_DURATION_MIN="2.8" \
    DROP_HEIGHT_LOSS="0.05" \
    PHYSICS_VARIANT_LABEL="phase07_v2_eval_${cell}_${method}" \
    BODY_MASS_SCALE="1.0" \
    SHAPE_FRICTION_SCALE="1.0" \
    OBJECT_MASS_KG="$mass" \
    OBJECT_FRICTION_MU="$friction" \
    FILL_LABEL="$fill" \
    NOMINAL_VISUAL_FILL="$nominal_visual_fill" \
    VISUAL_FILL_CUE="$visual_fill_cue" \
    VISUAL_FILL_CUE_RENDERED="0" \
    PRE_RECORD_WARMUP_STEPS="15" \
    NUM_STEPS="420" \
    SAMPLE_STEPS="0,105,210,315,419" \
    VIDEO_FRAME_STRIDE="1" \
    VIDEO_FPS="12" \
    DEVICE="$DEVICE" \
    NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
      bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
        2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"
  fi

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase07_v2_${method}" \
  MASS_LABEL="$fill" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"

  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  validate_action_bridge_npz "$eval_tag"
  echo "=== PHASE07_V2_HELDOUT_CELL_END method=$method cell=$cell eval_tag=$eval_tag ==="
}

for cell in empty_high_misleading full_low_hidden three_quarter_low_misleading; do
  run_eval_cell "no_adaptation" "$cell" "lift_hold" ""
  run_eval_cell "no_curiosity_residual" "$cell" "lift_hold_learned_residual" "$NO_CURIOSITY_CHECKPOINT"
  run_eval_cell "curiosity_weighted" "$cell" "lift_hold_learned_residual" "$CURIOSITY_CHECKPOINT"
done

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$ACTIVE_THRESHOLD" "$EVAL_TAG_PREFIX" "$EVAL_TAG_SUFFIX" "$REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
active_threshold = float(sys.argv[3])
eval_tag_prefix = sys.argv[4]
eval_tag_suffix = sys.argv[5]
report_path = Path(sys.argv[6])
cells = ["empty_high_misleading", "full_low_hidden", "three_quarter_low_misleading"]
methods = ["no_adaptation", "no_curiosity_residual", "curiosity_weighted"]

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
beats_no_curiosity_all = True
for cell in cells:
    per_method = {}
    for method in methods:
        eval_tag = f"{eval_tag_prefix}_{cell}_{method}_{eval_tag_suffix}"
        row = metric_row(eval_tag)
        summary = load(root / "experiments" / "outputs" / f"{eval_tag}_summary.json")
        per_method[method] = {
            "eval_run_tag": eval_tag,
            "metric_row": row,
            "score_tuple": list(score(row)),
            "summary": f"experiments/outputs/{eval_tag}_summary.json",
            "metrics": f"experiments/outputs/{eval_tag}_metrics.json",
            "rollout_video": summary.get("rollout_video"),
            "contact_sheet": summary.get("contact_sheet"),
        }
    curiosity_score = tuple(per_method["curiosity_weighted"]["score_tuple"])
    no_adapt_score = tuple(per_method["no_adaptation"]["score_tuple"])
    no_curiosity_score = tuple(per_method["no_curiosity_residual"]["score_tuple"])
    safety_regressions_vs_no_adapt = []
    safety_regressions_vs_no_curiosity = []
    for key in ("max_slip_m", "contact_loss_frames", "drop_height_loss_m", "max_object_accel_m_s2"):
        cur = float(per_method["curiosity_weighted"]["metric_row"].get(key, 0.0))
        nad = float(per_method["no_adaptation"]["metric_row"].get(key, 0.0))
        ncr = float(per_method["no_curiosity_residual"]["metric_row"].get(key, 0.0))
        if cur > nad:
            safety_regressions_vs_no_adapt.append(f"{key}_regression")
        if cur > ncr:
            safety_regressions_vs_no_curiosity.append(f"{key}_regression")
    beats_no_adapt = curiosity_score > no_adapt_score and not safety_regressions_vs_no_adapt
    beats_no_curiosity = curiosity_score > no_curiosity_score and not safety_regressions_vs_no_curiosity
    beats_no_adaptation_all = beats_no_adaptation_all and beats_no_adapt
    beats_no_curiosity_all = beats_no_curiosity_all and beats_no_curiosity
    per_cell[cell] = {
        "methods": per_method,
        "active_threshold": active_threshold,
        "curiosity_beats_no_adaptation_without_safety_regression": beats_no_adapt,
        "curiosity_beats_no_curiosity_residual_without_safety_regression": beats_no_curiosity,
        "safety_regressions_vs_no_adaptation": safety_regressions_vs_no_adapt,
        "safety_regressions_vs_no_curiosity_residual": safety_regressions_vs_no_curiosity,
    }

status = (
    "pass_candidate_needs_manual_visual_and_mainstream_gate"
    if beats_no_adaptation_all and beats_no_curiosity_all
    else "open_not_satisfied"
)
payload = {
    "classification": "phase07_v2_heldout_eval_summary_v1",
    "run_tag": run_tag,
    "status": status,
    "not_training": True,
    "not_success_claim": True,
    "active_threshold": active_threshold,
    "eval_tag_prefix": eval_tag_prefix,
    "eval_tag_suffix": eval_tag_suffix,
    "held_out_cells": cells,
    "methods": methods,
    "per_cell": per_cell,
    "curiosity_beats_no_adaptation_all_cells_without_safety_regression": beats_no_adaptation_all,
    "curiosity_beats_no_curiosity_residual_all_cells_without_safety_regression": beats_no_curiosity_all,
    "manual_visual_inspection": "pending_direct_agent_check",
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
report = report_path
lines = [
    "# Phase07 V2 Held-Out Eval V1",
    "",
    "Evaluation only. Not training and not a success claim.",
    "",
    f"- status: `{status}`",
    f"- active threshold: `{active_threshold}`",
    f"- curiosity beats no-adaptation all cells without safety regression: `{beats_no_adaptation_all}`",
    f"- curiosity beats no-curiosity residual all cells without safety regression: `{beats_no_curiosity_all}`",
    "",
    "## Cells",
    "",
]
for cell, item in per_cell.items():
    lines.append(f"- `{cell}`: curiosity_vs_no_adaptation `{item['curiosity_beats_no_adaptation_without_safety_regression']}`, curiosity_vs_no_curiosity `{item['curiosity_beats_no_curiosity_residual_without_safety_regression']}`")
    for method, details in item["methods"].items():
        row = details["metric_row"]
        lines.append(f"  - `{method}`: hold `{row.get('hold_duration_s')}`, lift `{row.get('lift_height_m')}`, accel `{row.get('max_object_accel_m_s2')}`, video `{details.get('rollout_video')}`")
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE07_V2_HELDOUT_EVAL_END"
