#!/usr/bin/env bash
set -euo pipefail

# Run the remaining Phase07 ablation trainings and held-out evaluations inside
# an already-held Slurm allocation. This script must not be run on the login
# node. It records pending manual visual inspection, but does not claim final
# success.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
ABLATED_VARIANTS="${ABLATED_VARIANTS:-contact_only shuffled_contact delayed_contact no_learning_progress}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/visuals checkpoints

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local trainer venv python at $TRAINER_VENV/bin/python" >&2
  exit 4
fi

echo "PHASE07_REMAINING_ABLATION_QUEUE_START"
echo "ROOT=$ROOT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "ABLATED_VARIANTS=$ABLATED_VARIANTS"
echo "QUEUE_RESULT=not_final_curiosity_success"
echo "QUEUE_REASON=remaining_ablation_training_and_evaluation_only"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

echo "=== PHASE07_EXISTING_BASELINE_ACTION_BRIDGE_BACKFILL_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_candidate_action_bridge_backfill_in_alloc.sh"
echo "=== PHASE07_EXISTING_BASELINE_ACTION_BRIDGE_BACKFILL_END ==="

echo "=== PHASE07_MAINSTREAM_ADAPTER_CONVERSION_PREFLIGHT_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_mainstream_adapter_conversion_preflight_in_alloc.sh"
echo "=== PHASE07_MAINSTREAM_ADAPTER_CONVERSION_PREFLIGHT_END ==="

echo "=== PHASE07_MAINSTREAM_STAGE1_DATASET_INDEX_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_mainstream_stage1_dataset_index_in_alloc.sh"
echo "=== PHASE07_MAINSTREAM_STAGE1_DATASET_INDEX_END ==="

echo "=== PHASE07_STAGE1_NO_HELDOUT_LEAKAGE_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_stage1_no_heldout_leakage_in_alloc.sh"
echo "=== PHASE07_STAGE1_NO_HELDOUT_LEAKAGE_END ==="

validate_training_summary() {
  local variant="$1"
  local run_tag="$2"
  local config_path="$3"
  "$TRAINER_VENV/bin/python" - "$ROOT" "$variant" "$run_tag" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
variant = sys.argv[2]
run_tag = sys.argv[3]
config_path = root / sys.argv[4]
config = json.loads(config_path.read_text(encoding="utf-8"))
summary_path = root / config["output_dir"] / f"{run_tag}_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"missing training summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
failures = []
if summary.get("status") != "pass":
    failures.append(f"status={summary.get('status')}")
if summary.get("run_mode") != "train":
    failures.append(f"run_mode={summary.get('run_mode')}")
if summary.get("real_training_result") is not True:
    failures.append("real_training_result_not_true")
if summary.get("checkpoint_written") is not True:
    failures.append("checkpoint_not_written")
checkpoint = summary.get("checkpoint_path")
if not checkpoint or not (root / checkpoint).is_file():
    failures.append(f"missing_checkpoint={checkpoint}")
if summary.get("ablation_name") != variant:
    failures.append(f"ablation_name={summary.get('ablation_name')}")
if failures:
    raise SystemExit(";".join(failures))
print(json.dumps({
    "status": "pass",
    "variant": variant,
    "run_tag": run_tag,
    "summary": str(summary_path.relative_to(root)),
    "checkpoint": checkpoint,
}, indent=2, sort_keys=True))
PY
}

checkpoint_from_summary() {
  local run_tag="$1"
  local config_path="$2"
  "$TRAINER_VENV/bin/python" - "$ROOT" "$run_tag" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
config = json.loads((root / sys.argv[3]).read_text(encoding="utf-8"))
summary = json.loads((root / config["output_dir"] / f"{run_tag}_summary.json").read_text(encoding="utf-8"))
print(root / summary["checkpoint_path"])
PY
}

validate_action_bridge_npz() {
  local run_tag="$1"
  "$NEWTON_VENV/bin/python" - "$ROOT" "$run_tag" <<'PY'
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
if not npz_path.is_file():
    raise SystemExit(f"missing NPZ for action bridge validation: {npz_path}")
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
  local variant="$1"
  local cell="$2"
  local checkpoint="$3"
  local run_tag="phase07_eval_${cell}_${variant}_20260627"
  local physics_label="phase07_eval_${cell}_${variant}"
  local mass=""
  local friction=""
  local fill=""
  local nominal_visual_fill=""
  local visual_fill_cue=""

  case "$cell" in
    empty_high_misleading)
      mass="0.08"
      friction="1.2"
      fill="empty"
      nominal_visual_fill="0.0"
      visual_fill_cue="misleading_fill_cue"
      ;;
    full_low_hidden)
      mass="0.35"
      friction="0.35"
      fill="full"
      nominal_visual_fill="1.0"
      visual_fill_cue="hidden_fill_cue"
      ;;
    three_quarter_low_misleading)
      mass="0.29"
      friction="0.35"
      fill="three_quarter"
      nominal_visual_fill="0.75"
      visual_fill_cue="misleading_fill_cue"
      ;;
    *)
      echo "ERROR: unknown held-out cell: $cell" >&2
      exit 20
      ;;
  esac

  {
    printf 'RUN_TAG=%q\n' "$run_tag"
    printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
    printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
    printf 'SCENE=%q\n' "cube"
    printf 'TRACKED_OBJECT=%q\n' "existing_cup_asset"
    printf 'CONTROLLER_MODE=%q\n' "lift_hold_learned_residual"
    printf 'FINAL_HOLD_DURATION=%q\n' "3.0"
    printf 'LIFT_HEIGHT_MIN=%q\n' "0.12"
    printf 'HOLD_DURATION_MIN=%q\n' "2.0"
    printf 'DROP_HEIGHT_LOSS=%q\n' "0.05"
    printf 'PHYSICS_VARIANT_LABEL=%q\n' "$physics_label"
    printf 'BODY_MASS_SCALE=%q\n' "1.0"
    printf 'SHAPE_FRICTION_SCALE=%q\n' "1.0"
    printf 'OBJECT_MASS_KG=%q\n' "$mass"
    printf 'OBJECT_FRICTION_MU=%q\n' "$friction"
    printf 'FILL_LABEL=%q\n' "$fill"
    printf 'NOMINAL_VISUAL_FILL=%q\n' "$nominal_visual_fill"
    printf 'VISUAL_FILL_CUE=%q\n' "$visual_fill_cue"
    printf 'VISUAL_FILL_CUE_RENDERED=%q\n' "0"
    printf 'FEEDBACK_MIN_CONTACT_COUNT=%q\n' "58"
    printf 'FEEDBACK_ACCEL_THRESHOLD=%q\n' "6.5"
    printf 'FEEDBACK_HEIGHT_DROP_THRESHOLD=%q\n' "0.015"
    printf 'FEEDBACK_INITIAL_LIFT_DURATION_SCALE=%q\n' "1.65"
    printf 'FEEDBACK_LIFT_DURATION_SCALE_MAX=%q\n' "1.05"
    printf 'FEEDBACK_HOLD_HEIGHT_STEP=%q\n' "0.0005"
    printf 'FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=%q\n' "0.005"
    printf 'FEEDBACK_STABILIZATION_STEP=%q\n' "0.05"
    printf 'FEEDBACK_STABILIZATION_MAX=%q\n' "0.3"
    printf 'PRE_RECORD_WARMUP_STEPS=%q\n' "15"
    printf 'RESIDUAL_ADAPTER_CHECKPOINT=%q\n' "$checkpoint"
    printf 'RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=%q\n' "0.5"
    printf 'NUM_STEPS=%q\n' "360"
    printf 'SAMPLE_STEPS=%q\n' "0,90,180,270,359"
    printf 'VIDEO_FRAME_STRIDE=%q\n' "1"
    printf 'VIDEO_FPS=%q\n' "12"
    printf 'DEVICE=%q\n' "$DEVICE"
    printf 'NEWTON_CACHE_PATH=%q\n' "$ROOT/external/newton-assets-cache"
  } >"$ROOT/logs/newton/${run_tag}_env.sh"

  echo "=== PHASE07_EVAL_START variant=$variant cell=$cell run_tag=$run_tag ==="
  RUN_TAG="$run_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="cube" \
  TRACKED_OBJECT="existing_cup_asset" \
  CONTROLLER_MODE="lift_hold_learned_residual" \
  FINAL_HOLD_DURATION="3.0" \
  LIFT_HEIGHT_MIN="0.12" \
  HOLD_DURATION_MIN="2.0" \
  DROP_HEIGHT_LOSS="0.05" \
  PHYSICS_VARIANT_LABEL="$physics_label" \
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
  RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="0.5" \
  NUM_STEPS="360" \
  SAMPLE_STEPS="0,90,180,270,359" \
  VIDEO_FRAME_STRIDE="1" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh"

  RUN_TAG="$run_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase07_${variant}_ablation_learned_residual" \
  MASS_LABEL="$fill" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"

  RUN_TAG="$run_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh"
  validate_action_bridge_npz "$run_tag"
  echo "=== PHASE07_EVAL_END variant=$variant cell=$cell run_tag=$run_tag ==="
}

for variant in $ABLATED_VARIANTS; do
  config_path="experiments/configs/phase07_${variant}_residual_adapter_trainer_v1.json"
  if [[ ! -f "$config_path" ]]; then
    echo "ERROR: missing ablation config: $config_path" >&2
    exit 10
  fi
  run_tag="phase07_${variant}_residual_adapter_v1_train_20260627"
  if [[ "$variant" == "contact_only" ]]; then
    run_tag="phase07_contact_only_residual_adapter_v1_train_retry_20260627"
  fi
  {
    printf 'RUN_TAG=%q\n' "$run_tag"
    printf 'RUN_MODE=%q\n' "train"
    printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
    printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
    printf 'CONFIG=%q\n' "$ROOT/$config_path"
    printf 'DEVICE=%q\n' "$DEVICE"
    printf 'ALLOW_REAL_TRAINING=%q\n' "1"
    printf 'ABLATED_VARIANT=%q\n' "$variant"
    printf 'QUEUE_SCRIPT=%q\n' "$ROOT/experiments/configs/run_phase07_remaining_ablation_queue_in_alloc.sh"
  } >"$ROOT/logs/newton/${run_tag}_env.sh"
  echo "=== PHASE07_ABLATION_TRAIN_START variant=$variant run_tag=$run_tag ==="
  ALLOW_REAL_TRAINING=1 \
  RUN_TAG="$run_tag" \
  RUN_MODE="train" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  CONFIG="$ROOT/$config_path" \
  DEVICE="$DEVICE" \
    bash "$ROOT/experiments/configs/run_curiosity_weighted_residual_adapter_trainer_in_alloc.sh"
  validate_training_summary "$variant" "$run_tag" "$config_path"
  checkpoint="$(checkpoint_from_summary "$run_tag" "$config_path")"
  echo "PHASE07_ABLATION_CHECKPOINT variant=$variant checkpoint=$checkpoint"

  run_eval_cell "$variant" "empty_high_misleading" "$checkpoint"
  run_eval_cell "$variant" "full_low_hidden" "$checkpoint"
  run_eval_cell "$variant" "three_quarter_low_misleading" "$checkpoint"
  echo "=== PHASE07_ABLATION_COMPLETE variant=$variant run_tag=$run_tag ==="
done

echo "=== PHASE07_OFFICIAL_METHOD_READINESS_START ==="
NEWTON_VENV="$NEWTON_VENV" \
  bash "$ROOT/experiments/configs/run_phase07_official_method_readiness_in_alloc.sh"
echo "=== PHASE07_OFFICIAL_METHOD_READINESS_END ==="

echo "=== PHASE07_HELDOUT_COMPARISON_REPORT_START ==="
"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_phase07_heldout_comparison_report_v1.py" \
  --root "$ROOT" \
  --output "$ROOT/experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json" \
  --report "$ROOT/experiments/reports/2026-06-27_phase07_heldout_comparison_report_v1.md"
echo "=== PHASE07_HELDOUT_COMPARISON_REPORT_END ==="

echo "=== PHASE07_HARD_TRAINING_EVIDENCE_GATE_START ==="
"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/audit_phase07_hard_training_evidence_gate_v1.py" \
  --root "$ROOT" \
  --output "$ROOT/experiments/outputs/phase07_hard_training_evidence_gate_v1_20260627.json" \
  --report "$ROOT/experiments/reports/2026-06-27_phase07_hard_training_evidence_gate_v1.md"
echo "=== PHASE07_HARD_TRAINING_EVIDENCE_GATE_END ==="

echo "PHASE07_REMAINING_ABLATION_QUEUE_END"
echo "FINAL_STATUS=ablation_queue_complete_pending_manual_visual_inspection_and_report_update"
