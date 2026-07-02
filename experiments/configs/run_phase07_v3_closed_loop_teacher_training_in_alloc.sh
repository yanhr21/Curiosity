#!/usr/bin/env bash
set -euo pipefail

# Full Phase07 V3 closed-loop repair chain:
# 1. collect on-policy learned-residual rollouts on train/validation cells;
# 2. record scripted corrective teacher labels under candidate.teacher.*;
# 3. build a no-held-out-leakage preflight;
# 4. train a residual adapter for at least one GPU-hour;
# 5. evaluate on harder held-out cells with full rollout videos.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
RUN_TAG="${RUN_TAG:-phase07_v3_closed_loop_teacher_chain_v1_20260628}"
SOURCE_POLICY_CHECKPOINT="${SOURCE_POLICY_CHECKPOINT:-$ROOT/checkpoints/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_trainer_v1_20260628/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_v1_train_20260628.pt}"
SOURCE_POLICY_ACTIVE_THRESHOLD="${SOURCE_POLICY_ACTIVE_THRESHOLD:-0.5}"
PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-$ROOT/experiments/configs/phase07_v3_closed_loop_teacher_preflight_v1.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/experiments/configs/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1.json}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-phase07_v3_closed_loop_teacher_residual_adapter_v1_train_20260628}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628}"
EVAL_TAG_PREFIX="${EVAL_TAG_PREFIX:-phase07_v3_closed_loop_teacher_eval}"
EVAL_TAG_SUFFIX="${EVAL_TAG_SUFFIX:-20260628}"
REPORT_PATH="${REPORT_PATH:-$ROOT/experiments/reports/2026-06-28_phase07_v3_closed_loop_teacher_heldout_eval_v1.md}"
NO_CURIOSITY_CHECKPOINT="${NO_CURIOSITY_CHECKPOINT:-$ROOT/checkpoints/phase07_v3_repaired_base_residual_adapter_trainer_v1_20260628/phase07_v3_repaired_base_residual_adapter_v1_train_20260628.pt}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals checkpoints data/processed

if [[ ! -x "$NEWTON_VENV/bin/python" || ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: required envs must exist under envs/ before compute use." >&2
  exit 3
fi
if [[ ! -f "$SOURCE_POLICY_CHECKPOINT" ]]; then
  echo "ERROR: missing source policy checkpoint: $SOURCE_POLICY_CHECKPOINT" >&2
  exit 4
fi
if [[ ! -f "$NO_CURIOSITY_CHECKPOINT" ]]; then
  echo "ERROR: missing no-curiosity checkpoint: $NO_CURIOSITY_CHECKPOINT" >&2
  exit 5
fi

echo "PHASE07_V3_CLOSED_LOOP_TEACHER_CHAIN_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "SOURCE_POLICY_CHECKPOINT=$SOURCE_POLICY_CHECKPOINT"
echo "SOURCE_POLICY_ACTIVE_THRESHOLD=$SOURCE_POLICY_ACTIVE_THRESHOLD"
echo "PREFLIGHT_CONFIG=$PREFLIGHT_CONFIG"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"
echo "TRAIN_RUN_TAG=$TRAIN_RUN_TAG"
echo "EVAL_RUN_TAG=$EVAL_RUN_TAG"
echo "NOTE=closed_loop_dagger_style_training_chain_not_success_claim_until_heldout_and_mainstream_gates_pass"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,190p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

collect_cell() {
  local cell="$1"
  local mass="$2"
  local friction="$3"
  local fill="$4"
  local nominal_visual_fill="$5"
  local visual_fill_cue="$6"
  local run_tag="phase07_v3_closed_loop_teacher_${cell}_20260628"
  echo "=== PHASE07_V3_CLOSED_LOOP_SOURCE_CELL_START cell=$cell run_tag=$run_tag ==="
  RUN_TAG="$run_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="cube" \
  TRACKED_OBJECT="existing_cup_asset" \
  CONTROLLER_MODE="lift_hold_learned_residual" \
  FINAL_HOLD_DURATION="3.0" \
  LIFT_HEIGHT_MIN="0.12" \
  HOLD_DURATION_MIN="2.8" \
  DROP_HEIGHT_LOSS="0.05" \
  PHYSICS_VARIANT_LABEL="phase07_v3_closed_loop_teacher_${cell}" \
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
  RESIDUAL_ADAPTER_CHECKPOINT="$SOURCE_POLICY_CHECKPOINT" \
  RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="$SOURCE_POLICY_ACTIVE_THRESHOLD" \
  RECORD_SCRIPTED_TEACHER_LABELS="1" \
  NUM_STEPS="420" \
  SAMPLE_STEPS="0,105,210,315,419" \
  VIDEO_FRAME_STRIDE="10" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${run_tag}.log"
  echo "=== PHASE07_V3_CLOSED_LOOP_SOURCE_CELL_END cell=$cell run_tag=$run_tag ==="
}

collect_cell "quarter_low_truthful" "0.14" "0.35" "quarter" "0.25" "truthful_fill_cue"
collect_cell "quarter_medium_hidden" "0.14" "0.75" "quarter" "0.25" "hidden_fill_cue"
collect_cell "half_low_hidden" "0.21" "0.35" "half" "0.5" "hidden_fill_cue"
collect_cell "half_medium_truthful" "0.21" "0.75" "half" "0.5" "truthful_fill_cue"
collect_cell "three_quarter_medium_misleading" "0.29" "0.75" "three_quarter" "0.75" "misleading_fill_cue"
collect_cell "three_quarter_high_truthful" "0.29" "1.2" "three_quarter" "0.75" "truthful_fill_cue"
collect_cell "empty_medium_hidden" "0.08" "0.75" "empty" "0.0" "hidden_fill_cue"
collect_cell "full_medium_misleading" "0.35" "0.75" "full" "1.0" "misleading_fill_cue"

last_sanity="$ROOT/experiments/outputs/phase07_v3_closed_loop_teacher_full_medium_misleading_20260628_fresh_newton_sensor_contact_sanity.json"

echo "=== PHASE07_V3_CLOSED_LOOP_PREFLIGHT_START ==="
"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_phase07_closed_loop_teacher_preflight_v1.py" \
  --config "$PREFLIGHT_CONFIG" \
  --root "$ROOT" \
  --fresh-sanity-json "$last_sanity"
echo "=== PHASE07_V3_CLOSED_LOOP_PREFLIGHT_END ==="

echo "=== PHASE07_V3_CLOSED_LOOP_TRAIN_START ==="
CONFIG="$TRAIN_CONFIG" \
RUN_TAG="$TRAIN_RUN_TAG" \
RUN_MODE="train" \
NEWTON_VENV="$NEWTON_VENV" \
TRAINER_VENV="$TRAINER_VENV" \
DEVICE="$DEVICE" \
  bash "$ROOT/experiments/configs/run_residual_adapter_trainer_in_alloc.sh"
echo "=== PHASE07_V3_CLOSED_LOOP_TRAIN_END ==="

closed_loop_checkpoint="$ROOT/checkpoints/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1_20260628/${TRAIN_RUN_TAG}.pt"
if [[ ! -f "$closed_loop_checkpoint" ]]; then
  echo "ERROR: expected closed-loop checkpoint not found: $closed_loop_checkpoint" >&2
  exit 6
fi

echo "=== PHASE07_V3_CLOSED_LOOP_HELDOUT_EVAL_START ==="
RUN_TAG="$EVAL_RUN_TAG" \
EVAL_TAG_PREFIX="$EVAL_TAG_PREFIX" \
EVAL_TAG_SUFFIX="$EVAL_TAG_SUFFIX" \
REPORT_PATH="$REPORT_PATH" \
NEWTON_VENV="$NEWTON_VENV" \
TRAINER_VENV="$TRAINER_VENV" \
DEVICE="$DEVICE" \
ACTIVE_THRESHOLD="0.5" \
NO_CURIOSITY_CHECKPOINT="$NO_CURIOSITY_CHECKPOINT" \
CURIOSITY_CHECKPOINT="$closed_loop_checkpoint" \
  bash "$ROOT/experiments/configs/run_phase07_v2_heldout_eval_in_alloc.sh"
echo "=== PHASE07_V3_CLOSED_LOOP_HELDOUT_EVAL_END ==="

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$TRAIN_RUN_TAG" "$EVAL_RUN_TAG" "$closed_loop_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
train_run_tag = sys.argv[3]
eval_run_tag = sys.argv[4]
checkpoint = Path(sys.argv[5])
payload = {
    "classification": "phase07_v3_closed_loop_teacher_chain_summary_v1",
    "status": "heldout_eval_completed_pending_manual_visual_and_hard_gate_audit",
    "run_tag": run_tag,
    "train_run_tag": train_run_tag,
    "eval_run_tag": eval_run_tag,
    "checkpoint": str(checkpoint),
    "preflight_manifest": "data/processed/phase07_v3_closed_loop_teacher_preflight_v1_20260628/manifest.json",
    "train_summary": f"experiments/outputs/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1_20260628/{train_run_tag}_summary.json",
    "heldout_summary": f"experiments/outputs/{eval_run_tag}_summary.json",
    "success_claim": False,
    "not_success_claim_until_manual_visual_baseline_and_mainstream_gates_pass": True,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "PHASE07_V3_CLOSED_LOOP_TEACHER_CHAIN_END"
