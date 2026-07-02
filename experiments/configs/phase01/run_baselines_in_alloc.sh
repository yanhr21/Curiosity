#!/usr/bin/env bash
set -euo pipefail

# Run Phase 01 held-out baselines inside a Curiosity-owned tmux-held H200
# allocation. This is evaluation/baseline evidence, not curiosity success.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_base_$(date +%Y%m%d_%H%M%S)}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
BASELINE_CONFIG="${BASELINE_CONFIG:-$ROOT/experiments/configs/phase01/baselines.json}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-phase01/core/baselines}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/baselines}"
VISUAL_PHASE_DIR="${VISUAL_PHASE_DIR:-phase01/core/baselines}"
NUM_STEPS="${NUM_STEPS:-1800}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,300,600,900,1200,1500,1799}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-3}"
VIDEO_FPS="${VIDEO_FPS:-20}"
FINAL_HOLD_DURATION="${FINAL_HOLD_DURATION:-12.0}"
LIFT_HEIGHT_MIN="${LIFT_HEIGHT_MIN:-0.12}"
HOLD_DURATION_MIN="${HOLD_DURATION_MIN:-8.0}"
DROP_HEIGHT_LOSS="${DROP_HEIGHT_LOSS:-0.05}"
RUN_NO_ADAPTATION="${RUN_NO_ADAPTATION:-1}"
RUN_SCRIPTED_FEEDBACK="${RUN_SCRIPTED_FEEDBACK:-1}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR" \
  "$ROOT/experiments/outputs/$OUTPUT_SUBDIR" \
  "$ROOT/experiments/reports/phase01/core/baselines" \
  "$ROOT/experiments/visuals/$VISUAL_PHASE_DIR"

if [[ ! -f "$BASELINE_CONFIG" ]]; then
  echo "ERROR: missing Phase 01 baseline config: $BASELINE_CONFIG" >&2
  exit 3
fi

echo "PHASE01_BASELINES_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "BASELINE_CONFIG=$BASELINE_CONFIG"
echo "OUTPUT_SUBDIR=$OUTPUT_SUBDIR"
echo "LOG_SUBDIR=$LOG_SUBDIR"
echo "VISUAL_PHASE_DIR=$VISUAL_PHASE_DIR"
echo "NOTE=baseline_evaluation_not_training_not_curiosity_success"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

map_visual_cue() {
  case "${1:-not_specified}" in
    truthful*|truthful_fill|truthful_fill_cue) echo "truthful_fill_cue" ;;
    hidden*|hidden_fill|hidden_fill_cue) echo "hidden_fill_cue" ;;
    misleading*|misleading_empty|misleading_full|misleading_fill|misleading_fill_cue) echo "misleading_fill_cue" ;;
    *) echo "not_specified" ;;
  esac
}

map_family_scene() {
  case "$1" in
    cup_like_official) echo "cube existing_cup_asset" ;;
    box_procedural) echo "cube official_object" ;;
    cylinder_procedural) echo "pen official_object" ;;
    *) echo "unsupported unsupported" ;;
  esac
}

apply_modality_mask() {
  local eval_tag="$1"
  local mask_mode="$2"
  "$NEWTON_VENV/bin/python" - "$ROOT" "$eval_tag" "$mask_mode" "$OUTPUT_SUBDIR" "$VISUAL_PHASE_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
run_tag = sys.argv[2]
mask_mode = sys.argv[3]
output_subdir = sys.argv[4]
visual_phase_dir = sys.argv[5]
output_root = root / "experiments" / "outputs"
if output_subdir:
    output_root = output_root / output_subdir
npz_path = output_root / f"{run_tag}.npz"
summary_path = output_root / f"{run_tag}_summary.json"
visual_validation_path = output_root / f"{run_tag}_visual_validation.json"

if not npz_path.exists():
    raise SystemExit(f"missing npz for modality mask: {npz_path}")

with np.load(npz_path, allow_pickle=False) as data:
    arrays = {key: data[key].copy() for key in data.files}

mask_id = {
    "vision_contact": 0,
    "contact_only_masked_vision": 1,
    "vision_only_masked_contact": 2,
    "alternating_mask": 3,
}.get(mask_mode, -1)

vision_source = arrays.get("newton.camera.object_z")
contact_source = arrays.get("newton.panda.rigid_contact_count")
if vision_source is None or contact_source is None:
    raise SystemExit("missing required arrays for modality mask")

vision_mask = np.ones_like(vision_source, dtype=np.int32)
contact_mask = np.ones_like(contact_source, dtype=np.int32)

if mask_mode == "contact_only_masked_vision":
    if "newton.camera.color_rgba" in arrays:
        arrays["newton.camera.color_rgba"] = np.zeros_like(arrays["newton.camera.color_rgba"])
    if "newton.camera.depth" in arrays:
        arrays["newton.camera.depth"] = np.zeros_like(arrays["newton.camera.depth"])
    vision_mask[...] = 0
elif mask_mode == "vision_only_masked_contact":
    arrays["newton.panda.rigid_contact_count"] = np.zeros_like(arrays["newton.panda.rigid_contact_count"])
    contact_mask[...] = 0
elif mask_mode == "alternating_mask":
    if "newton.camera.color_rgba" in arrays and arrays["newton.camera.color_rgba"].shape[0] > 0:
        arrays["newton.camera.color_rgba"][1::2] = 0
    if "newton.camera.depth" in arrays and arrays["newton.camera.depth"].shape[0] > 0:
        arrays["newton.camera.depth"][1::2] = 0
    arrays["newton.panda.rigid_contact_count"][::2] = 0
    vision_mask[1::2] = 0
    contact_mask[::2] = 0

arrays["candidate.modality.mask_mode_id"] = np.asarray([mask_id], dtype=np.int32)
arrays["candidate.modality.vision_available_mask"] = vision_mask
arrays["candidate.modality.contact_available_mask"] = contact_mask
np.savez_compressed(npz_path, **arrays)

summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
summary["phase01_modality_mask"] = {
    "status": "applied" if mask_mode != "vision_contact" else "identity",
    "mask_mode": mask_mode,
    "source_namespace": "candidate.modality.*",
    "applied_after_newton_export_inside_h200_allocation": True,
    "note": "Masking is a dataset-modality intervention over exported arrays; raw rollout visuals remain available for manual inspection.",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
visual_summary = root / "experiments" / "visuals" / visual_phase_dir / run_tag / "summary.json"
if visual_summary.parent.exists():
    visual_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

visual_validation = json.loads(visual_validation_path.read_text(encoding="utf-8")) if visual_validation_path.exists() else {}
visual_validation["phase01_modality_mask"] = summary["phase01_modality_mask"]
visual_validation_path.write_text(json.dumps(visual_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary["phase01_modality_mask"], indent=2, sort_keys=True))
PY
}

run_cell_method() {
  local method="$1"
  local cell="$2"
  local family="$3"
  local mass="$4"
  local friction="$5"
  local raw_visual_cue="$6"
  local mask_mode="$7"
  local com_x="${8:-0}"
  local scene tracked visual_cue
  local controller_mode="lift_hold"
  if [[ "$method" == "scripted_feedback" ]]; then
    controller_mode="lift_hold_feedback"
  fi
  read -r scene tracked < <(map_family_scene "$family")
  if [[ "$scene" == "unsupported" ]]; then
    echo "ERROR: unsupported Phase 01 baseline family: $family" >&2
    exit 5
  fi
  visual_cue="$(map_visual_cue "$raw_visual_cue")"
  local eval_tag="${RUN_TAG}_${method}_${cell}"
  local log_path="$ROOT/logs/newton/$LOG_SUBDIR/${eval_tag}.log"
  echo "=== PHASE01_BASELINE_CELL_START method=$method cell=$cell eval_tag=$eval_tag ==="
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="$scene" \
  TRACKED_OBJECT="$tracked" \
  CONTROLLER_MODE="$controller_mode" \
  FINAL_HOLD_DURATION="$FINAL_HOLD_DURATION" \
  LIFT_HEIGHT_MIN="$LIFT_HEIGHT_MIN" \
  HOLD_DURATION_MIN="$HOLD_DURATION_MIN" \
  DROP_HEIGHT_LOSS="$DROP_HEIGHT_LOSS" \
  PHYSICS_VARIANT_LABEL="phase01_${method}_${cell}" \
  BODY_MASS_SCALE="1.0" \
  SHAPE_FRICTION_SCALE="1.0" \
  OBJECT_MASS_KG="$mass" \
  OBJECT_FRICTION_MU="$friction" \
  OBJECT_COM_OFFSET_XYZ="$com_x,0,0" \
  GRASP_OFFSET_DELTA_XYZ="0,0,0" \
  FILL_LABEL="$cell" \
  NOMINAL_VISUAL_FILL="0.5" \
  VISUAL_FILL_CUE="$visual_cue" \
  VISUAL_FILL_CUE_RENDERED="0" \
  PRE_RECORD_WARMUP_STEPS="60" \
  NUM_STEPS="$NUM_STEPS" \
  SAMPLE_STEPS="$SAMPLE_STEPS" \
  VIDEO_FRAME_STRIDE="$VIDEO_FRAME_STRIDE" \
  VIDEO_FPS="$VIDEO_FPS" \
  DEVICE="$DEVICE" \
  OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
  LOG_SUBDIR="$LOG_SUBDIR" \
  VISUAL_PHASE_DIR="$VISUAL_PHASE_DIR" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
      2>&1 | tee "$log_path"
  apply_modality_mask "$eval_tag" "$mask_mode"
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase01_${method}" \
  MASS_LABEL="$mass" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_direct_agent_check" \
  OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  echo "=== PHASE01_BASELINE_CELL_END method=$method cell=$cell eval_tag=$eval_tag ==="
}

run_cell() {
  local cell="$1"
  local family="$2"
  local mass="$3"
  local friction="$4"
  local visual_cue="$5"
  local mask_mode="$6"
  local com_x="${7:-0}"
  if [[ "$RUN_NO_ADAPTATION" == "1" ]]; then
    run_cell_method "no_adaptation" "$cell" "$family" "$mass" "$friction" "$visual_cue" "$mask_mode" "$com_x"
  fi
  if [[ "$RUN_SCRIPTED_FEEDBACK" == "1" ]]; then
    run_cell_method "scripted_feedback" "$cell" "$family" "$mass" "$friction" "$visual_cue" "$mask_mode" "$com_x"
  fi
}

run_cell "heldout_cup_full_low_hidden" "cup_like_official" "0.35" "0.35" "hidden_fill" "vision_contact" "0"
run_cell "heldout_cup_empty_high_misleading" "cup_like_official" "0.08" "1.20" "misleading_empty" "alternating_mask" "0"
run_cell "heldout_box_heavy_low_large_offset" "box_procedural" "0.42" "0.35" "not_specified" "alternating_mask" "0.018"
run_cell "heldout_cylinder_heavy_low_masked_vision" "cylinder_procedural" "0.38" "0.25" "not_specified" "contact_only_masked_vision" "0"

echo "PHASE01_BASELINES_EXIT=0"
