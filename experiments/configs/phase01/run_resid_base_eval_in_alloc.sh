#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_resid_base_eval_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/phase01/core/resid/base/p01_resid_base_a1_20260630_0307.pt}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-phase01/core/resid/base_eval}"
LOG_SUBDIR="${LOG_SUBDIR:-phase01/core/resid/base_eval}"
VISUAL_PHASE_DIR="${VISUAL_PHASE_DIR:-phase01/core/resid/base_eval}"
NUM_STEPS="${NUM_STEPS:-1800}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,300,600,900,1200,1500,1799}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-3}"
VIDEO_FPS="${VIDEO_FPS:-20}"
FINAL_HOLD_DURATION="${FINAL_HOLD_DURATION:-12.0}"
LIFT_HEIGHT_MIN="${LIFT_HEIGHT_MIN:-0.12}"
HOLD_DURATION_MIN="${HOLD_DURATION_MIN:-8.0}"
DROP_HEIGHT_LOSS="${DROP_HEIGHT_LOSS:-0.05}"
METHOD_LABEL="${METHOD_LABEL:-no_curiosity_resid}"
METHOD_NAME="${METHOD_NAME:-phase01_no_curiosity_residual}"
METHOD_REPORT_TITLE="${METHOD_REPORT_TITLE:-Phase 01 No-Curiosity Residual Held-Out Eval}"
METHOD_REPORT_NOTE="${METHOD_REPORT_NOTE:-This is learned non-curiosity baseline evaluation, not curiosity success.}"
METHOD_SUMMARY_CLASSIFICATION="${METHOD_SUMMARY_CLASSIFICATION:-phase01_no_curiosity_residual_heldout_eval_summary_v1}"
METHOD_NOT_CURIOSITY_SUCCESS="${METHOD_NOT_CURIOSITY_SUCCESS:-1}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR" \
  "$ROOT/experiments/outputs/$OUTPUT_SUBDIR" \
  "$ROOT/experiments/reports/phase01/core/resid" \
  "$ROOT/experiments/visuals/$VISUAL_PHASE_DIR"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing residual checkpoint $CHECKPOINT" >&2
  exit 3
fi
if [[ ! -x "$NEWTON_VENV/bin/python" || ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: required envs missing under envs/." >&2
  exit 4
fi

echo "PHASE01_RESID_BASE_EVAL_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_SUBDIR=$OUTPUT_SUBDIR"
echo "LOG_SUBDIR=$LOG_SUBDIR"
echo "VISUAL_PHASE_DIR=$VISUAL_PHASE_DIR"
echo "METHOD_LABEL=$METHOD_LABEL"
echo "METHOD_NAME=$METHOD_NAME"
echo "NOTE=heldout_evaluation_of_${METHOD_LABEL}_not_success_until_comparison"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

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
output_root = root / "experiments" / "outputs" / output_subdir
npz_path = output_root / f"{run_tag}.npz"
summary_path = output_root / f"{run_tag}_summary.json"
visual_validation_path = output_root / f"{run_tag}_visual_validation.json"
with np.load(npz_path, allow_pickle=False) as data:
    arrays = {key: data[key].copy() for key in data.files}
mask_id = {"vision_contact": 0, "contact_only_masked_vision": 1, "vision_only_masked_contact": 2, "alternating_mask": 3}.get(mask_mode, -1)
vision_source = arrays["newton.camera.object_z"]
contact_source = arrays["newton.panda.rigid_contact_count"]
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
    "controller_modality_mask_mode": mask_mode,
    "applied_after_newton_export_inside_h200_allocation": True
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if visual_validation_path.exists():
    visual = json.loads(visual_validation_path.read_text(encoding="utf-8"))
    visual["phase01_modality_mask"] = summary["phase01_modality_mask"]
    visual_validation_path.write_text(json.dumps(visual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_cell() {
  local cell="$1"
  local family="$2"
  local mass="$3"
  local friction="$4"
  local visual_cue="$5"
  local mask_mode="$6"
  local com_x="${7:-0}"
  local scene tracked eval_tag log_path
  read -r scene tracked < <(map_family_scene "$family")
  if [[ "$scene" == "unsupported" ]]; then
    echo "ERROR: unsupported family $family" >&2
    exit 5
  fi
  eval_tag="${RUN_TAG}_${METHOD_LABEL}_${cell}"
  log_path="$ROOT/logs/newton/$LOG_SUBDIR/${eval_tag}.log"
  echo "=== PHASE01_RESID_BASE_EVAL_CELL_START cell=$cell eval_tag=$eval_tag ==="
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="$scene" \
  TRACKED_OBJECT="$tracked" \
  CONTROLLER_MODE="lift_hold_learned_residual" \
  RESIDUAL_ADAPTER_CHECKPOINT="$CHECKPOINT" \
  RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="0.5" \
  CONTROLLER_MODALITY_MASK_MODE="$mask_mode" \
  FINAL_HOLD_DURATION="$FINAL_HOLD_DURATION" \
  LIFT_HEIGHT_MIN="$LIFT_HEIGHT_MIN" \
  HOLD_DURATION_MIN="$HOLD_DURATION_MIN" \
  DROP_HEIGHT_LOSS="$DROP_HEIGHT_LOSS" \
  PHYSICS_VARIANT_LABEL="phase01_no_curiosity_resid_${cell}" \
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
  BASELINE_NAME="$METHOD_NAME" \
  MASS_LABEL="$mass" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_mp4_and_direct_review" \
  OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  echo "=== PHASE01_RESID_BASE_EVAL_CELL_END cell=$cell eval_tag=$eval_tag ==="
}

run_cell "heldout_cup_full_low_hidden" "cup_like_official" "0.35" "0.35" "hidden_fill_cue" "vision_contact" "0"
run_cell "heldout_cup_empty_high_misleading" "cup_like_official" "0.08" "1.20" "misleading_fill_cue" "alternating_mask" "0"
run_cell "heldout_box_heavy_low_large_offset" "box_procedural" "0.42" "0.35" "not_specified" "alternating_mask" "0.018"
run_cell "heldout_cylinder_heavy_low_masked_vision" "cylinder_procedural" "0.38" "0.25" "not_specified" "contact_only_masked_vision" "0"

"$TRAINER_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$OUTPUT_SUBDIR" "$CHECKPOINT" "$METHOD_LABEL" "$METHOD_REPORT_TITLE" "$METHOD_REPORT_NOTE" "$METHOD_SUMMARY_CLASSIFICATION" "$METHOD_NOT_CURIOSITY_SUCCESS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
output_subdir = sys.argv[3]
checkpoint = Path(sys.argv[4])
method_label = sys.argv[5]
report_title = sys.argv[6]
report_note = sys.argv[7]
summary_classification = sys.argv[8]
not_curiosity_success = sys.argv[9] == "1"
output_root = root / "experiments" / "outputs" / output_subdir
report_path = root / "experiments" / "reports" / "phase01" / "core" / "resid" / f"{run_tag}_summary.md"
summary_path = output_root / f"{run_tag}_summary.json"
rows = []
for metrics_path in sorted(output_root.glob(f"{run_tag}_{method_label}_*_metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = dict((payload.get("rows") or [{}])[0])
    row["metrics_json"] = str(metrics_path.relative_to(root))
    rows.append(row)
success_count = sum(1 for row in rows if row.get("status") == "success")
summary = {
    "classification": summary_classification,
    "status": "pass" if len(rows) == 4 else "fail",
    "run_tag": run_tag,
    "method_label": method_label,
    "checkpoint": str(checkpoint.relative_to(root)),
    "heldout_cell_count": len(rows),
    "success_count": success_count,
    "rows": rows,
    "not_curiosity_success": not_curiosity_success,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    f"# {report_title}",
    "",
    f"- status: `{summary['status']}`",
    f"- run tag: `{run_tag}`",
    f"- checkpoint: `{summary['checkpoint']}`",
    f"- held-out cells: `{len(rows)}`",
    f"- successes: `{success_count}`",
    "",
    report_note,
    "",
    "## Cells",
    "",
]
for row in rows:
    lines.append(
        f"- `{row.get('pose_seed')}` status `{row.get('status')}` lift `{row.get('lift_height_m')}` "
        f"hold `{row.get('hold_duration_s')}` slip `{row.get('max_slip_m')}` "
        f"contact_loss `{row.get('contact_loss_frames')}` accel `{row.get('max_object_accel_m_s2')}`"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if summary["status"] == "pass" else 1)
PY

echo "PHASE01_RESID_BASE_EVAL_END"
