#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_src_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase01/src_collect.json}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: missing config $CONFIG" >&2
  exit 3
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing Newton venv $NEWTON_VENV/bin/python" >&2
  exit 4
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing trainer venv $TRAINER_VENV/bin/python" >&2
  exit 5
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required to verify H200 GPU evidence." >&2
  exit 6
fi
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 01 source collection requires H200; observed GPU names:" >&2
  printf '%s\n' "$gpu_names" >&2
  exit 7
fi

OUTPUT_SUBDIR="$("$TRAINER_VENV/bin/python" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["output_subdir"])
PY
)"
LOG_SUBDIR="$("$TRAINER_VENV/bin/python" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["log_subdir"])
PY
)"
VISUAL_PHASE_DIR="$("$TRAINER_VENV/bin/python" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["visual_phase_dir"])
PY
)"
mkdir -p "$ROOT/logs/newton/$LOG_SUBDIR" "$ROOT/experiments/outputs/$OUTPUT_SUBDIR" "$ROOT/experiments/visuals/$VISUAL_PHASE_DIR" "$ROOT/experiments/reports/phase01/core"

echo "PHASE01_SRC_COLLECT_START"
echo "RUN_TAG=$RUN_TAG"
echo "CONFIG=$CONFIG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "DEVICE=$DEVICE"
echo "OUTPUT_SUBDIR=$OUTPUT_SUBDIR"
echo "LOG_SUBDIR=$LOG_SUBDIR"
echo "VISUAL_PHASE_DIR=$VISUAL_PHASE_DIR"
echo "NOTE=train_only_corrective_source_collection_not_training_not_curiosity_success"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,220p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

json_get() {
  "$TRAINER_VENV/bin/python" - "$CONFIG" "$1" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
cur = payload
for part in sys.argv[2].split("."):
    cur = cur[part]
print(cur)
PY
}

json_bool01() {
  case "$(json_get "$1")" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) echo "1" ;;
    0|false|False|FALSE|no|No|NO|off|Off|OFF) echo "0" ;;
    *)
      echo "ERROR: expected boolean-like JSON value at $1" >&2
      exit 9
      ;;
  esac
}

NUM_STEPS="$(json_get num_steps)"
SAMPLE_STEPS="$(json_get sample_steps)"
VIDEO_FRAME_STRIDE="$(json_get video_frame_stride)"
VIDEO_FPS="$(json_get video_fps)"
FINAL_HOLD_DURATION="$(json_get final_hold_duration_s)"
HOLD_DURATION_MIN="$(json_get hold_duration_min_s)"
LIFT_HEIGHT_MIN="$(json_get lift_height_min_m)"
DROP_HEIGHT_LOSS="$(json_get drop_height_loss_m)"
FEEDBACK_MIN_CONTACT_COUNT="$(json_get feedback.min_contact_count)"
FEEDBACK_ACCEL_THRESHOLD="$(json_get feedback.accel_threshold_m_s2)"
FEEDBACK_HEIGHT_DROP_THRESHOLD="$(json_get feedback.height_drop_threshold_m)"
FEEDBACK_INITIAL_LIFT_DURATION_SCALE="$(json_get feedback.initial_lift_duration_scale)"
FEEDBACK_LIFT_DURATION_SCALE_MAX="$(json_get feedback.lift_duration_scale_max)"
FEEDBACK_HOLD_HEIGHT_STEP="$(json_get feedback.hold_height_step_m)"
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="$(json_get feedback.hold_height_offset_max_m)"
FEEDBACK_STABILIZATION_STEP="$(json_get feedback.stabilization_step_s)"
FEEDBACK_STABILIZATION_MAX="$(json_get feedback.stabilization_max_s)"
FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT="$(json_bool01 feedback.apply_initial_waypoint_adjustment)"

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
    "applied_after_newton_export_inside_h200_allocation": True
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if visual_validation_path.exists():
    visual = json.loads(visual_validation_path.read_text(encoding="utf-8"))
    visual["phase01_modality_mask"] = summary["phase01_modality_mask"]
    visual_validation_path.write_text(json.dumps(visual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary["phase01_modality_mask"], indent=2, sort_keys=True))
PY
}

run_export() {
  local method="$1"
  local controller_mode="$2"
  local cell="$3"
  local family="$4"
  local mass="$5"
  local friction="$6"
  local visual_cue="$7"
  local mask_mode="$8"
  local com_xyz="$9"
  local scene tracked eval_tag log_path
  read -r scene tracked < <(map_family_scene "$family")
  if [[ "$scene" == "unsupported" ]]; then
    echo "ERROR: unsupported family $family" >&2
    exit 8
  fi
  eval_tag="${RUN_TAG}_${method}_${cell}"
  log_path="$ROOT/logs/newton/$LOG_SUBDIR/${eval_tag}.log"
  echo "=== PHASE01_SRC_CELL_START method=$method cell=$cell eval_tag=$eval_tag ==="
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
  PHYSICS_VARIANT_LABEL="phase01_src_${method}_${cell}" \
  BODY_MASS_SCALE="1.0" \
  SHAPE_FRICTION_SCALE="1.0" \
  OBJECT_MASS_KG="$mass" \
  OBJECT_FRICTION_MU="$friction" \
  OBJECT_COM_OFFSET_XYZ="$com_xyz" \
  GRASP_OFFSET_DELTA_XYZ="0,0,0" \
  FILL_LABEL="$cell" \
  NOMINAL_VISUAL_FILL="0.5" \
  VISUAL_FILL_CUE="$visual_cue" \
  VISUAL_FILL_CUE_RENDERED="0" \
  FEEDBACK_MIN_CONTACT_COUNT="$FEEDBACK_MIN_CONTACT_COUNT" \
  FEEDBACK_ACCEL_THRESHOLD="$FEEDBACK_ACCEL_THRESHOLD" \
  FEEDBACK_HEIGHT_DROP_THRESHOLD="$FEEDBACK_HEIGHT_DROP_THRESHOLD" \
  FEEDBACK_INITIAL_LIFT_DURATION_SCALE="$FEEDBACK_INITIAL_LIFT_DURATION_SCALE" \
  FEEDBACK_LIFT_DURATION_SCALE_MAX="$FEEDBACK_LIFT_DURATION_SCALE_MAX" \
  FEEDBACK_HOLD_HEIGHT_STEP="$FEEDBACK_HOLD_HEIGHT_STEP" \
  FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="$FEEDBACK_HOLD_HEIGHT_OFFSET_MAX" \
  FEEDBACK_STABILIZATION_STEP="$FEEDBACK_STABILIZATION_STEP" \
  FEEDBACK_STABILIZATION_MAX="$FEEDBACK_STABILIZATION_MAX" \
  FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT="$FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT" \
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
  BASELINE_NAME="phase01_src_${method}" \
  MASS_LABEL="$mass" \
  FRICTION_LABEL="$friction" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="pending_if_source_gate_passes" \
  OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh"
  echo "=== PHASE01_SRC_CELL_END method=$method cell=$cell eval_tag=$eval_tag ==="
}

"$TRAINER_VENV/bin/python" - "$CONFIG" <<'PY' > "$ROOT/experiments/outputs/$OUTPUT_SUBDIR/${RUN_TAG}_cells.tsv"
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for cell in payload["train_cells"]:
    print("\t".join(str(cell[key]) for key in [
        "cell", "family", "object_mass_kg", "object_friction_mu", "visual_cue", "mask_mode", "object_com_offset_xyz"
    ]))
PY

while IFS=$'\t' read -r cell family mass friction visual_cue mask_mode com_xyz; do
  run_export "no" "lift_hold" "$cell" "$family" "$mass" "$friction" "$visual_cue" "$mask_mode" "$com_xyz"
  run_export "fb" "lift_hold_feedback" "$cell" "$family" "$mass" "$friction" "$visual_cue" "$mask_mode" "$com_xyz"
done < "$ROOT/experiments/outputs/$OUTPUT_SUBDIR/${RUN_TAG}_cells.tsv"

set +e
"$TRAINER_VENV/bin/python" "$ROOT/experiments/configs/phase01/build_src_gate.py" \
  --root "$ROOT" \
  --config "$CONFIG" \
  --run-tag "$RUN_TAG"
gate_exit=$?
set -e
echo "PHASE01_SRC_GATE_EXIT=$gate_exit"
echo "PHASE01_SRC_COLLECT_END"
exit "$gate_exit"
