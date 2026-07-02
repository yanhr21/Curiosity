#!/usr/bin/env bash
set -euo pipefail

# Collect Phase07 V2 train/validation source rollouts inside a held allocation.
# This is not training and not a success claim. The generated source manifest
# remains manual-visual-pending until direct inspection writes pass JSONs.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase07_v2_stabilization_source_collection_v1.json}"
RUN_TAG="${RUN_TAG:-phase07_v2_stabilization_source_collection_v1_20260628}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/visuals experiments/configs

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local trainer venv python at $TRAINER_VENV/bin/python" >&2
  exit 4
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: missing config: $CONFIG" >&2
  exit 5
fi

echo "PHASE07_V2_STABILIZATION_SOURCE_COLLECTION_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "CONFIG=$CONFIG"
echo "NOTE=train_validation_source_collection_only_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

readarray -t cell_lines < <("$NEWTON_VENV/bin/python" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
common = config["common_rollout"]
for cell in config["cells"]:
    values = [
        cell["cell"],
        cell["split"],
        cell["fill_label"],
        cell["friction_label"],
        cell["visual_fill_cue"],
        str(cell["object_mass_kg"]),
        str(cell["object_friction_mu"]),
        str(cell["nominal_visual_fill"]),
        str(common["final_hold_duration"]),
        str(common["hold_duration_min"]),
        str(common["feedback_lift_duration_scale_max"]),
        str(int(bool(common.get("feedback_apply_initial_waypoint_adjustment", True)))),
        str(common["feedback_stabilization_step"]),
        str(common["feedback_stabilization_max"]),
        str(common["num_steps"]),
        str(common["sample_steps"]),
        str(common["video_frame_stride"]),
        str(common["video_fps"]),
    ]
    print("\t".join(values))
PY
)

for line in "${cell_lines[@]}"; do
  IFS=$'\t' read -r cell split fill_label friction_label visual_fill_cue object_mass object_friction nominal_fill final_hold hold_min lift_scale_max apply_initial_adjustment stab_step stab_max num_steps sample_steps video_stride video_fps <<<"$line"
  eval_tag="${RUN_TAG}_${cell}"
  manual_json="$ROOT/experiments/outputs/${eval_tag}_manual_visual_inspection.json"
  echo "=== PHASE07_V2_SOURCE_CELL_START cell=$cell split=$split eval_tag=$eval_tag ==="
  "$NEWTON_VENV/bin/python" - "$manual_json" "$eval_tag" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_tag = sys.argv[2]
payload = {
    "classification": "manual_visual_inspection_v1",
    "run_tag": run_tag,
    "checked_on": "pending",
    "status": "pending_direct_agent_check",
    "visual_status": "pending_direct_agent_check",
    "curiosity_success_claim_valid": False,
    "reason_not_success_claim": "Source rollout requires direct manual visual inspection before it may feed the V2 source runner.",
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  TRAINER_VENV="$TRAINER_VENV" \
  SCENE="cube" \
  TRACKED_OBJECT="existing_cup_asset" \
  CONTROLLER_MODE="lift_hold_feedback" \
  FINAL_HOLD_DURATION="$final_hold" \
  LIFT_HEIGHT_MIN="0.12" \
  HOLD_DURATION_MIN="$hold_min" \
  DROP_HEIGHT_LOSS="0.05" \
  PHYSICS_VARIANT_LABEL="phase07_v2_stabilization_source_${cell}" \
  BODY_MASS_SCALE="1.0" \
  SHAPE_FRICTION_SCALE="1.0" \
  OBJECT_MASS_KG="$object_mass" \
  OBJECT_FRICTION_MU="$object_friction" \
  FILL_LABEL="$fill_label" \
  NOMINAL_VISUAL_FILL="$nominal_fill" \
  VISUAL_FILL_CUE="$visual_fill_cue" \
  VISUAL_FILL_CUE_RENDERED="0" \
  FEEDBACK_MIN_CONTACT_COUNT="58" \
  FEEDBACK_ACCEL_THRESHOLD="6.5" \
  FEEDBACK_HEIGHT_DROP_THRESHOLD="0.015" \
  FEEDBACK_INITIAL_LIFT_DURATION_SCALE="1.65" \
  FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT="$apply_initial_adjustment" \
  FEEDBACK_LIFT_DURATION_SCALE_MAX="$lift_scale_max" \
  FEEDBACK_HOLD_HEIGHT_STEP="0.0005" \
  FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="0.005" \
  FEEDBACK_STABILIZATION_STEP="$stab_step" \
  FEEDBACK_STABILIZATION_MAX="$stab_max" \
  PRE_RECORD_WARMUP_STEPS="15" \
  NUM_STEPS="$num_steps" \
  SAMPLE_STEPS="$sample_steps" \
  VIDEO_FRAME_STRIDE="$video_stride" \
  VIDEO_FPS="$video_fps" \
  DEVICE="$DEVICE" \
  NEWTON_CACHE_PATH="$ROOT/external/newton-assets-cache" \
    bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}.log"

  RUN_TAG="$eval_tag" \
  NEWTON_VENV="$NEWTON_VENV" \
  BASELINE_NAME="phase07_v2_stabilization_source_scripted_feedback" \
  MASS_LABEL="$fill_label" \
  FRICTION_LABEL="$friction_label" \
  POSE_SEED="$cell" \
  MANUAL_VISUAL_INSPECTION="experiments/outputs/${eval_tag}_manual_visual_inspection.json" \
    bash "$ROOT/experiments/configs/run_lift_hold_metrics_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}_metrics.log"

  RUN_TAG="$eval_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${eval_tag}_accel_peak_analysis.log"
  echo "=== PHASE07_V2_SOURCE_CELL_END cell=$cell split=$split eval_tag=$eval_tag ==="
done

"$NEWTON_VENV/bin/python" - "$ROOT" "$CONFIG" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
run_tag = sys.argv[3]
config = json.loads(config_path.read_text(encoding="utf-8"))
common = config["common_rollout"]
manifest_path = root / config["output_manifest"]
summary_path = root / config["summary_output"]
source_candidates = []
manual_pending = []
failures = []
for cell in config["cells"]:
    cell_name = cell["cell"]
    eval_tag = f"{run_tag}_{cell_name}"
    summary_file = root / "experiments" / "outputs" / f"{eval_tag}_summary.json"
    metrics_file = root / "experiments" / "outputs" / f"{eval_tag}_metrics.json"
    accel_file = root / "experiments" / "outputs" / f"{eval_tag}_accel_peak_analysis.json"
    visual_file = root / "experiments" / "outputs" / f"{eval_tag}_visual_validation.json"
    sanity_file = root / "experiments" / "outputs" / f"{eval_tag}_fresh_newton_sensor_contact_sanity.json"
    manual_file = root / "experiments" / "outputs" / f"{eval_tag}_manual_visual_inspection.json"
    for path in (summary_file, metrics_file, accel_file, visual_file, sanity_file, manual_file):
        if not path.is_file():
            failures.append(f"missing:{path.relative_to(root)}")
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else {}
    accel = json.loads(accel_file.read_text(encoding="utf-8")) if accel_file.is_file() else {}
    visual = json.loads(visual_file.read_text(encoding="utf-8")) if visual_file.is_file() else {}
    sanity = json.loads(sanity_file.read_text(encoding="utf-8")) if sanity_file.is_file() else {}
    manual = json.loads(manual_file.read_text(encoding="utf-8")) if manual_file.is_file() else {}
    rows = metrics.get("rows") or []
    row = rows[0] if rows else {}
    if manual.get("status") != config["manual_visual_gate"]["pass_status"]:
        manual_pending.append(eval_tag)
    candidate = {
        "name": f"phase07_v2_{cell_name}_stabilization09_source_v1",
        "status": "promoted_source_candidate",
        "run_tag": eval_tag,
        "slurm_job_id": str(summary.get("slurm_job_id", "")),
        "split": cell["split"],
        "cell": cell_name,
        "fill_label": cell["fill_label"],
        "friction_label": cell["friction_label"],
        "visual_fill_cue": cell["visual_fill_cue"],
        "visual_fill_cue_rendering_status": "metadata_only_not_rendered",
        "object_mass_kg": cell["object_mass_kg"],
        "object_friction_mu": cell["object_friction_mu"],
        "nominal_visual_fill": cell["nominal_visual_fill"],
        "held_out_generalization_cell": False,
        "pre_record_warmup_steps": common["pre_record_warmup_steps"],
        "parameter_overrides": {
            "scene": common["scene"],
            "tracked_object": common["tracked_object"],
            "controller_mode": common["controller_mode"],
            "feedback_min_contact_count": common["feedback_min_contact_count"],
            "feedback_accel_threshold_m_s2": common["feedback_accel_threshold"],
            "feedback_height_drop_threshold_m": common["feedback_height_drop_threshold"],
            "feedback_initial_lift_duration_scale": common["feedback_initial_lift_duration_scale"],
            "feedback_lift_duration_scale_max": common["feedback_lift_duration_scale_max"],
            "feedback_hold_height_step_m": common["feedback_hold_height_step"],
            "feedback_hold_height_offset_max_m": common["feedback_hold_height_offset_max"],
            "feedback_stabilization_step_s": common["feedback_stabilization_step"],
            "feedback_stabilization_max_s": common["feedback_stabilization_max"],
            "num_steps": common["num_steps"],
            "video_frame_stride": common["video_frame_stride"],
            "video_fps": common["video_fps"],
        },
        "outputs": {
            "fresh_official_newton_sanity": f"experiments/outputs/{eval_tag}_fresh_newton_sensor_contact_sanity.json",
            "summary_json": f"experiments/outputs/{eval_tag}_summary.json",
            "run_status": f"experiments/outputs/{eval_tag}_run_status.json",
            "visual_validation": f"experiments/outputs/{eval_tag}_visual_validation.json",
            "manual_visual_inspection": f"experiments/outputs/{eval_tag}_manual_visual_inspection.json",
            "metrics_json": f"experiments/outputs/{eval_tag}_metrics.json",
            "metrics_csv": f"experiments/outputs/{eval_tag}_metrics.csv",
            "accel_peak_analysis": f"experiments/outputs/{eval_tag}_accel_peak_analysis.json",
            "npz": f"experiments/outputs/{eval_tag}.npz",
            "contact_sheet": f"experiments/visuals/{eval_tag}/contact_sheet.png",
            "frame_browser": f"experiments/visuals/{eval_tag}/frame_browser.html",
            "rollout_video": f"experiments/visuals/{eval_tag}/rollout_video.gif",
            "video_frames_dir": f"experiments/visuals/{eval_tag}/video_frames",
            "log": f"logs/newton/{eval_tag}.log",
            "metrics_log": f"logs/newton/{eval_tag}_metrics.log",
            "accel_peak_log": f"logs/newton/{eval_tag}_accel_peak_analysis.log",
        },
        "observed": {
            "fresh_official_newton_sanity": sanity.get("status"),
            "visual_validation": visual.get("status"),
            "manual_visual_inspection": manual.get("status"),
            "metrics_status": metrics.get("status"),
            "accel_peak_status": accel.get("status"),
            "object_not_dropped": row.get("object_not_dropped"),
            "contact_loss_frames": row.get("contact_loss_frames"),
            "max_contact_proxy": row.get("max_contact_proxy"),
            "lift_height_m": row.get("lift_height_m"),
            "hold_duration_s": row.get("hold_duration_s"),
            "max_slip_m": row.get("max_slip_m"),
            "max_object_accel_m_s2": row.get("max_object_accel_m_s2"),
            "success_per_contact_proxy_integral": row.get("success_per_contact_proxy_integral"),
            "rollout_video_frame_count": (summary.get("video_export") or {}).get("frame_count"),
        },
        "promotion_decision": (
            "promoted_as_phase07_harder_task_residual_label_source_candidate"
            if cell["split"] == "train"
            else "promoted_as_phase07_harder_task_validation_residual_label_source_candidate"
        ),
        "interpretation": "V2 stabilization-range source candidate. Manual visual inspection is required before this source may feed training.",
    }
    source_candidates.append(candidate)

status = (
    "phase07_v2_stabilization_source_candidates_complete_training_not_started"
    if not manual_pending and not failures
    else "phase07_v2_stabilization_source_candidates_manual_visual_pending"
)
manifest = {
    "classification": "phase07_v2_stabilization_source_manifest_v1",
    "phase": "07_harder_task_progression",
    "status": status,
    "purpose": "Track V2 train/validation scripted-feedback source candidates with a wider stabilization action range before any harder-task residual or curiosity training.",
    "training_started": False,
    "no_model_created": True,
    "no_placeholder_model": True,
    "not_training": True,
    "not_success_claim": True,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
    "source_spec": str(config_path.relative_to(root)),
    "source_runner_config": "experiments/configs/phase07_v2_stabilization_residual_label_source_runner_v1.json",
    "held_out_cells_reserved_for_evaluation": config["held_out_cells_reserved_for_evaluation"],
    "required_train_cells": [cell["cell"] for cell in config["cells"] if cell["split"] == "train"],
    "required_validation_cells": [cell["cell"] for cell in config["cells"] if cell["split"] == "validation"],
    "manual_visual_pending_run_tags": manual_pending,
    "source_candidates": source_candidates,
    "failures": failures,
}
summary = {
    "classification": "phase07_v2_stabilization_source_collection_summary_v1",
    "status": status,
    "run_tag": run_tag,
    "source_manifest": str(manifest_path.relative_to(root)),
    "source_candidate_count": len(source_candidates),
    "manual_visual_pending_run_tags": manual_pending,
    "failures": failures,
    "not_training": True,
    "not_success_claim": True,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "PHASE07_V2_STABILIZATION_SOURCE_COLLECTION_END"
