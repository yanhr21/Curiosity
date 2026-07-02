#!/usr/bin/env bash
set -euo pipefail

# Direct export runner for controller-mode experiments that must avoid stale
# wrapper caching on compute nodes. Always uses the trainer venv for the export
# process and explicitly passes residual checkpoint arguments when provided.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-newton_panda_hydro_direct_$(date +%Y%m%d_%H%M%S)}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
SCENE="${SCENE:-pen}"
TRACKED_OBJECT="${TRACKED_OBJECT:-official_object}"
CONTROLLER_MODE="${CONTROLLER_MODE:-official_pick_place}"
FINAL_HOLD_DURATION="${FINAL_HOLD_DURATION:-1.0}"
LIFT_HEIGHT_MIN="${LIFT_HEIGHT_MIN:-0.12}"
HOLD_DURATION_MIN="${HOLD_DURATION_MIN:-2.0}"
DROP_HEIGHT_LOSS="${DROP_HEIGHT_LOSS:-0.05}"
PHYSICS_VARIANT_LABEL="${PHYSICS_VARIANT_LABEL:-nominal}"
BODY_MASS_SCALE="${BODY_MASS_SCALE:-1.0}"
SHAPE_FRICTION_SCALE="${SHAPE_FRICTION_SCALE:-1.0}"
OBJECT_MASS_KG="${OBJECT_MASS_KG:-}"
OBJECT_FRICTION_MU="${OBJECT_FRICTION_MU:-}"
GRASP_OFFSET_DELTA_XYZ="${GRASP_OFFSET_DELTA_XYZ:-0,0,0}"
FILL_LABEL="${FILL_LABEL:-not_specified}"
NOMINAL_VISUAL_FILL="${NOMINAL_VISUAL_FILL:--1.0}"
VISUAL_FILL_CUE="${VISUAL_FILL_CUE:-not_specified}"
VISUAL_FILL_CUE_RENDERED="${VISUAL_FILL_CUE_RENDERED:-0}"
FEEDBACK_MIN_CONTACT_COUNT="${FEEDBACK_MIN_CONTACT_COUNT:-20}"
FEEDBACK_ACCEL_THRESHOLD="${FEEDBACK_ACCEL_THRESHOLD:-6.5}"
FEEDBACK_HEIGHT_DROP_THRESHOLD="${FEEDBACK_HEIGHT_DROP_THRESHOLD:-0.015}"
FEEDBACK_INITIAL_LIFT_DURATION_SCALE="${FEEDBACK_INITIAL_LIFT_DURATION_SCALE:-1.35}"
FEEDBACK_LIFT_DURATION_SCALE_MAX="${FEEDBACK_LIFT_DURATION_SCALE_MAX:-2.25}"
FEEDBACK_HOLD_HEIGHT_STEP="${FEEDBACK_HOLD_HEIGHT_STEP:-0.003}"
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="${FEEDBACK_HOLD_HEIGHT_OFFSET_MAX:-0.03}"
FEEDBACK_STABILIZATION_STEP="${FEEDBACK_STABILIZATION_STEP:-0.25}"
FEEDBACK_STABILIZATION_MAX="${FEEDBACK_STABILIZATION_MAX:-2.0}"
FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT="${FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT:-1}"
PRE_RECORD_WARMUP_STEPS="${PRE_RECORD_WARMUP_STEPS:-0}"
RESIDUAL_ADAPTER_CHECKPOINT="${RESIDUAL_ADAPTER_CHECKPOINT:-}"
RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="${RESIDUAL_ADAPTER_ACTIVE_THRESHOLD:-0.5}"
RECORD_SCRIPTED_TEACHER_LABELS="${RECORD_SCRIPTED_TEACHER_LABELS:-0}"
NUM_STEPS="${NUM_STEPS:-240}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,60,120,180,239}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-0}"
VIDEO_FPS="${VIDEO_FPS:-12}"
DEVICE="${DEVICE:-cuda:0}"
NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-$ROOT/external/newton-assets-cache}"
VISUAL_PHASE_DIR="${VISUAL_PHASE_DIR:-}"
if [[ -z "$VISUAL_PHASE_DIR" ]]; then
  if [[ "$RUN_TAG" =~ (phase[0-9][0-9]) ]]; then
    VISUAL_PHASE_DIR="${BASH_REMATCH[1]}"
  else
    VISUAL_PHASE_DIR="unphased"
  fi
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs "experiments/visuals/$VISUAL_PHASE_DIR"

if [[ ! -x "$NEWTON_VENV/bin/python" || ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing required local env under envs/." >&2
  exit 3
fi
if [[ "$CONTROLLER_MODE" == *residual* && ! -f "$RESIDUAL_ADAPTER_CHECKPOINT" ]]; then
  echo "ERROR: residual controller requires RESIDUAL_ADAPTER_CHECKPOINT." >&2
  exit 4
fi

source "$NEWTON_VENV/bin/activate"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NEWTON_CACHE_PATH
export PYTHONPATH="$ROOT/envs/newton/.venv/lib/python3.10/site-packages:$ROOT/external/newton:$ROOT/src:${PYTHONPATH:-}"

sanity_log="$ROOT/logs/newton/${RUN_TAG}_official_sensor_contact_sanity.log"
sanity_json="$ROOT/experiments/outputs/${RUN_TAG}_fresh_newton_sensor_contact_sanity.json"
summary_json="$ROOT/experiments/outputs/${RUN_TAG}_summary.json"
npz_path="$ROOT/experiments/outputs/${RUN_TAG}.npz"
visual_root="$ROOT/experiments/visuals/${VISUAL_PHASE_DIR}/${RUN_TAG}"
visual_validation="$ROOT/experiments/outputs/${RUN_TAG}_visual_validation.json"
run_status="$ROOT/experiments/outputs/${RUN_TAG}_run_status.json"

echo "RUN_TAG=$RUN_TAG"
echo "RUNNER_VERSION=20260629_direct_controller_mode_metrics"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "EXPORT_PYTHON=$TRAINER_VENV/bin/python"
echo "VISUAL_PHASE_DIR=$VISUAL_PHASE_DIR"
echo "VISUAL_ROOT=$visual_root"
echo "CONTROLLER_MODE=$CONTROLLER_MODE"
echo "RESIDUAL_ADAPTER_CHECKPOINT=$RESIDUAL_ADAPTER_CHECKPOINT"

echo "=== OFFICIAL_NEWTON_SANITY_START ==="
set +e
(
  cd "$ROOT/external/newton"
  timeout 900 "$NEWTON_VENV/bin/python" -m newton.examples.sensors.example_sensor_contact --device "$DEVICE" --viewer null --num-frames 160 --test --quiet
) >"$sanity_log" 2>&1
sanity_exit=$?
set -e
"$NEWTON_VENV/bin/python" - "$sanity_log" "$sanity_json" "$sanity_exit" "$DEVICE" <<'PY'
import json
import sys
from pathlib import Path
log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
exit_code = int(sys.argv[3])
device = sys.argv[4]
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
payload = {
    "classification": "fresh_official_newton_sensor_contact_sanity",
    "status": "pass" if exit_code == 0 and "Traceback" not in text else "fail",
    "command": ["timeout", "900", "python", "-m", "newton.examples.sensors.example_sensor_contact", "--device", device, "--viewer", "null", "--num-frames", "160", "--test", "--quiet"],
    "cwd": "external/newton",
    "exit_code": exit_code,
    "traceback_absent": "Traceback" not in text,
    "log": str(log_path),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pass":
    raise SystemExit(1)
PY
echo "=== OFFICIAL_NEWTON_SANITY_END ==="

task_args=(--fill-label "$FILL_LABEL" --nominal-visual-fill "$NOMINAL_VISUAL_FILL" --visual-fill-cue "$VISUAL_FILL_CUE")
if [[ "$VISUAL_FILL_CUE_RENDERED" == "1" || "$VISUAL_FILL_CUE_RENDERED" == "true" ]]; then
  task_args+=(--visual-fill-cue-rendered)
fi
physics_args=(--physics-variant-label "$PHYSICS_VARIANT_LABEL" --body-mass-scale "$BODY_MASS_SCALE" --shape-friction-scale "$SHAPE_FRICTION_SCALE")
if [[ -n "$OBJECT_MASS_KG" ]]; then
  physics_args+=(--object-mass-kg "$OBJECT_MASS_KG")
fi
if [[ -n "$OBJECT_FRICTION_MU" ]]; then
  physics_args+=(--object-friction-mu "$OBJECT_FRICTION_MU")
fi
residual_args=()
if [[ -n "$RESIDUAL_ADAPTER_CHECKPOINT" ]]; then
  residual_args+=(--residual-adapter-checkpoint "$RESIDUAL_ADAPTER_CHECKPOINT" --residual-adapter-active-threshold "$RESIDUAL_ADAPTER_ACTIVE_THRESHOLD")
fi
teacher_args=()
if [[ "$RECORD_SCRIPTED_TEACHER_LABELS" == "1" || "$RECORD_SCRIPTED_TEACHER_LABELS" == "true" ]]; then
  teacher_args+=(--record-scripted-teacher-labels)
fi
feedback_initial_args=(--feedback-apply-initial-waypoint-adjustment)
if [[ "$FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT" == "0" ]]; then
  feedback_initial_args=(--no-feedback-apply-initial-waypoint-adjustment)
fi

echo "=== NEWTON_CAMERA_EXPORT_START ==="
"$TRAINER_VENV/bin/python" experiments/configs/newton_panda_hydro_tiled_camera_export.py \
  --output-dir "$visual_root" \
  --summary "$summary_json" \
  --npz "$npz_path" \
  --num-steps "$NUM_STEPS" \
  --sample-steps "$SAMPLE_STEPS" \
  --video-frame-stride "$VIDEO_FRAME_STRIDE" \
  --video-fps "$VIDEO_FPS" \
  --scene "$SCENE" \
  --tracked-object "$TRACKED_OBJECT" \
  --controller-mode "$CONTROLLER_MODE" \
  --final-hold-duration "$FINAL_HOLD_DURATION" \
  --lift-height-min "$LIFT_HEIGHT_MIN" \
  --hold-duration-min "$HOLD_DURATION_MIN" \
  --drop-height-loss "$DROP_HEIGHT_LOSS" \
  --grasp-offset-delta-xyz="$GRASP_OFFSET_DELTA_XYZ" \
  "${task_args[@]}" \
  --feedback-min-contact-count "$FEEDBACK_MIN_CONTACT_COUNT" \
  --feedback-accel-threshold "$FEEDBACK_ACCEL_THRESHOLD" \
  --feedback-height-drop-threshold "$FEEDBACK_HEIGHT_DROP_THRESHOLD" \
  --feedback-initial-lift-duration-scale "$FEEDBACK_INITIAL_LIFT_DURATION_SCALE" \
  --feedback-lift-duration-scale-max "$FEEDBACK_LIFT_DURATION_SCALE_MAX" \
  --feedback-hold-height-step "$FEEDBACK_HOLD_HEIGHT_STEP" \
  --feedback-hold-height-offset-max "$FEEDBACK_HOLD_HEIGHT_OFFSET_MAX" \
  --feedback-stabilization-step "$FEEDBACK_STABILIZATION_STEP" \
  --feedback-stabilization-max "$FEEDBACK_STABILIZATION_MAX" \
  "${feedback_initial_args[@]}" \
  --pre-record-warmup-steps "$PRE_RECORD_WARMUP_STEPS" \
  "${physics_args[@]}" \
  "${residual_args[@]}" \
  "${teacher_args[@]}"
echo "=== NEWTON_CAMERA_EXPORT_END ==="

"$NEWTON_VENV/bin/python" experiments/configs/validate_newton_visual_preview.py "$visual_root" --min-frames 4 --output "$visual_validation"
"$NEWTON_VENV/bin/python" - "$run_status" "$RUN_TAG" "$SLURM_JOB_ID" "$sanity_json" "$summary_json" "$visual_validation" <<'PY'
import json
import sys
from pathlib import Path
out, run_tag, slurm_job_id, sanity_json, summary_json, visual_validation = sys.argv[1:]
sanity = json.loads(Path(sanity_json).read_text(encoding="utf-8"))
summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
visual = json.loads(Path(visual_validation).read_text(encoding="utf-8"))
payload = {
    "classification": "newton_panda_hydro_camera_export_direct_run_status",
    "runner_version": "20260629_direct_controller_mode_metrics",
    "status": "pass_downstream_blocked" if sanity.get("status") == "pass" and summary.get("status") == "pass" and visual.get("status") == "pass" else "fail",
    "run_tag": run_tag,
    "slurm_job_id": slurm_job_id,
    "sanity_json": sanity_json,
    "summary_json": summary_json,
    "visual_validation": visual_validation,
    "controller_mode": summary.get("controller_mode"),
    "task_metrics": summary.get("task_metrics"),
    "contact_sheet": summary.get("contact_sheet"),
    "rollout_video": summary.get("rollout_video"),
    "npz": summary.get("npz"),
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if payload["status"] == "pass_downstream_blocked" else 1)
PY
