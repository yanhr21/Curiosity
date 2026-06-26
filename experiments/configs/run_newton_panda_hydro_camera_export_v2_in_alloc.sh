#!/usr/bin/env bash
set -euo pipefail

# Run Newton Panda hydro SensorTiledCamera export in a held allocation.
# V2 explicitly supports controller-mode and metric thresholds.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-newton_panda_hydro_camera_export_$(date +%Y%m%d_%H%M%S)}"
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
FEEDBACK_MIN_CONTACT_COUNT="${FEEDBACK_MIN_CONTACT_COUNT:-20}"
FEEDBACK_ACCEL_THRESHOLD="${FEEDBACK_ACCEL_THRESHOLD:-6.5}"
FEEDBACK_HEIGHT_DROP_THRESHOLD="${FEEDBACK_HEIGHT_DROP_THRESHOLD:-0.015}"
FEEDBACK_INITIAL_LIFT_DURATION_SCALE="${FEEDBACK_INITIAL_LIFT_DURATION_SCALE:-1.35}"
FEEDBACK_LIFT_DURATION_SCALE_MAX="${FEEDBACK_LIFT_DURATION_SCALE_MAX:-2.25}"
FEEDBACK_HOLD_HEIGHT_STEP="${FEEDBACK_HOLD_HEIGHT_STEP:-0.003}"
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="${FEEDBACK_HOLD_HEIGHT_OFFSET_MAX:-0.03}"
FEEDBACK_STABILIZATION_STEP="${FEEDBACK_STABILIZATION_STEP:-0.25}"
FEEDBACK_STABILIZATION_MAX="${FEEDBACK_STABILIZATION_MAX:-2.0}"
PRE_RECORD_WARMUP_STEPS="${PRE_RECORD_WARMUP_STEPS:-0}"
RESIDUAL_ADAPTER_CHECKPOINT="${RESIDUAL_ADAPTER_CHECKPOINT:-}"
RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="${RESIDUAL_ADAPTER_ACTIVE_THRESHOLD:-0.5}"
NUM_STEPS="${NUM_STEPS:-240}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,60,120,180,239}"
DEVICE="${DEVICE:-cuda:0}"
NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-$ROOT/external/newton-assets-cache}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/visuals

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
if [[ "$CONTROLLER_MODE" == "lift_hold_learned_residual" ]]; then
  if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
    echo "ERROR: learned residual evaluation requires local trainer venv at $TRAINER_VENV/bin/python" >&2
    exit 5
  fi
  if [[ -z "$RESIDUAL_ADAPTER_CHECKPOINT" || ! -f "$RESIDUAL_ADAPTER_CHECKPOINT" ]]; then
    echo "ERROR: learned residual evaluation requires RESIDUAL_ADAPTER_CHECKPOINT to point at an existing checkpoint." >&2
    exit 6
  fi
fi

source "$NEWTON_VENV/bin/activate"
export NEWTON_CACHE_PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

EXPORT_PYTHON="$NEWTON_VENV/bin/python"
if [[ "$CONTROLLER_MODE" == "lift_hold_learned_residual" ]]; then
  EXPORT_PYTHON="$TRAINER_VENV/bin/python"
  export PYTHONPATH="$ROOT/envs/newton/.venv/lib/python3.10/site-packages:$ROOT/external/newton:$ROOT/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
fi

for asset_file in \
  "franka_emika_panda/urdf/fr3_franka_hand.urdf" \
  "manipulation_objects/pad/model.usda" \
  "manipulation_objects/cup/model.usda"; do
  if ! find -L "$NEWTON_CACHE_PATH" -maxdepth 6 -type f -path "*/$asset_file" -print -quit | grep -q .; then
    echo "ERROR: missing cached Newton asset file $asset_file under $NEWTON_CACHE_PATH; prefetch locally before compute use." >&2
    exit 4
  fi
done

sanity_log="$ROOT/logs/newton/${RUN_TAG}_official_sensor_contact_sanity.log"
sanity_json="$ROOT/experiments/outputs/${RUN_TAG}_fresh_newton_sensor_contact_sanity.json"
summary_json="$ROOT/experiments/outputs/${RUN_TAG}_summary.json"
npz_path="$ROOT/experiments/outputs/${RUN_TAG}.npz"
visual_root="$ROOT/experiments/visuals/${RUN_TAG}"
visual_validation="$ROOT/experiments/outputs/${RUN_TAG}_visual_validation.json"
run_status="$ROOT/experiments/outputs/${RUN_TAG}_run_status.json"

echo "RUN_TAG=$RUN_TAG"
echo "RUNNER_VERSION=20260627_v2_controller_mode_metrics"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "TRAINER_VENV=$TRAINER_VENV"
echo "EXPORT_PYTHON=$EXPORT_PYTHON"
echo "NEWTON_CACHE_PATH=$NEWTON_CACHE_PATH"
echo "SCENE=$SCENE"
echo "TRACKED_OBJECT=$TRACKED_OBJECT"
echo "CONTROLLER_MODE=$CONTROLLER_MODE"
echo "FINAL_HOLD_DURATION=$FINAL_HOLD_DURATION"
echo "LIFT_HEIGHT_MIN=$LIFT_HEIGHT_MIN"
echo "HOLD_DURATION_MIN=$HOLD_DURATION_MIN"
echo "DROP_HEIGHT_LOSS=$DROP_HEIGHT_LOSS"
echo "PHYSICS_VARIANT_LABEL=$PHYSICS_VARIANT_LABEL"
echo "BODY_MASS_SCALE=$BODY_MASS_SCALE"
echo "SHAPE_FRICTION_SCALE=$SHAPE_FRICTION_SCALE"
echo "OBJECT_MASS_KG=$OBJECT_MASS_KG"
echo "OBJECT_FRICTION_MU=$OBJECT_FRICTION_MU"
echo "FEEDBACK_MIN_CONTACT_COUNT=$FEEDBACK_MIN_CONTACT_COUNT"
echo "FEEDBACK_ACCEL_THRESHOLD=$FEEDBACK_ACCEL_THRESHOLD"
echo "FEEDBACK_HEIGHT_DROP_THRESHOLD=$FEEDBACK_HEIGHT_DROP_THRESHOLD"
echo "FEEDBACK_INITIAL_LIFT_DURATION_SCALE=$FEEDBACK_INITIAL_LIFT_DURATION_SCALE"
echo "FEEDBACK_LIFT_DURATION_SCALE_MAX=$FEEDBACK_LIFT_DURATION_SCALE_MAX"
echo "FEEDBACK_HOLD_HEIGHT_STEP=$FEEDBACK_HOLD_HEIGHT_STEP"
echo "FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=$FEEDBACK_HOLD_HEIGHT_OFFSET_MAX"
echo "FEEDBACK_STABILIZATION_STEP=$FEEDBACK_STABILIZATION_STEP"
echo "FEEDBACK_STABILIZATION_MAX=$FEEDBACK_STABILIZATION_MAX"
echo "PRE_RECORD_WARMUP_STEPS=$PRE_RECORD_WARMUP_STEPS"
echo "RESIDUAL_ADAPTER_CHECKPOINT=$RESIDUAL_ADAPTER_CHECKPOINT"
echo "RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=$RESIDUAL_ADAPTER_ACTIVE_THRESHOLD"
echo "NUM_STEPS=$NUM_STEPS"
echo "SAMPLE_STEPS=$SAMPLE_STEPS"
echo "DEVICE=$DEVICE"

echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

echo "=== OFFICIAL_NEWTON_SANITY_START ==="
set +e
(
  cd "$ROOT/external/newton"
  timeout 900 "$NEWTON_VENV/bin/python" -m newton.examples.sensors.example_sensor_contact --device "$DEVICE" --viewer null --num-frames 160 --test --quiet
) >"$sanity_log" 2>&1
sanity_exit=$?
set -e
"$NEWTON_VENV/bin/python" - "$sanity_log" "$sanity_json" "$sanity_exit" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
exit_code = int(sys.argv[3])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
payload = {
    "status": "pass" if exit_code == 0 and "Traceback" not in text else "fail",
    "classification": "fresh_official_newton_sensor_contact_sanity",
    "command": ["timeout", "900", "python", "-m", "newton.examples.sensors.example_sensor_contact", "--device", "cuda:0", "--viewer", "null", "--num-frames", "160", "--test", "--quiet"],
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

echo "=== NEWTON_CAMERA_EXPORT_START ==="
physics_args=(
  --physics-variant-label "$PHYSICS_VARIANT_LABEL"
  --body-mass-scale "$BODY_MASS_SCALE"
  --shape-friction-scale "$SHAPE_FRICTION_SCALE"
)
if [[ -n "$OBJECT_MASS_KG" ]]; then
  physics_args+=(--object-mass-kg "$OBJECT_MASS_KG")
fi
if [[ -n "$OBJECT_FRICTION_MU" ]]; then
  physics_args+=(--object-friction-mu "$OBJECT_FRICTION_MU")
fi
residual_args=()
if [[ "$CONTROLLER_MODE" == "lift_hold_learned_residual" ]]; then
  residual_args+=(
    --residual-adapter-checkpoint "$RESIDUAL_ADAPTER_CHECKPOINT"
    --residual-adapter-active-threshold "$RESIDUAL_ADAPTER_ACTIVE_THRESHOLD"
  )
fi
"$EXPORT_PYTHON" experiments/configs/newton_panda_hydro_tiled_camera_export.py \
  --output-dir "$visual_root" \
  --summary "$summary_json" \
  --npz "$npz_path" \
  --num-steps "$NUM_STEPS" \
  --sample-steps "$SAMPLE_STEPS" \
  --scene "$SCENE" \
  --tracked-object "$TRACKED_OBJECT" \
  --controller-mode "$CONTROLLER_MODE" \
  --final-hold-duration "$FINAL_HOLD_DURATION" \
  --lift-height-min "$LIFT_HEIGHT_MIN" \
  --hold-duration-min "$HOLD_DURATION_MIN" \
  --drop-height-loss "$DROP_HEIGHT_LOSS" \
  --feedback-min-contact-count "$FEEDBACK_MIN_CONTACT_COUNT" \
  --feedback-accel-threshold "$FEEDBACK_ACCEL_THRESHOLD" \
  --feedback-height-drop-threshold "$FEEDBACK_HEIGHT_DROP_THRESHOLD" \
  --feedback-initial-lift-duration-scale "$FEEDBACK_INITIAL_LIFT_DURATION_SCALE" \
  --feedback-lift-duration-scale-max "$FEEDBACK_LIFT_DURATION_SCALE_MAX" \
  --feedback-hold-height-step "$FEEDBACK_HOLD_HEIGHT_STEP" \
  --feedback-hold-height-offset-max "$FEEDBACK_HOLD_HEIGHT_OFFSET_MAX" \
  --feedback-stabilization-step "$FEEDBACK_STABILIZATION_STEP" \
  --feedback-stabilization-max "$FEEDBACK_STABILIZATION_MAX" \
  --pre-record-warmup-steps "$PRE_RECORD_WARMUP_STEPS" \
  "${physics_args[@]}" \
  "${residual_args[@]}"
echo "=== NEWTON_CAMERA_EXPORT_END ==="

"$NEWTON_VENV/bin/python" experiments/configs/validate_newton_visual_preview.py \
  "$visual_root" \
  --min-frames 5 \
  --output "$visual_validation"

"$NEWTON_VENV/bin/python" - "$run_status" "$RUN_TAG" "$SLURM_JOB_ID" "$sanity_json" "$summary_json" "$visual_validation" <<'PY'
import json
import sys
from pathlib import Path

out, run_tag, slurm_job_id, sanity_json, summary_json, visual_validation = sys.argv[1:]
sanity = json.loads(Path(sanity_json).read_text(encoding="utf-8"))
summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
visual = json.loads(Path(visual_validation).read_text(encoding="utf-8"))
payload = {
    "status": "pass_downstream_blocked" if sanity.get("status") == "pass" and summary.get("status") == "pass" and visual.get("status") == "pass" else "fail",
    "classification": "newton_panda_hydro_camera_export_run_status",
    "runner_version": "20260627_v2_controller_mode_metrics",
    "run_tag": run_tag,
    "slurm_job_id": slurm_job_id,
    "sanity_json": sanity_json,
    "summary_json": summary_json,
    "visual_validation": visual_validation,
    "scene": summary.get("scene"),
    "tracked_object": summary.get("tracked_object"),
    "controller_mode": summary.get("controller_mode"),
    "controller_type": summary.get("controller_type"),
    "final_hold_duration": summary.get("final_hold_duration"),
    "physics_variant": summary.get("physics_variant"),
    "object_physics_adapter": summary.get("object_physics_adapter"),
    "task_metrics": summary.get("task_metrics"),
    "num_steps": summary.get("num_steps"),
    "sample_steps": summary.get("sample_steps"),
    "frame_browser": summary.get("frame_browser"),
    "contact_sheet": summary.get("contact_sheet"),
    "npz": summary.get("npz"),
    "downstream_use": "blocked_until_manual_visual_inspection_pass",
    "generated_trex_fields": [],
    "schema_promotion": "blocked",
    "no_model_or_training": summary.get("controller_mode") != "lift_hold_learned_residual",
    "model_evaluation": summary.get("controller_mode") == "lift_hold_learned_residual",
    "residual_adapter_checkpoint": summary.get("scripted_feedback", {}).get("residual_adapter_checkpoint"),
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if payload["status"] == "pass_downstream_blocked" else 1)
PY
