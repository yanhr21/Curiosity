#!/usr/bin/env bash
set -euo pipefail

# Validation-only closed-loop threshold repair for the Phase07 curiosity
# residual checkpoint. This runs Newton interaction rollouts and full videos
# on validation cells only. It does not train, does not touch held-out cells,
# and does not claim success.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/phase07_closed_loop_threshold_repair_v1.json}"
RUN_TAG="${RUN_TAG:-phase07_closed_loop_threshold_repair_v1_20260628}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv python at $NEWTON_VENV/bin/python" >&2
  exit 3
fi
if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
  echo "ERROR: missing local trainer venv python at $TRAINER_VENV/bin/python" >&2
  exit 4
fi

checkpoint="$("$NEWTON_VENV/bin/python" - "$CONFIG" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
print(root / config["current_checkpoint"])
PY
)"
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing current checkpoint: $checkpoint" >&2
  exit 5
fi

echo "PHASE07_CLOSED_LOOP_THRESHOLD_REPAIR_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "DEVICE=$DEVICE"
echo "CONFIG=$CONFIG"
echo "CHECKPOINT=$checkpoint"
echo "NOTE=validation_only_closed_loop_interaction_repair_not_training_not_success_claim"
echo "=== AGENTS_REREAD_HEAD ==="
sed -n '1,180p' AGENTS.md
echo "=== AGENTS_REREAD_END ==="

run_one() {
  local cell="$1"
  local fill="$2"
  local visual_cue="$3"
  local mass="$4"
  local friction="$5"
  local nominal_visual_fill="$6"
  local threshold="$7"
  local threshold_tag="${threshold/./p}"
  local run_tag="${RUN_TAG}_${cell}_thr_${threshold_tag}"
  local visual_fill_cue="$visual_cue"

  echo "=== PHASE07_THRESHOLD_REPAIR_ROLLOUT_START cell=$cell threshold=$threshold run_tag=$run_tag ==="
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
  PHYSICS_VARIANT_LABEL="phase07_threshold_repair_${cell}_thr_${threshold_tag}" \
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
  RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="$threshold" \
  NUM_STEPS="360" \
  SAMPLE_STEPS="0,90,180,270,359" \
  VIDEO_FRAME_STRIDE="1" \
  VIDEO_FPS="12" \
  DEVICE="$DEVICE" \
  bash "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh" \
    2>&1 | tee "$ROOT/logs/newton/${run_tag}.log"

  RUN_TAG="$run_tag" NEWTON_VENV="$NEWTON_VENV" TOP_K="12" \
    bash "$ROOT/experiments/configs/run_lift_hold_accel_peak_analysis_in_alloc.sh" \
      2>&1 | tee "$ROOT/logs/newton/${run_tag}_accel_peak_analysis.log"
  echo "=== PHASE07_THRESHOLD_REPAIR_ROLLOUT_END cell=$cell threshold=$threshold run_tag=$run_tag ==="
}

run_one "empty_medium_hidden" "empty" "hidden_fill_cue" "0.08" "0.7" "0.0" "0.5"
run_one "empty_medium_hidden" "empty" "hidden_fill_cue" "0.08" "0.7" "0.0" "0.65"
run_one "empty_medium_hidden" "empty" "hidden_fill_cue" "0.08" "0.7" "0.0" "0.8"
run_one "empty_medium_hidden" "empty" "hidden_fill_cue" "0.08" "0.7" "0.0" "0.95"
run_one "full_medium_misleading" "full" "misleading_fill_cue" "0.35" "0.7" "1.0" "0.5"
run_one "full_medium_misleading" "full" "misleading_fill_cue" "0.35" "0.7" "1.0" "0.65"
run_one "full_medium_misleading" "full" "misleading_fill_cue" "0.35" "0.7" "1.0" "0.8"
run_one "full_medium_misleading" "full" "misleading_fill_cue" "0.35" "0.7" "1.0" "0.95"

"$NEWTON_VENV/bin/python" - "$ROOT" "$RUN_TAG" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_tag = sys.argv[2]
config_path = Path(sys.argv[3])
config = json.loads(config_path.read_text(encoding="utf-8"))
thresholds = config["thresholds"]
cells = config["validation_cells"]

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def threshold_tag(value):
    return str(value).replace(".", "p")

entries = []
for cell in cells:
    for threshold in thresholds:
        tag = f"{run_tag}_{cell['cell']}_thr_{threshold_tag(threshold)}"
        summary_path = root / "experiments" / "outputs" / f"{tag}_summary.json"
        visual_path = root / "experiments" / "outputs" / f"{tag}_visual_validation.json"
        accel_path = root / "experiments" / "outputs" / f"{tag}_accel_peak_analysis.json"
        summary = load(summary_path)
        visual = load(visual_path)
        accel = load(accel_path)
        task = summary.get("task_metrics", {})
        video = summary.get("video_export", {})
        per_world = task.get("per_world", [{}]) if isinstance(task, dict) else [{}]
        world0 = per_world[0] if per_world and isinstance(per_world[0], dict) else {}
        max_lift = max(float(x) for x in summary.get("max_lift", [0.0]))
        final_lift = float(world0.get("final_lift", 0.0))
        hold_duration = float(world0.get("longest_hold_s", 0.0))
        success = bool(task.get("success_all_worlds", False) if isinstance(task, dict) else False)
        events = accel.get("events", []) if isinstance(accel, dict) else []
        max_accel = float(events[0].get("accel_norm_m_s2", 0.0)) if events else 0.0
        status_ok = summary.get("status") == "pass" and visual.get("status") == "pass" and video.get("status") == "pass"
        score = (
            1.0 if success else 0.0,
            1.0 if status_ok else 0.0,
            hold_duration,
            -max_accel,
            max_lift,
        )
        entries.append({
            "cell": cell["cell"],
            "threshold": threshold,
            "run_tag": tag,
            "summary": str(summary_path.relative_to(root)),
            "visual_validation": str(visual_path.relative_to(root)),
            "accel_peak_analysis": str(accel_path.relative_to(root)),
            "rollout_video": summary.get("rollout_video"),
            "status_ok": status_ok,
            "success": success,
            "hold_duration_s": hold_duration,
            "max_lift_m": max_lift,
            "final_lift_m": final_lift,
            "max_object_accel_m_s2": max_accel,
            "score_tuple": list(score),
        })

by_threshold = {}
for threshold in thresholds:
    selected = [item for item in entries if item["threshold"] == threshold]
    valid = all(item["status_ok"] for item in selected)
    aggregate = (
        min(item["score_tuple"][0] for item in selected),
        min(item["score_tuple"][1] for item in selected),
        sum(item["hold_duration_s"] for item in selected) / len(selected),
        -max(item["max_object_accel_m_s2"] for item in selected),
        sum(item["max_lift_m"] for item in selected) / len(selected),
    )
    by_threshold[str(threshold)] = {
        "threshold": threshold,
        "status": "pass" if valid else "fail",
        "aggregate_score_tuple": list(aggregate),
        "entries": selected,
    }

passing = [item for item in by_threshold.values() if item["status"] == "pass"]
best = max(passing or by_threshold.values(), key=lambda item: tuple(item["aggregate_score_tuple"]))
payload = {
    "classification": "phase07_closed_loop_threshold_repair_summary_v1",
    "status": "pass" if passing else "fail",
    "run_tag": run_tag,
    "config": str(config_path.relative_to(root)),
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "validation_only": True,
    "held_out_cells_forbidden_for_threshold_selection": config["held_out_cells_forbidden_for_threshold_selection"],
    "thresholds": thresholds,
    "entries": entries,
    "by_threshold": by_threshold,
    "selected_threshold": best["threshold"],
    "selected_threshold_reason": "validation_only_success_then_status_then_hold_then_safety_accel_then_lift_not_heldout_tuned",
    "not_training": True,
    "not_success_claim": True,
    "not_official_trex_method": True,
    "schema_promotion": "blocked",
    "generated_trex_fields": [],
}
out = root / config["outputs"]["summary"]
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
report = root / config["outputs"]["report"]
lines = [
    "# Phase07 Closed-Loop Threshold Repair V1",
    "",
    "Validation-only Newton interaction sweep for the current curiosity checkpoint. This is not training and not a success claim.",
    "",
    f"- status: `{payload['status']}`",
    f"- selected threshold: `{payload['selected_threshold']}`",
    f"- summary: `{out.relative_to(root)}`",
    "",
    "## Thresholds",
    "",
]
for threshold, item in by_threshold.items():
    lines.append(f"- `{threshold}`: status `{item['status']}`, aggregate `{item['aggregate_score_tuple']}`")
lines.extend(["", "## Held-Out Protection", "", "Held-out cells were not used for threshold selection."])
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "pass":
    raise SystemExit(1)
PY

echo "PHASE07_CLOSED_LOOP_THRESHOLD_REPAIR_END"
