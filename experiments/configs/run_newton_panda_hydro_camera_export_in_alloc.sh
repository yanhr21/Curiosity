#!/usr/bin/env bash
set -euo pipefail

# Run official Newton Panda hydro SensorTiledCamera export in a held allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-newton_panda_hydro_camera_export_$(date +%Y%m%d_%H%M%S)}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
SCENE="${SCENE:-pen}"
TRACKED_OBJECT="${TRACKED_OBJECT:-official_object}"
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

source "$NEWTON_VENV/bin/activate"
export NEWTON_CACHE_PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

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
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NEWTON_VENV=$NEWTON_VENV"
echo "NEWTON_CACHE_PATH=$NEWTON_CACHE_PATH"
echo "SCENE=$SCENE"
echo "TRACKED_OBJECT=$TRACKED_OBJECT"
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
"$NEWTON_VENV/bin/python" experiments/configs/newton_panda_hydro_tiled_camera_export.py \
  --output-dir "$visual_root" \
  --summary "$summary_json" \
  --npz "$npz_path" \
  --num-steps "$NUM_STEPS" \
  --sample-steps "$SAMPLE_STEPS" \
  --scene "$SCENE" \
  --tracked-object "$TRACKED_OBJECT"
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
    "run_tag": run_tag,
    "slurm_job_id": slurm_job_id,
    "sanity_json": sanity_json,
    "summary_json": summary_json,
    "visual_validation": visual_validation,
    "scene": summary.get("scene"),
    "tracked_object": summary.get("tracked_object"),
    "num_steps": summary.get("num_steps"),
    "sample_steps": summary.get("sample_steps"),
    "frame_browser": summary.get("frame_browser"),
    "contact_sheet": summary.get("contact_sheet"),
    "npz": summary.get("npz"),
    "downstream_use": "blocked_until_manual_visual_inspection_pass",
    "generated_trex_fields": [],
    "schema_promotion": "blocked",
    "no_model_or_training": True,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if payload["status"] == "pass_downstream_blocked" else 1)
PY
