#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_panda_usd_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_v1.3}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
NUM_FRAMES="${NUM_FRAMES:-240}"
SCENE="${SCENE:-cube}"
WORLD_COUNT="${WORLD_COUNT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
VIS_DIR="${VIS_DIR:-$ROOT/experiments/visuals/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/newton_hydro/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$VIS_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for GPU evidence." >&2
  exit 4
fi

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 00 reference tactile USD export requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

newton_commit="$(git -C "$NEWTON_ROOT" rev-parse HEAD)"
usd_path="$VIS_DIR/panda_hydro.usd"
run_log="$LOG_DIR/newton_panda_hydro_usd.log"
summary_json="$OUTPUT_DIR/newton_hydro_usd_summary.json"
summary_md="$REPORT_DIR/newton_hydro_usd.md"

echo "PHASE00_NEWTON_HYDRO_USD_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$newton_commit"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VIS_DIR=$VIS_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "USD_PATH=$usd_path"
echo "NOTE=official_newton_hydro_visual_export_not_training_not_curiosity_success"

set +e
(
  cd "$NEWTON_ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1800 "$NEWTON_VENV/bin/python" -m newton.examples.robot.example_robot_panda_hydro \
    --device "$DEVICE" \
    --viewer usd \
    --output-path "$usd_path" \
    --num-frames "$NUM_FRAMES" \
    --test \
    --quiet \
    --scene "$SCENE" \
    --world-count "$WORLD_COUNT"
) >"$run_log" 2>&1
run_exit=$?
set -e

"$NEWTON_VENV/bin/python" - "$summary_json" "$summary_md" "$run_log" "$run_exit" "$RUN_TAG" "$newton_commit" "$gpu_names" "$usd_path" "$NUM_FRAMES" "$SCENE" "$WORLD_COUNT" <<'PY'
import json
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
run_log = Path(sys.argv[3])
run_exit = int(sys.argv[4])
run_tag = sys.argv[5]
newton_commit = sys.argv[6]
gpu_names = sys.argv[7]
usd_path = Path(sys.argv[8])
num_frames = int(sys.argv[9])
scene = sys.argv[10]
world_count = int(sys.argv[11])

text = run_log.read_text(encoding="utf-8", errors="replace") if run_log.exists() else ""
usd_exists = usd_path.exists()
usd_size = usd_path.stat().st_size if usd_exists else 0
status = "pass" if run_exit == 0 and "Traceback" not in text and usd_size > 0 else "fail"
payload = {
    "classification": "phase00_newton_official_panda_hydro_usd_v1",
    "run_tag": run_tag,
    "status": status,
    "not_training_result": True,
    "not_curiosity_success": True,
    "official_newton_hydro_base_evidence": status == "pass",
    "gpu_names": gpu_names,
    "newton_commit": newton_commit,
    "command": (
        "python -m newton.examples.robot.example_robot_panda_hydro "
        "--device cuda:0 --viewer usd --output-path panda_hydro.usd "
        f"--num-frames {num_frames} --test --quiet --scene {scene} "
        f"--world-count {world_count}"
    ),
    "run_exit": run_exit,
    "traceback_absent": "Traceback" not in text,
    "run_log": str(run_log),
    "usd_path": str(usd_path),
    "usd_exists": usd_exists,
    "usd_size_bytes": usd_size,
    "missing_for_dense_tactile_success": [
        "pad_resolved_pressure_or_compression_maps",
        "pad_resolved_force_and_shear_timeseries",
        "synchronized_mp4_or_dense_frame_video_equivalent",
        "manual_visual_inspection_record",
    ],
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
summary_md.write_text(
    "# Phase 00 Newton Official Panda Hydro USD\n\n"
    f"- run_tag: `{run_tag}`\n"
    f"- status: `{status}`\n"
    f"- Newton commit: `{newton_commit}`\n"
    f"- USD: `{usd_path}`\n"
    f"- USD size bytes: `{usd_size}`\n"
    f"- log: `{run_log}`\n"
    "\nClassification: official Newton hydro base visual evidence only. "
    "This is not dense tactile success, training, or curiosity success.\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
if status != "pass":
    raise SystemExit(1)
PY

echo "PHASE00_NEWTON_HYDRO_USD_END"
