#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_hydro_bench_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_v1.3}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-cube}"
WORLD_COUNT="${WORLD_COUNT:-1}"
NUM_FRAMES="${NUM_FRAMES:-720}"
BENCHMARK_SECONDS="${BENCHMARK_SECONDS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/newton_hydro/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

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
  echo "ERROR: Phase 00 hydro benchmark requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

newton_commit="$(git -C "$NEWTON_ROOT" rev-parse HEAD)"
run_log="$LOG_DIR/newton_hydro_benchmark.log"
summary_path="$OUTPUT_DIR/newton_hydro_benchmark_summary.json"
report_path="$REPORT_DIR/newton_hydro_benchmark.md"

echo "PHASE00_NEWTON_HYDRO_BENCHMARK_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$newton_commit"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=official_newton_hydro_null_viewer_benchmark_not_training_not_curiosity_success"

set +e
(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1200 "$NEWTON_VENV/bin/python" -m newton.examples.robot.example_robot_panda_hydro \
    --device "$DEVICE" \
    --viewer null \
    --benchmark "$BENCHMARK_SECONDS" \
    --num-frames "$NUM_FRAMES" \
    --quiet \
    --scene "$SCENE" \
    --world-count "$WORLD_COUNT"
) 2>&1 | tee "$run_log"
run_status="${PIPESTATUS[0]}"
set -e

"$NEWTON_VENV/bin/python" - "$summary_path" "$report_path" "$run_log" "$run_status" "$RUN_TAG" "$SLURM_JOB_ID" "$(hostname)" "$gpu_names" "$newton_commit" "$SCENE" "$WORLD_COUNT" "$NUM_FRAMES" "$BENCHMARK_SECONDS" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
run_log = Path(sys.argv[3])
run_status = int(sys.argv[4])
run_tag, slurm_job_id, host, gpu_names, newton_commit = sys.argv[5:10]
scene, world_count, num_frames, benchmark_seconds = sys.argv[10:14]
text = run_log.read_text(encoding="utf-8", errors="replace")
match = re.search(r"Benchmark:\s+([0-9.]+)\s+FPS\s+\((\d+)\s+frames in\s+([0-9.]+)s\)", text)
fps = float(match.group(1)) if match else 0.0
bench_frames = int(match.group(2)) if match else 0
elapsed_s = float(match.group(3)) if match else 0.0
summary = {
    "classification": "phase00_official_newton_hydro_null_viewer_benchmark_v1",
    "run_tag": run_tag,
    "status": "pass" if run_status == 0 and match else "failed_or_no_benchmark_line",
    "run_exit": run_status,
    "not_training_result": True,
    "not_curiosity_success": True,
    "official_example": "newton.examples.robot.example_robot_panda_hydro",
    "scene": scene,
    "world_count": int(world_count),
    "requested_num_frames": int(num_frames),
    "benchmark_seconds": int(benchmark_seconds),
    "benchmark_fps": fps,
    "benchmark_frames": bench_frames,
    "benchmark_elapsed_s": elapsed_s,
    "historical_reference_fps": 82.0,
    "runtime_acceptable_fps": 80.0,
    "minimum_accepted_fps": 60.0,
    "meets_target_82_fps": fps >= 82.0,
    "meets_runtime_around80_fps": fps >= 80.0,
    "meets_minimum_60_fps": fps >= 60.0,
    "fps_note": "Official Newton null-viewer benchmark throughput. 82 FPS is historical reference only; around 80 FPS is acceptable for continuing tactile export/Gate checks. No tactile export, no per-frame host copies, no video render.",
    "slurm_job_id": slurm_job_id,
    "host": host,
    "gpu_names": gpu_names,
    "newton_commit": newton_commit,
    "log_path": str(run_log),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
report_path.write_text(
    "# Phase 00 Official Newton Hydro Benchmark\n\n"
    f"- run_tag: `{run_tag}`\n"
    f"- status: `{summary['status']}`\n"
    f"- official example: `{summary['official_example']}`\n"
    f"- benchmark FPS: `{fps}`\n"
    f"- frames/elapsed: `{bench_frames}` / `{elapsed_s}` s\n"
    f"- meets historical 82 FPS reference: `{summary['meets_target_82_fps']}`\n"
    f"- meets around-80 FPS continuation threshold: `{summary['meets_runtime_around80_fps']}`\n"
    f"- meets 60 FPS minimum: `{summary['meets_minimum_60_fps']}`\n"
    f"- log: `{run_log}`\n\n"
    "This is a runtime benchmark only, not training and not curiosity success. "
    "The historical 82 FPS reference must not block tactile export when the "
    "run is stable around 80 FPS.\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(0 if summary["status"] == "pass" else 1)
PY

echo "PHASE00_NEWTON_HYDRO_BENCHMARK_END"
exit "$run_status"
