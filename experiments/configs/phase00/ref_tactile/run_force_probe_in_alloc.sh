#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_force_probe_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_v1.3}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-cube}"
NUM_FRAMES="${NUM_FRAMES:-240}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/newton_hydro/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_force_probe.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if ! printf '%s\n' "$gpu_names" | grep -qi 'H200'; then
  echo "ERROR: Phase 00 force probe requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

run_log="$LOG_DIR/force_probe.log"
echo "PHASE00_FORCE_PROBE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "NOTE=direct_force_probe_not_training_not_curiosity_success"

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1200 "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_force_probe.py" \
    --run-tag "$RUN_TAG" \
    --device "$DEVICE" \
    --scene "$SCENE" \
    --num-frames "$NUM_FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE00_FORCE_PROBE_END"
