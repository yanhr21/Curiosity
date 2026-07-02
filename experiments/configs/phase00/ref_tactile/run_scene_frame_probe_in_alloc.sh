#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_scene_cam_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_main}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-cube}"
STEPS="${STEPS:-180}"
SAMPLES="${SAMPLES:-12}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
VIS_DIR="${VIS_DIR:-$ROOT/experiments/visuals/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/newton_hydro/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$VIS_DIR" "$REPORT_DIR"

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"

echo "PHASE00_SCENE_FRAME_PROBE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$(git -C "$NEWTON_ROOT" rev-parse HEAD)"
echo "VIS_DIR=$VIS_DIR"

"$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_scene_frame_probe.py" \
  --output-dir "$OUTPUT_DIR" \
  --visual-dir "$VIS_DIR" \
  --report-dir "$REPORT_DIR" \
  --run-tag "$RUN_TAG" \
  --scene "$SCENE" \
  --device "$DEVICE" \
  --steps "$STEPS" \
  --samples "$SAMPLES" \
  --width "$WIDTH" \
  --height "$HEIGHT"

echo "PHASE00_SCENE_FRAME_PROBE_END"
