#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_mjw_direct_tac_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_main}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
NUM_FRAMES="${NUM_FRAMES:-240}"
SCENE="${SCENE:-cube}"
MAP_SIZE="${MAP_SIZE:-32}"
FPS="${FPS:-30}"
MATERIAL_LABEL="${MATERIAL_LABEL:-official_default}"
OVERRIDE_MU="${OVERRIDE_MU:-}"
OVERRIDE_KH="${OVERRIDE_KH:-}"
SCENE_CAMERA="${SCENE_CAMERA:-1}"
SCENE_CAMERA_WIDTH="${SCENE_CAMERA_WIDTH:-256}"
SCENE_CAMERA_HEIGHT="${SCENE_CAMERA_HEIGHT:-256}"
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

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_mjw_direct_tactile_export.py"; do
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
  echo "ERROR: Phase 00 candidate MJWarp direct tactile export requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

newton_commit="$(git -C "$NEWTON_ROOT" rev-parse HEAD)"
run_log="$LOG_DIR/candidate_mjw_direct_tactile.log"

echo "PHASE00_MJW_DIRECT_TACTILE_EXPORT_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$newton_commit"
echo "DEVICE=$DEVICE"
echo "NUM_FRAMES=$NUM_FRAMES"
echo "SCENE=$SCENE"
echo "MAP_SIZE=$MAP_SIZE"
echo "FPS=$FPS"
echo "MATERIAL_LABEL=$MATERIAL_LABEL"
echo "OVERRIDE_MU=$OVERRIDE_MU"
echo "OVERRIDE_KH=$OVERRIDE_KH"
echo "SCENE_CAMERA=$SCENE_CAMERA"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VIS_DIR=$VIS_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=candidate_mjwarp_direct_force_tactile_not_training_not_curiosity_success"

extra_args=("--material-label" "$MATERIAL_LABEL")
if [[ -n "$OVERRIDE_MU" ]]; then
  extra_args+=("--override-mu" "$OVERRIDE_MU")
fi
if [[ -n "$OVERRIDE_KH" ]]; then
  extra_args+=("--override-kh" "$OVERRIDE_KH")
fi
if [[ "$SCENE_CAMERA" == "1" || "$SCENE_CAMERA" == "true" || "$SCENE_CAMERA" == "TRUE" ]]; then
  extra_args+=("--scene-camera" "--scene-camera-width" "$SCENE_CAMERA_WIDTH" "--scene-camera-height" "$SCENE_CAMERA_HEIGHT")
fi

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1800 "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_mjw_direct_tactile_export.py" \
    --run-tag "$RUN_TAG" \
    --device "$DEVICE" \
    --scene "$SCENE" \
    --num-frames "$NUM_FRAMES" \
    --map-size "$MAP_SIZE" \
    --fps "$FPS" \
    "${extra_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --visual-dir "$VIS_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE00_MJW_DIRECT_TACTILE_EXPORT_END"
