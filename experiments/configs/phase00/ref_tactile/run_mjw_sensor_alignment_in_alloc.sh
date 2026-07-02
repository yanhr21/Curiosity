#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_mjw_sensor_align_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_main}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
NUM_FRAMES="${NUM_FRAMES:-240}"
SCENE="${SCENE:-cube}"
MATERIAL_LABEL="${MATERIAL_LABEL:-steel_spec_v1}"
OVERRIDE_MU="${OVERRIDE_MU:-0.3}"
OVERRIDE_KH="${OVERRIDE_KH:-1000000000000}"
MAX_RELATIVE_RMSE="${MAX_RELATIVE_RMSE:-0.25}"
MIN_MEAN_COSINE="${MIN_MEAN_COSINE:-0.80}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/mujoco_align/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/mujoco_align/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/mujoco_align/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_mjw_sensor_alignment_probe.py"; do
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
  echo "ERROR: Phase 00 MJWarp/SensorContact alignment requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

newton_commit="$(git -C "$NEWTON_ROOT" rev-parse HEAD)"
run_log="$LOG_DIR/mjw_sensor_alignment.log"

echo "PHASE00_MJW_SENSOR_ALIGNMENT_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$newton_commit"
echo "DEVICE=$DEVICE"
echo "NUM_FRAMES=$NUM_FRAMES"
echo "SCENE=$SCENE"
echo "MATERIAL_LABEL=$MATERIAL_LABEL"
echo "OVERRIDE_MU=$OVERRIDE_MU"
echo "OVERRIDE_KH=$OVERRIDE_KH"
echo "MAX_RELATIVE_RMSE=$MAX_RELATIVE_RMSE"
echo "MIN_MEAN_COSINE=$MIN_MEAN_COSINE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=diagnostic_only_candidate_mjwarp_vs_sensorcontact_alignment_not_training"

extra_args=("--material-label" "$MATERIAL_LABEL")
if [[ -n "$OVERRIDE_MU" ]]; then
  extra_args+=("--override-mu" "$OVERRIDE_MU")
fi
if [[ -n "$OVERRIDE_KH" ]]; then
  extra_args+=("--override-kh" "$OVERRIDE_KH")
fi

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 1800 "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_mjw_sensor_alignment_probe.py" \
    --run-tag "$RUN_TAG" \
    --device "$DEVICE" \
    --scene "$SCENE" \
    --num-frames "$NUM_FRAMES" \
    "${extra_args[@]}" \
    --max-relative-rmse "$MAX_RELATIVE_RMSE" \
    --min-mean-cosine "$MIN_MEAN_COSINE" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE00_MJW_SENSOR_ALIGNMENT_END"
