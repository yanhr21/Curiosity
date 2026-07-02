#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_ref_diag_$(date +%Y%m%d_%H%M%S)}"
TACCEL_ROOT="${TACCEL_ROOT:-$ROOT/external/Taccel}"
TACCEL_VENV="${TACCEL_VENV:-$ROOT/envs/taccel/.venv}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-48}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/diag/$RUN_TAG}"
VISUAL_DIR="${VISUAL_DIR:-$ROOT/experiments/visuals/phase00/ref_tactile/diag/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/diag/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/diag/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$VISUAL_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$TACCEL_ROOT" "$TACCEL_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_ref_tactile_diagnostic.py"; do
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
  echo "ERROR: Phase 00 reference tactile diagnostic requires an H200 allocation." >&2
  printf '%s\n' "$gpu_names" >&2
  exit 5
fi

taccel_commit="$(git -C "$TACCEL_ROOT" rev-parse HEAD)"

echo "PHASE00_REF_TACTILE_DIAGNOSTIC_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "GPU_NAMES=$gpu_names"
echo "TACCEL_ROOT=$TACCEL_ROOT"
echo "TACCEL_COMMIT=$taccel_commit"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VISUAL_DIR=$VISUAL_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=environment_diagnostic_only_not_training_not_curiosity_success"

diag_log="$LOG_DIR/ref_tactile_diagnostic.log"
(
  cd "$TACCEL_ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$TACCEL_ROOT:$ROOT/src:${PYTHONPATH:-}"
  export TACCEL_PTX_DIR="$OUTPUT_DIR/ptx"
  export TACCEL_PTX_ARCH="${TACCEL_PTX_ARCH:-86}"
  timeout 7200 "$TACCEL_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_ref_tactile_diagnostic.py" \
    --run-tag "$RUN_TAG" \
    --output-dir "$OUTPUT_DIR" \
    --visual-dir "$VISUAL_DIR" \
    --report-dir "$REPORT_DIR" \
    --device "$DEVICE" \
    --steps "$STEPS"
) >"$diag_log" 2>&1

echo "DIAG_LOG=$diag_log"
echo "PHASE00_REF_TACTILE_DIAGNOSTIC_END"
