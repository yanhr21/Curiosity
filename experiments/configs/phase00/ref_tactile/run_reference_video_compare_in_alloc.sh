#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_refcmp_$(date +%Y%m%d_%H%M%S)}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_main}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-$ROOT/0780e5ec3fdb26b63ae63de0f49f07c4.mp4}"
CANDIDATE_VIDEO="${CANDIDATE_VIDEO:-$ROOT/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile.avi}"
CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_direct_steel_v1_20260701_060500/candidate_mjw_direct_tactile_summary.json}"
SAMPLES="${SAMPLES:-12}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/ref_compare/$RUN_TAG}"
VIS_DIR="${VIS_DIR:-$ROOT/experiments/visuals/phase00/ref_tactile/ref_compare/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/ref_compare/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/ref_compare/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$VIS_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_ROOT" "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_reference_video_compare.py" "$REFERENCE_VIDEO" "$CANDIDATE_VIDEO" "$CANDIDATE_SUMMARY"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

run_log="$LOG_DIR/reference_video_compare.log"
newton_commit="$(git -C "$NEWTON_ROOT" rev-parse HEAD)"

echo "PHASE00_REFERENCE_VIDEO_COMPARE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "NEWTON_ROOT=$NEWTON_ROOT"
echo "NEWTON_COMMIT=$newton_commit"
echo "REFERENCE_VIDEO=$REFERENCE_VIDEO"
echo "CANDIDATE_VIDEO=$CANDIDATE_VIDEO"
echo "CANDIDATE_SUMMARY=$CANDIDATE_SUMMARY"
echo "SAMPLES=$SAMPLES"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VIS_DIR=$VIS_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=reference_video_comparison_not_training_not_curiosity_success"

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$NEWTON_ROOT:$ROOT/src:${PYTHONPATH:-}"
  timeout 900 "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_reference_video_compare.py" \
    --run-tag "$RUN_TAG" \
    --reference-video "$REFERENCE_VIDEO" \
    --candidate-video "$CANDIDATE_VIDEO" \
    --candidate-summary "$CANDIDATE_SUMMARY" \
    --samples "$SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --visual-dir "$VIS_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE00_REFERENCE_VIDEO_COMPARE_END"
