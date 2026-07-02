#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_chan_audit_$(date +%Y%m%d_%H%M%S)}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-$ROOT/0780e5ec3fdb26b63ae63de0f49f07c4.mp4}"
CANDIDATE_VIDEO="${CANDIDATE_VIDEO:-$ROOT/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile.avi}"
CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_summary.json}"
GATE_REVIEW_SUMMARY="${GATE_REVIEW_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_review_v2_20260701_080800/phase00_gate_review_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/channel_audit/$RUN_TAG}"
VIS_DIR="${VIS_DIR:-$ROOT/experiments/visuals/phase00/ref_tactile/channel_audit/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/channel_audit/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/channel_audit/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$VIS_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_channel_semantic_audit.py" "$REFERENCE_VIDEO" "$CANDIDATE_VIDEO" "$CANDIDATE_SUMMARY" "$GATE_REVIEW_SUMMARY"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

run_log="$LOG_DIR/channel_semantic_audit.log"

echo "PHASE00_CHANNEL_SEMANTIC_AUDIT_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REFERENCE_VIDEO=$REFERENCE_VIDEO"
echo "CANDIDATE_VIDEO=$CANDIDATE_VIDEO"
echo "CANDIDATE_SUMMARY=$CANDIDATE_SUMMARY"
echo "GATE_REVIEW_SUMMARY=$GATE_REVIEW_SUMMARY"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "VIS_DIR=$VIS_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=channel_semantic_audit_not_training_not_curiosity_success"

(
  cd "$ROOT"
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
  "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_channel_semantic_audit.py" \
    --run-tag "$RUN_TAG" \
    --reference-video "$REFERENCE_VIDEO" \
    --candidate-video "$CANDIDATE_VIDEO" \
    --candidate-summary "$CANDIDATE_SUMMARY" \
    --gate-review-summary "$GATE_REVIEW_SUMMARY" \
    --output-dir "$OUTPUT_DIR" \
    --visual-dir "$VIS_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE00_CHANNEL_SEMANTIC_AUDIT_END"
