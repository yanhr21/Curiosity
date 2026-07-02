#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_gate_review_$(date +%Y%m%d_%H%M%S)}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
BENCHMARK_SUMMARY="${BENCHMARK_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_main_20260701_035529/newton_hydro_benchmark_summary.json}"
CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_marker_v1_20260701_074200/candidate_mjw_direct_tactile_summary.json}"
REFERENCE_COMPARE_SUMMARY="${REFERENCE_COMPARE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_marker_v1_20260701_074900/reference_video_compare_summary.json}"
ALIGNMENT_SUMMARY="${ALIGNMENT_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_summary.json}"
CHANNEL_AUDIT_SUMMARY="${CHANNEL_AUDIT_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_audit_v1_20260701_082100/channel_semantic_audit_summary.json}"
SEMANTIC_REFERENCE_MATRIX="${SEMANTIC_REFERENCE_MATRIX:-$ROOT/experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json}"
SEMANTIC_BRIDGE_SPEC="${SEMANTIC_BRIDGE_SPEC:-$ROOT/experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json}"
REFERENCE_ENV_AVAILABILITY_SUMMARY="${REFERENCE_ENV_AVAILABILITY_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json}"
REFERENCE_ASSET_AVAILABILITY_SUMMARY="${REFERENCE_ASSET_AVAILABILITY_SUMMARY:-$ROOT/experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json}"
REFERENCE_ASSET_REUSE_PLAN="${REFERENCE_ASSET_REUSE_PLAN:-$ROOT/experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json}"
UNIVTAC_SANITY_SUMMARY="${UNIVTAC_SANITY_SUMMARY:-}"
TACAUCHY_SANITY_SUMMARY="${TACAUCHY_SANITY_SUMMARY:-}"
ISAACLAB_TACSL_SANITY_SUMMARY="${ISAACLAB_TACSL_SANITY_SUMMARY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/gate_review/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/gate_review/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/gate_review/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_gate_review.py" "$BENCHMARK_SUMMARY" "$CANDIDATE_SUMMARY" "$REFERENCE_COMPARE_SUMMARY" "$ALIGNMENT_SUMMARY" "$CHANNEL_AUDIT_SUMMARY" "$SEMANTIC_REFERENCE_MATRIX" "$SEMANTIC_BRIDGE_SPEC" "$REFERENCE_ENV_AVAILABILITY_SUMMARY" "$REFERENCE_ASSET_AVAILABILITY_SUMMARY" "$REFERENCE_ASSET_REUSE_PLAN"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done
for optional_path in "$UNIVTAC_SANITY_SUMMARY" "$TACAUCHY_SANITY_SUMMARY" "$ISAACLAB_TACSL_SANITY_SUMMARY"; do
  if [[ -n "$optional_path" && ! -e "$optional_path" ]]; then
    echo "ERROR: optional sanity summary was set but does not exist: $optional_path" >&2
    exit 4
  fi
done

run_log="$LOG_DIR/phase00_gate_review.log"

echo "PHASE00_GATE_REVIEW_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "BENCHMARK_SUMMARY=$BENCHMARK_SUMMARY"
echo "CANDIDATE_SUMMARY=$CANDIDATE_SUMMARY"
echo "REFERENCE_COMPARE_SUMMARY=$REFERENCE_COMPARE_SUMMARY"
echo "ALIGNMENT_SUMMARY=$ALIGNMENT_SUMMARY"
echo "CHANNEL_AUDIT_SUMMARY=$CHANNEL_AUDIT_SUMMARY"
echo "SEMANTIC_REFERENCE_MATRIX=$SEMANTIC_REFERENCE_MATRIX"
echo "SEMANTIC_BRIDGE_SPEC=$SEMANTIC_BRIDGE_SPEC"
echo "REFERENCE_ENV_AVAILABILITY_SUMMARY=$REFERENCE_ENV_AVAILABILITY_SUMMARY"
echo "REFERENCE_ASSET_AVAILABILITY_SUMMARY=$REFERENCE_ASSET_AVAILABILITY_SUMMARY"
echo "REFERENCE_ASSET_REUSE_PLAN=$REFERENCE_ASSET_REUSE_PLAN"
echo "UNIVTAC_SANITY_SUMMARY=${UNIVTAC_SANITY_SUMMARY:-missing}"
echo "TACAUCHY_SANITY_SUMMARY=${TACAUCHY_SANITY_SUMMARY:-missing}"
echo "ISAACLAB_TACSL_SANITY_SUMMARY=${ISAACLAB_TACSL_SANITY_SUMMARY:-missing}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NOTE=gate_review_not_training_not_curiosity_success"

extra_args=()
if [[ -n "$UNIVTAC_SANITY_SUMMARY" ]]; then
  extra_args+=(--univtac-sanity-summary "$UNIVTAC_SANITY_SUMMARY")
fi
if [[ -n "$TACAUCHY_SANITY_SUMMARY" ]]; then
  extra_args+=(--tacauchy-sanity-summary "$TACAUCHY_SANITY_SUMMARY")
fi
if [[ -n "$ISAACLAB_TACSL_SANITY_SUMMARY" ]]; then
  extra_args+=(--isaaclab-tacsl-sanity-summary "$ISAACLAB_TACSL_SANITY_SUMMARY")
fi

(
  cd "$ROOT"
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
  "$NEWTON_VENV/bin/python" "$ROOT/src/newton_tactile_curiosity/phase00_gate_review.py" \
    --run-tag "$RUN_TAG" \
    --root "$ROOT" \
    --benchmark-summary "$BENCHMARK_SUMMARY" \
    --candidate-summary "$CANDIDATE_SUMMARY" \
    --reference-compare-summary "$REFERENCE_COMPARE_SUMMARY" \
    --alignment-summary "$ALIGNMENT_SUMMARY" \
    --channel-audit-summary "$CHANNEL_AUDIT_SUMMARY" \
    --semantic-reference-matrix "$SEMANTIC_REFERENCE_MATRIX" \
    --semantic-bridge-spec "$SEMANTIC_BRIDGE_SPEC" \
    --reference-env-availability-summary "$REFERENCE_ENV_AVAILABILITY_SUMMARY" \
    --reference-asset-availability-summary "$REFERENCE_ASSET_AVAILABILITY_SUMMARY" \
    --reference-asset-reuse-plan "$REFERENCE_ASSET_REUSE_PLAN" \
    "${extra_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR"
) 2>&1 | tee "$run_log"

echo "PHASE00_GATE_REVIEW_END"
