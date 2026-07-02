#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_gate00f_bundle_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/gate00f_bundle/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/gate00f_bundle/$RUN_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/gate00f_bundle/$RUN_TAG}"
ALLOW_BLOCKER_SANITY="${ALLOW_BLOCKER_SANITY:-0}"
REQUIRE_RUNTIME_PREFLIGHT="${REQUIRE_RUNTIME_PREFLIGHT:-1}"
RUNTIME_REGISTRY="${RUNTIME_REGISTRY:-$ROOT/experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json}"
BENCHMARK_SUMMARY="${BENCHMARK_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_8c501_hot_r2_v1_20260701_162800/newton_hydro_benchmark_summary.json}"
CANDIDATE_SUMMARY="${CANDIDATE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_summary.json}"
REFERENCE_COMPARE_SUMMARY="${REFERENCE_COMPARE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_cont_20260701_1925/reference_video_compare_summary.json}"
CHANNEL_AUDIT_SUMMARY="${CHANNEL_AUDIT_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_8c501_cont_20260701_1926/channel_semantic_audit_summary.json}"
ALIGNMENT_SUMMARY="${ALIGNMENT_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_summary.json}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUTPUT_DIR"

for required_candidate_path in "$BENCHMARK_SUMMARY" "$CANDIDATE_SUMMARY" "$REFERENCE_COMPARE_SUMMARY" "$CHANNEL_AUDIT_SUMMARY" "$ALIGNMENT_SUMMARY"; do
  if [[ ! -e "$required_candidate_path" ]]; then
    echo "ERROR: missing required 8c501 candidate evidence path: $required_candidate_path" >&2
    exit 3
  fi
done

bundle_log="$LOG_DIR/gate00f_reference_bundle.log"
summary_json="$OUTPUT_DIR/gate00f_reference_bundle_summary.json"
report_md="$REPORT_DIR/gate00f_reference_bundle.md"

univtac_tag="${RUN_TAG}_univtac"
tacauchy_tag="${RUN_TAG}_tacauchy"
tacsl_tag="${RUN_TAG}_tacsl"
gate_tag="${RUN_TAG}_gate"
preflight_tag="${RUN_TAG}_preflight"

univtac_summary="$ROOT/experiments/outputs/phase00/ref_tactile/reference_sanity/$univtac_tag/univtac_official_reference_sanity_summary.json"
tacauchy_summary="$ROOT/experiments/outputs/phase00/ref_tactile/reference_sanity/$tacauchy_tag/tacauchy_official_reference_sanity_summary.json"
tacsl_summary="$ROOT/experiments/outputs/phase00/ref_tactile/reference_sanity/$tacsl_tag/isaaclab_tacsl_official_sanity_summary.json"
gate_summary="$ROOT/experiments/outputs/phase00/ref_tactile/gate_review/$gate_tag/phase00_gate_review_summary.json"
runtime_preflight_summary="${GATE00F_RUNTIME_PREFLIGHT_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/runtime_preflight/$preflight_tag/gate00f_runtime_preflight_summary.json}"

check_executable_env() {
  local name="$1"
  local explicit_var="$2"
  local default_a="$3"
  local default_b="$4"
  local explicit_value="${!explicit_var:-}"
  if [[ -n "$explicit_value" && -x "$explicit_value" ]]; then
    return 0
  fi
  if [[ -x "$default_a" || -x "$default_b" ]]; then
    return 0
  fi
  if [[ "$ALLOW_BLOCKER_SANITY" == "1" ]]; then
    return 0
  fi
  echo "ERROR: missing executable environment for $name." >&2
  echo "Set $explicit_var or set ALLOW_BLOCKER_SANITY=1 to record blocker summaries." >&2
  return 1
}

{
  echo "GATE00F_REFERENCE_BUNDLE_START"
  echo "RUN_TAG=$RUN_TAG"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "HOSTNAME=$(hostname)"
  echo "ALLOW_BLOCKER_SANITY=$ALLOW_BLOCKER_SANITY"
  echo "REQUIRE_RUNTIME_PREFLIGHT=$REQUIRE_RUNTIME_PREFLIGHT"
  echo "RUNTIME_REGISTRY=$RUNTIME_REGISTRY"
  echo "RUNTIME_PREFLIGHT_SUMMARY=$runtime_preflight_summary"
  echo "BENCHMARK_SUMMARY=$BENCHMARK_SUMMARY"
  echo "CANDIDATE_SUMMARY=$CANDIDATE_SUMMARY"
  echo "REFERENCE_COMPARE_SUMMARY=$REFERENCE_COMPARE_SUMMARY"
  echo "CHANNEL_AUDIT_SUMMARY=$CHANNEL_AUDIT_SUMMARY"
  echo "ALIGNMENT_SUMMARY=$ALIGNMENT_SUMMARY"
  echo "NOTE=official_reference_bundle_not_training_not_curiosity_success"

  runtime_preflight_status="not_required"
  if [[ "$REQUIRE_RUNTIME_PREFLIGHT" != "0" ]]; then
    if [[ ! -f "$runtime_preflight_summary" ]]; then
      RUNTIME_REGISTRY="$RUNTIME_REGISTRY" RUN_TAG="$preflight_tag" \
        bash "$ROOT/experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh"
    fi
    runtime_preflight_status="$(jq -r '.status // "missing"' "$runtime_preflight_summary" 2>/dev/null || echo missing)"
    if [[ "$runtime_preflight_status" != "pass_gate00f_runtime_preflight" ]]; then
      jq -n \
        --arg run_tag "$RUN_TAG" \
        --arg classification "gate00f_reference_bundle_not_training_not_curiosity_success" \
        --arg slurm_job_id "${SLURM_JOB_ID:-}" \
        --arg host "$(hostname)" \
        --arg allow_blocker_sanity "$ALLOW_BLOCKER_SANITY" \
        --arg runtime_preflight_summary "$runtime_preflight_summary" \
        --arg runtime_preflight_status "$runtime_preflight_status" \
        '{
          run_tag: $run_tag,
          classification: $classification,
          status: "fail_gate00f_bundle_runtime_preflight_not_passed",
          slurm_job_id: $slurm_job_id,
          host: $host,
          allow_blocker_sanity: $allow_blocker_sanity,
          runtime_preflight_summary: $runtime_preflight_summary,
          runtime_preflight_status: $runtime_preflight_status,
          curiosity_training_allowed: false
        }' >"$summary_json"

      cat >"$report_md" <<EOF
# Gate 00F Reference Bundle

- Run tag: \`$RUN_TAG\`
- Slurm job: \`${SLURM_JOB_ID:-}\`
- Host: \`$(hostname)\`
- Status: \`fail_gate00f_bundle_runtime_preflight_not_passed\`
- Runtime preflight summary: \`$runtime_preflight_summary\`
- Runtime preflight status: \`$runtime_preflight_status\`

The Gate 00F bundle requires runtime preflight to pass before official
reference sanity commands run. This is not training, not tactile gate
completion, and not curiosity success.
EOF

      echo "SUMMARY_JSON=$summary_json"
      echo "REPORT_MD=$report_md"
      echo "GATE00F_REFERENCE_BUNDLE_END_PREFLIGHT_NOT_PASSED"
      exit 1
    fi
  else
    check_executable_env "UniVTAC" "UNIVTAC_PYTHON" "$ROOT/envs/univtac/conda/bin/python" "$ROOT/envs/univtac/.venv/bin/python"
    check_executable_env "TaCauchy" "TACAUCHY_PYTHON" "$ROOT/envs/tacauchy/conda/bin/python" "$ROOT/envs/tacauchy/.venv/bin/python"
    check_executable_env "IsaacLab TacSL" "ISAACLAB_TACSL_PYTHON" "$ROOT/envs/isaaclab_tacsl/conda/bin/python" "$ROOT/envs/isaaclab_tacsl/.venv/bin/python"
  fi

  RUNTIME_REGISTRY="$RUNTIME_REGISTRY" TARGET=univtac RUN_TAG="$univtac_tag" \
    bash "$ROOT/experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh"

  RUNTIME_REGISTRY="$RUNTIME_REGISTRY" TARGET=tacauchy RUN_TAG="$tacauchy_tag" \
    bash "$ROOT/experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh"

  RUNTIME_REGISTRY="$RUNTIME_REGISTRY" RUN_TAG="$tacsl_tag" \
    bash "$ROOT/experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh"

  RUN_TAG="$gate_tag" \
    BENCHMARK_SUMMARY="$BENCHMARK_SUMMARY" \
    CANDIDATE_SUMMARY="$CANDIDATE_SUMMARY" \
    REFERENCE_COMPARE_SUMMARY="$REFERENCE_COMPARE_SUMMARY" \
    CHANNEL_AUDIT_SUMMARY="$CHANNEL_AUDIT_SUMMARY" \
    ALIGNMENT_SUMMARY="$ALIGNMENT_SUMMARY" \
    UNIVTAC_SANITY_SUMMARY="$univtac_summary" \
    TACAUCHY_SANITY_SUMMARY="$tacauchy_summary" \
    ISAACLAB_TACSL_SANITY_SUMMARY="$tacsl_summary" \
    bash "$ROOT/experiments/configs/phase00/ref_tactile/run_phase00_gate_review_in_alloc.sh"

  univtac_status="$(jq -r '.status // "missing"' "$univtac_summary" 2>/dev/null || echo missing)"
  tacauchy_status="$(jq -r '.status // "missing"' "$tacauchy_summary" 2>/dev/null || echo missing)"
  tacsl_status="$(jq -r '.status // "missing"' "$tacsl_summary" 2>/dev/null || echo missing)"
  gate_status="$(jq -r '.status // "missing"' "$gate_summary" 2>/dev/null || echo missing)"
  gate00f_status="$(jq -r '.gate_00f_official_semantic_validation_status // "missing"' "$gate_summary" 2>/dev/null || echo missing)"

  cat >"$summary_json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "classification": "gate00f_reference_bundle_not_training_not_curiosity_success",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "host": "$(hostname)",
  "allow_blocker_sanity": "$ALLOW_BLOCKER_SANITY",
  "runtime_registry": "$RUNTIME_REGISTRY",
  "runtime_preflight_summary": "$runtime_preflight_summary",
  "runtime_preflight_status": "$runtime_preflight_status",
  "benchmark_summary": "$BENCHMARK_SUMMARY",
  "candidate_summary": "$CANDIDATE_SUMMARY",
  "reference_compare_summary": "$REFERENCE_COMPARE_SUMMARY",
  "channel_audit_summary": "$CHANNEL_AUDIT_SUMMARY",
  "alignment_summary": "$ALIGNMENT_SUMMARY",
  "univtac_summary": "$univtac_summary",
  "univtac_status": "$univtac_status",
  "tacauchy_summary": "$tacauchy_summary",
  "tacauchy_status": "$tacauchy_status",
  "isaaclab_tacsl_summary": "$tacsl_summary",
  "isaaclab_tacsl_status": "$tacsl_status",
  "gate_review_summary": "$gate_summary",
  "gate_review_status": "$gate_status",
  "gate00f_status": "$gate00f_status",
  "curiosity_training_allowed": false
}
EOF

  cat >"$report_md" <<EOF
# Gate 00F Reference Bundle

- Run tag: \`$RUN_TAG\`
- Slurm job: \`${SLURM_JOB_ID:-}\`
- Host: \`$(hostname)\`
- Allow blocker sanity: \`$ALLOW_BLOCKER_SANITY\`
- Runtime registry: \`$RUNTIME_REGISTRY\`
- Runtime preflight summary: \`$runtime_preflight_summary\`
- Runtime preflight status: \`$runtime_preflight_status\`
- Classification: \`gate00f_reference_bundle_not_training_not_curiosity_success\`

## Candidate Evidence For Gate Review

- Benchmark summary: \`$BENCHMARK_SUMMARY\`
- Candidate tactile summary: \`$CANDIDATE_SUMMARY\`
- Reference comparison summary: \`$REFERENCE_COMPARE_SUMMARY\`
- Channel audit summary: \`$CHANNEL_AUDIT_SUMMARY\`
- Alignment summary: \`$ALIGNMENT_SUMMARY\`

## Outputs

- UniVTAC summary: \`$univtac_summary\`
- UniVTAC status: \`$univtac_status\`
- TaCauchy summary: \`$tacauchy_summary\`
- TaCauchy status: \`$tacauchy_status\`
- IsaacLab TacSL summary: \`$tacsl_summary\`
- IsaacLab TacSL status: \`$tacsl_status\`
- Gate review summary: \`$gate_summary\`
- Gate review status: \`$gate_status\`
- Gate 00F status: \`$gate00f_status\`

This bundle is an official-reference sanity and gate-review orchestration path.
It is not training, not tactile gate completion by itself, and not curiosity
success.
EOF

  echo "SUMMARY_JSON=$summary_json"
  echo "REPORT_MD=$report_md"
  echo "GATE00F_REFERENCE_BUNDLE_END"
} 2>&1 | tee "$bundle_log"
