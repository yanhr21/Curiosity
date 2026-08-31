#!/usr/bin/env bash
# Poll the official release and recompute the fail-closed training admission gate.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/release_training_monitor_v1}"
POLL_SECONDS="${2:-600}"
MAX_POLLS="${3:-30}"
WAN_BASE_AUDIT="${WAN_BASE_AUDIT:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/wan_base_h200_audit_v1/WAN22_BASE_AUDIT.json}"
SUGAR_MANIFEST_RESULT="${SUGAR_MANIFEST_RESULT:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/RESULT.json}"
SUGAR_DATA_DIVERSITY_RESULT="${SUGAR_DATA_DIVERSITY_RESULT:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/training_data_diversity_v1/SUGAR_TRAINING_DATA_DIVERSITY.json}"
PROMPT_CASE_RESULT="${PROMPT_CASE_RESULT:-$PROJECT_ROOT/experiments/demo_following/zero_wam_official_v1/frozen_prompt_gate_cases_v1/FROZEN_PROMPT_GATE_CASES_RESULT.json}"
PROMPT_GATE_RESULT="${PROMPT_GATE_RESULT:-}"
ADAPTER_AUDIT_RESULT="${ADAPTER_AUDIT_RESULT:-}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained Slurm allocation" >&2
    exit 2
fi
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$DEVICE_NAME" != *H200* ]]; then
    echo "expected retained H200 allocation, found: $DEVICE_NAME" >&2
    exit 2
fi
if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || ((POLL_SECONDS < 60)); then
    echo "poll interval must be an integer of at least 60 seconds" >&2
    exit 2
fi
if ! [[ "$MAX_POLLS" =~ ^[0-9]+$ ]] || ((MAX_POLLS < 1)); then
    echo "max polls must be a positive integer" >&2
    exit 2
fi
if [[ ! -f "$WAN_BASE_AUDIT" || ! -f "$SUGAR_MANIFEST_RESULT" || ! -f "$SUGAR_DATA_DIVERSITY_RESULT" || ! -f "$PROMPT_CASE_RESULT" ]]; then
    echo "required Wan base, SUGAR manifest, diversity or prompt-case audit is missing" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
for ((poll_index = 1; poll_index <= MAX_POLLS; poll_index += 1)); do
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    poll_dir=$(printf "%s/poll_%03d_%s" "$OUTPUT_ROOT" "$poll_index" "$timestamp")
    release_dir="$poll_dir/release"
    admission_dir="$poll_dir/admission"
    mkdir -p "$poll_dir"

    set +e
    bash "$PROJECT_ROOT/scripts/sugar/demo_following/run_zero_wam_official_release_audit.sh" \
        "$release_dir" >"$poll_dir/release_audit.log" 2>&1
    release_rc=$?
    set -e
    if [[ "$release_rc" -ne 0 && "$release_rc" -ne 3 ]]; then
        echo "official release audit failed unexpectedly with rc=$release_rc" >&2
        exit "$release_rc"
    fi
    release_result="$release_dir/audit/OFFICIAL_RELEASE_AUDIT.json"

    admission_args=(
        --release-audit "$release_result"
        --wan-base-audit "$WAN_BASE_AUDIT"
        --sugar-manifest-result "$SUGAR_MANIFEST_RESULT"
        --sugar-data-diversity-result "$SUGAR_DATA_DIVERSITY_RESULT"
        --prompt-case-result "$PROMPT_CASE_RESULT"
        --output-dir "$admission_dir"
    )
    if [[ -n "$PROMPT_GATE_RESULT" && -f "$PROMPT_GATE_RESULT" ]]; then
        admission_args+=(--prompt-gate-result "$PROMPT_GATE_RESULT")
    fi
    if [[ -n "$ADAPTER_AUDIT_RESULT" && -f "$ADAPTER_AUDIT_RESULT" ]]; then
        admission_args+=(--adapter-audit-result "$ADAPTER_AUDIT_RESULT")
    fi
    "$PYTHON_BIN" \
        "$PROJECT_ROOT/scripts/sugar/demo_following/audit_zero_wam_training_admission.py" \
        "${admission_args[@]}" >/dev/null
    admission_result="$admission_dir/ZERO_WAM_TRAINING_ADMISSION.json"

    release_available=$(jq -r '.release_available' "$release_result")
    release_passed=$(jq -r '.passed' "$release_result")
    training_allowed=$(jq -r '.training_allowed' "$admission_result")
    next_branch=$(jq -r '.automatic_next_branch' "$admission_result")
    jq -n \
        --arg protocol "zero_wam_release_training_monitor_v1" \
        --arg timestamp "$timestamp" \
        --argjson poll_index "$poll_index" \
        --argjson release_rc "$release_rc" \
        --argjson release_available "$release_available" \
        --argjson release_passed "$release_passed" \
        --argjson training_allowed "$training_allowed" \
        --arg next_branch "$next_branch" \
        '{
            protocol: $protocol,
            timestamp_utc: $timestamp,
            poll_index: $poll_index,
            release_audit_rc: $release_rc,
            release_available: $release_available,
            release_passed: $release_passed,
            training_allowed: $training_allowed,
            automatic_next_branch: $next_branch
        }' >"$poll_dir/MONITOR_STATE.json"
    ln -sfn "$(basename "$poll_dir")" "$OUTPUT_ROOT/latest"
    jq -c '.' "$poll_dir/MONITOR_STATE.json"

    if [[ "$training_allowed" == "true" ]]; then
        exit 0
    fi
    if [[ "$release_available" == "true" ]]; then
        # The external artifact changed.  Stop polling so the next autonomous
        # stage can inspect its unknown schema and strict-load it faithfully.
        exit 4
    fi
    if ((poll_index < MAX_POLLS)); then
        sleep "$POLL_SECONDS"
    fi
done

exit 3
