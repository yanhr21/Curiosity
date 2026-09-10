#!/usr/bin/env bash
# Freeze the dual real-camera BPP render contract from admitted physical traces.

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
BASE="${1:-$ROOT/experiments/demo_following/bpp_official_v1/sugar_five_variant_corpus_v1}"
ADMISSION="$BASE/action_admission_v1"
ADMITTED="$ADMISSION/BPP_ADMITTED_VARIANTS.jsonl"
CANDIDATES="${CANDIDATES:-$ROOT/scripts/sugar/demo_following/config/bpp_dual_camera_candidates_v1.json}"
AUDIT_ROOT="$BASE/camera_visibility_v1"
CONTRACT="$AUDIT_ROOT/BPP_DUAL_CAMERA_CONTRACT.json"
KIT_ARGS="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "BPP camera visibility audit requires a retained Slurm allocation" >&2
    exit 2
fi
if [[ "$(hostname)" == login* || "$(hostname)" == mgmtserver* ]]; then
    echo "BPP camera visibility audit is forbidden on a login node" >&2
    exit 2
fi
jq -e '.passed == true and .admitted_trajectory_count == 995' \
    "$ADMISSION/RESULT.json" >/dev/null
if [[ "$(wc -l < "$ADMITTED")" -ne 995 ]]; then
    echo "admitted manifest must contain exactly 995 rows" >&2
    exit 2
fi

select_trace() {
    local task=$1
    local selected
    selected=$(jq -r --arg task "$task" '
        select(.task == $task and .source_motion_id == 0 and .admitted_variant_rank == 0)
        | .trace_path
    ' "$ADMITTED")
    if [[ -z "$selected" || "$selected" == *$'\n'* ]]; then
        echo "expected one rank-zero admitted trace for $task source 0" >&2
        exit 2
    fi
    printf '%s\n' "$ROOT/$selected"
}

carry_trace=$(select_trace CarryBox)
kick_trace=$(select_trace KickBox)
carry_output="$AUDIT_ROOT/carry_source000"
kick_output="$AUDIT_ROOT/kick_source000"

if [[ ! -f "$carry_output/RESULT.json" ]]; then
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_following/audit_bpp_dual_camera_visibility.py" \
        --trace "$carry_trace" --source-motion-id 0 \
        --candidate-contract "$CANDIDATES" --output-dir "$carry_output" \
        --headless --device cuda:0 --kit_args "$KIT_ARGS"
fi
jq -e '.passed == true and .task == "CarryBox" and .source_motion_id == 0' \
    "$carry_output/RESULT.json" >/dev/null

if [[ ! -f "$kick_output/RESULT.json" ]]; then
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_following/audit_bpp_dual_camera_visibility.py" \
        --trace "$kick_trace" --source-motion-id 0 \
        --candidate-contract "$CANDIDATES" --output-dir "$kick_output" \
        --headless --device cuda:0 --kit_args "$KIT_ARGS"
fi
jq -e '.passed == true and .task == "KickBox" and .source_motion_id == 0' \
    "$kick_output/RESULT.json" >/dev/null

if [[ ! -f "$CONTRACT" ]]; then
    "$PYTHON_BIN" \
        "$ROOT/scripts/sugar/demo_following/finalize_bpp_dual_camera_contract.py" \
        --candidate-contract "$CANDIDATES" \
        --carry-result "$carry_output/RESULT.json" \
        --kick-result "$kick_output/RESULT.json" \
        --action-admission "$ADMISSION" \
        --output "$CONTRACT"
fi
jq -e '
    .protocol == "sugar_bpp_dual_camera_render_contract_v1"
    and .passed_visibility_audit == true
    and .camera_keys == ["agentview_rgb", "eye_in_hand_rgb"]
    and .neighbor_isolation_lower_bound_m > .camera_far_clip_m
' "$CONTRACT" >/dev/null

echo "BPP_DUAL_CAMERA_VISIBILITY_COMPLETE contract=$CONTRACT"
