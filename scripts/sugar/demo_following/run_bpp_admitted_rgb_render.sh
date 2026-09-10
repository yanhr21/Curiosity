#!/usr/bin/env bash
# Render only the five automatically admitted physical variants per SUGAR motion.

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
BASE="${1:-$ROOT/experiments/demo_following/bpp_official_v1/sugar_five_variant_corpus_v1}"
ADMISSION="$BASE/action_admission_v1"
ADMITTED="$ADMISSION/BPP_ADMITTED_VARIANTS.jsonl"
RGB_ROOT="$BASE/robot_rgb_admitted_10hz_v1"
CONTRACT="${CONTRACT:-$ROOT/scripts/sugar/demo_following/config/bpp_five_variant_corpus_v1.json}"
CAMERA_CONTRACT="${CAMERA_CONTRACT:-$BASE/camera_visibility_v1/BPP_DUAL_CAMERA_CONTRACT.json}"
SENSOR_AUDIT="$BASE/sensorimotor_corpus_v1"
KIT_ARGS="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "BPP RGB rendering requires a retained Slurm compute allocation" >&2
    exit 2
fi
if [[ "$(hostname)" == login* || "$(hostname)" == mgmtserver* ]]; then
    echo "BPP RGB rendering is forbidden on a login node" >&2
    exit 2
fi
jq -e '
    .protocol == "sugar_bpp_dual_camera_render_contract_v1"
    and .passed_visibility_audit == true
    and .camera_keys == ["agentview_rgb", "eye_in_hand_rgb"]
' "$CAMERA_CONTRACT" >/dev/null
jq -e '.passed == true and .admitted_trajectory_count == 995' \
    "$ADMISSION/RESULT.json" >/dev/null
if [[ "$(wc -l < "$ADMITTED")" -ne 995 ]]; then
    echo "admitted manifest must contain exactly 995 rows" >&2
    exit 2
fi

mapfile -t trace_paths < <(jq -r '.trace_path' "$ADMITTED" | sort -u)
for trace_relative in "${trace_paths[@]}"; do
    trace="$ROOT/$trace_relative"
    variant_id=$(jq -r --arg trace "$trace_relative" \
        'select(.trace_path == $trace) | .candidate_variant_id' "$ADMITTED" | sort -u)
    if [[ "$variant_id" == *$'\n'* || -z "$variant_id" ]]; then
        echo "trace maps to zero or multiple candidate variants: $trace_relative" >&2
        exit 2
    fi
    mapfile -t source_ids < <(
        jq -r --arg trace "$trace_relative" \
            'select(.trace_path == $trace) | .source_motion_id' "$ADMITTED" | sort -n
    )
    if [[ "${#source_ids[@]}" -eq 0 ]]; then
        echo "admitted trace has no source IDs: $trace_relative" >&2
        exit 2
    fi
    variant_root="$RGB_ROOT/candidate_variant_${variant_id}"
    shard=$(basename "$(dirname "$trace")")
    render_result="$variant_root/RENDER_RESULT_${shard}.json"
    if [[ -f "$render_result" ]]; then
        jq -e --argjson variant "$variant_id" --argjson count "${#source_ids[@]}" '
            .passed == true
            and .bpp_candidate_variant_id == $variant
            and .trajectory_count == $count
            and .frame_count_per_trajectory == 141
            and .camera_keys == ["agentview_rgb", "eye_in_hand_rgb"]
            and .total_rgb_frames == ($count * 141 * 2)
            and .resolution == [320, 320]
        ' "$render_result" >/dev/null
        continue
    fi

    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_following/render_zero_wam_robot_action_corpus.py" \
        --trace "$trace" \
        --output-root "$variant_root" \
        --source-motion-ids "${source_ids[@]}" \
        --bpp-candidate-variant-id "$variant_id" \
        --bpp-camera-contract "$CAMERA_CONTRACT" \
        --frame-stride 5 --camera-width 320 --camera-height 320 \
        --enable_cameras --headless --device cuda:0 --kit_args "$KIT_ARGS"
    jq -e --argjson variant "$variant_id" --argjson count "${#source_ids[@]}" '
        .passed == true
        and .bpp_candidate_variant_id == $variant
        and .trajectory_count == $count
        and .camera_keys == ["agentview_rgb", "eye_in_hand_rgb"]
        and .total_rgb_frames == ($count * 141 * 2)
    ' "$render_result" >/dev/null
done

if [[ ! -f "$SENSOR_AUDIT/RESULT.json" ]]; then
    "$PYTHON_BIN" \
        "$ROOT/scripts/sugar/demo_following/audit_bpp_sensorimotor_corpus.py" \
        --contract "$CONTRACT" \
        --action-admission "$ADMISSION" \
        --rgb-root "$RGB_ROOT" \
        --camera-contract "$CAMERA_CONTRACT" \
        --output-dir "$SENSOR_AUDIT"
fi
jq -e '
    .passed == true
    and .trajectory_count == 995
    and .rgb_frame_count == 280590
    and .train_five_action_interval_count == 112000
    and .train_target_samples_per_epoch == 560000
' "$SENSOR_AUDIT/RESULT.json" >/dev/null

echo "BPP_ADMITTED_RGB_RENDER_COMPLETE root=$RGB_ROOT audit=$SENSOR_AUDIT"
