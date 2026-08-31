#!/usr/bin/env bash
# Freeze the official-SUGAR action/RGB ICL dataset without training a local WAM.

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
BASE="${1:-$ROOT/experiments/demo_following/zero_wam_official_v1}"
ACTION_ROOT="$BASE/action_corpus_v1"
ACTION_AUDIT="$BASE/action_grounding_audit_v2"
ROBOT_RGB="$BASE/robot_rgb_isolated_10hz_v2"
ICL_MANIFEST="$BASE/icl_manifest_v2"
DATA_DIVERSITY="$BASE/training_data_diversity_v1"
PROMPT_GATE_CASES="$BASE/frozen_prompt_gate_cases_v1"
TRAINING_SCHEDULE="$BASE/bounded_posttraining_schedule_v1"
MOTION_DISJOINT_CASES="$BASE/motion_disjoint_closed_loop_cases_v2"
PROMPT_ROOT="$BASE/prompt_rgb_isolated_v1"
KIT_ARGS="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained GPU Slurm allocation" >&2
    exit 2
fi
if [[ ! -d "$ACTION_ROOT" ]]; then
    bash "$ROOT/scripts/sugar/demo_reward/collect_deployable_goal_core_corpus.sh" \
        "$ACTION_ROOT"
fi

render_prompt_shard() {
    local task=$1
    local first=$2
    local last=$3
    local result="$PROMPT_ROOT/RENDER_RESULT_${task}_$(printf '%03d' "$first")_$(printf '%03d' "$last").json"
    if [[ -f "$result" ]]; then
        jq -e '.passed == true and .neighbor_center_is_beyond_far_clip == true' \
            "$result" >/dev/null
        return
    fi
    mapfile -t motion_ids < <(seq "$first" "$last")
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_following/render_xirl_reference_corpus.py" \
        --task "$task" --motion-ids "${motion_ids[@]}" \
        --output-root "$PROMPT_ROOT" --frames-per-motion 64 \
        --camera-width 320 --camera-height 320 --env-spacing 30 \
        --enable_cameras --headless --device cuda:0 --kit_args "$KIT_ARGS"
    jq -e '.passed == true and .neighbor_center_is_beyond_far_clip == true' \
        "$result" >/dev/null
}

render_prompt_shard CarryBox 0 24
render_prompt_shard CarryBox 25 49
render_prompt_shard CarryBox 50 74
render_prompt_shard CarryBox 75 99
render_prompt_shard KickBox 0 24
render_prompt_shard KickBox 25 49
render_prompt_shard KickBox 50 74
render_prompt_shard KickBox 75 98

if [[ ! -f "$ACTION_AUDIT/RESULT.json" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_zero_wam_sugar_action_corpus.py" \
        --action-corpus "$ACTION_ROOT" \
        --prompt-corpus "$PROMPT_ROOT" \
        --output-dir "$ACTION_AUDIT"
fi
jq -e '.passed == true' "$ACTION_AUDIT/RESULT.json" >/dev/null

while IFS= read -r trace; do
    shard=$(basename "$(dirname "$trace")")
    render_result="$ROBOT_RGB/RENDER_RESULT_${shard}.json"
    if [[ -f "$render_result" ]]; then
        jq -e '.passed == true' "$render_result" >/dev/null
        continue
    fi
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_following/render_zero_wam_robot_action_corpus.py" \
        --trace "$trace" \
        --output-root "$ROBOT_RGB" \
        --frame-stride 5 --camera-width 320 --camera-height 320 \
        --enable_cameras --headless --device cuda:0 --kit_args "$KIT_ARGS"
    jq -e '.passed == true' "$render_result" >/dev/null
done < <(rg --files -uu "$ACTION_ROOT" | rg '/TRACE\.npz$' | sort)

if [[ ! -f "$ICL_MANIFEST/RESULT.json" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/build_zero_wam_sugar_icl_manifest.py" \
        --action-audit-dir "$ACTION_AUDIT" \
        --robot-rgb-root "$ROBOT_RGB" \
        --output-dir "$ICL_MANIFEST"
fi
jq -e '.passed == true' "$ICL_MANIFEST/RESULT.json" >/dev/null

if [[ ! -f "$DATA_DIVERSITY/SUGAR_TRAINING_DATA_DIVERSITY.json" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_zero_wam_sugar_data_diversity.py" \
        --manifest "$ICL_MANIFEST/ICL_MANIFEST.jsonl" \
        --project-root "$ROOT" \
        --output-dir "$DATA_DIVERSITY"
fi
jq -e '.passed == true' "$DATA_DIVERSITY/SUGAR_TRAINING_DATA_DIVERSITY.json" >/dev/null

if [[ ! -f "$PROMPT_GATE_CASES/FROZEN_PROMPT_GATE_CASES_RESULT.json" ]]; then
    "$PYTHON_BIN" \
        "$ROOT/scripts/sugar/demo_following/build_zero_wam_frozen_prompt_gate_cases.py" \
        --source-manifest "$ICL_MANIFEST/ICL_MANIFEST.jsonl" \
        --output-dir "$PROMPT_GATE_CASES"
fi
jq -e '.passed == true' "$PROMPT_GATE_CASES/FROZEN_PROMPT_GATE_CASES_RESULT.json" >/dev/null

if [[ ! -f "$TRAINING_SCHEDULE/BOUNDED_POSTTRAINING_SCHEDULE_RESULT.json" ]]; then
    "$PYTHON_BIN" \
        "$ROOT/scripts/sugar/demo_following/build_zero_wam_bounded_posttraining_schedule.py" \
        --source-manifest "$ICL_MANIFEST/ICL_MANIFEST.jsonl" \
        --output-dir "$TRAINING_SCHEDULE"
fi
jq -e '.passed == true' "$TRAINING_SCHEDULE/BOUNDED_POSTTRAINING_SCHEDULE_RESULT.json" >/dev/null

if [[ ! -f "$MOTION_DISJOINT_CASES/MOTION_DISJOINT_CLOSED_LOOP_CASES_RESULT.json" ]]; then
    "$PYTHON_BIN" \
        "$ROOT/scripts/sugar/demo_following/build_zero_wam_motion_disjoint_closed_loop_cases.py" \
        --source-manifest "$ICL_MANIFEST/ICL_MANIFEST.jsonl" \
        --output-dir "$MOTION_DISJOINT_CASES" \
        --self-test
fi
jq -e '.passed == true' \
    "$MOTION_DISJOINT_CASES/MOTION_DISJOINT_CLOSED_LOOP_CASES_RESULT.json" >/dev/null
