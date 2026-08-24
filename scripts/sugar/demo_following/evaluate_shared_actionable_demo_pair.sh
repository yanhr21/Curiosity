#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
RUN_ROOT=${RUN_ROOT:-$ROOT/experiments/demo_following/shared_actionable_demo_conditioning_v1/seed161591}
CONFIG=$RUN_ROOT/update_0064/protocol.json
EVALUATOR=$ROOT/scripts/sugar/demo_following/evaluate_matched_fixed_teacher.py
ANALYZER=$ROOT/scripts/sugar/demo_following/analyze_behavior_adherence.py
TRACE_RENDERER=$ROOT/scripts/sugar/demo_following/render_frozen_trace_behavior.py
EVAL_ROOT=$RUN_ROOT/evaluation_same_checkpoint_update0064
AUDIT_ROOT=$RUN_ROOT/behavior_adherence_same_checkpoint_update0064
VIDEO_ROOT=$RUN_ROOT/videos_same_checkpoint_update0064
KIT_ARGS="--/renderer/enabled= --/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "evaluation requires a retained compute allocation" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing evaluation on a login node" >&2; exit 2 ;;
esac

export PYTHONPATH="$ROOT/scripts/sugar/smp:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export DISPLAY=
unset CURIOSITY_TACSL_R15_USD
unset CURIOSITY_TACSL_LEFT_MOUNT_TRANSLATION_OFFSET
unset CURIOSITY_TACSL_RIGHT_MOUNT_TRANSLATION_OFFSET

for option in correct unrelated; do
    output=$EVAL_ROOT/$option
    if [[ -f "$output/RESULT.json" && -f "$output/TRACE.npz" ]]; then
        "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$output/RESULT.json"
        continue
    fi
    if [[ -e "$output" ]]; then
        echo "incomplete shared evaluation requires inspection: $output" >&2
        exit 2
    fi
    mkdir -p "$EVAL_ROOT"
    (
        cd "$ROOT/SUGAR"
        ISAACLAB_TMP_ROOT="/tmp/Curiosity_shared_eval_${option}_${SLURM_JOB_ID}" \
        SUGAR_UNITREE_TMP_ROOT="/tmp/Curiosity_shared_eval_unitree_${option}_${SLURM_JOB_ID}" \
        VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
        "$PYTHON" "$EVALUATOR" \
            --config "$CONFIG" \
            --arm shared_balanced_conditioning \
            --selected-demo-option "$option" \
            --output-dir "$output" \
            --updates 64 \
            --steps 400 \
            --seed 171591 \
            --fast-exit-after-evidence \
            --headless \
            --device cuda:0 \
            --kit_args "$KIT_ARGS"
    ) > "$EVAL_ROOT/${option}_console.log" 2>&1
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$output/RESULT.json"
done

if [[ -e "$AUDIT_ROOT" && ! -f "$AUDIT_ROOT/RESULT.json" ]]; then
    echo "incomplete shared behavior audit requires inspection: $AUDIT_ROOT" >&2
    exit 2
fi
if [[ ! -f "$AUDIT_ROOT/RESULT.json" ]]; then
    "$PYTHON" "$ANALYZER" \
        --correct-trace "$EVAL_ROOT/correct/TRACE.npz" \
        --unrelated-trace "$EVAL_ROOT/unrelated/TRACE.npz" \
        --policy-update 64 \
        --same-checkpoint-condition-swap \
        --output-dir "$AUDIT_ROOT"
fi

"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["matched_checks"]; assert p["protocol"] == "same_checkpoint_condition_swap_behavior_audit_v1"; assert p["status"] == "complete_existing_trace_audit"; assert p["semantic_directions_total"] == 4; assert 0 <= p["semantic_directions_observed"] <= 4; assert m["frame_budget_equal"] and m["profile_count_equal"] and m["initial_state_exact_match"]' "$AUDIT_ROOT/RESULT.json"
if [[ ! -f "$VIDEO_ROOT/RENDER_PROOF.json" ]]; then
    if [[ -e "$VIDEO_ROOT" ]]; then
        echo "incomplete shared video output requires inspection: $VIDEO_ROOT" >&2
        exit 2
    fi
    "$PYTHON" "$TRACE_RENDERER" \
        --correct-trace "$EVAL_ROOT/correct/TRACE.npz" \
        --unrelated-trace "$EVAL_ROOT/unrelated/TRACE.npz" \
        --output-dir "$VIDEO_ROOT" \
        --source-env 0 \
        --policy-update 64 \
        --shared-checkpoint
fi
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$VIDEO_ROOT/RENDER_PROOF.json"
printf 'shared_same_checkpoint_evaluation_complete=%s\n' "$RUN_ROOT"
