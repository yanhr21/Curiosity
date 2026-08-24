#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
UPDATE=${1:-}
DESIGN=${2:-same_teacher_reward_only}
RUN_ROOT_OVERRIDE=${3:-}
if [[ ! "$UPDATE" =~ ^[0-9]+$ ]] || (( UPDATE < 64 || UPDATE % 64 != 0 )); then
    echo "usage: $0 UPDATE_MULTIPLE_OF_64 [design] [run_root]" >&2
    exit 2
fi
case "$DESIGN" in
    same_teacher_reward_only|teacher_floor_overfit|phase_event_reward_only) ;;
    *) echo "unknown matched demo design: $DESIGN" >&2; exit 2 ;;
esac
PADDED_UPDATE=$(printf '%04d' "$UPDATE")
EVALUATOR="$ROOT/scripts/sugar/demo_following/evaluate_matched_fixed_teacher.py"
RENDERER="$ROOT/scripts/sugar/demo_following/render_demo_and_actual.py"
ANALYZER="$ROOT/scripts/sugar/demo_following/analyze_behavior_adherence.py"
if [[ -n "$RUN_ROOT_OVERRIDE" ]]; then
    RUN_ROOT=$(realpath -m "$RUN_ROOT_OVERRIDE")
else
    if [[ "$DESIGN" == "phase_event_reward_only" ]]; then
        RUN_ROOT="$ROOT/experiments/demo_following/matched_phase_event_reward_reference_aware_v2/seed161587"
    else
        RUN_ROOT="$ROOT/experiments/demo_following/matched_reward_identity_same_teacher_v1/seed161581"
    fi
fi
CONFIG="$RUN_ROOT/correct/update_${PADDED_UPDATE}/protocol.json"
EVAL_ROOT="$RUN_ROOT/evaluation_update${PADDED_UPDATE}"
VIDEO_ROOT="$RUN_ROOT/videos_update${PADDED_UPDATE}"
arms=(same_teacher_correct_reward same_teacher_unrelated_reward)
training_seed=$(
    "$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["shared_runtime"]["sim_and_policy_seed"])' \
        "$CONFIG"
)
eval_seed=$((training_seed + 10000))
renderer_design_args=(--same-teacher-reward-only)
evaluation_updates="$UPDATE"
renderer_source_env=0
if [[ "$DESIGN" == "phase_event_reward_only" ]]; then
    if (( UPDATE != 64 )); then
        echo "phase-event first evaluation is fixed to update 32 and 64" >&2
        exit 2
    fi
    evaluation_updates="32,64"
    renderer_source_env=20
fi
EVAL_KIT_ARGS="--/renderer/enabled= --/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"
RENDER_KIT_ARGS="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "evaluation requires a retained srun compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing evaluation/rendering on a login host" >&2; exit 2 ;;
esac
reuse_video=0
if [[ -e "$VIDEO_ROOT" ]]; then
    if [[ ! -f "$VIDEO_ROOT/RENDER_PROOF.json" ]]; then
        echo "incomplete video directory requires inspection: $VIDEO_ROOT" >&2
        exit 2
    fi
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$VIDEO_ROOT/RENDER_PROOF.json"
    reuse_video=1
fi

export PYTHONPATH="$ROOT/scripts/sugar/smp:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
# Cluster H200 rendering must use the installed NVIDIA Vulkan ICD.  Leaving
# discovery implicit can select an incompatible loader and fail during scene
# initialization with VK_ERROR_DEVICE_LOST.
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=
unset CURIOSITY_TACSL_R15_USD
unset CURIOSITY_TACSL_LEFT_MOUNT_TRANSLATION_OFFSET
unset CURIOSITY_TACSL_RIGHT_MOUNT_TRANSLATION_OFFSET

if [[ "${TEACHER_ONLY_GATE:-0}" == "1" ]]; then
    gate_root=${TEACHER_GATE_OUTPUT:-"$RUN_ROOT/teacher_only_gate_no_tactile_v2"}
    if [[ -f "$gate_root/RESULT.json" && -f "$gate_root/TRACE.npz" ]]; then
        "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values()) and p["checks"]["demo_control_has_no_tactile_scene"] and p["no_tactile_scene"]["passed"]' "$gate_root/RESULT.json"
        exit 0
    fi
    if [[ -e "$gate_root" ]]; then
        echo "incomplete teacher-only gate requires inspection: $gate_root" >&2
        exit 2
    fi
    (
        cd "$ROOT/SUGAR"
        ISAACLAB_TMP_ROOT="/tmp/Curiosity_teacher_gate_${SLURM_JOB_ID}" \
        SUGAR_UNITREE_TMP_ROOT="/tmp/Curiosity_teacher_gate_unitree_${SLURM_JOB_ID}" \
        VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
        "$PYTHON" "$EVALUATOR" \
            --config "$CONFIG" \
            --arm same_teacher_correct_reward \
            --output-dir "$gate_root" \
            --updates "$UPDATE" \
            --steps 400 \
            --seed 171581 \
            --teacher-only-zero-residual \
            --fast-exit-after-evidence \
            --headless \
            --device cuda:0 \
            --kit_args "$EVAL_KIT_ARGS"
    ) > "${gate_root}.log" 2>&1
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values()) and p["checks"]["demo_control_has_no_tactile_scene"] and p["no_tactile_scene"]["passed"]' "$gate_root/RESULT.json"
    exit 0
fi

if [[ "$DESIGN" == "phase_event_reward_only" ]]; then
    "$PYTHON" - \
        "$RUN_ROOT/correct/update_${PADDED_UPDATE}/proof.json" \
        "$RUN_ROOT/unrelated/update_${PADDED_UPDATE}/proof.json" \
        "$RUN_ROOT/correct/update_${PADDED_UPDATE}/protocol.json" \
        "$RUN_ROOT/unrelated/update_${PADDED_UPDATE}/protocol.json" <<'PY'
import json
import sys

proofs = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3]]
protocols = [json.load(open(path, encoding="utf-8")) for path in sys.argv[3:5]]
for proof, selected_option in zip(proofs, ("correct", "unrelated"), strict=True):
    checks = proof.get("checks", {})
    audit = proof.get("demo_event_reward", {}).get("final_frozen_audit", {})
    assert proof.get("passed") is True and checks and all(checks.values())
    assert checks.get("demo_event_phase_and_prefix_are_causal") is True
    assert proof.get("protocol") == "sugar_phase_event_reward_matched_policy_v1"
    assert proof.get("seed") == 161587 and proof.get("action_seed") == 161588
    assert proof.get("num_envs") == 20 and proof.get("num_updates") == 64
    assert proof.get("demo_event_reward", {}).get("selected_option") == selected_option
    assert audit.get("phase_source") == "reset_reference_frame_plus_causal_control_clock"
    assert audit.get("initial_episode_steps_supplied") is True
    assert audit.get("initial_episode_steps_min") == 197
    assert audit.get("initial_episode_steps_max") == 197
records = [proof.get("no_tactile_startup_physics") for proof in proofs]
assert protocols[0] == protocols[1], "correct/unrelated protocols differ"
assert all(isinstance(record, dict) and record.get("passed") for record in records)
assert records[0]["values"] == records[1]["values"], (
    "correct/unrelated startup physics differ"
)
PY
fi

mkdir -p "$EVAL_ROOT"
for arm in "${arms[@]}"; do
    if [[ "$arm" == "wrong_teacher_correct_reward" || "$arm" == "same_teacher_correct_reward" ]]; then
        short=correct
    else
        short=unrelated
    fi
    if [[ -f "$EVAL_ROOT/$short/RESULT.json" && -f "$EVAL_ROOT/$short/TRACE.npz" ]]; then
        "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$EVAL_ROOT/$short/RESULT.json"
        continue
    fi
    if [[ -e "$EVAL_ROOT/$short" ]]; then
        echo "incomplete evaluation requires inspection: $EVAL_ROOT/$short" >&2
        exit 2
    fi
    (
        cd "$ROOT/SUGAR"
        ISAACLAB_TMP_ROOT="/tmp/Curiosity_matched_eval${UPDATE}_${short}_${SLURM_JOB_ID}" \
        SUGAR_UNITREE_TMP_ROOT="/tmp/Curiosity_matched_eval${UPDATE}_unitree_${short}_${SLURM_JOB_ID}" \
        VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
        "$PYTHON" "$EVALUATOR" \
            --config "$CONFIG" \
            --arm "$arm" \
            --output-dir "$EVAL_ROOT/$short" \
            --updates "$evaluation_updates" \
            --steps 400 \
            --seed "$eval_seed" \
            --fast-exit-after-evidence \
            --headless \
            --device cuda:0 \
            --kit_args "$EVAL_KIT_ARGS"
    ) > "$EVAL_ROOT/${short}_console.log" 2>&1
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$EVAL_ROOT/$short/RESULT.json"
done

if [[ "$DESIGN" == "phase_event_reward_only" ]]; then
    for policy_update in 32 64; do
        audit_dir="$RUN_ROOT/behavior_adherence_update$(printf '%04d' "$policy_update")"
        if [[ -e "$audit_dir" && ! -f "$audit_dir/RESULT.json" ]]; then
            echo "incomplete behavior audit requires inspection: $audit_dir" >&2
            exit 2
        fi
        if [[ ! -f "$audit_dir/RESULT.json" ]]; then
            "$PYTHON" "$ANALYZER" \
                --correct-trace "$EVAL_ROOT/correct/TRACE.npz" \
                --unrelated-trace "$EVAL_ROOT/unrelated/TRACE.npz" \
                --policy-update "$policy_update" \
                --output-dir "$audit_dir"
        fi
    done
fi

if [[ "$reuse_video" == "1" ]]; then
    exit 0
fi

(
    cd "$ROOT/SUGAR"
    ISAACLAB_TMP_ROOT="/tmp/Curiosity_matched_render${UPDATE}_${SLURM_JOB_ID}" \
    SUGAR_UNITREE_TMP_ROOT="/tmp/Curiosity_matched_render${UPDATE}_unitree_${SLURM_JOB_ID}" \
    "$PYTHON" "$RENDERER" \
        --correct-trace "$EVAL_ROOT/correct/TRACE.npz" \
        --unrelated-trace "$EVAL_ROOT/unrelated/TRACE.npz" \
        --output-dir "$VIDEO_ROOT" \
        --matched-endpoint \
        --actual-source-env "$renderer_source_env" \
        "${renderer_design_args[@]}" \
        --headless \
        --enable_cameras \
        --device cuda:0 \
        --kit_args "$RENDER_KIT_ARGS"
) > "$EVAL_ROOT/render_console.log" 2>&1

"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' "$VIDEO_ROOT/RENDER_PROOF.json"
