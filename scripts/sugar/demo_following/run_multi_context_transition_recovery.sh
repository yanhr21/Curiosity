#!/usr/bin/env bash
# Train one shared serious controller over three online physical handoffs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/multi_context_transition_recovery_seed171642_v1}"
DEVICE="${2:-cuda:0}"
TRAIN_SEED="${TRAIN_SEED_OVERRIDE:-171642}"
EVAL_SEED="${EVAL_SEED_OVERRIDE:-181652}"
VIDEO_SEED="${VIDEO_SEED_OVERRIDE:-181653}"
NUM_ENVS="${NUM_ENVS_OVERRIDE:-64}"
POLICY_TOPOLOGY="${POLICY_TOPOLOGY_OVERRIDE:-selected_expert_residual}"
PREFIXES=(41 49 57)
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
case "$POLICY_TOPOLOGY" in
    selected_expert_residual)
        TASK="Sugar-G129dof-KickBox-FrozenExpert-Transition"
        TOPOLOGY_LABEL="Multi-context residual"
        ;;
    causal_action_composition)
        TASK="Sugar-G129dof-KickBox-CausalActionComposition"
        TOPOLOGY_LABEL="Causal action composition"
        ;;
    *) echo "Unknown policy topology: $POLICY_TOPOLOGY" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_multi_context_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""
export SUGAR_CROSS_SKILL_RECOVERY=1
export SUGAR_CROSS_SKILL_CARRY_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/tracker.pt"
export SUGAR_CROSS_SKILL_KICK_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt"
export SUGAR_CROSS_SKILL_CARRY_GENERATOR_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/generator.ckpt"
export SUGAR_CROSS_SKILL_CARRY_PREFIX_STEPS=41
export SUGAR_CROSS_SKILL_CARRY_PREFIX_SCHEDULE=41,49,57
export SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0
export SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1
export SUGAR_TRANSITION_SELECTED_SKILL_ID=-1
export SUGAR_TRANSITION_RECOVERY_REWARD=1
export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$OUTPUT_ROOT/train/prefix_audit.json"
unset SUGAR_CONDITIONAL_TINYMDM_REWARD

cd "$ROOT/SUGAR"
"$PYTHON_BIN" -u scripts/sugar_rl/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" --max_iterations 65 --seed "$TRAIN_SEED" \
    --log_dir "$OUTPUT_ROOT/train" --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
test -s "$OUTPUT_ROOT/train/model_pre_update.pt"
test -s "$OUTPUT_ROOT/train/model_64.pt"
test -s "$OUTPUT_ROOT/train/prefix_audit.json"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_frozen_expert_transition_checkpoints.py" \
    --shared-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json"

for prefix in "${PREFIXES[@]}"; do
    for endpoint in learned pre_update; do
        checkpoint="$OUTPUT_ROOT/train/model_64.pt"
        if [[ "$endpoint" == "pre_update" ]]; then
            checkpoint="$OUTPUT_ROOT/train/model_pre_update.pt"
        fi
        "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
            --checkpoint "$checkpoint" \
            --output-dir "$OUTPUT_ROOT/evaluation/prefix${prefix}/${endpoint}_kick" \
            --transition-selected-skill-id 1 --carry-prefix-steps "$prefix" \
            --policy-topology "$POLICY_TOPOLOGY" \
            --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
            --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
    done
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
    --checkpoint "$OUTPUT_ROOT/train/model_64.pt" \
    --output-dir "$OUTPUT_ROOT/evaluation/prefix41/learned_carry" \
    --transition-selected-skill-id 0 --carry-prefix-steps 41 \
    --policy-topology "$POLICY_TOPOLOGY" \
    --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_shared_frozen_expert_transition.py" \
    --kick "$OUTPUT_ROOT/evaluation/prefix41/learned_kick/RESULT.json" \
    --carry "$OUTPUT_ROOT/evaluation/prefix41/learned_carry/RESULT.json" \
    --training-audit "$OUTPUT_ROOT/train/prefix_audit.json" \
    --checkpoint-audit "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json" \
    --training-seed "$TRAIN_SEED" --expected-recovery-reward 1 \
    --expected-policy-topology "$POLICY_TOPOLOGY" \
    --output "$OUTPUT_ROOT/CONDITION_RESULT.json"

summary_args=()
for prefix in "${PREFIXES[@]}"; do
    summary_args+=(
        --comparison "$prefix"
        "$OUTPUT_ROOT/evaluation/prefix${prefix}/learned_kick/RESULT.json"
        "$OUTPUT_ROOT/evaluation/prefix${prefix}/pre_update_kick/RESULT.json"
    )
done
"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_multi_context_transition_recovery.py" \
    "${summary_args[@]}" \
    --training-audit "$OUTPUT_ROOT/train/prefix_audit.json" \
    --checkpoint-audit "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json" \
    --training-seed "$TRAIN_SEED" --expected-schedule 41,49,57 \
    --expected-policy-topology "$POLICY_TOPOLOGY" \
    --output "$OUTPUT_ROOT/RESULT.json"

for prefix in "${PREFIXES[@]}"; do
    video_dir="$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/prefix${prefix}"
    mkdir -p "$video_dir"
    for endpoint in learned pre_update; do
        checkpoint="$OUTPUT_ROOT/train/model_64.pt"
        label="${TOPOLOGY_LABEL} learned Kick: prefix ${prefix}"
        if [[ "$endpoint" == "pre_update" ]]; then
            checkpoint="$OUTPUT_ROOT/train/model_pre_update.pt"
            label="Exact pre-update Kick: prefix ${prefix}"
        fi
        "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
            --checkpoint "$checkpoint" --output "$video_dir/${endpoint}_kick.mp4" \
            --label "$label" --steps 250 --seed "$VIDEO_SEED" \
            --carry-prefix-steps "$prefix" --transition-selected-skill-id 1 \
            --policy-topology "$POLICY_TOPOLOGY" \
            --headless --device "$DEVICE" \
            --kit_args="--/renderer/multiGpu/enabled=false"
    done
    "$FFMPEG_BIN" -hide_banner -loglevel error -y \
        -i "$video_dir/learned_kick.mp4" -i "$video_dir/pre_update_kick.mp4" \
        -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p \
        -movflags +faststart "$video_dir/learned_vs_pre_update_prefix${prefix}.mp4"
done
