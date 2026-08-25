#!/usr/bin/env bash
# Fixed-context learnability diagnostic for the failure-rich prefix41 seed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/frozen_expert_transition_failure_overfit_seed181630_v1}"
DEVICE="${2:-cuda:0}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_transition_overfit_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""
export SUGAR_CROSS_SKILL_RECOVERY=1
export SUGAR_CROSS_SKILL_CARRY_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/tracker.pt"
export SUGAR_CROSS_SKILL_KICK_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt"
export SUGAR_CROSS_SKILL_CARRY_GENERATOR_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/generator.ckpt"
export SUGAR_CROSS_SKILL_CARRY_PREFIX_STEPS=41
export SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0
export SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1
export SUGAR_TRANSITION_SELECTED_SKILL_ID=1
export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$OUTPUT_ROOT/train/prefix_audit.json"
unset SUGAR_CONDITIONAL_TINYMDM_REWARD

cd "$ROOT/SUGAR"
"$PYTHON_BIN" -u scripts/sugar_rl/train.py \
    --task Sugar-G129dof-KickBox-FrozenExpert-Transition \
    --num_envs 20 --max_iterations 257 --seed 181630 \
    --log_dir "$OUTPUT_ROOT/train" --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"

for iteration in 64 128 192 256; do
    test -s "$OUTPUT_ROOT/train/model_${iteration}.pt"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_frozen_expert_transition_checkpoints.py" \
    --shared-root "$OUTPUT_ROOT" --post-iteration 256 \
    --output "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json"

for spec in pre:model_pre_update:-1 64:model_64:64 128:model_128:128 192:model_192:192 256:model_256:256; do
    label="${spec%%:*}"
    rest="${spec#*:}"
    checkpoint="${rest%%:*}"
    iteration="${rest##*:}"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "$OUTPUT_ROOT/train/${checkpoint}.pt" \
        --output-dir "$OUTPUT_ROOT/evaluation/${label}" \
        --transition-selected-skill-id 1 --carry-prefix-steps 41 \
        --num-envs 20 --steps 250 --seed 181630 --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
    test -s "$OUTPUT_ROOT/evaluation/${label}/RESULT.json"
    if [[ "$iteration" != "-1" ]]; then
        "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_shared_transition_learning.py" \
            --learned "$OUTPUT_ROOT/evaluation/${label}/RESULT.json" \
            --pre-update "$OUTPUT_ROOT/evaluation/pre/RESULT.json" \
            --training-seed 181630 --learned-iteration "$iteration" \
            --output "$OUTPUT_ROOT/evaluation/${label}/LEARNING_RESULT.json"
    fi
done

summary_args=()
for iteration in 64 128 192 256; do
    summary_args+=(--learning-result "$OUTPUT_ROOT/evaluation/${iteration}/LEARNING_RESULT.json")
done
"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_transition_failure_overfit.py" \
    "${summary_args[@]}" --checkpoint-audit "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json" \
    --output "$OUTPUT_ROOT/RESULT.json"
