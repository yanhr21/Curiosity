#!/usr/bin/env bash
# Train and freeze-evaluate matched correct-Kick and wrong-Carry SMP reward arms.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/conditional_smp_recovery_prefix41_v1}"
DEVICE="${2:-cuda:0}"
TRAIN_SEED="${TRAIN_SEED_OVERRIDE:-171632}"
EVAL_SEED="${EVAL_SEED_OVERRIDE:-181632}"
NUM_ENVS="${NUM_ENVS_OVERRIDE:-64}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
RESUME_MATCHED_PAIR="${RESUME_MATCHED_PAIR:-0}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" && "$RESUME_MATCHED_PAIR" != "1" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_conditional_smp_recovery_${SLURM_JOB_ID:-local}"
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
export SUGAR_ACTOR_CRITIC_WARM_START_EXPLORATION_STD=0.05
export SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0
export SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1
export SUGAR_CONDITIONAL_TINYMDM_REWARD=1
export SUGAR_CONDITIONAL_TINYMDM_CONFIG="$ROOT/experiments/demo_following/conditional_taskwide_smp_v1/prior/diffusion_config.yaml"
export SUGAR_CONDITIONAL_TINYMDM_CHECKPOINT="$ROOT/experiments/demo_following/conditional_taskwide_smp_v1/prior/model.pt"
export SUGAR_CONDITIONAL_TINYMDM_CALIBRATION="$ROOT/experiments/demo_following/conditional_taskwide_smp_v1/reward_calibration/RESULT.json"
export SUGAR_CONDITIONAL_TINYMDM_REWARD_SEED=190001
export SUGAR_CONDITIONAL_TINYMDM_REWARD_MODE="${REWARD_MODE_OVERRIDE:-occupancy}"
export SUGAR_CONDITIONAL_TINYMDM_TASK_WEIGHT=0.5
export SUGAR_CONDITIONAL_TINYMDM_SMP_WEIGHT=0.5

cd "$ROOT/SUGAR"
for arm_spec in correct_kick:1 wrong_carry:0; do
    arm="${arm_spec%%:*}"
    class_id="${arm_spec##*:}"
    arm_root="$OUTPUT_ROOT/$arm"
    if [[ "$RESUME_MATCHED_PAIR" == "1" \
        && -s "$arm_root/train/model_pre_update.pt" \
        && -s "$arm_root/train/model_64.pt" \
        && -s "$arm_root/train/prefix_audit.json" ]]; then
        echo "Reusing complete matched endpoint: $arm_root/train/model_64.pt"
        continue
    fi
    if [[ -e "$arm_root" ]]; then
        echo "Refusing ambiguous partial arm during resume: $arm_root" >&2
        exit 2
    fi
    export SUGAR_CONDITIONAL_TINYMDM_CLASS_ID="$class_id"
    export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$arm_root/train/prefix_audit.json"
    "$PYTHON_BIN" -u scripts/sugar_rl/train.py \
        --task Sugar-G129dof-KickBox-Carry9-Recovery \
        --num_envs "$NUM_ENVS" --max_iterations 65 --seed "$TRAIN_SEED" \
        --log_dir "$arm_root/train" \
        --actor_critic_warm_start_checkpoint_path "$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt" \
        --teacher_ckpt "$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt" \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
    test -s "$arm_root/train/model_pre_update.pt"
    test -s "$arm_root/train/model_64.pt"
    test -s "$arm_root/train/prefix_audit.json"
done

for arm in correct_kick wrong_carry; do
    evaluation_root="$OUTPUT_ROOT/$arm/evaluation"
    if [[ "$RESUME_MATCHED_PAIR" == "1" \
        && -s "$evaluation_root/RESULT.json" ]]; then
        echo "Reusing complete frozen evaluation: $evaluation_root/RESULT.json"
        continue
    fi
    if [[ -e "$evaluation_root" ]]; then
        echo "Refusing ambiguous partial evaluation during resume: $evaluation_root" >&2
        exit 2
    fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "$OUTPUT_ROOT/$arm/train/model_64.pt" \
        --output-dir "$evaluation_root" \
        --carry-prefix-steps 41 --num-envs 20 --steps 250 --seed "$EVAL_SEED" \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
    test -s "$evaluation_root/RESULT.json"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/smp/summarize_conditional_smp_recovery_pair.py" \
    --correct "$OUTPUT_ROOT/correct_kick/evaluation/RESULT.json" \
    --wrong "$OUTPUT_ROOT/wrong_carry/evaluation/RESULT.json" \
    --correct-audit "$OUTPUT_ROOT/correct_kick/train/prefix_audit.json" \
    --wrong-audit "$OUTPUT_ROOT/wrong_carry/train/prefix_audit.json" \
    --training-seed "$TRAIN_SEED" \
    --output "$OUTPUT_ROOT/PAIR_RESULT.json"
test -s "$OUTPUT_ROOT/PAIR_RESULT.json"
