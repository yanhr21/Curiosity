#!/usr/bin/env bash
# Train, freeze-evaluate and render one matched selected-endpoint transition pair.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/frozen_expert_transition_prefix41_seed171637_v1}"
DEVICE="${2:-cuda:0}"
TRAIN_SEED="${TRAIN_SEED_OVERRIDE:-171637}"
EVAL_SEED="${EVAL_SEED_OVERRIDE:-181642}"
VIDEO_SEED="${VIDEO_SEED_OVERRIDE:-181643}"
NUM_ENVS="${NUM_ENVS_OVERRIDE:-64}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_frozen_transition_${SLURM_JOB_ID:-local}"
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
unset SUGAR_CONDITIONAL_TINYMDM_REWARD

cd "$ROOT/SUGAR"
for arm_spec in correct_kick:1 wrong_carry:0; do
    arm="${arm_spec%%:*}"
    skill_id="${arm_spec##*:}"
    arm_root="$OUTPUT_ROOT/$arm"
    export SUGAR_TRANSITION_SELECTED_SKILL_ID="$skill_id"
    export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$arm_root/train/prefix_audit.json"
    "$PYTHON_BIN" -u scripts/sugar_rl/train.py \
        --task Sugar-G129dof-KickBox-FrozenExpert-Transition \
        --num_envs "$NUM_ENVS" --max_iterations 65 --seed "$TRAIN_SEED" \
        --log_dir "$arm_root/train" --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
    test -s "$arm_root/train/model_pre_update.pt"
    test -s "$arm_root/train/model_64.pt"
    test -s "$arm_root/train/prefix_audit.json"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_frozen_expert_transition_checkpoints.py" \
    --pair-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json"

for arm_spec in correct_kick:1 wrong_carry:0; do
    arm="${arm_spec%%:*}"
    skill_id="${arm_spec##*:}"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "$OUTPUT_ROOT/$arm/train/model_64.pt" \
        --output-dir "$OUTPUT_ROOT/$arm/evaluation" \
        --transition-selected-skill-id "$skill_id" --carry-prefix-steps 41 \
        --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_frozen_expert_transition_pair.py" \
    --correct "$OUTPUT_ROOT/correct_kick/evaluation/RESULT.json" \
    --wrong "$OUTPUT_ROOT/wrong_carry/evaluation/RESULT.json" \
    --correct-audit "$OUTPUT_ROOT/correct_kick/train/prefix_audit.json" \
    --wrong-audit "$OUTPUT_ROOT/wrong_carry/train/prefix_audit.json" \
    --training-seed "$TRAIN_SEED" --output "$OUTPUT_ROOT/PAIR_RESULT.json"

mkdir -p "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}"
for arm_spec in correct_kick:1 wrong_carry:0; do
    arm="${arm_spec%%:*}"
    skill_id="${arm_spec##*:}"
    label="Correct: Kick endpoint"
    if [[ "$skill_id" == "0" ]]; then
        label="Wrong: Carry endpoint"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
        --checkpoint "$OUTPUT_ROOT/$arm/train/model_64.pt" \
        --output "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/${arm}_actual_world.mp4" \
        --label "$label" --steps 250 --seed "$VIDEO_SEED" --carry-prefix-steps 41 \
        --transition-selected-skill-id "$skill_id" --headless --device "$DEVICE" \
        --kit_args="--/renderer/multiGpu/enabled=false"
done

"$FFMPEG_BIN" -hide_banner -loglevel error -y \
    -i "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/correct_kick_actual_world.mp4" \
    -i "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/wrong_carry_actual_world.mp4" \
    -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/correct_kick_vs_wrong_carry_seed${VIDEO_SEED}.mp4"
