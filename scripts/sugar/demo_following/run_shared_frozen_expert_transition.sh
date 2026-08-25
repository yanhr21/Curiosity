#!/usr/bin/env bash
# Train one balanced-condition transition policy, then swap only its condition.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/shared_frozen_expert_transition_prefix41_seed171638_v1}"
DEVICE="${2:-cuda:0}"
TRAIN_SEED="${TRAIN_SEED_OVERRIDE:-171638}"
EVAL_SEED="${EVAL_SEED_OVERRIDE:-181644}"
VIDEO_SEED="${VIDEO_SEED_OVERRIDE:-181645}"
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
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_shared_transition_${SLURM_JOB_ID:-local}"
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
export SUGAR_TRANSITION_SELECTED_SKILL_ID=-1
export SUGAR_TRANSITION_RECOVERY_REWARD="${SUGAR_TRANSITION_RECOVERY_REWARD_OVERRIDE:-0}"
export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$OUTPUT_ROOT/train/prefix_audit.json"
unset SUGAR_CONDITIONAL_TINYMDM_REWARD

cd "$ROOT/SUGAR"
"$PYTHON_BIN" -u scripts/sugar_rl/train.py \
    --task Sugar-G129dof-KickBox-FrozenExpert-Transition \
    --num_envs "$NUM_ENVS" --max_iterations 65 --seed "$TRAIN_SEED" \
    --log_dir "$OUTPUT_ROOT/train" --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
test -s "$OUTPUT_ROOT/train/model_pre_update.pt"
test -s "$OUTPUT_ROOT/train/model_64.pt"
test -s "$OUTPUT_ROOT/train/prefix_audit.json"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_frozen_expert_transition_checkpoints.py" \
    --shared-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json"

for condition_spec in kick:1 carry:0; do
    condition="${condition_spec%%:*}"
    skill_id="${condition_spec##*:}"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "$OUTPUT_ROOT/train/model_64.pt" \
        --output-dir "$OUTPUT_ROOT/evaluation/$condition" \
        --transition-selected-skill-id "$skill_id" --carry-prefix-steps 41 \
        --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
    --checkpoint "$OUTPUT_ROOT/train/model_pre_update.pt" \
    --output-dir "$OUTPUT_ROOT/evaluation/kick_pre_update" \
    --transition-selected-skill-id 1 --carry-prefix-steps 41 \
    --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
test -s "$OUTPUT_ROOT/evaluation/kick_pre_update/RESULT.json"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_shared_frozen_expert_transition.py" \
    --kick "$OUTPUT_ROOT/evaluation/kick/RESULT.json" \
    --carry "$OUTPUT_ROOT/evaluation/carry/RESULT.json" \
    --training-audit "$OUTPUT_ROOT/train/prefix_audit.json" \
    --checkpoint-audit "$OUTPUT_ROOT/CHECKPOINT_AUDIT.json" \
    --training-seed "$TRAIN_SEED" \
    --expected-recovery-reward "$SUGAR_TRANSITION_RECOVERY_REWARD" \
    --output "$OUTPUT_ROOT/RESULT.json"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_shared_transition_learning.py" \
    --learned "$OUTPUT_ROOT/evaluation/kick/RESULT.json" \
    --pre-update "$OUTPUT_ROOT/evaluation/kick_pre_update/RESULT.json" \
    --training-seed "$TRAIN_SEED" --output "$OUTPUT_ROOT/LEARNING_RESULT.json"

mkdir -p "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}"
for condition_spec in kick:1 carry:0; do
    condition="${condition_spec%%:*}"
    skill_id="${condition_spec##*:}"
    label="Same checkpoint: Kick condition"
    if [[ "$skill_id" == "0" ]]; then label="Same checkpoint: Carry condition"; fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
        --checkpoint "$OUTPUT_ROOT/train/model_64.pt" \
        --output "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/${condition}_actual_world.mp4" \
        --label "$label" --steps 250 --seed "$VIDEO_SEED" --carry-prefix-steps 41 \
        --transition-selected-skill-id "$skill_id" --headless --device "$DEVICE" \
        --kit_args="--/renderer/multiGpu/enabled=false"
done

"$FFMPEG_BIN" -hide_banner -loglevel error -y \
    -i "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/kick_actual_world.mp4" \
    -i "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/carry_actual_world.mp4" \
    -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT_ROOT/videos_seed${VIDEO_SEED}/same_checkpoint_kick_vs_carry_seed${VIDEO_SEED}.mp4"
