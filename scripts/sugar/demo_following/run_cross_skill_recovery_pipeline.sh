#!/usr/bin/env bash
# Reproduce the fixed Carry-9 -> Kick recovery overfit and matched evidence.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/cross_skill_recovery_reproduction}"
DEVICE="${2:-cuda:0}"
TRAIN_SEED=171629
EVAL_SEED=181629
CARRY_PREFIX_STEPS="${CARRY_PREFIX_STEPS:-9}"
TRAIN_SEED="${TRAIN_SEED_OVERRIDE:-$TRAIN_SEED}"
EVAL_SEED="${EVAL_SEED_OVERRIDE:-$EVAL_SEED}"
NUM_ENVS="${NUM_ENVS_OVERRIDE:-1024}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run this pipeline inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_cross_skill_recovery_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export DISPLAY=""
export SUGAR_CROSS_SKILL_RECOVERY=1
export SUGAR_CROSS_SKILL_CARRY_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/tracker.pt"
export SUGAR_CROSS_SKILL_KICK_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt"
export SUGAR_CROSS_SKILL_CARRY_GENERATOR_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/generator.ckpt"
export SUGAR_CROSS_SKILL_CARRY_PREFIX_STEPS="$CARRY_PREFIX_STEPS"
export SUGAR_CROSS_SKILL_PREFIX_AUDIT="$OUTPUT_ROOT/train/prefix_audit.json"
export SUGAR_ACTOR_CRITIC_WARM_START_EXPLORATION_STD=0.05
export SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0
export SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY="${SAFETY_PENALTY_OVERRIDE:-1}"

cd "$ROOT/SUGAR"
"$PYTHON_BIN" -u scripts/sugar_rl/train.py \
    --task Sugar-G129dof-KickBox-Carry9-Recovery \
    --num_envs "$NUM_ENVS" --max_iterations 65 --seed "$TRAIN_SEED" \
    --log_dir "$OUTPUT_ROOT/train" \
    --actor_critic_warm_start_checkpoint_path "$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt" \
    --teacher_ckpt "$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt" \
    --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
test -s "$OUTPUT_ROOT/train/model_pre_update.pt"
test -s "$OUTPUT_ROOT/train/model_64.pt"

for arm in baseline trained; do
    checkpoint="$OUTPUT_ROOT/train/model_pre_update.pt"
    if [[ "$arm" == trained ]]; then checkpoint="$OUTPUT_ROOT/train/model_64.pt"; fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "$checkpoint" --output-dir "$OUTPUT_ROOT/evaluation/$arm" \
        --carry-prefix-steps "$CARRY_PREFIX_STEPS" \
        --num-envs 20 --steps 250 --seed "$EVAL_SEED" \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
    test -s "$OUTPUT_ROOT/evaluation/$arm/RESULT.json"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_cross_skill_recovery_pair.py" \
    --baseline "$OUTPUT_ROOT/evaluation/baseline/RESULT.json" \
    --trained "$OUTPUT_ROOT/evaluation/trained/RESULT.json" \
    --output "$OUTPUT_ROOT/evaluation/PAIR_RESULT.json"

for arm in baseline trained; do
    checkpoint="$OUTPUT_ROOT/train/model_pre_update.pt"
    label="Official Kick Tracker before recovery training"
    if [[ "$arm" == trained ]]; then
        checkpoint="$OUTPUT_ROOT/train/model_64.pt"
        label="Kick recovery policy at update 64"
    fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
        --checkpoint "$checkpoint" --output "$OUTPUT_ROOT/videos/${arm}_actual_world.mp4" \
        --label "$label" --steps 250 --seed "$EVAL_SEED" \
        --carry-prefix-steps "$CARRY_PREFIX_STEPS" \
        --headless --device "$DEVICE" --kit_args="--/renderer/multiGpu/enabled=false"
    test -s "$OUTPUT_ROOT/videos/${arm}_actual_world.mp4"
done

"$FFMPEG_BIN" -hide_banner -loglevel error -y \
    -i "$OUTPUT_ROOT/videos/baseline_actual_world.mp4" \
    -i "$OUTPUT_ROOT/videos/trained_actual_world.mp4" \
    -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
    "$OUTPUT_ROOT/videos/matched_baseline_vs_learned_recovery_actual_world.mp4"
