#!/usr/bin/env bash
# Render one exact frozen-evaluation profile for both conditional reward arms.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENT_ROOT="${1:-$ROOT/experiments/demo_following/conditional_smp_recovery_prefix41_v1}"
EVAL_SEED="${2:-181633}"
DEVICE="${3:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
EXPERIMENT_ROOT="$(realpath -m "$EXPERIMENT_ROOT")"
VIDEO_ROOT="$EXPERIMENT_ROOT/videos_single_seed${EVAL_SEED}"
REWARD_MODE="$(jq -r '.experiment.reward_mode // "occupancy"' "$EXPERIMENT_ROOT/PAIR_RESULT.json")"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$VIDEO_ROOT" ]]; then
    echo "Refusing to overwrite $VIDEO_ROOT" >&2
    exit 2
fi
mkdir -p "$VIDEO_ROOT"

for arm in correct_kick wrong_carry; do
    if [[ "$REWARD_MODE" == "contrastive_progress" ]]; then
        label="Correct: Kick contrastive progress"
        if [[ "$arm" == wrong_carry ]]; then
            label="Wrong: Carry contrastive progress"
        fi
    else
        label="Correct ${REWARD_MODE} reward: Kick condition"
        if [[ "$arm" == wrong_carry ]]; then
            label="Wrong ${REWARD_MODE} reward: Carry condition"
        fi
    fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
        --checkpoint "$EXPERIMENT_ROOT/$arm/train/model_64.pt" \
        --output "$VIDEO_ROOT/${arm}_seed${EVAL_SEED}_actual_world.mp4" \
        --label "$label" --steps 250 --seed "$EVAL_SEED" --carry-prefix-steps 41 \
        --profile-index 0 \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/multiGpu/enabled=false"
    test -s "$VIDEO_ROOT/${arm}_seed${EVAL_SEED}_actual_world.mp4"
done

"$FFMPEG_BIN" -hide_banner -loglevel error -y \
    -i "$VIDEO_ROOT/correct_kick_seed${EVAL_SEED}_actual_world.mp4" \
    -i "$VIDEO_ROOT/wrong_carry_seed${EVAL_SEED}_actual_world.mp4" \
    -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 -pix_fmt yuv420p \
    -movflags +faststart \
    "$VIDEO_ROOT/correct_kick_vs_wrong_carry_seed${EVAL_SEED}.mp4"
test -s "$VIDEO_ROOT/correct_kick_vs_wrong_carry_seed${EVAL_SEED}.mp4"
