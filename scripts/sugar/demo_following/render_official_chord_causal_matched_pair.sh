#!/usr/bin/env bash
# Render only the final learned CHORD-OFF/ON policies and place them side by side.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PAIR_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_chord_causal_matched_seed171648_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
VIDEO_SEED=181667

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
test -s "$PAIR_ROOT/RESULT.json" || { echo "matched CHORD result is missing: $PAIR_ROOT" >&2; exit 2; }
jq -e '.structurally_valid == true' "$PAIR_ROOT/RESULT.json" >/dev/null

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_chord_pair_render_${SLURM_JOB_ID:-local}"
SUGAR_SOURCE_PREFIX=""
if [[ -n "${SUGAR_LOCAL_SOURCE_STAGING:-}" ]]; then
    test -d "$SUGAR_LOCAL_SOURCE_STAGING/sugar_rl/sugar_rl"
    test -d "$SUGAR_LOCAL_SOURCE_STAGING/sugar_il/sugar_il"
    SUGAR_SOURCE_PREFIX="$SUGAR_LOCAL_SOURCE_STAGING/sugar_rl:$SUGAR_LOCAL_SOURCE_STAGING/sugar_il:"
fi
export PYTHONPATH="${SUGAR_SOURCE_PREFIX}$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""
export SUGAR_CROSS_SKILL_RECOVERY=1
export SUGAR_CROSS_SKILL_CARRY_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/tracker.pt"
export SUGAR_CROSS_SKILL_KICK_TRACKER_CKPT="$ROOT/SUGAR/demo_ckpts/KickBox/tracker.pt"
export SUGAR_CROSS_SKILL_CARRY_GENERATOR_CKPT="$ROOT/SUGAR/demo_ckpts/CarryBox/generator.ckpt"
export SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0
export SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1
export SUGAR_TRANSITION_SELECTED_SKILL_ID=1
export SUGAR_TRANSITION_RECOVERY_REWARD=1
unset SUGAR_OFFICIAL_CHORD_REWARD SUGAR_OFFICIAL_CHORD_ROOT
unset SUGAR_OFFICIAL_CHORD_REFERENCE_GEOMETRY SUGAR_OFFICIAL_CHORD_OBJECT_USD
unset SUGAR_CONDITIONAL_TINYMDM_REWARD

valid_video() {
    "$FFMPEG_BIN" -hide_banner -loglevel error -i "$1" -f null - >/dev/null 2>&1
}

mkdir -p "$PAIR_ROOT/videos_off_vs_on"
for prefix in 37 45 53 61; do
    for arm in off on; do
        video_dir="$PAIR_ROOT/$arm/videos_seed${VIDEO_SEED}/prefix${prefix}"
        output="$video_dir/learned_kick.mp4"
        mkdir -p "$video_dir"
        if valid_video "$output"; then
            continue
        fi
        rm -f "$output"
        "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
            --checkpoint "$PAIR_ROOT/$arm/train/model_64.pt" --output "$output" \
            --label "Official CHORD ${arm^^}: learned Kick, prefix ${prefix}" \
            --steps 250 --seed "$VIDEO_SEED" --carry-prefix-steps "$prefix" \
            --transition-selected-skill-id 1 \
            --policy-topology causal_temporal_action_composition \
            --headless --device "$DEVICE" \
            --kit_args="--/renderer/multiGpu/enabled=false"
        valid_video "$output"
    done
    output="$PAIR_ROOT/videos_off_vs_on/chord_off_vs_on_prefix${prefix}.mp4"
    "$FFMPEG_BIN" -hide_banner -loglevel error -y \
        -i "$PAIR_ROOT/off/videos_seed${VIDEO_SEED}/prefix${prefix}/learned_kick.mp4" \
        -i "$PAIR_ROOT/on/videos_seed${VIDEO_SEED}/prefix${prefix}/learned_kick.mp4" \
        -filter_complex hstack=inputs=2 \
        -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "$output"
    valid_video "$output"
done

printf 'render_complete=1\n' > "$PAIR_ROOT/VIDEO_STATUS.env"
exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_OFFICIAL_CHORD_PAIR_RENDER_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
