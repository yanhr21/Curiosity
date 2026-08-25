#!/usr/bin/env bash
# Frozen learned/pre evaluation on unseen Carry-prefix lengths; no training.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/causal_composition_heldout_prefix33_65_v1}")"
DEVICE="${2:-cuda:0}"
SEED1_ROOT="$(realpath -m "${SEED1_ROOT:-$ROOT/experiments/demo_following/causal_action_composition_seed171644_autorun_v1}")"
SEED2_ROOT="$(realpath -m "${SEED2_ROOT:-$ROOT/experiments/demo_following/causal_action_composition_seed171644_autorun_v1_seed171645_replication}")"
PREFIXES=(33 65)

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "held-out evaluation requires a retained srun compute step" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_causal_heldout_${SLURM_JOB_ID}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export DISPLAY=""

run_pipeline() (
    set -euo pipefail
    mkdir -p "$OUTPUT_ROOT"
    seed_specs=(
        "171644 181656 181660 $SEED1_ROOT"
        "171645 181658 181661 $SEED2_ROOT"
    )
    summary_args=()
    for spec in "${seed_specs[@]}"; do
        read -r train_seed eval_seed video_seed checkpoint_root <<<"$spec"
        seed_root="$OUTPUT_ROOT/seed${train_seed}"
        summary_paths=()
        for prefix in "${PREFIXES[@]}"; do
            for endpoint in learned pre_update; do
                checkpoint="$checkpoint_root/train/model_64.pt"
                if [[ "$endpoint" == "pre_update" ]]; then
                    checkpoint="$checkpoint_root/train/model_pre_update.pt"
                fi
                result_dir="$seed_root/evaluation/prefix${prefix}/${endpoint}_kick"
                "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
                    --checkpoint "$checkpoint" --output-dir "$result_dir" \
                    --transition-selected-skill-id 1 --carry-prefix-steps "$prefix" \
                    --policy-topology causal_action_composition \
                    --num-envs 20 --steps 250 --seed "$eval_seed" \
                    --headless --device "$DEVICE" \
                    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
                summary_paths+=("$result_dir/RESULT.json")
            done
        done
        summary_args+=(
            --seed "$train_seed" "$eval_seed" "$checkpoint_root/CHECKPOINT_AUDIT.json"
            "${summary_paths[0]}" "${summary_paths[1]}"
            "${summary_paths[2]}" "${summary_paths[3]}"
        )

    done

    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_heldout_causal_composition.py" \
        "${summary_args[@]}" --heldout-prefixes 33,65 \
        --training-prefixes 41,49,57 --output "$OUTPUT_ROOT/RESULT.json"

    for spec in "${seed_specs[@]}"; do
        read -r train_seed _eval_seed video_seed checkpoint_root <<<"$spec"
        seed_root="$OUTPUT_ROOT/seed${train_seed}"
        for prefix in "${PREFIXES[@]}"; do
            video_dir="$seed_root/videos_seed${video_seed}/prefix${prefix}"
            mkdir -p "$video_dir"
            for endpoint in learned pre_update; do
                checkpoint="$checkpoint_root/train/model_64.pt"
                label="Held-out causal composition learned Kick: prefix ${prefix}"
                if [[ "$endpoint" == "pre_update" ]]; then
                    checkpoint="$checkpoint_root/train/model_pre_update.pt"
                    label="Held-out exact pre-update Kick: prefix ${prefix}"
                fi
                "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
                    --checkpoint "$checkpoint" --output "$video_dir/${endpoint}_kick.mp4" \
                    --label "$label" --steps 250 --seed "$video_seed" \
                    --carry-prefix-steps "$prefix" --transition-selected-skill-id 1 \
                    --policy-topology causal_action_composition \
                    --headless --device "$DEVICE" \
                    --kit_args="--/renderer/multiGpu/enabled=false"
            done
            "$FFMPEG_BIN" -hide_banner -loglevel error -y \
                -i "$video_dir/learned_kick.mp4" \
                -i "$video_dir/pre_update_kick.mp4" \
                -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 \
                -pix_fmt yuv420p -movflags +faststart \
                "$video_dir/learned_vs_pre_update_prefix${prefix}.mp4"
            "$FFMPEG_BIN" -hide_banner -loglevel error \
                -i "$video_dir/learned_vs_pre_update_prefix${prefix}.mp4" -f null -
        done
    done
)

set +e
run_pipeline
pipeline_rc=$?
set -e
mkdir -p "$OUTPUT_ROOT"
{
    printf 'pipeline_exit_code=%s\n' "$pipeline_rc"
    printf 'policy_training_or_optimizer_updates=0\n'
    printf 'heldout_prefixes=33,65\n'
} > "$OUTPUT_ROOT/PIPELINE_STATUS.env"
echo "CAUSAL_COMPOSITION_HELDOUT_RC=$pipeline_rc output=$OUTPUT_ROOT" >&2

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_CAUSAL_HELDOUT_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
