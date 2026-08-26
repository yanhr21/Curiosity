#!/usr/bin/env bash
# Frozen fitted-context audit for the completed dense-prefix causal composer.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1}")"
OUTPUT_ROOT="$(realpath -m "${2:-$ROOT/experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1_seen_context_audit}")"
DEVICE="${3:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
PREFIXES=(33 41 49 57 65)
TRAIN_SEED=171646
EVAL_SEED=181662

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "seen-context audit requires a retained GPU compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing GPU work on a login host" >&2; exit 2 ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_dense_seen_${SLURM_JOB_ID}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export DISPLAY=""

valid_evaluation() {
    local result="$1"
    local trace="${result%/RESULT.json}/trace.npz"
    [[ -s "$result" && -s "$trace" ]] && jq -e \
        '.protocol == "sugar_cross_skill_recovery_frozen_eval_v4" and .structurally_valid == true' \
        "$result" >/dev/null
}

run_pipeline() (
    set -euo pipefail
    test -s "$SOURCE_ROOT/train/model_64.pt"
    test -s "$SOURCE_ROOT/train/model_pre_update.pt"
    test -s "$SOURCE_ROOT/train/prefix_audit.json"
    test -s "$SOURCE_ROOT/CHECKPOINT_AUDIT.json"
    mkdir -p "$OUTPUT_ROOT/evaluation"

    cd "$ROOT/SUGAR"
    for prefix in "${PREFIXES[@]}"; do
        for endpoint in learned pre_update; do
            checkpoint="$SOURCE_ROOT/train/model_64.pt"
            if [[ "$endpoint" == "pre_update" ]]; then
                checkpoint="$SOURCE_ROOT/train/model_pre_update.pt"
            fi
            result_dir="$OUTPUT_ROOT/evaluation/prefix${prefix}/${endpoint}_kick"
            if valid_evaluation "$result_dir/RESULT.json"; then
                continue
            fi
            rm -rf "$result_dir"
            "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
                --checkpoint "$checkpoint" --output-dir "$result_dir" \
                --transition-selected-skill-id 1 --carry-prefix-steps "$prefix" \
                --policy-topology causal_action_composition \
                --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
                --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
        done
    done

    cd "$ROOT"
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
        --training-audit "$SOURCE_ROOT/train/prefix_audit.json" \
        --checkpoint-audit "$SOURCE_ROOT/CHECKPOINT_AUDIT.json" \
        --training-seed "$TRAIN_SEED" \
        --expected-schedule 33,41,49,57,65 \
        --expected-evaluation-schedule 33,41,49,57,65 \
        --expected-context-relation seen \
        --expected-policy-topology causal_action_composition \
        --output "$OUTPUT_ROOT/RESULT.json"

    # Visualize at most two profiles where learned and exact pre-update physical
    # outcomes differ. If there are none, show one central fitted context.
    mapfile -t targets < <(
        jq -r '[.contexts[] | .carry_prefix_steps as $prefix | .outcome_changes[] |
            [$prefix, .profile]] | unique | .[:2][] | @tsv' "$OUTPUT_ROOT/RESULT.json"
    )
    if [[ "${#targets[@]}" -eq 0 ]]; then
        targets=("49"$'\t'"0")
    fi
    for target in "${targets[@]}"; do
        IFS=$'\t' read -r prefix profile <<<"$target"
        video_dir="$OUTPUT_ROOT/videos_camera_seed${EVAL_SEED}/prefix${prefix}_profile${profile}"
        mkdir -p "$video_dir"
        for endpoint in learned pre_update; do
            checkpoint="$SOURCE_ROOT/train/model_64.pt"
            label="Learned composer | fitted prefix ${prefix}"
            if [[ "$endpoint" == "pre_update" ]]; then
                checkpoint="$SOURCE_ROOT/train/model_pre_update.pt"
                label="Exact pre-update Kick | fitted prefix ${prefix}"
            fi
            video="$video_dir/${endpoint}.mp4"
            if "$FFMPEG_BIN" -hide_banner -loglevel error -i "$video" -f null - \
                >/dev/null 2>&1; then
                continue
            fi
            rm -f "$video"
            cd "$ROOT/SUGAR"
            "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
                --checkpoint "$checkpoint" --output "$video" --label "$label" \
                --steps 250 --seed "$EVAL_SEED" --carry-prefix-steps "$prefix" \
                --transition-selected-skill-id 1 \
                --policy-topology causal_action_composition \
                --profile-index "$profile" --num-profiles 20 \
                --headless --device "$DEVICE" \
                --kit_args="--/renderer/multiGpu/enabled=false"
        done
        cd "$ROOT"
        paired="$video_dir/learned_vs_pre_update.mp4"
        "$FFMPEG_BIN" -hide_banner -loglevel error -y \
            -i "$video_dir/learned.mp4" -i "$video_dir/pre_update.mp4" \
            -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 \
            -pix_fmt yuv420p -movflags +faststart "$paired"
        "$FFMPEG_BIN" -hide_banner -loglevel error -i "$paired" -f null -
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
    printf 'source_training_seed=%s\n' "$TRAIN_SEED"
    printf 'evaluation_seed=%s\n' "$EVAL_SEED"
    printf 'fitted_prefixes=33,41,49,57,65\n'
} > "$OUTPUT_ROOT/PIPELINE_STATUS.env"
echo "DENSE_PREFIX_SEEN_CONTEXT_AUDIT_RC=$pipeline_rc output=$OUTPUT_ROOT" >&2

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_DENSE_PREFIX_SEEN_CONTEXT_AUDIT_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
