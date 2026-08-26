#!/usr/bin/env bash
# Frozen gate/residual attribution for the completed dense-prefix composer.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1}")"
OUTPUT_ROOT="$(realpath -m "${2:-$ROOT/experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1_frozen_ablation}")"
DEVICE="${3:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
PREFIXES=(37 45 53 61)
EVAL_SEED=181662
SOURCE_VIDEO_SEED=181663

valid_frozen_evaluation() {
    local result="$1"
    local trace="${result%/RESULT.json}/trace.npz"
    [[ -s "$result" && -s "$trace" ]] && jq -e \
        '.protocol == "sugar_cross_skill_recovery_frozen_eval_v4" and .structurally_valid == true' \
        "$result" >/dev/null
}

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "frozen composer ablation requires a retained GPU compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing GPU work on a login host" >&2; exit 2 ;;
esac

run_pipeline() (
    set -e
    mkdir -p "$OUTPUT_ROOT/checkpoints" "$OUTPUT_ROOT/evaluation"
    local source_checkpoint="$SOURCE_ROOT/train/model_64.pt"
    local gate_checkpoint="$OUTPUT_ROOT/checkpoints/model_64_gate_only.pt"
    local residual_checkpoint="$OUTPUT_ROOT/checkpoints/model_64_residual_only.pt"
    test -s "$source_checkpoint"
    test -s "$SOURCE_ROOT/train/model_pre_update.pt"
    test -s "$SOURCE_ROOT/RESULT.json"

    # Finish any source learned/pre videos left incomplete by an interrupted
    # rendering step. Completed, decodable files are reused without rerunning.
    for prefix in "${PREFIXES[@]}"; do
        local source_video_dir="$SOURCE_ROOT/videos_seed${SOURCE_VIDEO_SEED}/prefix${prefix}"
        mkdir -p "$source_video_dir"
        for endpoint in learned pre_update; do
            local source_video="$source_video_dir/${endpoint}_kick.mp4"
            if "$FFMPEG_BIN" -hide_banner -loglevel error \
                -i "$source_video" -f null - >/dev/null 2>&1; then
                continue
            fi
            rm -f "$source_video"
            local checkpoint="$source_checkpoint"
            local label="Causal action composition learned Kick: prefix ${prefix}"
            if [[ "$endpoint" == "pre_update" ]]; then
                checkpoint="$SOURCE_ROOT/train/model_pre_update.pt"
                label="Exact pre-update Kick: prefix ${prefix}"
            fi
            cd "$ROOT/SUGAR"
            "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
                --checkpoint "$checkpoint" --output "$source_video" \
                --label "$label" --steps 250 --seed "$SOURCE_VIDEO_SEED" \
                --carry-prefix-steps "$prefix" --transition-selected-skill-id 1 \
                --policy-topology causal_action_composition \
                --headless --device "$DEVICE" \
                --kit_args="--/renderer/multiGpu/enabled=false"
        done
        cd "$ROOT"
        local paired="$source_video_dir/learned_vs_pre_update_prefix${prefix}.mp4"
        if ! "$FFMPEG_BIN" -hide_banner -loglevel error \
            -i "$paired" -f null - >/dev/null 2>&1; then
            "$FFMPEG_BIN" -hide_banner -loglevel error -y \
                -i "$source_video_dir/learned_kick.mp4" \
                -i "$source_video_dir/pre_update_kick.mp4" \
                -filter_complex hstack=inputs=2 -c:v libx264 -crf 18 \
                -pix_fmt yuv420p -movflags +faststart "$paired"
        fi
        "$FFMPEG_BIN" -hide_banner -loglevel error -i "$paired" -f null -
    done

    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/make_causal_composer_ablation_checkpoint.py" \
        --source "$source_checkpoint" --output "$gate_checkpoint" --mode gate_only \
        --audit "$OUTPUT_ROOT/checkpoints/gate_only_audit.json"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/make_causal_composer_ablation_checkpoint.py" \
        --source "$source_checkpoint" --output "$residual_checkpoint" --mode residual_only \
        --audit "$OUTPUT_ROOT/checkpoints/residual_only_audit.json"

    cd "$ROOT/SUGAR"
    for prefix in "${PREFIXES[@]}"; do
        for arm in gate_only residual_only; do
            local checkpoint="$gate_checkpoint"
            if [[ "$arm" == "residual_only" ]]; then
                checkpoint="$residual_checkpoint"
            fi
            local result_dir="$OUTPUT_ROOT/evaluation/prefix${prefix}/${arm}"
            if valid_frozen_evaluation "$result_dir/RESULT.json"; then
                continue
            fi
            rm -rf "$result_dir"
            "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
                --checkpoint "$checkpoint" \
                --output-dir "$result_dir" \
                --transition-selected-skill-id 1 --carry-prefix-steps "$prefix" \
                --policy-topology causal_action_composition \
                --num-envs 20 --steps 250 --seed "$EVAL_SEED" --headless --device "$DEVICE" \
                --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"
        done
    done

    cd "$ROOT"
    local summary_args=()
    for prefix in "${PREFIXES[@]}"; do
        summary_args+=(
            --comparison "$prefix"
            "$SOURCE_ROOT/evaluation/prefix${prefix}/learned_kick/RESULT.json"
            "$OUTPUT_ROOT/evaluation/prefix${prefix}/gate_only/RESULT.json"
            "$OUTPUT_ROOT/evaluation/prefix${prefix}/residual_only/RESULT.json"
            "$SOURCE_ROOT/evaluation/prefix${prefix}/pre_update_kick/RESULT.json"
        )
    done
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_causal_composer_ablation.py" \
        "${summary_args[@]}" --expected-prefixes 37,45,53,61 \
        --evaluation-seed "$EVAL_SEED" \
        --gate-audit "$OUTPUT_ROOT/checkpoints/gate_only_audit.json" \
        --residual-audit "$OUTPUT_ROOT/checkpoints/residual_only_audit.json" \
        --output "$OUTPUT_ROOT/RESULT.json"

    mapfile -t visualization_targets < <(
        jq -r '.visualization_targets[] | [.prefix, .profile] | @tsv' \
            "$OUTPUT_ROOT/RESULT.json"
    )
    for target in "${visualization_targets[@]}"; do
        IFS=$'\t' read -r prefix profile <<<"$target"
        local video_dir="$OUTPUT_ROOT/videos_camera_seed${EVAL_SEED}/prefix${prefix}_profile${profile}"
        mkdir -p "$video_dir"
        local arms=(full gate_only residual_only exact_pre_update)
        for arm in "${arms[@]}"; do
            local checkpoint="$source_checkpoint"
            local label="Full learned composer"
            case "$arm" in
                gate_only) checkpoint="$gate_checkpoint"; label="Frozen gate only" ;;
                residual_only) checkpoint="$residual_checkpoint"; label="Frozen residual only" ;;
                exact_pre_update) checkpoint="$SOURCE_ROOT/train/model_pre_update.pt"; label="Exact pre-update Kick" ;;
            esac
            if "$FFMPEG_BIN" -hide_banner -loglevel error \
                -i "$video_dir/${arm}.mp4" -f null - >/dev/null 2>&1; then
                continue
            fi
            rm -f "$video_dir/${arm}.mp4"
            cd "$ROOT/SUGAR"
            "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_cross_skill_recovery_world.py" \
                --checkpoint "$checkpoint" --output "$video_dir/${arm}.mp4" \
                --label "$label | prefix ${prefix}" --steps 250 --seed "$EVAL_SEED" \
                --carry-prefix-steps "$prefix" --transition-selected-skill-id 1 \
                --policy-topology causal_action_composition \
                --profile-index "$profile" --num-profiles 20 \
                --headless --device "$DEVICE" \
                --kit_args="--/renderer/multiGpu/enabled=false"
        done
        cd "$ROOT"
        "$FFMPEG_BIN" -hide_banner -loglevel error -y \
            -i "$video_dir/full.mp4" -i "$video_dir/gate_only.mp4" \
            -i "$video_dir/residual_only.mp4" -i "$video_dir/exact_pre_update.mp4" \
            -filter_complex "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" \
            -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
            "$video_dir/full_gate_residual_pre.mp4"
        "$FFMPEG_BIN" -hide_banner -loglevel error \
            -i "$video_dir/full_gate_residual_pre.mp4" -f null -
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
    printf 'source_root=%s\n' "$SOURCE_ROOT"
    printf 'output_root=%s\n' "$OUTPUT_ROOT"
} > "$OUTPUT_ROOT/PIPELINE_STATUS.env"
echo "DENSE_PREFIX_COMPOSER_ABLATION_RC=$pipeline_rc output=$OUTPUT_ROOT" >&2

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_DENSE_PREFIX_COMPOSER_ABLATION_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
