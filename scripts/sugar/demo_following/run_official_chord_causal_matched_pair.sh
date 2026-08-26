#!/usr/bin/env bash
# Run one serious, one-variable CHORD OFF/ON causal temporal-composer pair.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_chord_causal_matched_seed171648_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OFFICIAL_ROOT="$ROOT/experiments/runtime_assets/official_chord_5654c50e"
REFERENCE="$ROOT/experiments/demo_following/sugar_demo_chord_geometry_v2/kick21/contact_geometry.npz"
OBJECT_USD="$ROOT/SUGAR/descriptions/objects/big_box/obj_aligned.usd"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "official CHORD pair requires a retained srun compute step" >&2
    exit 2
fi
for required in \
    "$OFFICIAL_ROOT/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/utils_jit.py" \
    "$REFERENCE" "$OBJECT_USD"; do
    test -s "$required" || { echo "missing required CHORD input: $required" >&2; exit 2; }
done
if [[ -e "$OUTPUT_ROOT" ]]; then
    if [[ ! -d "$OUTPUT_ROOT/off" ]]; then
        echo "existing CHORD pair root has no resumable OFF arm: $OUTPUT_ROOT" >&2
        exit 2
    fi
else
    mkdir -p "$OUTPUT_ROOT"
fi

export POLICY_TOPOLOGY_OVERRIDE=causal_temporal_action_composition
export TRAIN_PREFIXES_CSV_OVERRIDE=33,41,49,57,65
export EVAL_PREFIXES_CSV_OVERRIDE=37,45,53,61
export REQUIRE_DISJOINT_PREFIX_SCHEDULES=1
export TRAIN_SEED_OVERRIDE=171648
export EVAL_SEED_OVERRIDE=181666
export VIDEO_SEED_OVERRIDE=181667
export NUM_ENVS_OVERRIDE=64

unset SUGAR_OFFICIAL_CHORD_REWARD SUGAR_OFFICIAL_CHORD_ROOT
unset SUGAR_OFFICIAL_CHORD_REFERENCE_GEOMETRY SUGAR_OFFICIAL_CHORD_OBJECT_USD
bash "$ROOT/scripts/sugar/demo_following/run_multi_context_transition_recovery.sh" \
    "$OUTPUT_ROOT/off" "$DEVICE"

export SUGAR_OFFICIAL_CHORD_REWARD=1
export SUGAR_OFFICIAL_CHORD_ROOT="$OFFICIAL_ROOT"
export SUGAR_OFFICIAL_CHORD_REFERENCE_GEOMETRY="$REFERENCE"
export SUGAR_OFFICIAL_CHORD_OBJECT_USD="$OBJECT_USD"
bash "$ROOT/scripts/sugar/demo_following/run_multi_context_transition_recovery.sh" \
    "$OUTPUT_ROOT/on" "$DEVICE"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/analyze_official_chord_causal_pair.py" \
    --pair-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/CHORD_GEOMETRY_RESULT.json" \
    --headless --device "$DEVICE" \
    --kit_args="--/renderer/enabled=false --/renderer/multiGpu/enabled=false"

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_official_chord_matched_pair.py" \
    --off-root "$OUTPUT_ROOT/off" --on-root "$OUTPUT_ROOT/on" \
    --output "$OUTPUT_ROOT/RESULT.json"

if [[ "${SUGAR_SKIP_CROSS_SKILL_VIDEOS:-0}" != "1" ]]; then
    FFMPEG_BIN="${FFMPEG_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
    mkdir -p "$OUTPUT_ROOT/videos_off_vs_on"
    for prefix in 37 45 53 61; do
    off_video="$OUTPUT_ROOT/off/videos_seed181667/prefix${prefix}/learned_kick.mp4"
    on_video="$OUTPUT_ROOT/on/videos_seed181667/prefix${prefix}/learned_kick.mp4"
    output_video="$OUTPUT_ROOT/videos_off_vs_on/chord_off_vs_on_prefix${prefix}.mp4"
    "$FFMPEG_BIN" -hide_banner -loglevel error -y \
        -i "$off_video" -i "$on_video" \
        -filter_complex hstack=inputs=2 \
        -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "$output_video"
        "$FFMPEG_BIN" -hide_banner -loglevel error -i "$output_video" -f null -
    done
fi
printf 'pair_complete=1\nresult=%s\n' "$OUTPUT_ROOT/RESULT.json" > "$OUTPUT_ROOT/PAIR_STATUS.env"

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_OFFICIAL_CHORD_MATCHED_PAIR_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
