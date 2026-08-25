#!/usr/bin/env bash
# Frozen no-training official-Kick sweep over a predeclared Carry prefix grid.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/cross_skill_prefix_frontier_v1}"
CHECKPOINT="${2:-$ROOT/experiments/demo_following/cross_skill_recovery_v1/bcppo_update64_seed171629/model_pre_update.pt}"
DEVICE="${3:-cuda:0}"
EVAL_SEED=181630
PREFIX_STEPS=(9 17 25 33 41 49 57 65 73 81 89 97)
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
CHECKPOINT="$(realpath -m "$CHECKPOINT")"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run this sweep inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing frozen official-Kick checkpoint: $CHECKPOINT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_cross_skill_frontier_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export DISPLAY=""

cd "$ROOT/SUGAR"
for prefix_steps in "${PREFIX_STEPS[@]}"; do
    result_dir="$OUTPUT_ROOT/prefix_$(printf '%03d' "$prefix_steps")"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$result_dir" \
        --carry-prefix-steps "$prefix_steps" \
        --num-envs 20 --steps 250 --seed "$EVAL_SEED" \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
    test -s "$result_dir/RESULT.json"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/summarize_cross_skill_prefix_frontier.py" \
    --input-root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/FRONTIER_RESULT.json"
