#!/usr/bin/env bash
# Collect feature-complete frozen prefix-41 traces and score official TinyMDM priors.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
DEVICE="${1:-cuda:0}"
OUTPUT_ROOT="${2:-$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
PRIOR_ROOT="$ROOT/experiments/demo_following/selected_demo_smp_v1"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite $OUTPUT_ROOT" >&2
    exit 2
fi
for path in \
    "$ROOT/experiments/demo_following/cross_skill_recovery_prefix41_v1/train/model_pre_update.pt" \
    "$ROOT/experiments/demo_following/cross_skill_recovery_prefix41_v1/train/model_64.pt" \
    "$ROOT/experiments/demo_following/cross_skill_recovery_prefix41_safe_v1/train/model_64.pt" \
    "$PRIOR_ROOT/priors/carry45/model.pt" \
    "$PRIOR_ROOT/priors/kick21/model.pt" \
    "$PRIOR_ROOT/cross_score/RESULT.json"; do
    test -s "$path"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_recovery_tinymdm_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export DISPLAY=""

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT/SUGAR"
declare -A CHECKPOINTS=(
    [released_baseline]="$ROOT/experiments/demo_following/cross_skill_recovery_prefix41_v1/train/model_pre_update.pt"
    [unconstrained_update64]="$ROOT/experiments/demo_following/cross_skill_recovery_prefix41_v1/train/model_64.pt"
    [safety_update64]="$ROOT/experiments/demo_following/cross_skill_recovery_prefix41_safe_v1/train/model_64.pt"
)
for arm in released_baseline unconstrained_update64 safety_update64; do
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --checkpoint "${CHECKPOINTS[$arm]}" \
        --output-dir "$OUTPUT_ROOT/traces/$arm" \
        --carry-prefix-steps 41 --num-envs 20 --steps 250 --seed 181631 \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
    test -s "$OUTPUT_ROOT/traces/$arm/trace.npz"
    test -s "$OUTPUT_ROOT/traces/$arm/RESULT.json"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/smp/audit_cross_skill_recovery_tinymdm.py" \
    --arm released_baseline "$OUTPUT_ROOT/traces/released_baseline/trace.npz" \
    --arm unconstrained_update64 "$OUTPUT_ROOT/traces/unconstrained_update64/trace.npz" \
    --arm safety_update64 "$OUTPUT_ROOT/traces/safety_update64/trace.npz" \
    --prior-root "$PRIOR_ROOT" --output-dir "$OUTPUT_ROOT/score" \
    --device "$DEVICE" --chunk-size 256
test -s "$OUTPUT_ROOT/score/RESULT.json"
