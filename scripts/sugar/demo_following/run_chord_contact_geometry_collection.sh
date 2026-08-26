#!/usr/bin/env bash
# Collect live foot-box contact points for the known prefix-53 outcome boundary.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_ROOT="$ROOT/experiments/demo_following/causal_temporal_composition_dense_prefix_seed171648_v1"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/chord_contact_geometry_phase_aligned_prefix53_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac

run_arm() {
    local checkpoint="$1"
    local name="$2"
    local output="$OUTPUT_ROOT/$name"
    if [[ -f "$output/RESULT.json" ]]; then
        echo "CONTACT_GEOMETRY_REUSE arm=$name output=$output"
        return
    fi
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_cross_skill_recovery.py" \
        --headless --device "$DEVICE" \
        --checkpoint "$checkpoint" \
        --output-dir "$output" \
        --carry-prefix-steps 53 \
        --transition-selected-skill-id 1 \
        --policy-topology causal_temporal_action_composition \
        --num-envs 20 --steps 250 --seed 181666
}

run_arm "$SOURCE_ROOT/train/model_pre_update.pt" pre_update_kick
run_arm "$SOURCE_ROOT/train/model_64.pt" learned_kick

NATIVE_OUTPUT="$OUTPUT_ROOT/native_kick21"
if [[ -f "$NATIVE_OUTPUT/RESULT.json" ]]; then
    echo "CONTACT_GEOMETRY_REUSE arm=native_kick21 output=$NATIVE_OUTPUT"
else
    export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py" \
        --domain KickBox --selected-demo-option unrelated \
        --route-generator-with-expert \
        --shared-checkpoint "$ROOT/experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt" \
        --training-proof "$ROOT/experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json" \
        --output-dir "$NATIVE_OUTPUT" \
        --num-envs 20 --steps 650 --seed 171611 \
        --headless --device "$DEVICE" \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
fi

ANALYSIS_OUTPUT="$OUTPUT_ROOT/official_chord_representation"
if [[ -f "$ANALYSIS_OUTPUT/RESULT.json" ]]; then
    echo "CHORD_REPRESENTATION_REUSE output=$ANALYSIS_OUTPUT"
else
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_following/analyze_chord_contact_geometry.py" \
        --collection-root "$OUTPUT_ROOT" \
        --output-dir "$ANALYSIS_OUTPUT" \
        --official-chord-root "$ROOT/experiments/runtime_assets/official_chord_5654c50e"
fi

echo "CHORD_CONTACT_GEOMETRY_COLLECTION_READY output=$OUTPUT_ROOT"
exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_CHORD_CONTACT_GEOMETRY_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
