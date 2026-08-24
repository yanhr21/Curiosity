#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
CHECKPOINT=${CHECKPOINT:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt}
PROOF=${PROOF:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_generator_compatibility_audit_v1}
EVALUATOR=scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py

export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

run_arm() {
    "$PYTHON_BIN" -u "$EVALUATOR" \
        --domain "$1" --selected-demo-option "$2" \
        --route-generator-with-expert --causal-safe-fallback \
        --disable-observation-corruption \
        --shared-checkpoint "$CHECKPOINT" --training-proof "$PROOF" \
        --output-dir "$OUTPUT_ROOT/$3" --num-envs 20 --steps 650 --seed "$4" \
        --headless --device cuda:0 \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
}

run_arm CarryBox unrelated compatible_kick_on_small 171623
if run_arm KickBox correct incompatible_carry_on_big 171624; then
    echo "The incompatible arm unexpectedly passed; retaining it for the audit" >&2
fi
test -f "$OUTPUT_ROOT/incompatible_carry_on_big/RESULT.json"
"$PYTHON_BIN" scripts/sugar/demo_following/summarize_official_generator_compatibility.py \
    --input-root "$OUTPUT_ROOT"
