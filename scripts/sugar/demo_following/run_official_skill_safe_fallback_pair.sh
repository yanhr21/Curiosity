#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
CHECKPOINT=${CHECKPOINT:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt}
PROOF=${PROOF:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_skill_safe_fallback_v1}
EVALUATOR=scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py

export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

run_arm() {
    local domain=$1
    local option=$2
    local seed=$3
    local name=$4
    shift 4
    "$PYTHON_BIN" -u "$EVALUATOR" \
        --domain "$domain" \
        --selected-demo-option "$option" \
        --route-generator-with-expert \
        --shared-checkpoint "$CHECKPOINT" \
        --training-proof "$PROOF" \
        --output-dir "$OUTPUT_ROOT/$name" \
        --num-envs 20 \
        --steps 650 \
        --seed "$seed" \
        --disable-observation-corruption \
        --headless \
        --device cuda:0 \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false" \
        "$@"
}

run_arm CarryBox unrelated 171623 carry_kick_direct
run_arm CarryBox unrelated 171623 carry_kick_safe --causal-safe-fallback
if run_arm KickBox correct 171624 kick_carry_direct; then
    echo "Expected direct BIGBOX Carry candidate rejection, but it passed" >&2
    exit 1
fi
test -f "$OUTPUT_ROOT/kick_carry_direct/RESULT.json"
run_arm KickBox correct 171624 kick_carry_safe --causal-safe-fallback

"$PYTHON_BIN" scripts/sugar/demo_following/summarize_official_skill_safe_fallback.py \
    --input-root "$OUTPUT_ROOT"
echo "OFFICIAL_SKILL_SAFE_FALLBACK_COMPLETE output_root=$OUTPUT_ROOT"
