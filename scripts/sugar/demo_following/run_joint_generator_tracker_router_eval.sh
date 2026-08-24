#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
CHECKPOINT=${CHECKPOINT:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt}
PROOF=${PROOF:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_tracker_router_v1/seed161610/frozen_eval_joint_final}
EVALUATOR=scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py

export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

run_arm() {
    local domain=$1
    local option=$2
    local seed=$3
    local name=$4
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
        --headless \
        --device cuda:0 \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
}

run_arm CarryBox correct 171610 carry_carry45
run_arm CarryBox unrelated 171610 carry_kick21
if run_arm KickBox correct 171611 kick_carry45; then
    echo "Expected BIGBOX Carry45 transfer rejection, but the evaluator passed" >&2
    exit 1
fi
test -f "$OUTPUT_ROOT/kick_carry45/RESULT.json"
run_arm KickBox unrelated 171611 kick_kick21

echo "JOINT_GENERATOR_TRACKER_ROUTER_EVAL_COMPLETE output_root=$OUTPUT_ROOT"
