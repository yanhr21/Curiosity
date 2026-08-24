#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
CHECKPOINT=${CHECKPOINT:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt}
PROOF=${PROOF:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_skill_transition_context_goal_v1/seed171622}
SEED=${SEED:-171622}
EVALUATOR=scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py

export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

run_cell() {
    local domain=$1
    local asset=$2
    local goal=$3
    local name=$4
    set +e
    "$PYTHON_BIN" -u "$EVALUATOR" \
        --domain "$domain" \
        --selected-demo-option correct \
        --route-generator-with-expert \
        --scene-object-asset "$asset" \
        --object-nominal-mass-source "$asset" \
        --target-goal-source "$goal" \
        --shared-checkpoint "$CHECKPOINT" \
        --training-proof "$PROOF" \
        --output-dir "$OUTPUT_ROOT/$name" \
        --num-envs 20 \
        --steps 650 \
        --seed "$SEED" \
        --headless \
        --device cuda:0 \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false"
    local rc=$?
    set -e
    test -f "$OUTPUT_ROOT/$name/RESULT.json"
    printf 'CONTEXT_GOAL_CELL_COMPLETE name=%s evaluator_rc=%s\n' "$name" "$rc"
}

run_cell CarryBox small carry45 carry_context_carry_goal
run_cell CarryBox small kick21 carry_context_kick_goal
run_cell KickBox big carry45 kick_context_carry_goal
run_cell KickBox big kick21 kick_context_kick_goal

"$PYTHON_BIN" scripts/sugar/demo_following/summarize_carry_skill_context_goal_factorial.py \
    --input-root "$OUTPUT_ROOT"

echo "CARRY_SKILL_CONTEXT_GOAL_FACTORIAL_COMPLETE output_root=$OUTPUT_ROOT"
