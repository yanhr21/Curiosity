#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
CHECKPOINT=${CHECKPOINT:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt}
PROOF=${PROOF:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_skill_transition_geometry_mass_v1/seed171621}
SEED=${SEED:-171621}
EVALUATOR=scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py

export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

run_cell() {
    local geometry=$1
    local mass_source=$2
    local name=$3
    set +e
    "$PYTHON_BIN" -u "$EVALUATOR" \
        --domain CarryBox \
        --selected-demo-option correct \
        --route-generator-with-expert \
        --scene-object-asset "$geometry" \
        --object-nominal-mass-source "$mass_source" \
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
    printf 'GEOMETRY_MASS_CELL_COMPLETE name=%s evaluator_rc=%s\n' "$name" "$rc"
}

run_cell small small small_geometry_small_mass
run_cell small big small_geometry_big_mass
run_cell big small big_geometry_small_mass
run_cell big big big_geometry_big_mass

"$PYTHON_BIN" scripts/sugar/demo_following/summarize_carry_skill_geometry_mass_factorial.py \
    --input-root "$OUTPUT_ROOT"

echo "CARRY_SKILL_GEOMETRY_MASS_FACTORIAL_COMPLETE output_root=$OUTPUT_ROOT"
