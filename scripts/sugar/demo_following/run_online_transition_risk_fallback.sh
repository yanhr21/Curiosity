#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
CHECKPOINT=${CHECKPOINT:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/policy.pt}
PROOF=${PROOF:-experiments/demo_following/official_tracker_router_v1/seed161610/step_1000/proof.json}
RISK_PROOF=${RISK_PROOF:-experiments/demo_following/official_transition_risk_v1/frozen_eval_seed171626/RESULT.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_transition_risk_v1/online_fallback_seed171627}
EVALUATOR=scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:$ROOT/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

run_arm() {
    "$PYTHON_BIN" -u "$EVALUATOR" \
        --domain KickBox --selected-demo-option correct --route-generator-with-expert \
        --disable-observation-corruption \
        --shared-checkpoint "$CHECKPOINT" --training-proof "$PROOF" \
        --output-dir "$OUTPUT_ROOT/$1" --num-envs 20 --steps 650 --seed 171627 \
        --headless --device cuda:0 \
        --kit_args="--/renderer/enabled= --/renderer/multiGpu/enabled=false" \
        "${@:2}"
}

if run_arm direct_carry_on_big; then
    echo "Direct Carry45-on-BIGBOX unexpectedly passed; retaining trace" >&2
fi
test -f "$OUTPUT_ROOT/direct_carry_on_big/RESULT.json"
set +e
run_arm risk_latched_fallback --causal-transition-risk-fallback \
    --transition-risk-proof "$RISK_PROOF"
risk_rc=$?
set -e
test -f "$OUTPUT_ROOT/risk_latched_fallback/RESULT.json"

set +e
"$PYTHON_BIN" scripts/sugar/demo_following/summarize_online_transition_risk_fallback.py \
    --input-root "$OUTPUT_ROOT"
summary_rc=$?
set -e
if ((risk_rc != 0 || summary_rc != 0)); then
    exit 1
fi
