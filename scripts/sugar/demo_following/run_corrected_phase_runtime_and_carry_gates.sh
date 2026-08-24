#!/usr/bin/env bash
# Run corrected phase online smokes, then the matched frozen Carry scorer gate.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/experiments/demo_following/corrected_phase_runtime_gate_v1}
RUNNER="$ROOT/scripts/sugar/demo_following/run_matched_state_predictor.py"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "corrected phase runtime gates require a retained Slurm allocation" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "refusing to overwrite corrected phase runtime evidence: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT/online_smokes"
for arm in correct unrelated; do
    "$PYTHON_BIN" -u "$RUNNER" \
        --design phase_event_reward_only \
        --arm "$arm" \
        --endpoint-updates 64 \
        --stop-after-segment \
        --output-root "$OUTPUT_ROOT/probe_segments" \
        --runner-rollout-smoke-only \
        --probe-evidence-output "$OUTPUT_ROOT/online_smokes/$arm.json"
    jq -e '
        .passed == true and
        .policy_updates_executed == 0 and
        .checks.initial_phase_matches_restored_reference_frame == true and
        .frozen_model_audit.initial_episode_steps_supplied == true and
        .frozen_model_audit.initial_episode_steps_min == 197 and
        .frozen_model_audit.initial_episode_steps_max == 197
    ' "$OUTPUT_ROOT/online_smokes/$arm.json" >/dev/null
done

OUTPUT_ROOT="$OUTPUT_ROOT/frozen_carry" \
PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT/scripts/sugar/demo_following/run_corrected_phase_frozen_carry_gate.sh"

echo "CORRECTED_PHASE_RUNTIME_AND_CARRY_GATES_COMPLETE root=$OUTPUT_ROOT"
