#!/usr/bin/env bash
# Run the one predeclared phase-corrected matched pair, serially, then stop.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/experiments/demo_following/matched_phase_event_reward_reference_aware_v2}
LAUNCHER=$ROOT/scripts/sugar/native_tactile/launch_retained_child.sh
RUNNER=$ROOT/scripts/sugar/demo_following/run_matched_state_predictor.py

if [[ "${DEMO_POLICY_TRAINING_AUTHORIZED:-}" != "YES" ]]; then
    echo "explicit user authorization required: set DEMO_POLICY_TRAINING_AUTHORIZED=YES" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "run inside a retained srun compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing policy training on a login host" >&2; exit 2 ;;
esac

check_endpoint() {
    local arm=$1
    local proof="$OUTPUT_ROOT/seed161587/$arm/update_0064/proof.json"
    local checkpoint="$OUTPUT_ROOT/seed161587/$arm/update_0064/policy.pt"
    "$PYTHON_BIN" - "$proof" "$checkpoint" <<'PY'
import json
from pathlib import Path
import sys

proof = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
payload = json.loads(proof.read_text(encoding="utf-8"))
checks = payload.get("checks", {})
assert payload.get("passed") is True
assert checks and all(checks.values())
assert checks.get("demo_event_phase_and_prefix_are_causal") is True
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
PY
}

run_arm() {
    local arm=$1
    "$LAUNCHER" --foreground \
        --record "$OUTPUT_ROOT/$arm.process" \
        --status "$OUTPUT_ROOT/$arm.status" \
        --log "$OUTPUT_ROOT/$arm.log" \
        --tag "reference-aware-phase-event-$arm-64" -- \
        "$PYTHON_BIN" -u "$RUNNER" \
        --design phase_event_reward_only \
        --arm "$arm" \
        --output-root "$OUTPUT_ROOT" \
        --endpoint-updates 64 \
        --stop-after-segment \
        --policy-training-authorized
    check_endpoint "$arm"
}

run_arm correct
run_arm unrelated

printf 'matched_pair_complete output_root=%s; frozen evaluation not started\n' "$OUTPUT_ROOT"
