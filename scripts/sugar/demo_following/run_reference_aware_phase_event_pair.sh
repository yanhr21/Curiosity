#!/usr/bin/env bash
# Run the one predeclared phase-corrected matched pair, serially, then stop.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/experiments/demo_following/matched_phase_event_reward_reference_aware_v2}
LAUNCHER=$ROOT/scripts/sugar/native_tactile/launch_retained_child.sh
RUNNER=$ROOT/scripts/sugar/demo_following/run_matched_state_predictor.py

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
    "$PYTHON_BIN" - "$proof" "$checkpoint" "$arm" <<'PY'
import json
from pathlib import Path
import sys

proof = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
arm = sys.argv[3]
payload = json.loads(proof.read_text(encoding="utf-8"))
checks = payload.get("checks", {})
audit = payload.get("demo_event_reward", {}).get("final_frozen_audit", {})
assert payload.get("passed") is True
assert checks and all(checks.values())
assert checks.get("demo_event_phase_and_prefix_are_causal") is True
assert payload.get("protocol") == "sugar_phase_event_reward_matched_policy_v1"
assert payload.get("seed") == 161587
assert payload.get("action_seed") == 161588
assert payload.get("num_envs") == 20
assert payload.get("num_updates") == 64
assert payload.get("demo_event_reward", {}).get("selected_option") == arm
assert audit.get("phase_source") == "reset_reference_frame_plus_causal_control_clock"
assert audit.get("initial_episode_steps_supplied") is True
assert audit.get("initial_episode_steps_min") == 197
assert audit.get("initial_episode_steps_max") == 197
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
PY
}

check_pair() {
    "$PYTHON_BIN" - \
        "$OUTPUT_ROOT/seed161587/correct/update_0064/proof.json" \
        "$OUTPUT_ROOT/seed161587/unrelated/update_0064/proof.json" \
        "$OUTPUT_ROOT/seed161587/correct/update_0064/protocol.json" \
        "$OUTPUT_ROOT/seed161587/unrelated/update_0064/protocol.json" <<'PY'
import json
import sys

proofs = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:3]]
protocols = [json.load(open(path, encoding="utf-8")) for path in sys.argv[3:5]]
records = [proof.get("no_tactile_startup_physics") for proof in proofs]
assert protocols[0] == protocols[1]
assert all(isinstance(record, dict) and record.get("passed") for record in records)
assert records[0]["values"] == records[1]["values"]
PY
}

run_arm() {
    local arm=$1
    local arm_root="$OUTPUT_ROOT/seed161587/$arm/update_0064"
    local record="$OUTPUT_ROOT/$arm.process"
    local status="$OUTPUT_ROOT/$arm.status"
    local log="$OUTPUT_ROOT/$arm.log"

    if [[ -f "$arm_root/proof.json" && -f "$arm_root/policy.pt" ]]; then
        check_endpoint "$arm"
        printf 'reusing_complete_arm=%s\n' "$arm"
        return
    fi
    if [[ -e "$arm_root" || -e "$record" || -e "$status" || -e "$log" ]]; then
        echo "incomplete arm requires inspection; refusing to overwrite: $arm" >&2
        exit 2
    fi
    "$LAUNCHER" --foreground \
        --record "$record" \
        --status "$status" \
        --log "$log" \
        --tag "reference-aware-phase-event-$arm-64" -- \
        "$PYTHON_BIN" -u "$RUNNER" \
        --design phase_event_reward_only \
        --arm "$arm" \
        --output-root "$OUTPUT_ROOT" \
        --endpoint-updates 64 \
        --stop-after-segment
    check_endpoint "$arm"
}

run_arm correct
run_arm unrelated
check_pair

printf 'matched_pair_complete output_root=%s; endpoint proofs passed\n' "$OUTPUT_ROOT"
