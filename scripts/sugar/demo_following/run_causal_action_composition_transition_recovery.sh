#!/usr/bin/env bash
# Fixed multi-context diagnostic for state-dependent exact Carry/Kick composition.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/causal_action_composition_seed171644_v1}"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
REPLICATION_OUTPUT_ROOT="$(realpath -m "${REPLICATION_OUTPUT_ROOT:-${OUTPUT_ROOT}_seed171645_replication}")"
AGGREGATE_OUTPUT_ROOT="$(realpath -m "${AGGREGATE_OUTPUT_ROOT:-${OUTPUT_ROOT}_two_seed_aggregate}")"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "causal action composition requires a retained srun compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing GPU work on a login host" >&2; exit 2 ;;
esac

export POLICY_TOPOLOGY_OVERRIDE=causal_action_composition
export TRAIN_SEED_OVERRIDE="${TRAIN_SEED_OVERRIDE:-171644}"
export EVAL_SEED_OVERRIDE="${EVAL_SEED_OVERRIDE:-181656}"
export VIDEO_SEED_OVERRIDE="${VIDEO_SEED_OVERRIDE:-181657}"

set +e
bash "$ROOT/scripts/sugar/demo_following/run_multi_context_transition_recovery.sh" \
    "$OUTPUT_ROOT" "$DEVICE"
first_rc=$?
first_positive=0
decision_rc=0
replication_launched=0
replication_rc=0
aggregate_rc=0

if [[ "$first_rc" -eq 0 ]]; then
    first_positive=$("$PYTHON_BIN" - "$OUTPUT_ROOT/RESULT.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("policy_topology") != "causal_action_composition":
    raise SystemExit("causal-composition result topology drift")
positive = bool(
    result.get("checks", {}).get("aggregate_kick_safety_improvement") is True
    and result.get("conclusion")
    == "multi_context_training_improves_unseen_seed_kick_safety"
)
print(int(positive))
PY
    )
    decision_rc=$?
fi

if [[ "$first_rc" -eq 0 && "$decision_rc" -eq 0 && "$first_positive" -eq 1 ]]; then
    replication_launched=1
    export TRAIN_SEED_OVERRIDE="${REPLICATION_TRAIN_SEED_OVERRIDE:-171645}"
    export EVAL_SEED_OVERRIDE="${REPLICATION_EVAL_SEED_OVERRIDE:-181658}"
    export VIDEO_SEED_OVERRIDE="${REPLICATION_VIDEO_SEED_OVERRIDE:-181659}"
    bash "$ROOT/scripts/sugar/demo_following/run_multi_context_transition_recovery.sh" \
        "$REPLICATION_OUTPUT_ROOT" "$DEVICE"
    replication_rc=$?
    if [[ "$replication_rc" -eq 0 ]]; then
        "$PYTHON_BIN" \
            "$ROOT/scripts/sugar/demo_following/aggregate_multi_context_transition_recovery_seeds.py" \
            --result "$OUTPUT_ROOT/RESULT.json" \
            --result "$REPLICATION_OUTPUT_ROOT/RESULT.json" \
            --output "$AGGREGATE_OUTPUT_ROOT/RESULT.json"
        aggregate_rc=$?
    fi
fi

pipeline_rc=$first_rc
if [[ "$decision_rc" -ne 0 || "$replication_rc" -ne 0 || "$aggregate_rc" -ne 0 ]]; then
    pipeline_rc=1
fi
set -e
mkdir -p "$OUTPUT_ROOT"
{
    printf 'pipeline_exit_code=%s\n' "$pipeline_rc"
    printf 'first_seed_exit_code=%s\n' "$first_rc"
    printf 'first_seed_positive=%s\n' "$first_positive"
    printf 'automatic_decision_exit_code=%s\n' "$decision_rc"
    printf 'replication_launched=%s\n' "$replication_launched"
    printf 'replication_exit_code=%s\n' "$replication_rc"
    printf 'aggregate_exit_code=%s\n' "$aggregate_rc"
    printf 'replication_output=%s\n' "$REPLICATION_OUTPUT_ROOT"
    printf 'aggregate_output=%s\n' "$AGGREGATE_OUTPUT_ROOT"
} > "$OUTPUT_ROOT/PIPELINE_STATUS.env"
echo "CAUSAL_ACTION_COMPOSITION_PIPELINE_RC=$pipeline_rc output=$OUTPUT_ROOT" >&2

# Retain the granted GPU for review and follow-up even after a successful or
# failed child pipeline. The recorded launcher PGID remains the sole kill target.
exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_CAUSAL_ACTION_COMPOSITION_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
