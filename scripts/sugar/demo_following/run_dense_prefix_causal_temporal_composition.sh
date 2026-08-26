#!/usr/bin/env bash
# One matched serious temporal-composer run on seen/interleaved handoff prefixes.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/causal_temporal_composition_dense_prefix_seed171648_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "causal temporal composition requires a retained srun compute step" >&2
    exit 2
fi

export POLICY_TOPOLOGY_OVERRIDE=causal_temporal_action_composition
export TRAIN_PREFIXES_CSV_OVERRIDE=33,41,49,57,65
export EVAL_PREFIXES_CSV_OVERRIDE=37,45,53,61
export REQUIRE_DISJOINT_PREFIX_SCHEDULES=1
export TRAIN_SEED_OVERRIDE=171648
export EVAL_SEED_OVERRIDE=181666
export VIDEO_SEED_OVERRIDE=181667

set +e
bash "$ROOT/scripts/sugar/demo_following/run_multi_context_transition_recovery.sh" \
    "$OUTPUT_ROOT" "$DEVICE"
pipeline_rc=$?
seen_rc=0
if [[ "$pipeline_rc" -eq 0 ]]; then
    export SEEN_AUDIT_TRAIN_SEED_OVERRIDE=171648
    export SEEN_AUDIT_EVAL_SEED_OVERRIDE=181666
    export SEEN_AUDIT_POLICY_TOPOLOGY_OVERRIDE=causal_temporal_action_composition
    export SEEN_AUDIT_HOLD_AFTER_OVERRIDE=0
    bash "$ROOT/scripts/sugar/demo_following/run_dense_prefix_seen_context_audit.sh" \
        "$OUTPUT_ROOT" "${OUTPUT_ROOT}_seen_context_audit" "$DEVICE"
    seen_rc=$?
    if [[ "$seen_rc" -eq 0 ]]; then
        "$PYTHON_BIN" \
            "$ROOT/scripts/sugar/demo_following/summarize_temporal_composer_complete_grid.py" \
            --interleaved "$OUTPUT_ROOT/RESULT.json" \
            --seen "${OUTPUT_ROOT}_seen_context_audit/RESULT.json" \
            --output "${OUTPUT_ROOT}_complete_grid/RESULT.json"
        seen_rc=$?
    fi
fi
set -e
mkdir -p "$OUTPUT_ROOT"
if [[ "$seen_rc" -ne 0 ]]; then
    pipeline_rc=1
fi
{
    printf 'pipeline_exit_code=%s\n' "$pipeline_rc"
    printf 'seen_context_exit_code=%s\n' "$seen_rc"
    printf 'seen_context_output=%s\n' "${OUTPUT_ROOT}_seen_context_audit"
    printf 'complete_grid_output=%s\n' "${OUTPUT_ROOT}_complete_grid"
} > "$OUTPUT_ROOT/PIPELINE_STATUS.env"
echo "CAUSAL_TEMPORAL_COMPOSITION_PIPELINE_RC=$pipeline_rc output=$OUTPUT_ROOT" >&2

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_CAUSAL_TEMPORAL_COMPOSITION_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
