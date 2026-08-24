#!/usr/bin/env bash
# Run the corrected matched pair, evaluate/render it, then retain the allocated GPU.

set -uo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/experiments/demo_following/matched_phase_event_reward_reference_aware_v2}
TRAIN_SEED=${TRAIN_SEED:-161587}
ACTION_SEED=${ACTION_SEED:-161588}
LOCK_PATH="$ROOT/experiments/runtime_allocations/reference_aware_phase_event_pair.lock"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "matched pair requires a retained srun compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing matched policy training on a login host" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if flock -n 9; then
    set +e
    OUTPUT_ROOT="$OUTPUT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        TRAIN_SEED="$TRAIN_SEED" ACTION_SEED="$ACTION_SEED" \
        bash "$ROOT/scripts/sugar/demo_following/run_reference_aware_phase_event_pair.sh"
    pair_rc=$?
    eval_rc=76
    if [[ "$pair_rc" == "0" ]]; then
        bash "$ROOT/scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh" \
            64 phase_event_reward_only "$OUTPUT_ROOT/seed${TRAIN_SEED}"
        eval_rc=$?
    fi
    set -e
    flock -u 9
else
    pair_rc=75
    eval_rc=75
    echo "REFERENCE_AWARE_MATCHED_PAIR_SKIPPED lock_busy=$LOCK_PATH" >&2
fi
echo "REFERENCE_AWARE_MATCHED_PAIR_RC=$pair_rc output_root=$OUTPUT_ROOT" >&2
echo "REFERENCE_AWARE_MATCHED_EVALUATION_RC=$eval_rc output_root=$OUTPUT_ROOT" >&2

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_REFERENCE_AWARE_MATCHED_PAIR_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
