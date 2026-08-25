#!/usr/bin/env bash
# Fixed multi-context diagnostic for state-dependent exact Carry/Kick composition.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/causal_action_composition_seed171644_v1}"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

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
pipeline_rc=$?
set -e
mkdir -p "$OUTPUT_ROOT"
printf 'pipeline_exit_code=%s\n' "$pipeline_rc" > "$OUTPUT_ROOT/PIPELINE_STATUS.env"
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
