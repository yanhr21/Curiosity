#!/usr/bin/env bash
# Keep the retained allocation alive after corrected phase gates finish or fail.

set -uo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
job_id=${SLURM_JOB_ID:-unknown}
output_root=${OUTPUT_ROOT:-$ROOT/experiments/demo_following/corrected_phase_runtime_gate_job${job_id}}
lock_path="$ROOT/experiments/runtime_allocations/corrected_phase_gate_pipeline.lock"

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
if flock -n 9; then
    set +e
    OUTPUT_ROOT="$output_root" PYTHON_BIN="$PYTHON_BIN" \
        bash "$ROOT/scripts/sugar/demo_following/run_corrected_phase_runtime_and_carry_gates.sh"
    gate_rc=$?
    set -e
else
    gate_rc=75
    echo "CORRECTED_PHASE_GATE_PIPELINE_SKIPPED lock_busy=$lock_path" >&2
fi
echo "CORRECTED_PHASE_GATE_PIPELINE_RC=$gate_rc output_root=$output_root" >&2

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_CORRECTED_PHASE_GATES_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
