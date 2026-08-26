#!/usr/bin/env bash
# Build the official-TMR mismatch dataset and retain the granted GPU afterward.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_tmr_internal_reward_v1/motion_disjoint_predictor_dataset_suffix_v2}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
TMR_ROOT="$ROOT/experiments/runtime_assets/official_tmr"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "official TMR dataset build requires a retained srun compute step" >&2
    exit 2
fi

PYTHONPATH="$TMR_ROOT/runtime_deps:$TMR_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_reward/build_official_tmr_mismatch_predictor_dataset.py" \
    --tmr-root "$TMR_ROOT" --output-dir "$OUTPUT" --device "$DEVICE"

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_OFFICIAL_TMR_DATASET_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
