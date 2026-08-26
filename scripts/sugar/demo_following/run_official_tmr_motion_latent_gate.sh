#!/usr/bin/env bash
# Run the released TMR semantic feasibility gate, then retain the granted GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_tmr_semantic_gate_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
TMR_ROOT="$ROOT/experiments/runtime_assets/official_tmr"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "official TMR gate requires a retained srun compute step" >&2
    exit 2
fi
for required in \
    "$TMR_ROOT/models/tmr_humanml3d_guoh3dfeats/last_weights/motion_encoder.pt" \
    "$TMR_ROOT/stats/humanml3d/guoh3dfeats/mean.pt" \
    "$TMR_ROOT/runtime_deps/pytorch_lightning"; do
    test -e "$required" || { echo "missing official TMR input: $required" >&2; exit 2; }
done

PYTHONPATH="$TMR_ROOT/runtime_deps:$TMR_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_official_tmr_motion_latent.py" \
    --tmr-root "$TMR_ROOT" --output "$OUTPUT" --device "$DEVICE"

printf 'gate_complete=1\nresult=%s\n' "$OUTPUT/RESULT.json" > "$OUTPUT/STATUS.env"

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_OFFICIAL_TMR_GATE_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
